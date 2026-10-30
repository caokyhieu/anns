#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include "utils/euclidian_point.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"

using Id = uint32_t;
using Point = parlayANN::Euclidian_Point<uint8_t>;
using Pair = std::pair<float, Id>;
constexpr Id NIL = UINT32_MAX;
inline double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void require(bool b, const std::string& message) {
  if (!b) throw std::runtime_error(message);
}
struct Data {
  using Point = ::Point;
  Point::parameters params;
  uint32_t n=0, dim=0, stride=0;
  std::vector<uint8_t> bytes;
  Data() = default;
  Data(uint32_t count, uint32_t d): params(d), n(count), dim(d), stride((d+63)/64*64),
                                   bytes(size_t(count)*stride) {}
  explicit Data(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot open "+path);
    in.read(reinterpret_cast<char*>(&n),4);
    in.read(reinterpret_cast<char*>(&dim),4);
    require(bool(in) && n>0 && dim>0 && dim<=1024, "invalid u8bin header");
    require(std::filesystem::file_size(path)==8+uint64_t(n)*dim, "u8bin size mismatch: "+path);
    stride=(dim+63)/64*64;
    params=Point::parameters(dim);
    bytes.resize(size_t(n)*stride);
    const size_t block=65536;
    std::vector<uint8_t> temp(block*dim);
    for (size_t start=0; start<n; start+=block) {
      size_t count=std::min(block, size_t(n)-start);
      in.read(reinterpret_cast<char*>(temp.data()), count*dim);
      require(bool(in), "truncated u8bin");
      parlay::parallel_for(0,count,[&](size_t i) {
        std::memcpy(location(start+i),temp.data()+i*dim,dim);
      });
    }
  }
  size_t size() const { return n; }
  size_t dimension() const { return dim; }
  uint32_t get_dims() const { return dim; }
  uint8_t* location(size_t i) const { return const_cast<uint8_t*>(bytes.data()+i*stride); }
  Point operator[](size_t i) const { return Point(location(i),i,params); }
  Id append(const uint8_t* x) {
    require(n<NIL,"ID space exhausted");
    // Caller must not pass a pointer into bytes (which may reallocate).
    bytes.resize(size_t(n+1)*stride);
    std::memcpy(location(n),x,dim);
    return n++;
  }
  float distance(Id a, Id b) const { return (*this)[a].distance((*this)[b]); }
};
struct GroundTruth {
  uint32_t n=0,k=0;
  std::vector<Id> ids;
  explicit GroundTruth(const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    require(bool(in),"cannot open ground truth");
    in.read(reinterpret_cast<char*>(&n),4); in.read(reinterpret_cast<char*>(&k),4);
    require(bool(in) && n>0 && k>=10 && k<=10000,"invalid ground truth header");
    uint64_t size=8+uint64_t(n)*k*4;
    auto actual=std::filesystem::file_size(path);
    require(actual==size || actual==2*size-8,"ground truth must contain IDs, optionally followed by distances");
    ids.resize(size_t(n)*k);
    in.read(reinterpret_cast<char*>(ids.data()),ids.size()*4);
    require(bool(in),"truncated ground truth");
  }
};
inline double peak_mib() { rusage u{}; getrusage(RUSAGE_SELF,&u); return u.ru_maxrss/1024.; }

// Standard best-first bounded-beam graph search. A timestamp array is reused per worker.
struct SearchScratch {
  std::vector<uint32_t> seen;
  uint32_t epoch=0;
  void reset(size_t n) {
    if (seen.size()!=n) {seen.assign(n,0); epoch=0;}
    if (++epoch==0) {std::fill(seen.begin(),seen.end(),0); epoch=1;}
  }
};
template<class Neighbors>
std::vector<Id> graph_search(const Data& data, const uint8_t* query,
                            size_t beam, size_t k, Neighbors neighbors,
                            const std::vector<Id>& seeds) {
  thread_local SearchScratch scratch;
  scratch.reset(data.size());
  Point q(const_cast<uint8_t*>(query),-1,data.params);
  std::priority_queue<Pair> best;
  std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>> pending;
  auto visit=[&](Id p) {
    if (scratch.seen[p]==scratch.epoch) return;
    scratch.seen[p]=scratch.epoch;
    Pair item{data[p].distance(q),p};
    if (best.size()<beam || item<best.top()) {
      best.push(item); pending.push(item);
      if (best.size()>beam) best.pop();
    }
  };
  for (Id p:seeds) visit(p);
  while (!pending.empty()) {
    auto item=pending.top(); pending.pop();
    if (best.size()==beam && item>best.top()) break;
    for (Id p:neighbors(item.second)) visit(p);
  }
  while(best.size()>k) best.pop();
  std::vector<Id> result(best.size());
  for(size_t i=result.size();i>0;--i) {result[i-1]=best.top().second;best.pop();}
  return result;
}
