// Helpers shared by the native benchmarks (shard_bench, ivf_graph_bench): dataset and ground
// truth loaders, hnswlib spaces that count distance evaluations, JSONL output, arguments,
// routers over centroids (exact / HNSW / fastpipnn), k-means and PiPNN-carved centroids.
#pragma once
#include "fastpipnn.hpp"
#include "hnswlib/hnswlib.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_set>

using fastpipnn::Id;
using fastpipnn::now;
using fastpipnn::parallel_for;
using RowMat = fastpipnn::RowMat;

static void require(bool ok, const std::string& m) { if (!ok) throw std::runtime_error(m); }

struct U8 {
  uint32_t n = 0, d = 0;
  std::vector<uint8_t> v;
  const uint8_t* row(size_t i) const { return v.data() + i * d; }
};
static U8 load_u8(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  require(bool(in), "cannot open " + path);
  U8 x;
  in.read(reinterpret_cast<char*>(&x.n), 4); in.read(reinterpret_cast<char*>(&x.d), 4);
  x.v.resize(size_t(x.n) * x.d);
  in.read(reinterpret_cast<char*>(x.v.data()), x.v.size());
  require(bool(in), "truncated " + path);
  return x;
}
struct Truth { uint32_t n = 0, k = 0; std::vector<uint32_t> ids; };
static Truth load_gt(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  require(bool(in), "cannot open " + path);
  Truth t;
  in.read(reinterpret_cast<char*>(&t.n), 4); in.read(reinterpret_cast<char*>(&t.k), 4);
  t.ids.resize(size_t(t.n) * t.k);
  in.read(reinterpret_cast<char*>(t.ids.data()), t.ids.size() * 4);
  require(bool(in), "truncated " + path);
  return t;
}
static std::vector<size_t> parse_list(const std::string& s) {
  std::vector<size_t> out; std::istringstream in(s);
  for (std::string t; std::getline(in, t, ',');) out.push_back(std::stoul(t));
  return out;
}
static uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}
static inline float sq_u8(const uint8_t* a, const uint8_t* b, size_t d) { return fastpipnn::sqdist(a, b, d); }

// Distance evaluations are counted per thread, so routers and HNSW shards report work the
// same way fastpipnn does (one evaluation per vector compared with the query).
static thread_local uint64_t g_evals = 0;
struct ByteSpace : hnswlib::SpaceInterface<float> {
  size_t dim;
  explicit ByteSpace(size_t d) : dim(d) {}
  static float dist(const void* a, const void* b, const void* p) {
    ++g_evals;
    return sq_u8(static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), *static_cast<const size_t*>(p));
  }
  size_t get_data_size() override { return dim; }
  hnswlib::DISTFUNC<float> get_dist_func() override { return dist; }
  void* get_dist_func_param() override { return &dim; }
};
struct FloatSpace : hnswlib::SpaceInterface<float> {
  size_t dim;
  explicit FloatSpace(size_t d) : dim(d) {}
  static float dist(const void* a, const void* b, const void* p) {
    ++g_evals;
    return fastpipnn::sqdist(static_cast<const float*>(a), static_cast<const float*>(b), *static_cast<const size_t*>(p));
  }
  size_t get_data_size() override { return dim * sizeof(float); }
  hnswlib::DISTFUNC<float> get_dist_func() override { return dist; }
  void* get_dist_func_param() override { return &dim; }
};

struct Recorder {
  std::ofstream out;
  explicit Recorder(const std::string& path) : out(path, std::ios::app) { require(bool(out), "cannot write " + path); }
  void write(const std::string& json) { out << json << "\n"; out.flush(); std::cout << json << std::endl; }
};
static std::string num(double x) { std::ostringstream s; s.precision(6); s << x; return s.str(); }
static std::string list_json(const std::vector<double>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + num(v[i]);
  return s + "]";
}
struct Summary { double mean = 0, p50 = 0, p99 = 0; };
static Summary summarize(std::vector<double> v) {
  Summary s; if (v.empty()) return s;
  for (double x : v) s.mean += x; s.mean /= v.size();
  std::sort(v.begin(), v.end());
  s.p50 = v[v.size() / 2]; s.p99 = v[std::min(v.size() - 1, size_t(v.size() * 0.99))];
  return s;
}

struct Args {
  std::map<std::string, std::string> m;
  std::string get(const std::string& k, const std::string& f = "") const { auto it = m.find(k); return it == m.end() ? f : it->second; }
  size_t n(const std::string& k, size_t f) const { return std::stoul(get(k, std::to_string(f))); }
  double f(const std::string& k, double f) const { return std::stod(get(k, num(f))); }
};

// ---------------------------------------------------------------- routers over centroids
struct Router {
  std::string kind;
  size_t C = 0, d = 0;
  const float* cent = nullptr;  // C x d, row-major, owned by the caller
  std::unique_ptr<FloatSpace> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
  fastpipnn::Index<float> fast;
  double build_seconds = 0;

  void build(const std::string& k, const float* c, size_t count, size_t dim, size_t threads, const Args& a) {
    kind = k; cent = c; C = count; d = dim;
    double t = now();
    if (kind == "hnsw") {
      space = std::make_unique<FloatSpace>(d);
      hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), C, a.n("--router-m", 16), a.n("--router-efc", 200), 7);
      hnsw->addPoint(cent, 0);
      parallel_for(threads, C - 1, 64, [&](size_t i) { hnsw->addPoint(cent + (i + 1) * d, i + 1); });
    } else if (kind == "fastpipnn") {
      fastpipnn::Params p;
      p.degree = a.n("--router-degree", 48); p.reservoir = std::max<size_t>(p.reservoir, p.degree);
      p.leaf_size = a.n("--router-leaf", 512);
      p.top_leaders = a.n("--router-leaders", std::max<size_t>(3, std::min<size_t>(1000, C / 60)));
      p.local_k = a.n("--router-local-k", 16); p.alpha = a.f("--router-alpha", 1.5);
      p.fanout = parse_list(a.get("--router-fanout", "10,3")); p.threads = threads;
      fast.build(cent, C, d, d, p);
    } else {
      require(kind == "exact", "router must be exact, hnsw or fastpipnn");
    }
    build_seconds = now() - t;
  }
  // Nearest `probes` centroids, nearest first. `beam` is the HNSW ef / fastpipnn beam.
  std::vector<Id> route(const float* q, size_t probes, size_t beam, uint64_t& evals) const {
    probes = std::min(probes, C);
    std::vector<Id> out;
    if (kind == "exact") {
      std::vector<std::pair<float, Id>> all(C);
      for (size_t j = 0; j < C; ++j) all[j] = {fastpipnn::sqdist(q, cent + j * d, d), Id(j)};
      std::partial_sort(all.begin(), all.begin() + probes, all.end());
      for (size_t j = 0; j < probes; ++j) out.push_back(all[j].second);
      evals += C;
    } else if (kind == "hnsw") {
      g_evals = 0;
      hnsw->setEf(std::max(beam, probes));
      auto r = hnsw->searchKnn(q, probes);
      out.resize(r.size());
      for (size_t i = out.size(); i > 0; --i) { out[i - 1] = Id(r.top().second); r.pop(); }
      evals += g_evals;
    } else {
      size_t e = 0;
      out = fast.search(q, probes, std::max(beam, probes), &e);
      evals += e;
    }
    return out;
  }
};

// ---------------------------------------------------------------- centroid training
static std::vector<size_t> sample_rows(size_t n, size_t m, uint64_t seed) {
  std::vector<size_t> all(n); std::iota(all.begin(), all.end(), 0);
  std::mt19937_64 rng(seed);
  m = std::min(m, n);
  for (size_t i = 0; i < m; ++i) std::swap(all[i], all[i + rng() % (n - i)]);
  all.resize(m); std::sort(all.begin(), all.end());
  return all;
}
// Nearest centroid for each row of X (rows x d floats), by one GEMM per tile.
static void assign_rows(const float* X, size_t rows, const RowMat& C, const std::vector<float>& cn,
                        std::vector<Id>& out, size_t threads) {
  const size_t tile = 1024, d = C.cols(), k = C.rows();
  out.resize(rows);
  parallel_for(threads, (rows + tile - 1) / tile, 1, [&](size_t t) {
    size_t s = t * tile, r = std::min(tile, rows - s);
    RowMat S = Eigen::Map<const RowMat>(X + s * d, r, d) * C.transpose();
    for (size_t i = 0; i < r; ++i) {
      float best = std::numeric_limits<float>::max(); Id bj = 0;
      for (size_t j = 0; j < k; ++j) { float v = cn[j] - 2 * S(i, j); if (v < best) { best = v; bj = Id(j); } }
      out[s + i] = bj;
    }
  });
}
static std::vector<float> norms(const RowMat& C) {
  std::vector<float> n(C.rows());
  for (Eigen::Index j = 0; j < C.rows(); ++j) n[j] = C.row(j).squaredNorm();
  return n;
}
// Lloyd k-means on X; empty clusters are reseeded from random rows.
static RowMat kmeans(const RowMat& X, size_t k, size_t iters, size_t threads, uint64_t seed) {
  std::mt19937_64 rng(seed);
  size_t m = X.rows(), d = X.cols();
  RowMat C(k, d);
  auto init = sample_rows(m, k, seed + 1);
  for (size_t j = 0; j < k; ++j) C.row(j) = X.row(init[j]);
  std::vector<Id> a;
  for (size_t it = 0; it < iters; ++it) {
    assign_rows(X.data(), m, C, norms(C), a, threads);
    Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(k, d);
    std::vector<size_t> cnt(k, 0);
    for (size_t i = 0; i < m; ++i) { sum.row(a[i]) += X.row(i).cast<double>(); ++cnt[a[i]]; }
    for (size_t j = 0; j < k; ++j) {
      if (cnt[j]) C.row(j) = (sum.row(j) / double(cnt[j])).cast<float>();
      else C.row(j) = X.row(rng() % m);
    }
  }
  return C;
}
// PiPNN ball carving on X; one centroid (leaf mean) per leaf. leaf_size is chosen so the
// number of leaves lands near k; the actual count is returned in the matrix size.
static RowMat carve_centroids(const RowMat& X, size_t k, size_t threads, uint64_t seed) {
  size_t m = X.rows(), d = X.cols();
  fastpipnn::Params p;
  p.leaf_size = std::max<size_t>(8, size_t(1.6 * double(m) / double(k)));
  p.fanout = {1}; p.threads = threads; p.seed = seed;
  p.top_leaders = std::min<size_t>(1000, std::max<size_t>(3, k / 2));
  fastpipnn::Index<float> idx;
  auto leaves = idx.carve(X.data(), m, d, d, p);
  RowMat C(leaves.size(), d);
  for (size_t j = 0; j < leaves.size(); ++j) {
    Eigen::VectorXd s = Eigen::VectorXd::Zero(d);
    for (Id i : leaves[j]) s += X.row(i).transpose().cast<double>();
    C.row(j) = (s / double(leaves[j].size())).cast<float>().transpose();
  }
  return C;
}

