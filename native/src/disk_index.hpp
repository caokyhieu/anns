#pragma once
// On-disk graph index (DiskANN-style layout) for any graph over uint8 vectors: PiPNN
// (fastpipnn) or the level-0 layer of HNSW. RAM holds only PQ codes, the ID map and an
// optional cache of nodes near the entry point; full vectors and adjacency live on disk.
//
// Node file: 4 KB sectors. Sector 0 is a header; every later sector holds `per_sector`
// fixed-size records [vector (d bytes) | degree (uint32) | R neighbor IDs (uint32)], and a
// record never straddles a sector, so expanding a node costs at most one 4 KB read.
//
// Search (beam search, as in DiskANN): a candidate list of size L ordered by PQ distance;
// each round expands up to W unexpanded candidates, reading their sectors from disk. The
// full vector in each fetched record gives an exact distance, so the final top-k is ranked
// exactly without extra reads. With page expansion (as in Starling), every other node in a
// freshly read sector is expanded too, which pays off when the layout puts neighbours
// together (see order_*).
#include "fastpipnn.hpp"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace diskgraph {

using Id = uint32_t;
constexpr Id kNone = 0xFFFFFFFFu;
constexpr size_t kSector = 4096;

struct Graph {
  size_t n = 0, R = 0;
  Id entry = 0;
  std::vector<uint16_t> deg;
  std::vector<Id> adj;  // n x R
  const Id* nbrs(Id i) const { return adj.data() + size_t(i) * R; }
};

// ---------------------------------------------------------------- product quantization
struct PQ {
  size_t d = 0, M = 0, sub = 0;
  std::vector<float> cent;  // M x 256 x sub
  void train(const uint8_t* X, size_t n, size_t stride, size_t dim, size_t m, size_t sample, size_t iters,
             size_t threads, uint64_t seed) {
    d = dim; M = m;
    if (d % M) throw std::invalid_argument("PQ: dimension must be divisible by M");
    sub = d / M;
    std::mt19937_64 rng(seed);
    sample = std::min(sample, n);
    std::vector<size_t> rows(sample);
    for (auto& r : rows) r = rng() % n;
    cent.assign(M * 256 * sub, 0);
    fastpipnn::parallel_for(threads, M, 1, [&](size_t mi) {
      std::vector<float> x(sample * sub), c(256 * sub), sum(256 * sub);
      std::vector<uint32_t> cnt(256), a(sample);
      for (size_t i = 0; i < sample; ++i)
        for (size_t j = 0; j < sub; ++j) x[i * sub + j] = X[rows[i] * stride + mi * sub + j];
      std::mt19937_64 r2(seed + mi);
      for (size_t k = 0; k < 256; ++k) std::copy_n(&x[(r2() % sample) * sub], sub, &c[k * sub]);
      for (size_t it = 0; it < iters; ++it) {
        for (size_t i = 0; i < sample; ++i) {
          float best = std::numeric_limits<float>::max();
          for (size_t k = 0; k < 256; ++k) {
            float s = fastpipnn::sqdist(&x[i * sub], &c[k * sub], sub);
            if (s < best) { best = s; a[i] = uint32_t(k); }
          }
        }
        std::fill(sum.begin(), sum.end(), 0.f); std::fill(cnt.begin(), cnt.end(), 0);
        for (size_t i = 0; i < sample; ++i) { ++cnt[a[i]]; for (size_t j = 0; j < sub; ++j) sum[a[i] * sub + j] += x[i * sub + j]; }
        for (size_t k = 0; k < 256; ++k) {
          if (cnt[k]) for (size_t j = 0; j < sub; ++j) c[k * sub + j] = sum[k * sub + j] / float(cnt[k]);
          else std::copy_n(&x[(r2() % sample) * sub], sub, &c[k * sub]);
        }
      }
      std::copy(c.begin(), c.end(), cent.begin() + mi * 256 * sub);
    });
  }
  void encode(const uint8_t* X, size_t n, size_t stride, uint8_t* codes, size_t threads) const {
    fastpipnn::parallel_for(threads, n, 1024, [&](size_t i) {
      float x[512];
      for (size_t j = 0; j < d; ++j) x[j] = X[i * stride + j];
      for (size_t mi = 0; mi < M; ++mi) {
        float best = std::numeric_limits<float>::max(); uint8_t bk = 0;
        for (size_t k = 0; k < 256; ++k) {
          float s = fastpipnn::sqdist(x + mi * sub, &cent[(mi * 256 + k) * sub], sub);
          if (s < best) { best = s; bk = uint8_t(k); }
        }
        codes[i * M + mi] = bk;
      }
    });
  }
  void table(const uint8_t* q, float* tab) const {  // M x 256 partial distances
    for (size_t mi = 0; mi < M; ++mi)
      for (size_t k = 0; k < 256; ++k) {
        float s = 0;
        for (size_t j = 0; j < sub; ++j) { float t = float(q[mi * sub + j]) - cent[(mi * 256 + k) * sub + j]; s += t * t; }
        tab[mi * 256 + k] = s;
      }
  }
};

// ---------------------------------------------------------------- node orderings (layout)
// Returns perm with perm[new position] = old ID.
inline std::vector<Id> order_identity(size_t n) { std::vector<Id> p(n); std::iota(p.begin(), p.end(), Id(0)); return p; }
inline std::vector<Id> order_bfs(const Graph& g) {
  std::vector<Id> p; p.reserve(g.n);
  std::vector<uint8_t> seen(g.n, 0);
  auto bfs = [&](Id s) {
    size_t head = p.size(); p.push_back(s); seen[s] = 1;
    while (head < p.size()) {
      Id u = p[head++];
      for (size_t j = 0; j < g.deg[u]; ++j) { Id v = g.nbrs(u)[j]; if (!seen[v]) { seen[v] = 1; p.push_back(v); } }
    }
  };
  bfs(g.entry);
  for (Id i = 0; i < g.n; ++i) if (!seen[i]) bfs(i);
  return p;
}
// PiPNN ball carving with small leaves; leaves are laid out one after another, so vectors
// that are close in space share sectors.
inline std::vector<Id> order_carve(const uint8_t* X, size_t n, size_t d, size_t leaf, size_t threads, uint64_t seed) {
  fastpipnn::Params p;
  p.leaf_size = leaf; p.fanout = {1}; p.threads = threads; p.seed = seed;
  fastpipnn::Index<uint8_t> idx;
  auto leaves = idx.carve(X, n, d, d, p);
  std::vector<Id> perm; perm.reserve(n);
  std::vector<uint8_t> seen(n, 0);
  for (auto& l : leaves) for (Id i : l) if (!seen[i]) { seen[i] = 1; perm.push_back(i); }
  for (Id i = 0; i < n; ++i) if (!seen[i]) perm.push_back(i);
  return perm;
}

// ---------------------------------------------------------------- node file
struct Layout {
  size_t n = 0, d = 0, R = 0, rec = 0, per_sector = 0, sectors = 0;
  Id entry = 0;
  void init(size_t count, size_t dim, size_t degree) {
    n = count; d = dim; R = degree; rec = d + 4 + 4 * R; per_sector = kSector / rec;
    if (per_sector == 0) throw std::invalid_argument("record larger than a sector");
    sectors = (n + per_sector - 1) / per_sector;
  }
  size_t sector_of(Id i) const { return 1 + i / per_sector; }
  size_t slot_of(Id i) const { return i % per_sector; }
  uint64_t bytes() const { return uint64_t(1 + sectors) * kSector; }
};

inline void write_nodes(const std::string& path, const uint8_t* X, size_t stride, const Graph& g,
                        const std::vector<Id>& perm, Layout& L) {
  std::vector<Id> inv(g.n);
  for (size_t p = 0; p < g.n; ++p) inv[perm[p]] = Id(p);
  L.entry = inv[g.entry];
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  std::vector<uint8_t> sec(kSector, 0);
  uint64_t head[8] = {0x444b47524150484eull, L.n, L.d, L.R, L.entry, L.per_sector, L.rec, 0};
  std::memcpy(sec.data(), head, sizeof(head));
  bool ok = std::fwrite(sec.data(), 1, kSector, f) == kSector;
  for (size_t s = 0; s < L.sectors && ok; ++s) {
    std::fill(sec.begin(), sec.end(), 0);
    for (size_t j = 0; j < L.per_sector; ++j) {
      size_t p = s * L.per_sector + j;
      if (p >= g.n) break;
      Id old = perm[p];
      uint8_t* r = sec.data() + j * L.rec;
      std::memcpy(r, X + size_t(old) * stride, L.d);
      uint32_t deg = g.deg[old];
      std::memcpy(r + L.d, &deg, 4);
      Id* nb = reinterpret_cast<Id*>(r + L.d + 4);
      for (size_t k = 0; k < L.R; ++k) nb[k] = k < deg ? inv[g.nbrs(old)[k]] : kNone;
    }
    ok = std::fwrite(sec.data(), 1, kSector, f) == kSector;
  }
  ok = (std::fclose(f) == 0) && ok;
  if (!ok) throw std::runtime_error("short write " + path);
}

// ---------------------------------------------------------------- searcher
struct SearchStats { uint64_t reads = 0, cache_hits = 0, expanded = 0, pq_evals = 0, rounds = 0; };

class DiskIndex {
 public:
  Layout L;
  PQ pq;
  std::vector<uint8_t> codes;   // n x M, in file order
  std::vector<Id> orig;         // file position -> original ID
  std::string io = "direct";    // direct (O_DIRECT), buffered (page cache) or ram (file copy in RAM)
  int fd = -1;
  std::vector<uint8_t> ram;     // io == ram: whole node file
  std::unordered_map<Id, uint32_t> cache_pos;  // cached node -> offset into cache_recs
  std::vector<uint8_t> cache_recs;

  ~DiskIndex() { if (fd >= 0) ::close(fd); }

  void open(const std::string& path, const std::string& mode) {
    io = mode;
    if (fd >= 0) { ::close(fd); fd = -1; }
    ram.clear();
    if (io == "ram") {
      std::FILE* f = std::fopen(path.c_str(), "rb");
      if (!f) throw std::runtime_error("cannot read " + path);
      ram.resize(L.bytes());
      bool ok = std::fread(ram.data(), 1, ram.size(), f) == ram.size();
      std::fclose(f);
      if (!ok) throw std::runtime_error("short read " + path);
      return;
    }
    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (io == "direct") flags |= O_DIRECT;
#endif
    fd = ::open(path.c_str(), flags);
    if (fd < 0 && io == "direct") throw std::runtime_error("O_DIRECT open failed: " + std::string(std::strerror(errno)));
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    if (io == "buffered") posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
  }
  // Keep the records of the `count` nodes nearest the entry point (BFS order) in RAM.
  void build_cache(size_t count) {
    cache_pos.clear(); cache_recs.clear();
    if (count == 0) return;
    std::vector<Id> q{L.entry};
    std::vector<uint8_t> buf(kSector + kSector);
    uint8_t* sec = align(buf.data());
    std::unordered_map<Id, uint8_t> seen{{L.entry, 1}};
    for (size_t head = 0; head < q.size() && cache_pos.size() < count; ++head) {
      Id u = q[head];
      read_sector(L.sector_of(u), sec);
      const uint8_t* r = sec + L.slot_of(u) * L.rec;
      cache_pos[u] = uint32_t(cache_recs.size());
      cache_recs.insert(cache_recs.end(), r, r + L.rec);
      uint32_t deg; std::memcpy(&deg, r + L.d, 4);
      const Id* nb = reinterpret_cast<const Id*>(r + L.d + 4);
      for (size_t k = 0; k < deg; ++k) if (seen.emplace(nb[k], 1).second) q.push_back(nb[k]);
    }
  }
  uint64_t ram_bytes() const {
    return codes.size() + orig.size() * sizeof(Id) + pq.cent.size() * sizeof(float) + cache_recs.size() +
           cache_pos.size() * 16;
  }

  // Returns up to k (exact distance, original ID) pairs, nearest first.
  std::vector<std::pair<float, Id>> search(const uint8_t* q, size_t k, size_t Lsize, size_t W, bool page_expand,
                                           SearchStats& st) const {
    Scratch& s = scratch();
    s.reset(Lsize, L.R);
    pq.table(q, s.tab.data());
    auto adc = [&](Id i) {
      const uint8_t* c = codes.data() + size_t(i) * pq.M;
      float sum = 0;
      for (size_t m = 0; m < pq.M; ++m) sum += s.tab[m * 256 + c[m]];
      ++st.pq_evals;
      return sum;
    };
    auto insert = [&](float dist, Id id) {
      auto& cs = s.cand;
      if (cs.size() == Lsize && dist >= cs.back().dist) return;
      auto it = std::upper_bound(cs.begin(), cs.end(), dist, [](float v, const Cand& c) { return v < c.dist; });
      cs.insert(it, Cand{dist, id, false});
      if (cs.size() > Lsize) cs.pop_back();
    };
    auto expand = [&](Id id, const uint8_t* r) {
      ++st.expanded;
      s.result.push_back({fastpipnn::sqdist(r, q, L.d), orig[id]});
      uint32_t deg; std::memcpy(&deg, r + L.d, 4);
      const Id* nb = reinterpret_cast<const Id*>(r + L.d + 4);
      for (size_t j = 0; j < deg; ++j) {
        Id v = nb[j];
        if (s.state(v) != 0) continue;
        s.set(v, 1);
        insert(adc(v), v);
      }
    };
    s.set(L.entry, 1);
    insert(adc(L.entry), L.entry);
    std::vector<Id> pick;
    for (;;) {
      pick.clear();
      for (auto& c : s.cand) {
        if (!c.expanded) { c.expanded = true; pick.push_back(c.id); if (pick.size() == W) break; }
      }
      if (pick.empty()) break;
      ++st.rounds;
      // Fetch the sectors this round needs (cached nodes and sectors already read this query
      // cost nothing), then expand.
      std::vector<size_t> fresh;
      for (Id id : pick) {
        if (s.state(id) == 2) continue;
        if (cache_pos.count(id)) continue;
        size_t sec = L.sector_of(id);
        if (!s.sectors.count(sec)) { s.sectors[sec] = s.buffer(s.sectors.size()); read_sector(sec, s.sectors[sec]); ++st.reads; fresh.push_back(sec); }
      }
      for (Id id : pick) {
        if (s.state(id) == 2) continue;
        s.set(id, 2);
        auto c = cache_pos.find(id);
        if (c != cache_pos.end()) { ++st.cache_hits; expand(id, cache_recs.data() + c->second); continue; }
        expand(id, s.sectors[L.sector_of(id)] + L.slot_of(id) * L.rec);
      }
      if (page_expand) {
        for (size_t sec : fresh) {
          const uint8_t* buf = s.sectors[sec];
          for (size_t j = 0; j < L.per_sector; ++j) {
            Id id = Id((sec - 1) * L.per_sector + j);
            if (id >= L.n || s.state(id) == 2) continue;
            s.set(id, 2);
            for (auto& c : s.cand) if (c.id == id) { c.expanded = true; break; }
            expand(id, buf + j * L.rec);
          }
        }
      }
    }
    auto& res = s.result;
    std::sort(res.begin(), res.end());
    res.erase(std::unique(res.begin(), res.end(), [](auto& a, auto& b) { return a.second == b.second; }), res.end());
    if (res.size() > k) res.resize(k);
    return res;
  }

 private:
  struct Cand { float dist; Id id; bool expanded; };
  // Per-thread scratch: candidate list, open-addressing visit states, sector buffers.
  struct Scratch {
    std::vector<Cand> cand;
    std::vector<std::pair<float, Id>> result;
    std::vector<float> tab;
    std::vector<Id> keys; std::vector<uint32_t> tags; std::vector<uint8_t> vals; uint32_t epoch = 0; size_t mask = 0;
    std::unordered_map<size_t, uint8_t*> sectors;
    std::vector<std::vector<uint8_t>> bufs;
    void reset(size_t Lsize, size_t R) {
      cand.clear(); cand.reserve(Lsize + 1); result.clear(); sectors.clear();
      tab.resize(256 * 512);  // up to M = 512 sub-quantizers
      size_t want = 1; while (want < 16 * Lsize * std::max<size_t>(R, 8)) want <<= 1;
      if (keys.size() != want) { keys.assign(want, 0); tags.assign(want, 0); vals.assign(want, 0); epoch = 0; mask = want - 1; }
      if (++epoch == 0) { std::fill(tags.begin(), tags.end(), 0); epoch = 1; }
    }
    size_t find(Id id) const {
      size_t h = (size_t(id) * 0x9E3779B97F4A7C15ull) >> 20 & mask;
      while (tags[h] == epoch && keys[h] != id) h = (h + 1) & mask;
      return h;
    }
    uint8_t state(Id id) const { size_t h = find(id); return tags[h] == epoch ? vals[h] : 0; }
    void set(Id id, uint8_t v) { size_t h = find(id); keys[h] = id; tags[h] = epoch; vals[h] = v; }
    uint8_t* buffer(size_t i) {
      while (bufs.size() <= i) bufs.emplace_back(2 * kSector);
      return align(bufs[i].data());
    }
  };
  static uint8_t* align(uint8_t* p) { return reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(p) + kSector - 1) & ~uintptr_t(kSector - 1)); }
  static Scratch& scratch() { thread_local Scratch s; return s; }
  void read_sector(size_t sec, uint8_t* dst) const {
    if (io == "ram") { std::memcpy(dst, ram.data() + sec * kSector, kSector); return; }
    ssize_t got = ::pread(fd, dst, kSector, off_t(sec * kSector));
    if (got != ssize_t(kSector)) throw std::runtime_error("pread failed: " + std::string(std::strerror(errno)));
  }
};

}  // namespace diskgraph
