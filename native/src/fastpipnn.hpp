#pragma once
// From-scratch PiPNN graph build (arXiv:2602.21247), written against the paper and the
// authors' reference code (ParAlg/PiPNN) but sharing no code with it:
//   1. randomized ball carving: random leaders, GEMM routing to the f nearest leaders,
//      small-cluster merging, recursion until every leaf has <= leaf_size points;
//   2. leaves: one GEMM distance matrix per leaf, bidirected k-NN candidate edges;
//   3. HashPrune: per-point reservoir keyed by a sign-of-random-projection hash of q - p,
//      keeping the nearest candidate per hash bucket and evicting the farthest when full;
//   4. RobustPrune of each reservoir to the final degree.
// Unlike incremental builders (HNSW, Vamana), no step searches the partial graph, so
// all work is dense GEMMs plus independent per-point updates.
// Dependencies: C++17, std::thread, Eigen (header-only). Distances are squared L2.
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace fastpipnn {

using Id = uint32_t;
constexpr Id kNone = std::numeric_limits<Id>::max();
using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Bucket = std::vector<Id>;

struct Params {
  size_t leaf_size = 1024;              // maximum leaf size
  size_t min_leaf = 0;                  // clusters below this are merged; 0 means leaf_size/10
  size_t local_k = 2;                   // k of the leaf k-NN (edges are added in both directions)
  std::vector<size_t> fanout{10, 3};    // fanout per partition depth; deeper levels use 1
  size_t top_leaders = 1000;            // leaders sampled at depth 0
  double leader_fraction = 0.005;       // leaders at deeper levels, as a fraction of the bucket
  size_t max_leaders = 512;            // cap at every depth (the reference code uses 512)
  size_t max_depth = 8;                 // deeper buckets are split randomly (duplicate guard)
  size_t reservoir = 160;               // HashPrune reservoir capacity per point
  size_t hash_bits = 12;
  size_t degree = 64;                   // final out-degree bound R
  double alpha = 1.2;                   // RobustPrune alpha, applied to squared distances like upstream
  size_t threads = 0;                   // 0 means std::thread::hardware_concurrency()
  uint64_t seed = 555;
};

struct Stats {
  double sketch_seconds = 0, partition_seconds = 0, leaf_seconds = 0, prune_seconds = 0;
  double total_seconds = 0;
  size_t leaves = 0, leaf_points = 0, max_leaf = 0, depth = 0, random_splits = 0;
  uint64_t offers = 0;
  double mean_reservoir = 0, mean_degree = 0;
};

inline double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline float sqdist(const float* a, const float* b, size_t d) {
  float s = 0;
  for (size_t i = 0; i < d; ++i) { float t = a[i] - b[i]; s += t * t; }
  return s;
}
inline float sqdist(const uint8_t* a, const uint8_t* b, size_t d) {
  int32_t s = 0;  // exact for d <= 33,000
  for (size_t i = 0; i < d; ++i) { int32_t t = int32_t(a[i]) - int32_t(b[i]); s += t * t; }
  return float(s);
}
template <class T> inline void load(const T* src, float* dst, size_t d) {
  for (size_t i = 0; i < d; ++i) dst[i] = float(src[i]);
}

// Dynamic-scheduling parallel loop over [0, n) in chunks of `grain`.
template <class F> void parallel_for(size_t threads, size_t n, size_t grain, F&& f) {
  grain = std::max<size_t>(grain, 1);
  size_t workers = std::min(threads, (n + grain - 1) / grain);
  if (workers <= 1) { for (size_t i = 0; i < n; ++i) f(i); return; }
  std::atomic<size_t> next{0};
  auto work = [&] {
    for (;;) {
      size_t start = next.fetch_add(grain, std::memory_order_relaxed);
      if (start >= n) return;
      for (size_t i = start, end = std::min(n, start + grain); i < end; ++i) f(i);
    }
  };
  std::vector<std::thread> pool;
  for (size_t t = 1; t < workers; ++t) pool.emplace_back(work);
  work();
  for (auto& t : pool) t.join();
}

// Point closest to the dataset mean: a single well-connected search entry.
template <class T> Id medoid(const T* data, size_t n, size_t dim, size_t stride, size_t threads) {
  std::vector<double> mean(dim, 0);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < dim; ++j) mean[j] += data[i * stride + j];
  std::vector<float> center(dim);
  for (size_t j = 0; j < dim; ++j) center[j] = float(mean[j] / double(n));
  constexpr size_t chunk = 4096;
  size_t chunks = (n + chunk - 1) / chunk;
  std::vector<std::pair<float, Id>> best(chunks, {std::numeric_limits<float>::max(), 0});
  parallel_for(threads, chunks, 1, [&](size_t c) {
    std::vector<float> x(dim);
    for (size_t i = c * chunk; i < std::min(n, (c + 1) * chunk); ++i) {
      load(data + i * stride, x.data(), dim);
      best[c] = std::min(best[c], {sqdist(x.data(), center.data(), dim), Id(i)});
    }
  });
  return std::min_element(best.begin(), best.end())->second;
}

template <class T> class Index {
 public:
  struct Neighbors {
    const Id* b; const Id* e;
    const Id* begin() const { return b; }
    const Id* end() const { return e; }
    size_t size() const { return e - b; }
  };

  Params params;
  Stats stats;
  size_t n = 0, dim = 0, stride = 0;
  const T* data = nullptr;  // not owned; must outlive the index
  Id entry = 0;

  Neighbors neighbors(Id i) const {
    const Id* b = adjacency.data() + size_t(i) * params.degree;
    return {b, b + degrees[i]};
  }
  const T* point(Id i) const { return data + size_t(i) * stride; }

  void build(const T* points, size_t count, size_t dimension, size_t row_stride, Params p) {
    if (count < 2 || dimension == 0 || row_stride < dimension) throw std::invalid_argument("need >= 2 points");
    if (count >= kNone) throw std::invalid_argument("too many points for 32-bit IDs");
    if (p.leaf_size < 2 || p.local_k < 1 || p.hash_bits < 1 || p.hash_bits > 16 ||
        p.reservoir < 1 || p.reservoir > 65535 || p.degree < 1 || p.degree > 65535 ||
        p.top_leaders < 3 || p.max_leaders < 3 || !(p.alpha >= 1) || !(p.leader_fraction > 0) ||
        std::any_of(p.fanout.begin(), p.fanout.end(), [](size_t f) { return f < 1 || f > 64; }))
      throw std::invalid_argument("invalid PiPNN parameters");
    if (p.min_leaf == 0) p.min_leaf = std::max<size_t>(1, p.leaf_size / 10);
    if (p.threads == 0) p.threads = std::max(1u, std::thread::hardware_concurrency());
    params = p; stats = Stats{};
    n = count; dim = dimension; stride = row_stride; data = points;
    double start = now(), t = start;

    compute_sketches();
    slots = (params.reservoir + 31) / 32 * 32;  // whole 64-byte lines of hashes per point
    slot_hash.reset(new uint16_t[n * slots]);       // uninitialized: header sizes guard reads
    slot_dist.reset(new uint16_t[n * slots]);
    slot_id.reset(new Id[n * slots]);
    headers.reset(new Header[n]);
    parallel_for(params.threads, n, 4096, [&](size_t i) {
      headers[i].lock.store(0, std::memory_order_relaxed);
      headers[i].size = headers[i].worst = 0;
    });
    stats.sketch_seconds = now() - t; t = now();

    std::vector<Bucket> leaves = partition();
    stats.partition_seconds = now() - t; t = now();

    std::sort(leaves.begin(), leaves.end(), [](const Bucket& a, const Bucket& b) { return a.size() > b.size(); });
    std::atomic<uint64_t> offers{0};
    parallel_for(params.threads, leaves.size(), 1, [&](size_t i) {
      offers.fetch_add(build_leaf(leaves[i]), std::memory_order_relaxed);
    });
    stats.offers = offers.load();
    stats.leaves = leaves.size();
    for (auto& leaf : leaves) {
      stats.leaf_points += leaf.size();
      stats.max_leaf = std::max(stats.max_leaf, leaf.size());
    }
    leaves = {};
    stats.leaf_seconds = now() - t; t = now();

    prune_all();
    entry = medoid(data, n, dim, stride, params.threads);
    stats.prune_seconds = now() - t;
    stats.total_seconds = now() - start;
  }

  // Best-first beam search from the medoid. Returns up to k IDs, nearest first.
  std::vector<Id> search(const T* query, size_t k, size_t beam, size_t* evaluations = nullptr) const {
    thread_local std::vector<uint32_t> seen;
    thread_local uint32_t epoch = 0;
    if (seen.size() != n) { seen.assign(n, 0); epoch = 0; }
    if (++epoch == 0) { std::fill(seen.begin(), seen.end(), 0); epoch = 1; }
    using Pair = std::pair<float, Id>;
    std::priority_queue<Pair> best;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> frontier;
    size_t evals = 0;
    auto visit = [&](Id p) {
      if (seen[p] == epoch) return;
      seen[p] = epoch; ++evals;
      Pair item{sqdist(point(p), query, dim), p};
      if (best.size() < beam || item < best.top()) {
        best.push(item); frontier.push(item);
        if (best.size() > beam) best.pop();
      }
    };
    visit(entry);
    while (!frontier.empty()) {
      Pair item = frontier.top(); frontier.pop();
      if (best.size() == beam && item > best.top()) break;
      for (Id q : neighbors(item.second)) visit(q);
    }
    while (best.size() > k) best.pop();
    std::vector<Id> out(best.size());
    for (size_t i = out.size(); i > 0; --i) { out[i - 1] = best.top().second; best.pop(); }
    if (evaluations) *evaluations = evals;
    return out;
  }

 private:
  // HashPrune reservoirs, structure-of-arrays with `slots` entries per point: hash, distance
  // (top 16 bits of a nonnegative float, i.e. order-preserving bfloat16 truncation) and ID.
  // Slots are unsorted; a hash bucket is found with a SIMD compare over the hash row.
  struct Header { std::atomic<uint8_t> lock; uint16_t size, worst; };
  size_t slots = 0;
  std::unique_ptr<Header[]> headers;
  std::unique_ptr<uint16_t[]> slot_hash, slot_dist;
  std::unique_ptr<Id[]> slot_id;

  static constexpr size_t kSketchStride = 16;  // one 64-byte line per point, zero padded
  std::vector<float> sketches;                  // n x kSketchStride random projections
  std::vector<Id> adjacency;
  std::vector<uint16_t> degrees;

  // Total order (distance, then ID) makes HashPrune history-independent, so the
  // multithreaded build is deterministic even though offers arrive in any order.
  static uint64_t key(uint16_t d, Id id) { return uint64_t(d) << 32 | id; }

  static uint16_t bf16(float d) {
    if (!(d > 0)) return 0;  // also maps -0 and NaN to 0
    uint32_t bits; std::memcpy(&bits, &d, 4);
    return uint16_t(bits >> 16);
  }

  void compute_sketches() {
    const size_t bits = params.hash_bits;
    RowMat planes(bits, dim);
    std::mt19937_64 rng(params.seed);
    std::normal_distribution<float> normal;
    for (size_t i = 0; i < bits * dim; ++i) planes.data()[i] = normal(rng);
    sketches.assign(n * kSketchStride, 0.f);
    constexpr size_t chunk = 2048;
    parallel_for(params.threads, (n + chunk - 1) / chunk, 1, [&](size_t c) {
      size_t begin = c * chunk, rows = std::min(n, begin + chunk) - begin;
      RowMat x(rows, dim);
      for (size_t r = 0; r < rows; ++r) load(point(Id(begin + r)), x.data() + r * dim, dim);
      Eigen::Map<RowMat, 0, Eigen::OuterStride<>>(sketches.data() + begin * kSketchStride, rows, bits,
                                                  Eigen::OuterStride<>(kSketchStride)).noalias() = x * planes.transpose();
    });
  }

  // Bit b of hash(p, q) is the sign of <r_b, q - p>, i.e. sketch_b(q) > sketch_b(p).
  // Padding lanes are zero in both rows and so contribute 0 bits.
  uint16_t relative_hash(Id p, Id q) const {
    const float* a = sketches.data() + size_t(p) * kSketchStride;
    const float* c = sketches.data() + size_t(q) * kSketchStride;
#if defined(__AVX512F__)
    return uint16_t(_mm512_cmp_ps_mask(_mm512_loadu_ps(c), _mm512_loadu_ps(a), _CMP_GT_OQ));
#else
    uint16_t h = 0;
    for (size_t b = 0; b < params.hash_bits; ++b) h = uint16_t(h | (c[b] > a[b]) << b);
    return h;
#endif
  }

  void lock(Id p) {
    auto& l = headers[p].lock;
    while (l.exchange(1, std::memory_order_acquire))
      while (l.load(std::memory_order_relaxed)) {
#if defined(__x86_64__)
        _mm_pause();
#endif
      }
  }
  void unlock(Id p) { headers[p].lock.store(0, std::memory_order_release); }

  void recompute_worst(Id p) {
    const uint16_t* dist = slot_dist.get() + size_t(p) * slots;
    const Id* ids = slot_id.get() + size_t(p) * slots;
    size_t w = 0;
    for (size_t i = 1; i < headers[p].size; ++i) if (key(dist[i], ids[i]) > key(dist[w], ids[w])) w = i;
    headers[p].worst = uint16_t(w);
  }

  static size_t find_hash(const uint16_t* hashes, size_t count, uint16_t h) {
#if defined(__AVX512BW__)
    const __m512i v = _mm512_set1_epi16(short(h));
    for (size_t i = 0; i < count; i += 32) {
      __mmask32 live = count - i >= 32 ? ~__mmask32(0) : __mmask32((1u << (count - i)) - 1);
      __mmask32 hit = _mm512_mask_cmpeq_epi16_mask(live, _mm512_maskz_loadu_epi16(live, hashes + i), v);
      if (hit) return i + __builtin_ctz(hit);
    }
    return count;
#else
    size_t i = 0;
    while (i < count && hashes[i] != h) ++i;
    return i;
#endif
  }

  void prefetch_reservoir(Id p) const {
    __builtin_prefetch(&headers[p], 1, 3);
    __builtin_prefetch(slot_hash.get() + size_t(p) * slots, 1, 3);
    __builtin_prefetch(slot_dist.get() + size_t(p) * slots, 1, 3);
    __builtin_prefetch(slot_id.get() + size_t(p) * slots, 1, 3);
  }

  // HashPrune insertion of candidate q (at bf16 distance d) into p's reservoir; caller holds
  // p's lock. Hashes are unique per reservoir: a collision keeps the nearer candidate, and a
  // full reservoir evicts its farthest candidate.
  void offer_locked(Id p, Id q, uint16_t d) {
    Header& head = headers[p];
    uint16_t* hashes = slot_hash.get() + size_t(p) * slots;
    uint16_t* dist = slot_dist.get() + size_t(p) * slots;
    Id* ids = slot_id.get() + size_t(p) * slots;
    const size_t count = head.size;
    const uint64_t k = key(d, q);
    if (count == params.reservoir && k >= key(dist[head.worst], ids[head.worst])) return;
    const uint16_t h = relative_hash(p, q);
    size_t at = find_hash(hashes, count, h);
    if (at < count) {
      if (k < key(dist[at], ids[at])) {
        dist[at] = d; ids[at] = q;
        if (at == head.worst) recompute_worst(p);
      }
    } else if (count < params.reservoir) {
      hashes[at] = h; dist[at] = d; ids[at] = q;
      head.size = uint16_t(count + 1);
      if (count == 0 || k > key(dist[head.worst], ids[head.worst])) head.worst = uint16_t(at);
    } else {
      at = head.worst;
      hashes[at] = h; dist[at] = d; ids[at] = q;
      recompute_worst(p);
    }
  }

  // Level-synchronous randomized ball carving. Returns the leaves.
  std::vector<Bucket> partition() {
    std::vector<Bucket> leaves, active(1);
    active[0].resize(n);
    std::iota(active[0].begin(), active[0].end(), Id(0));
    struct Plan { size_t leaders = 0, fanout = 1; std::vector<float> coords, norms; std::vector<uint16_t> route; };
    constexpr size_t chunk = 1024;
    std::atomic<size_t> random_splits{0};
    for (size_t depth = 0; !active.empty(); ++depth) {
      std::vector<Bucket> work;
      for (auto& b : active) (b.size() <= params.leaf_size ? leaves : work).push_back(std::move(b));
      active.clear();
      if (work.empty()) break;
      stats.depth = depth + 1;
      std::vector<Plan> plans(work.size());
      parallel_for(params.threads, work.size(), 1, [&](size_t w) {
        const Bucket& b = work[w];
        Plan& plan = plans[w];
        size_t count = depth == 0 ? params.top_leaders : size_t(double(b.size()) * params.leader_fraction);
        count = std::min({std::max<size_t>(count, 3), params.max_leaders, b.size(), size_t(65535)});
        plan.leaders = count;
        plan.fanout = std::min(depth < params.fanout.size() ? params.fanout[depth] : 1, count);
        std::mt19937_64 rng(params.seed ^ (depth * 0x9E3779B97F4A7C15ull) ^ (w * 0xBF58476D1CE4E5B9ull));
        Bucket leaders(count);
        std::sample(b.begin(), b.end(), leaders.begin(), count, rng);
        plan.coords.resize(count * dim);
        plan.norms.resize(count);
        for (size_t j = 0; j < count; ++j) {
          load(point(leaders[j]), plan.coords.data() + j * dim, dim);
          plan.norms[j] = Eigen::Map<Eigen::VectorXf>(plan.coords.data() + j * dim, dim).squaredNorm();
        }
        plan.route.resize(b.size() * plan.fanout);
      });
      // Route every point to its `fanout` nearest leaders: argmin |l|^2 - 2<x, l> from one GEMM per tile.
      std::vector<std::pair<uint32_t, size_t>> tiles;
      for (size_t w = 0; w < work.size(); ++w)
        for (size_t s = 0; s < work[w].size(); s += chunk) tiles.push_back({uint32_t(w), s});
      parallel_for(params.threads, tiles.size(), 1, [&](size_t t) {
        auto [w, s] = tiles[t];
        const Bucket& b = work[w];
        Plan& plan = plans[w];
        size_t rows = std::min(chunk, b.size() - s), m = plan.leaders, f = plan.fanout;
        thread_local std::vector<float> x, scores;
        x.resize(rows * dim); scores.resize(rows * m);
        for (size_t r = 0; r < rows; ++r) load(point(b[s + r]), x.data() + r * dim, dim);
        Eigen::Map<RowMat> S(scores.data(), rows, m);
        S.noalias() = Eigen::Map<const RowMat>(x.data(), rows, dim) *
                      Eigen::Map<const RowMat>(plan.coords.data(), m, dim).transpose();
        float best_d[64]; uint16_t best_j[64];
        for (size_t r = 0; r < rows; ++r) {
          const float* row = scores.data() + r * m;
          size_t filled = 0;
          for (size_t j = 0; j < m; ++j) {
            float d = plan.norms[j] - 2 * row[j];
            if (filled == f && d >= best_d[f - 1]) continue;
            size_t at = filled < f ? filled++ : f - 1;
            while (at > 0 && best_d[at - 1] > d) { best_d[at] = best_d[at - 1]; best_j[at] = best_j[at - 1]; --at; }
            best_d[at] = d; best_j[at] = uint16_t(j);
          }
          std::copy(best_j, best_j + f, plan.route.begin() + (s + r) * f);
        }
      });
      std::vector<std::vector<Bucket>> next(work.size());
      parallel_for(params.threads, work.size(), 1, [&](size_t w) {
        const Bucket& b = work[w];
        Plan& plan = plans[w];
        std::vector<Bucket> clusters(plan.leaders);
        std::vector<size_t> counts(plan.leaders, 0);
        for (uint16_t j : plan.route) ++counts[j];
        for (size_t j = 0; j < plan.leaders; ++j) clusters[j].reserve(counts[j]);
        for (size_t i = 0; i < b.size(); ++i)
          for (size_t k = 0; k < plan.fanout; ++k) clusters[plan.route[i * plan.fanout + k]].push_back(b[i]);
        plan = Plan{};
        clusters.erase(std::remove_if(clusters.begin(), clusters.end(), [](const Bucket& c) { return c.empty(); }),
                       clusters.end());
        std::sort(clusters.begin(), clusters.end(), [](const Bucket& a, const Bucket& c) { return a.size() < c.size(); });
        // Greedily merge the smallest clusters while they stay within leaf_size.
        std::vector<Bucket> children;
        size_t i = 0;
        while (i < clusters.size() && clusters[i].size() < params.min_leaf) {
          Bucket merged = std::move(clusters[i++]);
          while (i < clusters.size() && merged.size() < params.min_leaf &&
                 merged.size() + clusters[i].size() <= params.leaf_size) {
            merged.insert(merged.end(), clusters[i].begin(), clusters[i].end());
            ++i;
          }
          std::sort(merged.begin(), merged.end());  // fanout can place a point in several clusters
          merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
          children.push_back(std::move(merged));
        }
        for (; i < clusters.size(); ++i) children.push_back(std::move(clusters[i]));
        std::mt19937_64 rng(params.seed + 31 * depth + w);
        for (auto& c : children) {
          // No progress (e.g. duplicate vectors) or too deep: split randomly into leaves.
          if (c.size() > params.leaf_size && (c.size() == b.size() || depth + 1 >= params.max_depth)) {
            random_splits.fetch_add(1, std::memory_order_relaxed);
            std::shuffle(c.begin(), c.end(), rng);
            size_t pieces = (c.size() + params.leaf_size - 1) / params.leaf_size;
            for (size_t k = 0; k < pieces; ++k)
              next[w].emplace_back(c.begin() + c.size() * k / pieces, c.begin() + c.size() * (k + 1) / pieces);
          } else {
            next[w].push_back(std::move(c));
          }
        }
      });
      work = {};
      for (auto& group : next)
        for (auto& c : group) active.push_back(std::move(c));
    }
    stats.random_splits = random_splits.load();
    return leaves;
  }

#if defined(__AVX512F__)
  // Branchless sorted insertion of 16 candidates into 16 top-K lists (masked blends).
  template <size_t K>
  static inline void insert16(__m512* bd, __m512i* bi, __m512 d, __m512i id) {
    __mmask16 lt[K];
    for (size_t t = 0; t < K; ++t) lt[t] = _mm512_cmp_ps_mask(d, bd[t], _CMP_LT_OQ);
    for (size_t t = K - 1; t > 0; --t) {
      bd[t] = _mm512_mask_blend_ps(lt[t - 1], _mm512_mask_blend_ps(lt[t], bd[t], d), bd[t - 1]);
      bi[t] = _mm512_mask_blend_epi32(lt[t - 1], _mm512_mask_blend_epi32(lt[t], bi[t], id), bi[t - 1]);
    }
    bd[0] = _mm512_mask_blend_ps(lt[0], bd[0], d);
    bi[0] = _mm512_mask_blend_epi32(lt[0], bi[0], id);
  }

  // Processes pairs (i, j) for j in whole 16-blocks below i: updates the lists of the j's in
  // place and returns row i's per-lane partial lists (K x 16) for a scalar merge.
  // Returns the first unprocessed j.
  template <size_t K>
  static size_t scan_row(const float* row, const float* norms, float ni, size_t i, size_t m,
                         float* kd, Id* ki, float* lane_d, Id* lane_i) {
    __m512 own_d[K], col_d[K];
    __m512i own_i[K], col_i[K];
    for (size_t t = 0; t < K; ++t) {
      own_d[t] = _mm512_set1_ps(std::numeric_limits<float>::infinity());
      own_i[t] = _mm512_set1_epi32(int(kNone));
    }
    const __m512 vni = _mm512_set1_ps(ni), minus2 = _mm512_set1_ps(-2.f);
    const __m512i vi = _mm512_set1_epi32(int(i));
    __m512i vj = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i step = _mm512_set1_epi32(16);
    size_t j = 0;
    for (; j + 16 <= i; j += 16, vj = _mm512_add_epi32(vj, step)) {
      __m512 d = _mm512_fmadd_ps(minus2, _mm512_loadu_ps(row + j), _mm512_add_ps(vni, _mm512_loadu_ps(norms + j)));
      for (size_t t = 0; t < K; ++t) {
        col_d[t] = _mm512_loadu_ps(kd + t * m + j);
        col_i[t] = _mm512_loadu_si512(ki + t * m + j);
      }
      insert16<K>(col_d, col_i, d, vi);
      for (size_t t = 0; t < K; ++t) {
        _mm512_storeu_ps(kd + t * m + j, col_d[t]);
        _mm512_storeu_si512(ki + t * m + j, col_i[t]);
      }
      insert16<K>(own_d, own_i, d, vj);
    }
    for (size_t t = 0; t < K; ++t) {
      _mm512_store_ps(lane_d + 16 * t, own_d[t]);
      _mm512_store_si512(lane_i + 16 * t, own_i[t]);
    }
    return j;
  }
#endif

  // Lower triangle of the leaf Gram matrix P P^T into `out` (row stride returned) and
  // squared norms. uint8 input uses an exact bfloat16 dot-product kernel when available:
  // 0..255 are exact in bf16 and fp32 sums of products stay exact below 2^24.
  size_t gram(const Bucket& ids, std::vector<float>& out, std::vector<float>& norms) const {
    const size_t m = ids.size();
    norms.resize(m);
#if defined(__AVX512BF16__)
    if constexpr (std::is_same_v<T, uint8_t>) {
      if (dim * 255 * 255 < (size_t(1) << 24)) {
        const size_t rows = (m + 7) / 8 * 8, ld = (m + 31) / 32 * 32, half = (dim + 1) / 2;
        thread_local std::vector<uint32_t> a, b;  // bf16 pairs: a row-major, b transposed
        a.assign(rows * half, 0);
        b.resize(half * ld);
        for (size_t i = 0; i < m; ++i) {
          const uint8_t* x = point(ids[i]);
          uint32_t sum = 0;
          for (size_t c = 0; c < dim; ++c) sum += uint32_t(x[c]) * x[c];
          norms[i] = float(sum);
          uint16_t* row = reinterpret_cast<uint16_t*>(a.data() + i * half);
          for (size_t c = 0; c < dim; ++c) row[c] = bf16(float(x[c]));
        }
        for (size_t t = 0; t < half; ++t)
          for (size_t i = 0; i < ld; ++i) b[t * ld + i] = i < m ? a[i * half + t] : 0;
        out.resize(rows * ld);
        for (size_t r0 = 0; r0 < rows; r0 += 8)
          for (size_t c0 = 0; c0 < r0 + 8 && c0 < ld; c0 += 32) {
            __m512 acc[8][2];
            for (auto& row : acc) row[0] = row[1] = _mm512_setzero_ps();
            for (size_t t = 0; t < half; ++t) {
              __m512bh b0 = (__m512bh)_mm512_loadu_si512(b.data() + t * ld + c0);
              __m512bh b1 = (__m512bh)_mm512_loadu_si512(b.data() + t * ld + c0 + 16);
              for (size_t r = 0; r < 8; ++r) {
                __m512bh pair = (__m512bh)_mm512_set1_epi32(int(a[(r0 + r) * half + t]));
                acc[r][0] = _mm512_dpbf16_ps(acc[r][0], pair, b0);
                acc[r][1] = _mm512_dpbf16_ps(acc[r][1], pair, b1);
              }
            }
            for (size_t r = 0; r < 8; ++r) {
              _mm512_storeu_ps(out.data() + (r0 + r) * ld + c0, acc[r][0]);
              _mm512_storeu_ps(out.data() + (r0 + r) * ld + c0 + 16, acc[r][1]);
            }
          }
        return ld;
      }
    }
#endif
    thread_local std::vector<float> x;
    x.resize(m * dim);
    for (size_t i = 0; i < m; ++i) load(point(ids[i]), x.data() + i * dim, dim);
    out.resize(m * m);
    Eigen::Map<const RowMat> P(x.data(), m, dim);
    Eigen::Map<RowMat> G(out.data(), m, m);
    Eigen::Map<Eigen::VectorXf>(norms.data(), m) = P.rowwise().squaredNorm();
    G.triangularView<Eigen::Lower>().setZero();
    G.selfadjointView<Eigen::Lower>().rankUpdate(P);
    return m;
  }

  // Exact k-NN inside one leaf from a single symmetric rank-update GEMM; each k-NN edge
  // is offered to HashPrune in both directions. Returns the number of offers.
  uint64_t build_leaf(const Bucket& ids) {
    const size_t m = ids.size();
    if (m < 2) return 0;
    const size_t k = std::min(params.local_k, m - 1);
    thread_local std::vector<float> dist, norms, knn_d;
    thread_local std::vector<Id> knn_i;
    // Top-k lists in structure-of-arrays form: slot t of point a is at t * m + a.
    knn_d.assign(k * m, std::numeric_limits<float>::infinity());
    knn_i.assign(k * m, kNone);
    for (size_t i = 0; i < m; ++i) {
      // Sketches and headers are read after the GEMM; random IDs otherwise miss in cache.
      __builtin_prefetch(sketches.data() + size_t(ids[i]) * kSketchStride, 0, 1);
      __builtin_prefetch(&headers[ids[i]], 0, 1);
    }
    const size_t ld = gram(ids, dist, norms);  // lower triangle of the Gram matrix, row stride ld
    float* kd = knn_d.data();
    Id* ki = knn_i.data();
    auto push = [&](size_t a, Id b, float d) {
      if (d >= kd[(k - 1) * m + a]) return;
      size_t t = k - 1;
      for (; t > 0 && kd[(t - 1) * m + a] > d; --t) { kd[t * m + a] = kd[(t - 1) * m + a]; ki[t * m + a] = ki[(t - 1) * m + a]; }
      kd[t * m + a] = d; ki[t * m + a] = b;
    };
    // Row i holds pairs (i, j < i); each pair may improve both lists. Point i's list is
    // untouched before row i, so it can be built from per-lane partial lists.
    for (size_t i = 1; i < m; ++i) {
      const float* row = dist.data() + i * ld;
      const float ni = norms[i];
      size_t j = 0;
#if defined(__AVX512F__)
      alignas(64) float lane_d[4 * 16];
      alignas(64) Id lane_i[4 * 16];
      switch (k) {  // compile-time K keeps the lane lists in registers
        case 1: j = scan_row<1>(row, norms.data(), ni, i, m, kd, ki, lane_d, lane_i); break;
        case 2: j = scan_row<2>(row, norms.data(), ni, i, m, kd, ki, lane_d, lane_i); break;
        case 3: j = scan_row<3>(row, norms.data(), ni, i, m, kd, ki, lane_d, lane_i); break;
        case 4: j = scan_row<4>(row, norms.data(), ni, i, m, kd, ki, lane_d, lane_i); break;
        default: break;
      }
      if (j > 0)
        for (size_t l = 0; l < k * 16; ++l) if (lane_i[l] != kNone) push(i, lane_i[l], lane_d[l]);
#endif
      for (; j < i; ++j) {
        float d = ni + norms[j] - 2 * row[j];
        push(i, Id(j), d);
        push(j, Id(i), d);
      }
    }
    // Group the directed offers (i -> j and j -> i) by target with a counting sort, so each
    // reservoir is locked once per leaf and the next targets can be prefetched.
    thread_local std::vector<uint32_t> start;
    thread_local std::vector<std::pair<Id, uint16_t>> grouped;  // (source local index, bf16 distance)
    start.assign(m + 2, 0);
    for (size_t t = 0; t < k; ++t)
      for (size_t i = 0; i < m; ++i)
        if (Id j = ki[t * m + i]; j != kNone) { ++start[i + 2]; ++start[j + 2]; }
    for (size_t i = 2; i <= m + 1; ++i) start[i] += start[i - 1];
    grouped.resize(start[m + 1]);
    for (size_t t = 0; t < k; ++t)
      for (size_t i = 0; i < m; ++i)
        if (Id j = ki[t * m + i]; j != kNone) {
          uint16_t d = bf16(kd[t * m + i]);
          grouped[start[i + 1]++] = {j, d};
          grouped[start[j + 1]++] = {Id(i), d};
        }
    constexpr size_t ahead = 8;
    for (size_t i = 0; i < std::min(ahead, m); ++i) prefetch_reservoir(ids[i]);
    for (size_t i = 0; i < m; ++i) {
      if (i + ahead < m) prefetch_reservoir(ids[i + ahead]);
      if (start[i] == start[i + 1]) continue;
      lock(ids[i]);
      for (size_t e = start[i]; e < start[i + 1]; ++e) offer_locked(ids[i], ids[grouped[e].first], grouped[e].second);
      unlock(ids[i]);
    }
    return grouped.size();
  }

  // RobustPrune each reservoir (exact distances recomputed), then free HashPrune state.
  void prune_all() {
    const size_t R = params.degree;
    const float alpha = float(params.alpha);
    adjacency.assign(n * R, kNone);
    degrees.assign(n, 0);
    parallel_for(params.threads, n, 64, [&](size_t i) {
      thread_local std::vector<std::pair<float, Id>> cand;
      cand.clear();
      const Id* ids = slot_id.get() + i * slots;
      const T* xi = point(Id(i));
      for (size_t s = 0; s < headers[i].size; ++s) cand.push_back({sqdist(xi, point(ids[s]), dim), ids[s]});
      std::sort(cand.begin(), cand.end());
      Id* out = adjacency.data() + i * R;
      size_t count = 0;
      for (auto [d, q] : cand) {
        if (count == R) break;
        const T* xq = point(q);
        bool keep = true;
        for (size_t s = 0; s < count && keep; ++s) keep = alpha * sqdist(point(out[s]), xq, dim) > d;
        if (keep) out[count++] = q;
      }
      degrees[i] = uint16_t(count);
    });
    double filled = 0;
    for (size_t i = 0; i < n; ++i) filled += headers[i].size;
    stats.mean_reservoir = filled / double(n);
    stats.mean_degree = std::accumulate(degrees.begin(), degrees.end(), 0.0) / double(n);
    headers.reset(); slot_hash.reset(); slot_dist.reset(); slot_id.reset();
    sketches = {};
  }
};

}  // namespace fastpipnn
