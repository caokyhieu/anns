// Sharded ANN designs for the distributed-PiPNN study (see DISTRIBUTED_EXPERIMENTS.md).
// One process simulates S shards (segments) and reports work per shard, so build and
// query costs can be read as "S workers in parallel" (wall = slowest shard) or as total CPU.
//
//   --design graph   D1: rows hashed to S shards, one graph per shard (fastpipnn or HNSW),
//                    every query searches every shard, the coordinator merges top-k.
//   --design ivf     D6 (LanceDB-style) and D5: centroids trained once on a sample
//                    (k-means for D6, PiPNN ball carving for D5), a router over the centroids
//                    (exact, HNSW or fastpipnn) on the query node, rows hashed to S segments,
//                    each segment assigns and RaBitQ-encodes its own rows; a query is routed
//                    once and the same nprobe lists are scanned in every segment, then
//                    candidates are re-ranked with full vectors and merged.
//   --design router  Router quality only: HNSW vs fastpipnn over C centroids.
//
// Distances are squared L2 on uint8 vectors. Queries run on one thread; per-shard times are
// measured one after another, and latency is reported as route time + slowest shard.
#include "bench_common.hpp"

// ---------------------------------------------------------------- RaBitQ (1-bit codes, 4-bit query)
struct RaBitQ {
  size_t d = 0, W = 0;
  RowMat P;                 // random orthogonal rotation, d x d
  void init(size_t dim, uint64_t seed) {
    d = dim; W = (d + 63) / 64;
    std::mt19937_64 rng(seed); std::normal_distribution<float> g;
    Eigen::MatrixXf G(d, d);
    for (size_t i = 0; i < d; ++i) for (size_t j = 0; j < d; ++j) G(i, j) = g(rng);
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(G);
    P = qr.householderQ();
  }
};
// One segment's IVF data, rows grouped by list.
struct Segment {
  std::vector<size_t> begin;      // C + 1 offsets
  std::vector<Id> ids;            // global row IDs
  std::vector<uint64_t> codes;    // W words per row
  std::vector<float> nr2, scale;  // |r|^2 and |r|^2 / sum|P r|
  double build_seconds = 0;
};

// Query prepared against one list: rotated residual quantized to 4 bits as bit planes.
struct ListQuery { float nq2 = 0, vmin = 0, delta = 0, sum = 0; uint64_t planes[4][4]; };

static inline float rabitq_estimate(const uint64_t* code, const ListQuery& q, size_t W, float nr2, float scale) {
  uint32_t ones = 0, acc = 0;
  for (size_t w = 0; w < W; ++w) {
    uint64_t c = code[w];
    ones += __builtin_popcountll(c);
    acc += __builtin_popcountll(c & q.planes[0][w]) + 2 * __builtin_popcountll(c & q.planes[1][w]) +
           4 * __builtin_popcountll(c & q.planes[2][w]) + 8 * __builtin_popcountll(c & q.planes[3][w]);
  }
  float set_sum = q.delta * float(acc) + q.vmin * float(ones);  // ~ sum of q'' over set bits
  return nr2 + q.nq2 - 2 * scale * (2 * set_sum - q.sum);
}

int main(int argc, char** argv) {
 try {
  Args a;
  for (int i = 1; i < argc; i += 2) { require(i + 1 < argc, "arguments come in pairs"); a.m[argv[i]] = argv[i + 1]; }
  const std::string design = a.get("--design", "graph");
  U8 base = load_u8(a.get("--base")), query = load_u8(a.get("--query"));
  Truth gt = load_gt(a.get("--gt"));
  require(base.d == query.d && gt.n == query.n, "dataset mismatch");
  const size_t d = base.d, N = base.n, nq = std::min<size_t>(query.n, a.n("--query-count", query.n));
  const size_t k = a.n("--k", 10), threads = a.n("--threads", std::thread::hardware_concurrency());
  const std::vector<size_t> shard_list = parse_list(a.get("--shards", "8"));
  size_t S = shard_list.at(0);
  require(k <= gt.k && S >= 1, "bad k or shards");
  Eigen::setNbThreads(1);
  Recorder rec(a.get("--output", "shard-results.jsonl"));
  const std::string dataset = a.get("--dataset", "custom");

  // Spot-check the ground truth against this base file.
  for (size_t c = 0; c < 3; ++c) {
    size_t qi = c * nq / 3;
    std::vector<float> all(N);
    parallel_for(threads, N, 4096, [&](size_t i) { all[i] = sq_u8(base.row(i), query.row(qi), d); });
    std::nth_element(all.begin(), all.begin() + k - 1, all.end());
    float kth = *std::max_element(all.begin(), all.begin() + k);
    for (size_t j = 0; j < k; ++j) require(sq_u8(base.row(gt.ids[qi * gt.k + j]), query.row(qi), d) <= kth, "ground truth mismatch");
  }
  auto recall = [&](size_t qi, const std::vector<std::pair<float, Id>>& top) {
    std::unordered_set<uint32_t> want(gt.ids.begin() + qi * gt.k, gt.ids.begin() + qi * gt.k + k);
    size_t hit = 0;
    for (size_t i = 0; i < std::min(k, top.size()); ++i) hit += want.count(top[i].second);
    return double(hit) / double(k);
  };
  // Rows -> shards by a hash of the row ID (balanced, independent of the vectors).
  const uint64_t seed = a.n("--seed", 7);
  std::vector<std::vector<Id>> rows;
  std::string common;
  auto setup = [&](size_t count) {
    require(count >= 1, "shards must be >= 1");
    S = count; rows.assign(S, {});
    for (size_t i = 0; i < N; ++i) rows[mix(i ^ (seed << 40)) % S].push_back(Id(i));
    common = "\"dataset\":\"" + dataset + "\",\"n\":" + std::to_string(N) + ",\"dim\":" + std::to_string(d) +
             ",\"queries\":" + std::to_string(nq) + ",\"k\":" + std::to_string(k) +
             ",\"shards\":" + std::to_string(S) + ",\"threads\":" + std::to_string(threads);
  };
  setup(S);

  if (design == "graph") {
    // ------------------------------------------------------------ D1
    const std::string index = a.get("--index", "fastpipnn");
    std::vector<std::vector<uint8_t>> data(S);
    std::vector<std::unique_ptr<fastpipnn::Index<uint8_t>>> fast(S);
    std::vector<std::unique_ptr<hnswlib::HierarchicalNSW<float>>> hnsw(S);
    ByteSpace space(d);
    std::vector<double> shard_build(S);
    for (size_t s = 0; s < S; ++s) {
      data[s].resize(rows[s].size() * d);
      for (size_t i = 0; i < rows[s].size(); ++i) std::memcpy(data[s].data() + i * d, base.row(rows[s][i]), d);
      double t = now();
      if (index == "fastpipnn") {
        fastpipnn::Params p;
        p.degree = a.n("--degree", 64); p.leaf_size = a.n("--leaf-size", 1024); p.local_k = a.n("--local-k", 2);
        p.alpha = a.f("--alpha", 1.2); p.fanout = {10, 3}; p.threads = threads;
        p.top_leaders = std::min<size_t>(1000, std::max<size_t>(3, rows[s].size() / 1000));
        fast[s] = std::make_unique<fastpipnn::Index<uint8_t>>();
        fast[s]->build(data[s].data(), rows[s].size(), d, d, p);
      } else {
        require(index == "hnsw", "--index must be fastpipnn or hnsw");
        hnsw[s] = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, rows[s].size(), a.n("--m", 32), a.n("--ef-construction", 200), 100 + s);
        hnsw[s]->addPoint(data[s].data(), 0);
        parallel_for(threads, rows[s].size() - 1, 64, [&](size_t i) { hnsw[s]->addPoint(data[s].data() + (i + 1) * d, i + 1); });
      }
      shard_build[s] = now() - t;
      std::cerr << "built shard " << s << " (" << rows[s].size() << " rows) in " << shard_build[s] << " s\n";
    }
    double wall = *std::max_element(shard_build.begin(), shard_build.end()), cpu = 0;
    for (double x : shard_build) cpu += x;
    rec.write("{\"kind\":\"build\",\"design\":\"D1\",\"index\":\"" + index + "\"," + common +
              ",\"fixed_seconds\":0,\"shard_seconds_max\":" + num(wall) + ",\"shard_seconds_sum\":" + num(cpu) +
              ",\"wall_seconds\":" + num(wall) + ",\"shard_seconds\":" + list_json(shard_build) + "}");
    for (size_t beam : parse_list(a.get("--beams", "10,16,32,64,128,256"))) {
      std::vector<double> rc(nq), lat(nq), work_time(nq), evals(nq);
      for (size_t qi = 0; qi < nq; ++qi) {
        const uint8_t* q = query.row(qi);
        std::vector<std::pair<float, Id>> merged;
        double slowest = 0, total = 0; uint64_t e = 0;
        for (size_t s = 0; s < S; ++s) {
          double t = now();
          if (index == "fastpipnn") {
            size_t ev = 0;
            auto r = fast[s]->search(q, k, std::max(beam, k), &ev);
            e += ev;
            for (Id id : r) merged.push_back({sq_u8(data[s].data() + size_t(id) * d, q, d), rows[s][id]});
          } else {
            g_evals = 0;
            hnsw[s]->setEf(std::max(beam, k));
            auto r = hnsw[s]->searchKnn(q, k);
            e += g_evals;
            while (!r.empty()) { merged.push_back({r.top().first, rows[s][r.top().second]}); r.pop(); }
          }
          double dt = now() - t; slowest = std::max(slowest, dt); total += dt;
        }
        double t = now();
        std::partial_sort(merged.begin(), merged.begin() + std::min(k, merged.size()), merged.end());
        double merge = now() - t;
        rc[qi] = recall(qi, merged); lat[qi] = (slowest + merge) * 1e3; work_time[qi] = (total + merge) * 1e3; evals[qi] = double(e);
      }
      auto R = summarize(rc), L = summarize(lat), Wt = summarize(work_time), E = summarize(evals);
      rec.write("{\"kind\":\"query\",\"design\":\"D1\",\"index\":\"" + index + "\"," + common + ",\"beam\":" + std::to_string(beam) +
                ",\"recall\":" + num(R.mean) + ",\"shards_touched\":" + std::to_string(S) + ",\"distance_evals\":" + num(E.mean) +
                ",\"latency_ms_mean\":" + num(L.mean) + ",\"latency_ms_p50\":" + num(L.p50) + ",\"latency_ms_p99\":" + num(L.p99) +
                ",\"cpu_ms_per_query\":" + num(Wt.mean) + "}");
    }
    return 0;
  }

  // Centroids for ivf and router designs.
  const std::string partition = a.get("--partition", "kmeans");
  const size_t lists = a.n("--lists", 1024);
  double t0 = now();
  auto sample = sample_rows(N, a.n("--sample-per-list", 256) * lists, seed + 11);
  RowMat X(sample.size(), d);
  for (size_t i = 0; i < sample.size(); ++i) for (size_t j = 0; j < d; ++j) X(i, j) = base.row(sample[i])[j];
  double sample_seconds = now() - t0;
  t0 = now();
  RowMat cent = partition == "kmeans" ? kmeans(X, lists, a.n("--iters", 10), threads, seed + 12)
                                      : carve_centroids(X, lists, threads, seed + 12);
  require(partition == "kmeans" || partition == "carve", "--partition must be kmeans or carve");
  double train_seconds = now() - t0;
  const size_t C = cent.rows();
  std::vector<float> cn = norms(cent);
  std::string cfg, part_cfg = std::string(",\"partition\":\"") + partition + "\",\"lists\":" + std::to_string(C) +
                    ",\"sample\":" + std::to_string(sample.size());
  cfg = common + part_cfg;

  if (design == "router") {
    // ------------------------------------------------------------ router study
    std::vector<std::vector<float>> qf(nq, std::vector<float>(d));
    for (size_t qi = 0; qi < nq; ++qi) for (size_t j = 0; j < d; ++j) qf[qi][j] = query.row(qi)[j];
    Router exact; exact.build("exact", cent.data(), C, d, threads, a);
    std::vector<std::string> kinds;
    { std::istringstream in(a.get("--routers", "hnsw,fastpipnn")); for (std::string t; std::getline(in, t, ',');) kinds.push_back(t); }
    for (const std::string& kind : kinds) {
      Router r; r.build(kind, cent.data(), C, d, threads, a);
      for (size_t probes : parse_list(a.get("--nprobes", "8,32"))) {
        for (size_t beam : parse_list(a.get("--router-beams", "16,32,64,128,256"))) {
          if (beam < probes) continue;
          double hit = 0, ev = 0, ms = 0;
          for (size_t qi = 0; qi < nq; ++qi) {
            uint64_t e0 = 0, e1 = 0;
            auto want = exact.route(qf[qi].data(), probes, probes, e0);
            double t = now();
            auto got = r.route(qf[qi].data(), probes, beam, e1);
            ms += (now() - t) * 1e3;
            std::unordered_set<Id> w(want.begin(), want.end());
            for (Id g : got) hit += w.count(g);
            ev += double(e1);
          }
          rec.write("{\"kind\":\"router\",\"router\":\"" + kind + "\"," + cfg + ",\"router_build_seconds\":" + num(r.build_seconds) +
                    ",\"nprobe\":" + std::to_string(probes) + ",\"router_beam\":" + std::to_string(beam) +
                    ",\"router_recall\":" + num(hit / double(nq * probes)) + ",\"router_evals\":" + num(ev / nq) +
                    ",\"router_ms\":" + num(ms / nq) + ",\"exact_evals\":" + std::to_string(C) + "}");
        }
      }
    }
    return 0;
  }

  // ------------------------------------------------------------ D6 / D5: IVF + RaBitQ segments
  require(design == "ivf", "--design must be graph, ivf or router");
  const std::string quant = a.get("--quant", "rabitq");
  require(quant == "rabitq" || quant == "flat", "--quant must be rabitq or flat");
  t0 = now();
  RaBitQ rq; rq.init(d, seed + 13);
  RowMat rot_cent = cent * rq.P.transpose();  // P c for every centroid
  const double rotation_seconds = now() - t0;
  const size_t W = rq.W;
  require(W <= 4, "RaBitQ path supports d <= 256");
  Router exact_router; exact_router.build("exact", cent.data(), C, d, threads, a);
  std::vector<std::string> router_kinds;
  { std::istringstream in(a.get("--routers", "exact")); for (std::string t; std::getline(in, t, ',');) router_kinds.push_back(t); }
  for (const std::string& router_kind : router_kinds) {
  t0 = now();
  Router router; router.build(router_kind, cent.data(), C, d, threads, a);
  const double router_seconds = rotation_seconds + now() - t0;
  for (size_t shard_count : shard_list) {
  setup(shard_count);
  cfg = common + part_cfg;
  std::vector<Segment> seg(S);
  for (size_t s = 0; s < S; ++s) {
    Segment& g = seg[s];
    double t = now();
    const size_t m = rows[s].size();
    RowMat Xs(m, d);
    for (size_t i = 0; i < m; ++i) for (size_t j = 0; j < d; ++j) Xs(i, j) = base.row(rows[s][i])[j];
    std::vector<Id> lst;
    assign_rows(Xs.data(), m, cent, cn, lst, threads);
    std::vector<size_t> cnt(C + 1, 0);
    for (Id j : lst) ++cnt[j + 1];
    for (size_t j = 0; j < C; ++j) cnt[j + 1] += cnt[j];
    g.begin = cnt;
    std::vector<size_t> pos(cnt.begin(), cnt.end() - 1), slot(m);
    g.ids.resize(m);
    for (size_t i = 0; i < m; ++i) { slot[i] = pos[lst[i]]++; g.ids[slot[i]] = rows[s][i]; }
    if (quant == "rabitq") {
      g.codes.assign(m * W, 0); g.nr2.resize(m); g.scale.resize(m);
      const size_t tile = 1024;
      parallel_for(threads, (m + tile - 1) / tile, 1, [&](size_t ti) {
        size_t b = ti * tile, r = std::min(tile, m - b);
        RowMat R(r, d);
        for (size_t i = 0; i < r; ++i) R.row(i) = Xs.row(b + i) - cent.row(lst[b + i]);
        RowMat PR = R * rq.P.transpose();
        for (size_t i = 0; i < r; ++i) {
          size_t at = slot[b + i];
          float l1 = 0;
          for (size_t j = 0; j < d; ++j) {
            float v = PR(i, j); l1 += std::fabs(v);
            if (v > 0) g.codes[at * W + j / 64] |= 1ull << (j % 64);
          }
          float n2 = R.row(i).squaredNorm();
          g.nr2[at] = n2; g.scale[at] = l1 > 0 ? n2 / l1 : 0.f;
        }
      });
    }
    g.build_seconds = now() - t;
  }
  std::vector<double> shard_build(S);
  double wall = 0, cpu = 0;
  for (size_t s = 0; s < S; ++s) { shard_build[s] = seg[s].build_seconds; wall = std::max(wall, shard_build[s]); cpu += shard_build[s]; }
  double fixed = sample_seconds + train_seconds + router_seconds;
  std::vector<double> sizes(C, 0);
  for (auto& g : seg) for (size_t j = 0; j < C; ++j) sizes[j] += double(g.begin[j + 1] - g.begin[j]);
  std::sort(sizes.begin(), sizes.end());
  const std::string name = partition == "kmeans" ? "D6" : "D5";
  cfg += ",\"router\":\"" + router_kind + "\",\"quant\":\"" + quant + "\"";
  rec.write("{\"kind\":\"build\",\"design\":\"" + name + "\"," + cfg + ",\"sample_seconds\":" + num(sample_seconds) +
            ",\"train_seconds\":" + num(train_seconds) + ",\"router_seconds\":" + num(router_seconds) +
            ",\"fixed_seconds\":" + num(fixed) + ",\"shard_seconds_max\":" + num(wall) + ",\"shard_seconds_sum\":" + num(cpu) +
            ",\"wall_seconds\":" + num(fixed + wall) + ",\"list_size_min\":" + num(sizes.front()) +
            ",\"list_size_p50\":" + num(sizes[C / 2]) + ",\"list_size_max\":" + num(sizes.back()) +
            ",\"shard_seconds\":" + list_json(shard_build) + "}");

  const size_t router_beam = a.n("--router-beam", 64);
  const std::string rerank_mode = a.get("--rerank", "global");
  require(rerank_mode == "global" || rerank_mode == "segment", "--rerank must be global or segment");
  const bool global_rerank = quant == "rabitq" && rerank_mode == "global";
  cfg += ",\"rerank\":\"" + rerank_mode + "\"";
  for (size_t refine : parse_list(a.get("--refines", "10"))) {
   for (size_t probes : parse_list(a.get("--nprobes", "1,2,4,8,16,32,64"))) {
    if (probes > C) continue;
    std::vector<double> rc(nq), lat(nq), work_time(nq), scanned(nq), reranked(nq), routed(nq), rhit(nq);
    std::vector<float> qf(d), qrot(d);
    for (size_t qi = 0; qi < nq; ++qi) {
      const uint8_t* q = query.row(qi);
      double t = now();
      for (size_t j = 0; j < d; ++j) qf[j] = q[j];
      uint64_t re = 0;
      auto probe = router.route(qf.data(), probes, router_beam, re);
      std::vector<ListQuery> lq(probe.size());
      if (quant == "rabitq") {
        Eigen::Map<Eigen::VectorXf>(qrot.data(), d) = rq.P * Eigen::Map<const Eigen::VectorXf>(qf.data(), d);
        for (size_t p = 0; p < probe.size(); ++p) {
          ListQuery& L = lq[p];
          std::vector<float> r(d);
          float lo = std::numeric_limits<float>::max(), hi = -lo;
          for (size_t j = 0; j < d; ++j) { r[j] = qrot[j] - rot_cent(probe[p], j); lo = std::min(lo, r[j]); hi = std::max(hi, r[j]); L.sum += r[j]; L.nq2 += r[j] * r[j]; }
          L.vmin = lo; L.delta = hi > lo ? (hi - lo) / 15.f : 1.f;
          std::memset(L.planes, 0, sizeof(L.planes));
          for (size_t j = 0; j < d; ++j) {
            uint32_t u = uint32_t(std::lround((r[j] - lo) / L.delta));
            for (size_t b = 0; b < 4; ++b) if (u >> b & 1) L.planes[b][j / 64] |= 1ull << (j % 64);
          }
        }
      }
      double route_time = now() - t, slowest = 0, total = 0;
      size_t sc = 0, rr = 0;
      std::vector<std::pair<float, Id>> merged;
      const size_t keep = quant == "rabitq" ? k * refine : k;
      for (size_t s = 0; s < S; ++s) {
        double ts = now();
        const Segment& g = seg[s];
        std::priority_queue<std::pair<float, Id>> heap;  // max-heap on (estimate, global ID)
        for (size_t p = 0; p < probe.size(); ++p) {
          size_t b = g.begin[probe[p]], e = g.begin[probe[p] + 1];
          sc += e - b;
          for (size_t i = b; i < e; ++i) {
            float est = quant == "rabitq" ? rabitq_estimate(&g.codes[i * W], lq[p], W, g.nr2[i], g.scale[i])
                                          : sq_u8(base.row(g.ids[i]), q, d);
            if (heap.size() < keep) heap.push({est, g.ids[i]});
            else if (est < heap.top().first) { heap.pop(); heap.push({est, g.ids[i]}); }
          }
        }
        std::vector<std::pair<float, Id>> cand;
        while (!heap.empty()) { cand.push_back(heap.top()); heap.pop(); }
        if (quant == "rabitq" && !global_rerank) {
          rr += cand.size();
          for (auto& c : cand) c.first = sq_u8(base.row(c.second), q, d);
        }
        std::sort(cand.begin(), cand.end());
        if (cand.size() > (global_rerank ? keep : k)) cand.resize(global_rerank ? keep : k);
        merged.insert(merged.end(), cand.begin(), cand.end());
        double dt = now() - ts; slowest = std::max(slowest, dt); total += dt;
      }
      t = now();
      if (global_rerank) {
        // Estimates from different segments are comparable (shared centroids and rotation), so the
        // coordinator keeps the global best k*refine estimates and reranks only those.
        std::sort(merged.begin(), merged.end());
        if (merged.size() > keep) merged.resize(keep);
        rr += merged.size();
        for (auto& c : merged) c.first = sq_u8(base.row(c.second), q, d);
      }
      std::partial_sort(merged.begin(), merged.begin() + std::min(k, merged.size()), merged.end());
      double merge = now() - t;
      uint64_t ee = 0;
      auto want = exact_router.route(qf.data(), probes, probes, ee);
      std::unordered_set<Id> w(want.begin(), want.end());
      double hit = 0; for (Id p : probe) hit += w.count(p);
      rc[qi] = recall(qi, merged); lat[qi] = (route_time + slowest + merge) * 1e3; work_time[qi] = (route_time + total + merge) * 1e3;
      scanned[qi] = double(sc); reranked[qi] = double(rr); routed[qi] = double(re); rhit[qi] = hit / double(probes);
    }
    auto R = summarize(rc), L = summarize(lat), Wt = summarize(work_time), Sc = summarize(scanned), Rr = summarize(reranked),
         Ro = summarize(routed), Rh = summarize(rhit);
    rec.write("{\"kind\":\"query\",\"design\":\"" + name + "\"," + cfg + ",\"nprobe\":" + std::to_string(probes) +
              ",\"refine\":" + std::to_string(refine) + ",\"router_beam\":" + std::to_string(router_beam) +
              ",\"recall\":" + num(R.mean) + ",\"router_recall\":" + num(Rh.mean) + ",\"shards_touched\":" + std::to_string(S) +
              ",\"codes_scanned\":" + num(Sc.mean) + ",\"reranked\":" + num(Rr.mean) + ",\"router_evals\":" + num(Ro.mean) +
              ",\"latency_ms_mean\":" + num(L.mean) + ",\"latency_ms_p50\":" + num(L.p50) + ",\"latency_ms_p99\":" + num(L.p99) +
              ",\"cpu_ms_per_query\":" + num(Wt.mean) + "}");
   }
  }
  }  // shard counts
  }  // routers
  return 0;
 } catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << std::endl;
  return 1;
 }
}
