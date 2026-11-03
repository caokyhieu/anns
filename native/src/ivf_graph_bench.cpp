// IVF with graphs in the two places LanceDB uses HNSW, LanceDB's choice vs PiPNN (see IVF_GRAPH.md).
//
//   1. assignment  every row to its nearest centroid
//        flat       exact, one GEMM per tile against all C centroids
//        hnsw       Lance's SimpleIndex: HNSW over the centroids, m 12, ef_construction 15,
//                   top-1 search at ef 15 (rust/lance-index/src/vector/utils.rs)
//        fastpipnn  fastpipnn graph over the centroids, top-1 search at beam B ("fastpipnn:B")
//   2. sub-index   a graph inside every partition (Lance's IVF_HNSW_FLAT)
//        flat       no graph: scan the probed partitions
//        hnsw       Lance's defaults: m 20 (level-0 degree 40), ef_construction 150
//        fastpipnn  one fastpipnn graph per partition, degree 40 by default
//
// Queries follow Lance: score the query against all C centroids, keep the nprobe nearest (Lance
// has no router over centroids), search each probed partition for its top k with ef / beam,
// merge. Distances are exact squared L2 on uint8 vectors (no SQ/PQ), so differences come from
// the graphs and the assignment only. Partitions are built in parallel, one thread each, and
// queries run on one thread.
#include "bench_common.hpp"

// One partition's sub-index. Rows are stored contiguously in `Partitions::data`.
struct SubIndex {
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
  std::unique_ptr<fastpipnn::Index<uint8_t>> fast;
  double seconds = 0;
};

// Rows grouped by partition (CSR) with a contiguous copy of their vectors.
struct Partitions {
  size_t C = 0, d = 0;
  std::vector<size_t> begin;     // C + 1 offsets
  std::vector<Id> ids;           // global row IDs, grouped by partition
  std::vector<uint8_t> data;     // the same rows' vectors, in the same order
  const uint8_t* row(size_t i) const { return data.data() + i * d; }
  size_t size(size_t j) const { return begin[j + 1] - begin[j]; }
};

static Partitions group(const U8& base, const std::vector<Id>& assign, size_t C) {
  Partitions P; P.C = C; P.d = base.d;
  P.begin.assign(C + 1, 0);
  for (Id j : assign) ++P.begin[j + 1];
  for (size_t j = 0; j < C; ++j) P.begin[j + 1] += P.begin[j];
  std::vector<size_t> pos(P.begin.begin(), P.begin.end() - 1);
  P.ids.resize(assign.size()); P.data.resize(assign.size() * P.d);
  for (size_t i = 0; i < assign.size(); ++i) {
    size_t at = pos[assign[i]]++;
    P.ids[at] = Id(i);
    std::memcpy(P.data.data() + at * P.d, base.row(i), P.d);
  }
  return P;
}

// Beam search over a fastpipnn graph with caller-owned visited marks, so a thread that searches
// many small graphs in turn does not look up per-index scratch. Same algorithm as
// fastpipnn::Index::search: best-first from the medoid, stop when the frontier is worse than
// the beam's worst. Returns (distance, local ID) pairs, nearest first.
// `entries` > 0 also seeds the beam with that many rows spread over the partition (every
// (m / entries)-th row; row order is the dataset's, unrelated to the vectors), all counted as
// distance evaluations. It plays the role of HNSW's upper layers: a graph whose clusters are
// not linked (see reachable_fraction) is still searched from a seed near the query.
struct Visited { std::vector<uint32_t> mark; uint32_t epoch = 0; };
static std::vector<std::pair<float, Id>> graph_search(const fastpipnn::Index<uint8_t>& g, const uint8_t* q, size_t k,
                                                      size_t beam, Visited& v, uint64_t& evals, size_t entries = 0) {
  if (v.mark.size() < g.n) { v.mark.assign(g.n, 0); v.epoch = 0; }
  if (++v.epoch == 0) { std::fill(v.mark.begin(), v.mark.end(), 0); v.epoch = 1; }
  using Pair = std::pair<float, Id>;
  std::priority_queue<Pair> best;
  std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> frontier;
  auto visit = [&](Id p) {
    if (v.mark[p] == v.epoch) return;
    v.mark[p] = v.epoch; ++evals;
    Pair item{fastpipnn::sqdist(g.point(p), q, g.dim), p};
    if (best.size() < beam || item < best.top()) {
      best.push(item); frontier.push(item);
      if (best.size() > beam) best.pop();
    }
  };
  visit(g.entry);
  for (size_t s = 0, E = std::min(entries, g.n); s < E; ++s) visit(Id(s * g.n / E));
  while (!frontier.empty()) {
    Pair item = frontier.top(); frontier.pop();
    if (best.size() == beam && item > best.top()) break;
    for (Id n : g.neighbors(item.second)) visit(n);
  }
  while (best.size() > k) best.pop();
  std::vector<Pair> out(best.size());
  for (size_t i = out.size(); i > 0; --i) { out[i - 1] = best.top(); best.pop(); }
  return out;
}

// Rows reachable from the entry by following out-edges (BFS), for a fastpipnn graph or an
// HNSW level-0 graph given as a neighbor callback.
template <class Neighbors> static size_t reachable(size_t n, Id entry, Neighbors&& nb) {
  std::vector<char> seen(n, 0);
  std::vector<Id> stack{entry};
  seen[entry] = 1;
  size_t count = 1;
  while (!stack.empty()) {
    Id p = stack.back(); stack.pop_back();
    nb(p, [&](Id q) { if (!seen[q]) { seen[q] = 1; ++count; stack.push_back(q); } });
  }
  return count;
}

int main(int argc, char** argv) {
 try {
  Args a;
  for (int i = 1; i < argc; i += 2) { require(i + 1 < argc, "arguments come in pairs"); a.m[argv[i]] = argv[i + 1]; }
  U8 base = load_u8(a.get("--base")), query = load_u8(a.get("--query"));
  Truth gt = load_gt(a.get("--gt"));
  require(base.d == query.d && gt.n == query.n, "dataset mismatch");
  const size_t d = base.d, N = base.n, nq = std::min<size_t>(query.n, a.n("--query-count", query.n));
  const size_t k = a.n("--k", 10), threads = a.n("--threads", std::thread::hardware_concurrency());
  const size_t C = a.n("--lists", 1024), seed = a.n("--seed", 7);
  const size_t min_graph = a.n("--min-graph", 64);  // smaller partitions get no graph and are scanned
  require(k <= gt.k && C >= 1, "bad k or lists");
  Eigen::setNbThreads(1);
  Recorder rec(a.get("--output", "ivf-graph-results.jsonl"));
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

  // ------------------------------------------------------------ centroids (k-means, as Lance)
  double t0 = now();
  auto sample = sample_rows(N, a.n("--sample-per-list", 256) * C, seed + 11);
  RowMat X(sample.size(), d);
  for (size_t i = 0; i < sample.size(); ++i) for (size_t j = 0; j < d; ++j) X(i, j) = base.row(sample[i])[j];
  RowMat cent = kmeans(X, C, a.n("--iters", 10), threads, seed + 12);
  X.resize(0, 0);
  const double train_seconds = now() - t0;
  std::vector<float> cn = norms(cent);
  std::cerr << "trained " << C << " centroids in " << train_seconds << " s\n";
  const std::string common = "\"tag\":\"" + a.get("--tag", "") + "\",\"dataset\":\"" + dataset + "\",\"n\":" + std::to_string(N) + ",\"dim\":" + std::to_string(d) +
                             ",\"queries\":" + std::to_string(nq) + ",\"k\":" + std::to_string(k) +
                             ",\"lists\":" + std::to_string(C) + ",\"threads\":" + std::to_string(threads) +
                             ",\"train_seconds\":" + num(train_seconds);

  // Base rows as floats for the GEMM path and the graph searches over centroids.
  std::vector<float> basef(N * d);
  parallel_for(threads, N, 4096, [&](size_t i) { for (size_t j = 0; j < d; ++j) basef[i * d + j] = base.row(i)[j]; });
  t0 = now();
  std::vector<Id> exact;
  assign_rows(basef.data(), N, cent, cn, exact, threads);
  const double flat_seconds = now() - t0;

  // Query vectors as floats, and each query's nprobe list (exact, as Lance) computed per query below.
  std::vector<std::string> assigns, subs;
  { std::istringstream in(a.get("--assign", "flat,hnsw,fastpipnn:15")); for (std::string t; std::getline(in, t, ',');) assigns.push_back(t); }
  { std::istringstream in(a.get("--sub", "flat,hnsw,fastpipnn")); for (std::string t; std::getline(in, t, ',');) subs.push_back(t); }
  const auto nprobes = parse_list(a.get("--nprobes", "1,2,4,8,16,32"));
  const auto efs = parse_list(a.get("--efs", "10,20,40,80,160"));
  const auto entry_list = parse_list(a.get("--entries", "0"));  // extra seeds for fastpipnn searches

  for (const std::string& spec : assigns) {
    // -------------------------------------------------------- 1. assignment
    std::string kind = spec.substr(0, spec.find(':'));
    size_t beam = spec.find(':') == std::string::npos ? 15 : std::stoul(spec.substr(spec.find(':') + 1));
    std::vector<Id> lst;
    double graph_seconds = 0, assign_seconds = 0, evals_per_row = double(C);
    if (kind == "flat") {
      lst = exact; assign_seconds = flat_seconds;
    } else {
      require(kind == "hnsw" || kind == "fastpipnn", "--assign entries must be flat, hnsw or fastpipnn[:beam]");
      Args ra = a;  // Lance's SimpleIndex parameters unless overridden
      if (!ra.m.count("--router-m")) ra.m["--router-m"] = "12";
      if (!ra.m.count("--router-efc")) ra.m["--router-efc"] = "15";
      Router r; r.build(kind, cent.data(), C, d, threads, ra);
      graph_seconds = r.build_seconds;
      lst.resize(N);
      std::atomic<uint64_t> total{0};
      t0 = now();
      parallel_for(threads, N, 256, [&](size_t i) {
        uint64_t e = 0;
        lst[i] = r.route(basef.data() + i * d, 1, beam, e).at(0);
        total.fetch_add(e, std::memory_order_relaxed);
      });
      assign_seconds = now() - t0;
      evals_per_row = double(total.load()) / double(N);
    }
    // Exact when the chosen centroid is as close as the true nearest (ties count as exact).
    std::atomic<size_t> hits{0};
    parallel_for(threads, N, 4096, [&](size_t i) {
      if (lst[i] == exact[i]) { hits.fetch_add(1, std::memory_order_relaxed); return; }
      const float* x = basef.data() + i * d;
      if (fastpipnn::sqdist(x, cent.data() + size_t(lst[i]) * d, d) <= fastpipnn::sqdist(x, cent.data() + size_t(exact[i]) * d, d))
        hits.fetch_add(1, std::memory_order_relaxed);
    });
    Partitions P = group(base, lst, C);
    std::vector<double> sizes(C);
    size_t empty = 0;
    for (size_t j = 0; j < C; ++j) { sizes[j] = double(P.size(j)); empty += P.size(j) == 0; }
    std::sort(sizes.begin(), sizes.end());
    const std::string acfg = common + ",\"assign\":\"" + kind + "\",\"assign_beam\":" + std::to_string(kind == "flat" ? 0 : beam);
    rec.write("{\"kind\":\"assign\"," + acfg + ",\"graph_seconds\":" + num(graph_seconds) + ",\"assign_seconds\":" + num(assign_seconds) +
              ",\"flat_seconds\":" + num(flat_seconds) + ",\"evals_per_row\":" + num(evals_per_row) +
              ",\"exact_fraction\":" + num(double(hits.load()) / double(N)) + ",\"empty_lists\":" + std::to_string(empty) +
              ",\"list_size_min\":" + num(sizes.front()) + ",\"list_size_p50\":" + num(sizes[C / 2]) +
              ",\"list_size_max\":" + num(sizes.back()) + "}");

    for (const std::string& sub : subs) {
      // ------------------------------------------------------ 2. sub-index in every partition
      require(sub == "flat" || sub == "hnsw" || sub == "fastpipnn", "--sub entries must be flat, hnsw or fastpipnn");
      std::vector<SubIndex> idx(C);
      ByteSpace space(d);
      std::vector<size_t> order(C); std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return P.size(x) > P.size(y); });
      t0 = now();
      if (sub != "flat") {
        parallel_for(threads, C, 1, [&](size_t o) {
          const size_t j = order[o], m = P.size(j);
          if (m < std::max<size_t>(min_graph, 2)) return;
          const uint8_t* rows = P.row(P.begin[j]);
          double t = now();
          if (sub == "hnsw") {
            idx[j].hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, m, a.n("--m", 20), a.n("--ef-construction", 150), seed + j);
            for (size_t i = 0; i < m; ++i) idx[j].hnsw->addPoint(rows + i * d, i);
          } else {
            fastpipnn::Params p;
            p.degree = a.n("--degree", 40); p.reservoir = std::max<size_t>(a.n("--reservoir", 160), p.degree);
            // Leaf size follows the partition unless given: m / 8 within [64, 1024]. A leaf as large
            // as the partition makes the graph a plain k-NN graph (IVF_GRAPH.md §3).
            p.leaf_size = a.m.count("--leaf-size") ? a.n("--leaf-size", 1024) : std::clamp<size_t>(m / 8, 64, 1024);
            p.local_k = a.n("--local-k", 2); p.alpha = a.f("--alpha", 1.2);
            p.fanout = parse_list(a.get("--fanout", "10,3")); p.threads = 1; p.seed = seed + j;
            p.leader_fraction = a.f("--leader-fraction", 0.005);
            // Depth-0 leaders scale with the partition: m / leaf_size, which is fastpipnn's own
            // default at 1M points (1000 leaders, leaf 1024). A fixed small count leaves a small
            // partition in one or two leaves, i.e. a k-NN graph that does not link its clusters.
            p.top_leaders = std::min<size_t>(1000, std::max<size_t>(3, size_t(a.f("--leaders-per-leaf", 1.0) * double(m) / double(p.leaf_size))));
            idx[j].fast = std::make_unique<fastpipnn::Index<uint8_t>>();
            idx[j].fast->build(rows, m, d, d, p);
          }
          idx[j].seconds = now() - t;
        });
      }
      const double sub_wall = now() - t0;
      double sub_cpu = 0, slowest = 0, edges = 0, graph_rows = 0, reached = 0;
      for (size_t j = 0; j < C; ++j) {
        sub_cpu += idx[j].seconds; slowest = std::max(slowest, idx[j].seconds);
        const size_t m = P.size(j);
        if (idx[j].fast) {
          const auto& g = *idx[j].fast;
          graph_rows += double(m);
          for (size_t i = 0; i < m; ++i) edges += g.neighbors(Id(i)).size();
          reached += double(reachable(m, g.entry, [&](Id p, auto&& f) { for (Id q : g.neighbors(p)) f(q); }));
        }
        if (idx[j].hnsw) {
          auto& h = *idx[j].hnsw;
          graph_rows += double(m);
          for (size_t i = 0; i < m; ++i) edges += *h.get_linklist0(hnswlib::tableint(i)) & 0xffff;
          // Level 0 from the top entry point; HNSW's upper layers can still reach other parts.
          reached += double(reachable(m, Id(h.enterpoint_node_), [&](Id p, auto&& f) {
            const unsigned* l = h.get_linklist0(hnswlib::tableint(p));
            for (unsigned s = 0, deg = *l & 0xffff; s < deg; ++s) f(Id(l[1 + s]));
          }));
        }
      }
      const std::string cfg = acfg + ",\"sub\":\"" + sub + "\"";
      rec.write("{\"kind\":\"build\"," + cfg + ",\"graph_seconds\":" + num(graph_seconds) + ",\"assign_seconds\":" + num(assign_seconds) +
                ",\"sub_seconds\":" + num(sub_wall) + ",\"sub_cpu_seconds\":" + num(sub_cpu) + ",\"sub_slowest_seconds\":" + num(slowest) +
                ",\"total_seconds\":" + num(train_seconds + graph_seconds + assign_seconds + sub_wall) +
                ",\"mean_degree\":" + num(graph_rows > 0 ? edges / graph_rows : 0) + ",\"graph_rows\":" + num(graph_rows) +
                ",\"reachable_fraction\":" + num(graph_rows > 0 ? reached / graph_rows : 0) + "}");
      std::cerr << "built " << spec << " + " << sub << " in " << sub_wall << " s\n";

      // ------------------------------------------------------ queries
      Visited visited;
      std::vector<float> qf(d);
      std::vector<std::pair<float, Id>> cd(C);
      for (size_t probes : nprobes) {
        if (probes > C) continue;
        for (size_t seeds : (sub == "fastpipnn" ? entry_list : std::vector<size_t>{0}))
        for (size_t ef : (sub == "flat" ? std::vector<size_t>{0} : efs)) {
          std::vector<double> rc(nq), lat(nq), ev(nq);
          for (size_t qi = 0; qi < nq; ++qi) {
            const uint8_t* q = query.row(qi);
            double t = now();
            for (size_t j = 0; j < d; ++j) qf[j] = q[j];
            for (size_t j = 0; j < C; ++j) cd[j] = {cn[j] - 2 * Eigen::Map<const Eigen::VectorXf>(cent.data() + j * d, d).dot(
                                                                 Eigen::Map<const Eigen::VectorXf>(qf.data(), d)), Id(j)};
            std::partial_sort(cd.begin(), cd.begin() + probes, cd.end());
            uint64_t e = C;
            std::vector<std::pair<float, Id>> merged;
            for (size_t p = 0; p < probes; ++p) {
              const size_t j = cd[p].second, b = P.begin[j], m = P.size(j);
              if (idx[j].hnsw) {
                g_evals = 0;
                idx[j].hnsw->setEf(std::max(ef, k));
                auto r = idx[j].hnsw->searchKnn(q, k);
                e += g_evals;
                while (!r.empty()) { merged.push_back({r.top().first, P.ids[b + r.top().second]}); r.pop(); }
              } else if (idx[j].fast) {
                for (auto& [dist, id] : graph_search(*idx[j].fast, q, k, std::max(ef, k), visited, e, seeds)) merged.push_back({dist, P.ids[b + id]});
              } else {
                for (size_t i = 0; i < m; ++i) merged.push_back({sq_u8(P.row(b + i), q, d), P.ids[b + i]});
                e += m;
              }
            }
            std::partial_sort(merged.begin(), merged.begin() + std::min(k, merged.size()), merged.end());
            lat[qi] = (now() - t) * 1e3;
            rc[qi] = recall(qi, merged); ev[qi] = double(e);
          }
          auto R = summarize(rc), L = summarize(lat), E = summarize(ev);
          rec.write("{\"kind\":\"query\"," + cfg + ",\"nprobe\":" + std::to_string(probes) + ",\"ef\":" + std::to_string(ef) + ",\"entries\":" + std::to_string(seeds) +
                    ",\"recall\":" + num(R.mean) + ",\"distance_evals\":" + num(E.mean) + ",\"latency_ms_mean\":" + num(L.mean) +
                    ",\"latency_ms_p50\":" + num(L.p50) + ",\"latency_ms_p99\":" + num(L.p99) + "}");
        }
      }
    }
  }
  return 0;
 } catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << std::endl;
  return 1;
 }
}
