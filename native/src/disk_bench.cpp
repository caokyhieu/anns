// On-disk graph benchmark: PiPNN (fastpipnn) vs HNSW level 0 in the same disk layout and
// search code (disk_index.hpp), plus the in-RAM fastpipnn search for reference.
//
//   disk_bench --base B.u8bin --query Q.u8bin --gt G.ibin --graph fastpipnn|hnsw
//              --work DIR [--orders none,bfs,carve] [--io direct] [--beams 16,32,64,128]
//              [--widths 4] [--page-expand 0,1] [--cache-nodes 0,10000] [--pq-m 32]
//              [--query-threads 4] [--output disk-results.jsonl]
//
// The base file is memory-mapped (not copied). The graph is saved to DIR/<graph>.graph and
// reused on later runs. Each layout is written to DIR/<graph>-<order>.nodes.
#include "disk_index.hpp"
#include "hnswlib/hnswlib.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/mman.h>
#include <unordered_set>

using diskgraph::Id;
using fastpipnn::now;

static void require(bool ok, const std::string& m) { if (!ok) throw std::runtime_error(m); }
static std::vector<size_t> nums(const std::string& s) {
  std::vector<size_t> v; std::istringstream in(s);
  for (std::string t; std::getline(in, t, ',');) v.push_back(std::stoul(t));
  return v;
}
static std::vector<std::string> words(const std::string& s) {
  std::vector<std::string> v; std::istringstream in(s);
  for (std::string t; std::getline(in, t, ',');) v.push_back(t);
  return v;
}
static std::string num(double x) { std::ostringstream s; s.precision(6); s << x; return s.str(); }

struct Mapped {  // read-only mmap of a .u8bin file
  uint32_t n = 0, d = 0;
  const uint8_t* rows = nullptr;
  size_t bytes = 0; void* base = nullptr;
  explicit Mapped(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    require(fd >= 0, "cannot open " + path);
    struct stat st; fstat(fd, &st); bytes = st.st_size;
    base = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    require(base != MAP_FAILED, "mmap failed " + path);
    std::memcpy(&n, base, 4); std::memcpy(&d, static_cast<uint8_t*>(base) + 4, 4);
    require(bytes == 8 + uint64_t(n) * d, "size mismatch " + path);
    rows = static_cast<const uint8_t*>(base) + 8;
  }
  ~Mapped() { munmap(base, bytes); }
  const uint8_t* row(size_t i) const { return rows + i * d; }
};

static thread_local uint64_t g_evals = 0;
struct ByteSpace : hnswlib::SpaceInterface<float> {
  size_t dim;
  explicit ByteSpace(size_t d) : dim(d) {}
  static float dist(const void* a, const void* b, const void* p) {
    ++g_evals;
    return fastpipnn::sqdist(static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), *static_cast<const size_t*>(p));
  }
  size_t get_data_size() override { return dim; }
  hnswlib::DISTFUNC<float> get_dist_func() override { return dist; }
  void* get_dist_func_param() override { return &dim; }
};

static bool exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }

int main(int argc, char** argv) {
 try {
  std::map<std::string, std::string> a;
  for (int i = 1; i < argc; i += 2) { require(i + 1 < argc, "arguments come in pairs"); a[argv[i]] = argv[i + 1]; }
  auto get = [&](const std::string& k, const std::string& f) { return a.count(k) ? a[k] : f; };
  const size_t threads = std::stoul(get("--threads", std::to_string(std::thread::hardware_concurrency())));
  const size_t qthreads = std::stoul(get("--query-threads", "4")), k = std::stoul(get("--k", "10"));
  const std::string graph_kind = get("--graph", "fastpipnn"), work = get("--work", "data/disk");
  const std::string dataset = get("--dataset", "custom");
  Eigen::setNbThreads(1);
  Mapped base(get("--base", "")), query(get("--query", ""));
  require(base.d == query.d, "dimension mismatch");
  std::vector<uint32_t> gt;
  uint32_t gn = 0, gk = 0;
  {
    std::ifstream in(get("--gt", ""), std::ios::binary);
    require(bool(in), "cannot open ground truth");
    in.read(reinterpret_cast<char*>(&gn), 4); in.read(reinterpret_cast<char*>(&gk), 4);
    gt.resize(size_t(gn) * gk);
    in.read(reinterpret_cast<char*>(gt.data()), gt.size() * 4);
    require(bool(in) && gn == query.n && gk >= k, "bad ground truth");
  }
  const size_t n = base.n, d = base.d, nq = std::min<size_t>(query.n, std::stoul(get("--query-count", "1000000")));
  ::mkdir(work.c_str(), 0755);
  std::ofstream out(get("--output", "disk-results.jsonl"), std::ios::app);

  // ---- graph: load if saved, otherwise build and save
  diskgraph::Graph g;
  const std::string gpath = work + "/" + graph_kind + ".graph";
  double build_seconds = 0;
  fastpipnn::Index<uint8_t> fast;  // also used for the in-RAM reference search
  const Id medoid = fastpipnn::medoid(base.rows, n, d, d, threads);
  if (graph_kind == "fastpipnn") {
    if (exists(gpath)) {
      fast.load_graph(gpath, base.rows, d);
      build_seconds = std::stod(get("--known-build-seconds", "0"));
    } else {
      fastpipnn::Params p;
      p.degree = std::stoul(get("--degree", "64")); p.leaf_size = std::stoul(get("--leaf-size", "1024"));
      p.local_k = std::stoul(get("--local-k", "2")); p.alpha = std::stod(get("--alpha", "1.2"));
      p.fanout = {10, 3}; p.threads = threads;
      double t = now();
      fast.build(base.rows, n, d, d, p);
      build_seconds = now() - t;
      fast.save_graph(gpath);
    }
    g.n = n; g.R = fast.params.degree; g.entry = medoid;
    g.deg.resize(n); g.adj.assign(n * g.R, diskgraph::kNone);
    for (size_t i = 0; i < n; ++i) {
      auto nb = fast.neighbors(Id(i));
      g.deg[i] = uint16_t(nb.size());
      std::copy(nb.begin(), nb.end(), g.adj.begin() + i * g.R);
    }
  } else {
    require(graph_kind == "hnsw", "--graph must be fastpipnn or hnsw");
    const size_t M = std::stoul(get("--m", "32"));
    g.n = n; g.R = 2 * M; g.entry = medoid;
    if (exists(gpath)) {
      std::FILE* f = std::fopen(gpath.c_str(), "rb");
      g.deg.resize(n); g.adj.resize(n * g.R);
      require(std::fread(g.deg.data(), 2, n, f) == n && std::fread(g.adj.data(), 4, g.adj.size(), f) == g.adj.size(), "bad hnsw graph file");
      std::fclose(f);
      build_seconds = std::stod(get("--known-build-seconds", "0"));
    } else {
      ByteSpace space(d);
      double t = now();
      hnswlib::HierarchicalNSW<float> h(&space, n, M, std::stoul(get("--ef-construction", "200")), 100);
      h.addPoint(base.row(0), 0);
      fastpipnn::parallel_for(threads, n - 1, 64, [&](size_t i) { h.addPoint(base.row(i + 1), i + 1); });
      build_seconds = now() - t;
      // Level-0 adjacency, translated from internal IDs to labels (= row numbers).
      g.deg.assign(n, 0); g.adj.assign(n * g.R, diskgraph::kNone);
      for (size_t c = 0; c < n; ++c) {
        size_t label = h.getExternalLabel(c);
        auto* ll = h.get_linklist0(c);
        size_t cnt = h.getListCount(ll);
        auto* nb = reinterpret_cast<hnswlib::tableint*>(ll + 1);
        g.deg[label] = uint16_t(cnt);
        for (size_t j = 0; j < cnt; ++j) g.adj[label * g.R + j] = Id(h.getExternalLabel(nb[j]));
      }
      std::FILE* f = std::fopen(gpath.c_str(), "wb");
      std::fwrite(g.deg.data(), 2, n, f); std::fwrite(g.adj.data(), 4, g.adj.size(), f);
      std::fclose(f);
    }
  }
  double mean_degree = 0;
  for (auto x : g.deg) mean_degree += x;
  mean_degree /= double(n);
  std::cerr << graph_kind << " graph ready, build " << build_seconds << " s, mean degree " << mean_degree << "\n";

  // ---- PQ codes (original order)
  diskgraph::PQ pq;
  double t = now();
  pq.train(base.rows, n, d, d, std::stoul(get("--pq-m", "32")), std::stoul(get("--pq-sample", "100000")), 10, threads, 7);
  std::vector<uint8_t> codes0(n * pq.M);
  pq.encode(base.rows, n, d, codes0.data(), threads);
  const double pq_seconds = now() - t;

  auto recall = [&](size_t qi, const std::vector<std::pair<float, Id>>& r) {
    std::unordered_set<uint32_t> want(gt.begin() + qi * gk, gt.begin() + qi * gk + k);
    size_t hit = 0;
    for (auto& x : r) hit += want.count(x.second);
    return double(hit) / double(k);
  };
  const std::string common = "\"dataset\":\"" + dataset + "\",\"n\":" + std::to_string(n) + ",\"dim\":" + std::to_string(d) +
      ",\"k\":" + std::to_string(k) + ",\"queries\":" + std::to_string(nq) + ",\"graph\":\"" + graph_kind +
      "\",\"degree\":" + std::to_string(g.R) + ",\"mean_degree\":" + num(mean_degree) +
      ",\"graph_build_seconds\":" + num(build_seconds) + ",\"pq_m\":" + std::to_string(pq.M) +
      ",\"pq_seconds\":" + num(pq_seconds);
  auto emit = [&](const std::string& s) { out << s << "\n"; out.flush(); std::cout << s << std::endl; };

  // ---- in-RAM reference (fastpipnn's own search over full vectors)
  if (graph_kind == "fastpipnn" && get("--ram-reference", "1") == "1") {
    for (size_t beam : nums(get("--beams", "16,32,64,128"))) {
      double rc = 0, ev = 0, ms = 0;
      for (size_t qi = 0; qi < nq; ++qi) {
        size_t e = 0;
        double t0 = now();
        auto ids = fast.search(query.row(qi), k, std::max(beam, k), &e);
        ms += (now() - t0) * 1e3;
        std::vector<std::pair<float, Id>> r;
        for (Id id : ids) r.push_back({0.f, id});
        rc += recall(qi, r); ev += double(e);
      }
      emit("{\"kind\":\"ram\"," + common + ",\"beam\":" + std::to_string(beam) + ",\"recall\":" + num(rc / nq) +
           ",\"distance_evals\":" + num(ev / nq) + ",\"latency_ms\":" + num(ms / nq) +
           ",\"ram_bytes\":" + std::to_string(uint64_t(n) * d + uint64_t(n) * g.R * 4) + "}");
    }
  }

  // ---- disk layouts
  for (const std::string& order : words(get("--orders", "none,bfs,carve"))) {
    t = now();
    std::vector<Id> perm = order == "none" ? diskgraph::order_identity(n)
                         : order == "bfs"  ? diskgraph::order_bfs(g)
                         : diskgraph::order_carve(base.rows, n, d, std::stoul(get("--order-leaf", "32")), threads, 11);
    require(order == "none" || order == "bfs" || order == "carve", "--orders: none, bfs, carve");
    const double order_seconds = now() - t;
    diskgraph::DiskIndex idx;
    idx.L.init(n, d, g.R);
    const std::string npath = work + "/" + graph_kind + "-" + order + ".nodes";
    t = now();
    diskgraph::write_nodes(npath, base.rows, d, g, perm, idx.L);
    const double write_seconds = now() - t;
    idx.pq = pq;
    idx.codes.resize(n * pq.M); idx.orig = perm;
    for (size_t p = 0; p < n; ++p) std::memcpy(&idx.codes[p * pq.M], &codes0[size_t(perm[p]) * pq.M], pq.M);
    // How often a node and its neighbour share a sector: what page expansion can exploit.
    double same = 0, edges = 0;
    {
      std::vector<Id> inv(n);
      for (size_t p = 0; p < n; ++p) inv[perm[p]] = Id(p);
      for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < g.deg[i]; ++j) { edges += 1; same += idx.L.sector_of(inv[i]) == idx.L.sector_of(inv[g.nbrs(Id(i))[j]]); }
    }
    for (const std::string& io : words(get("--io", "direct"))) {
      idx.open(npath, io);
      for (size_t cache : nums(get("--cache-nodes", "0"))) {
        idx.build_cache(cache);
        const std::string layout = common + ",\"order\":\"" + order + "\",\"order_seconds\":" + num(order_seconds) +
            ",\"write_seconds\":" + num(write_seconds) + ",\"disk_bytes\":" + std::to_string(idx.L.bytes()) +
            ",\"nodes_per_sector\":" + std::to_string(idx.L.per_sector) +
            ",\"neighbor_same_sector\":" + num(same / edges) + ",\"io\":\"" + io + "\",\"cache_nodes\":" +
            std::to_string(cache) + ",\"ram_bytes\":" + std::to_string(idx.ram_bytes());
        for (size_t pe : nums(get("--page-expand", "0,1")))
          for (size_t W : nums(get("--widths", "4")))
            for (size_t beam : nums(get("--beams", "16,32,64,128"))) {
              // Latency: one thread, queries one after another.
              double rc = 0, ms = 0;
              diskgraph::SearchStats st;
              std::vector<double> lat(nq);
              for (size_t qi = 0; qi < nq; ++qi) {
                double t0 = now();
                auto r = idx.search(query.row(qi), k, std::max(beam, k), W, pe, st);
                lat[qi] = (now() - t0) * 1e3; ms += lat[qi];
                rc += recall(qi, r);
              }
              std::sort(lat.begin(), lat.end());
              // Throughput: query-threads threads share the file descriptor.
              diskgraph::SearchStats st2;
              double t1 = now();
              fastpipnn::parallel_for(qthreads, nq, 1, [&](size_t qi) {
                diskgraph::SearchStats local;
                idx.search(query.row(qi), k, std::max(beam, k), W, pe, local);
              });
              double qps = double(nq) / (now() - t1);
              emit("{\"kind\":\"disk\"," + layout + ",\"page_expand\":" + std::to_string(pe) + ",\"width\":" +
                   std::to_string(W) + ",\"beam\":" + std::to_string(beam) + ",\"recall\":" + num(rc / nq) +
                   ",\"reads\":" + num(double(st.reads) / nq) + ",\"cache_hits\":" + num(double(st.cache_hits) / nq) +
                   ",\"expanded\":" + num(double(st.expanded) / nq) + ",\"rounds\":" + num(double(st.rounds) / nq) +
                   ",\"pq_evals\":" + num(double(st.pq_evals) / nq) + ",\"latency_ms\":" + num(ms / nq) +
                   ",\"latency_ms_p99\":" + num(lat[std::min(nq - 1, size_t(nq * 0.99))]) +
                   ",\"qps\":" + num(qps) + ",\"query_threads\":" + std::to_string(qthreads) + "}");
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
