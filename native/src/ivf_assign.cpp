// Row -> partition assignment against fixed IVF centroids, for feeding a Lance IVF index through
// `create_index(..., ivf_centroids=C, precomputed_partition_dataset=...)` (IVF_RQ.md).
//
//   ivf_assign --base base.fbin --centroids centroids.fbin --assign flat,hnsw:15,fastpipnn:32
//              --out-prefix data/gist/assign --output stats.jsonl [--threads T]
//
// `.fbin` = uint32 n, uint32 d, then n*d float32, row-major; the base file is memory-mapped and
// assigned in tiles, so it can be larger than RAM. Methods (squared L2):
//   flat         exact, one GEMM per tile against all centroids
//   hnsw:B       hnswlib over the centroids with Lance's SimpleIndex settings (M 12, efC 15), top-1 at ef B
//   fastpipnn:B  fastpipnn graph over the centroids (router settings of bench_common.hpp), top-1 at beam B
//   tree:t       two-level tree over the centroids (sqrt(C) coarse cells), each row searches its t nearest cells
// Each method writes <out-prefix>.<method>.u32 (n uint32 partition IDs, no header) and one JSON line
// with its time, distances per row, and the fraction of rows on their exact nearest centroid.
#include "bench_common.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct FMap {
  size_t n = 0, d = 0, bytes = 0;
  void* base = MAP_FAILED;
  const float* rows = nullptr;
  explicit FMap(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    require(fd >= 0, "cannot open " + path);
    struct stat st; fstat(fd, &st); bytes = size_t(st.st_size);
    base = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    require(base != MAP_FAILED, "mmap failed " + path);
    const uint32_t* h = static_cast<const uint32_t*>(base);
    n = h[0]; d = h[1];
    require(bytes == 8 + n * d * sizeof(float), "size mismatch in " + path);
    rows = reinterpret_cast<const float*>(static_cast<const char*>(base) + 8);
  }
  ~FMap() { if (base != MAP_FAILED) munmap(base, bytes); }
  const float* row(size_t i) const { return rows + i * d; }
};

// Tree assignment (LANCE_VS_PIPNN_BUILD.md, "tree_t") over fixed leaf centroids. The coarse level
// is k-means over the C leaves themselves (round(sqrt(C)) cells, each leaf under its nearest coarse
// centroid), since the leaves are Lance's flat k-means centroids rather than a trained tree. Each
// row takes its t nearest coarse centroids; per coarse cell, one GEMM scores the rows that chose it
// against that cell's leaves; the row keeps the nearest leaf seen. Rows go in chunks so the GEMMs
// stay matrix-shaped without sharing per-row state between threads.
struct Tree {
  RowMat coarse; std::vector<float> coarse_norm;
  std::vector<std::vector<Id>> kids;   // leaf IDs under each coarse cell
  std::vector<RowMat> kid_mat;         // their coordinates, one matrix per cell
  std::vector<std::vector<float>> kid_norm;
  double build_seconds = 0;
  void build(const RowMat& leaves, const std::vector<float>& leaf_norm, size_t cells, size_t threads, uint64_t seed) {
    double t = now();
    cells = std::max<size_t>(1, std::min(cells, size_t(leaves.rows())));
    coarse = kmeans(leaves, cells, 25, threads, seed);
    coarse_norm = norms(coarse);
    std::vector<Id> parent;
    assign_rows(leaves.data(), leaves.rows(), coarse, coarse_norm, parent, threads);
    kids.assign(cells, {});
    for (size_t j = 0; j < parent.size(); ++j) kids[parent[j]].push_back(Id(j));
    kid_mat.resize(cells); kid_norm.resize(cells);
    for (size_t c = 0; c < cells; ++c) {
      kid_mat[c].resize(kids[c].size(), leaves.cols());
      for (size_t i = 0; i < kids[c].size(); ++i) { kid_mat[c].row(i) = leaves.row(kids[c][i]); kid_norm[c].push_back(leaf_norm[kids[c][i]]); }
    }
    build_seconds = now() - t;
  }
  // Assigns rows [0, n) of X; returns distance evaluations per row.
  double assign(const float* X, size_t n, size_t d, size_t t, std::vector<Id>& out, size_t threads) const {
    const size_t K = coarse.rows(), chunk = 4096;
    t = std::min(t, K);
    out.resize(n);
    std::atomic<uint64_t> evals{0};
    parallel_for(threads, (n + chunk - 1) / chunk, 1, [&](size_t ci) {
      const size_t s = ci * chunk, r = std::min(chunk, n - s);
      Eigen::Map<const RowMat> Xc(X + s * d, r, d);
      RowMat S = Xc * coarse.transpose();
      std::vector<std::vector<Id>> rows_of(K);
      std::vector<std::pair<float, Id>> cd(K);
      for (size_t i = 0; i < r; ++i) {
        for (size_t c = 0; c < K; ++c) cd[c] = {coarse_norm[c] - 2 * S(i, c), Id(c)};
        std::partial_sort(cd.begin(), cd.begin() + t, cd.end());
        for (size_t j = 0; j < t; ++j) rows_of[cd[j].second].push_back(Id(i));
      }
      std::vector<float> best(r, std::numeric_limits<float>::max());
      uint64_t e = uint64_t(r) * K;
      for (size_t c = 0; c < K; ++c) {
        const auto& rows = rows_of[c];
        if (rows.empty() || kids[c].empty()) continue;
        RowMat Q(rows.size(), d);
        for (size_t i = 0; i < rows.size(); ++i) Q.row(i) = Xc.row(rows[i]);
        RowMat D = Q * kid_mat[c].transpose();
        e += uint64_t(rows.size()) * kids[c].size();
        for (size_t i = 0; i < rows.size(); ++i)
          for (size_t j = 0; j < kids[c].size(); ++j) {
            float v = kid_norm[c][j] - 2 * D(i, j);
            if (v < best[rows[i]]) { best[rows[i]] = v; out[s + rows[i]] = kids[c][j]; }
          }
      }
      evals.fetch_add(e, std::memory_order_relaxed);
    });
    return double(evals.load()) / double(n);
  }
};

int main(int argc, char** argv) {
 try {
  Args a;
  for (int i = 1; i < argc; i += 2) { require(i + 1 < argc, "arguments come in pairs"); a.m[argv[i]] = argv[i + 1]; }
  FMap base(a.get("--base")), cfile(a.get("--centroids"));
  require(base.d == cfile.d, "base and centroids differ in dimension");
  const size_t N = base.n, d = base.d, C = cfile.n, threads = a.n("--threads", std::thread::hardware_concurrency());
  Eigen::setNbThreads(1);
  RowMat cent = Eigen::Map<const RowMat>(cfile.rows, C, d);
  std::vector<float> cn = norms(cent);
  const std::string prefix = a.get("--out-prefix");
  Recorder rec(a.get("--output", prefix + ".jsonl"));
  const std::string common = "\"tag\":\"" + a.get("--tag", "") + "\",\"n\":" + std::to_string(N) + ",\"dim\":" + std::to_string(d) +
                             ",\"lists\":" + std::to_string(C) + ",\"threads\":" + std::to_string(threads);
  auto write = [&](const std::string& name, const std::vector<Id>& v) {
    std::string path = prefix + "." + name + ".u32";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    require(f && std::fwrite(v.data(), sizeof(Id), v.size(), f) == v.size(), "cannot write " + path);
    std::fclose(f);
  };

  // Fault the mapped base in once, so method timings measure compute rather than whichever method
  // happens to read the file from disk first.
  double t0 = now();
  madvise(const_cast<void*>(static_cast<const void*>(base.base)), base.bytes, MADV_WILLNEED);
  std::atomic<uint64_t> sink{0};
  parallel_for(threads, (base.bytes + 4095) / 4096, 256, [&](size_t pg) {
    sink.fetch_add(static_cast<const uint8_t*>(base.base)[pg * 4096], std::memory_order_relaxed);
  });
  const double load_seconds = now() - t0;
  std::cerr << "read " << base.bytes / 1e9 << " GB in " << load_seconds << " s\n";

  // Exact assignment is the reference for every method (and is written if "flat" is requested).
  t0 = now();
  std::vector<Id> exact(N);
  const size_t tile = 1 << 16;
  for (size_t s = 0; s < N; s += tile) {
    std::vector<Id> part;
    assign_rows(base.row(s), std::min(tile, N - s), cent, cn, part, threads);
    std::copy(part.begin(), part.end(), exact.begin() + s);
  }
  const double flat_seconds = now() - t0;
  std::cerr << "exact assignment of " << N << " rows to " << C << " centroids in " << flat_seconds << " s\n";

  std::istringstream specs(a.get("--assign", "flat,hnsw:15,fastpipnn:16,fastpipnn:32"));
  for (std::string spec; std::getline(specs, spec, ',');) {
    const std::string kind = spec.substr(0, spec.find(':'));
    const size_t beam = spec.find(':') == std::string::npos ? 15 : std::stoul(spec.substr(spec.find(':') + 1));
    std::vector<Id> lst;
    double graph_seconds = 0, seconds = 0, evals = double(C);
    if (kind == "flat") {
      lst = exact; seconds = flat_seconds;
    } else if (kind == "tree") {
      // beam = t, the number of coarse cells each row searches.
      Tree tree; tree.build(cent, cn, size_t(std::lround(std::sqrt(double(C)))), threads, a.n("--seed", 7));
      graph_seconds = tree.build_seconds;
      t0 = now();
      evals = 0;
      for (size_t s = 0; s < N; s += tile) {
        std::vector<Id> part;
        const size_t r = std::min(tile, N - s);
        evals += tree.assign(base.row(s), r, d, beam, part, threads) * double(r);
        if (lst.empty()) lst.resize(N);
        std::copy(part.begin(), part.end(), lst.begin() + s);
      }
      seconds = now() - t0;
      evals /= double(N);
    } else {
      require(kind == "hnsw" || kind == "fastpipnn", "--assign entries must be flat, hnsw:B, fastpipnn:B or tree:t");
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
        lst[i] = r.route(base.row(i), 1, beam, e).at(0);
        total.fetch_add(e, std::memory_order_relaxed);
      });
      seconds = now() - t0;
      evals = double(total.load()) / double(N);
    }
    std::atomic<size_t> hits{0};
    parallel_for(threads, N, 4096, [&](size_t i) {
      if (lst[i] == exact[i] || fastpipnn::sqdist(base.row(i), cent.data() + size_t(lst[i]) * d, d) <=
                                    fastpipnn::sqdist(base.row(i), cent.data() + size_t(exact[i]) * d, d))
        hits.fetch_add(1, std::memory_order_relaxed);
    });
    std::vector<size_t> sizes(C, 0);
    for (Id j : lst) ++sizes[j];
    size_t empty = std::count(sizes.begin(), sizes.end(), size_t(0));
    std::sort(sizes.begin(), sizes.end());
    const std::string name = kind == "flat" ? "flat" : kind + std::to_string(beam);
    write(name, lst);
    rec.write("{\"kind\":\"assign\"," + common + ",\"assign\":\"" + kind + "\",\"assign_beam\":" + std::to_string(kind == "flat" ? 0 : beam) +
              ",\"file\":\"" + prefix + "." + name + ".u32\",\"graph_seconds\":" + num(graph_seconds) +
              ",\"assign_seconds\":" + num(seconds) + ",\"flat_seconds\":" + num(flat_seconds) + ",\"load_seconds\":" + num(load_seconds) + ",\"evals_per_row\":" + num(evals) +
              ",\"exact_fraction\":" + num(double(hits.load()) / double(N)) + ",\"empty_lists\":" + std::to_string(empty) +
              ",\"list_size_p50\":" + std::to_string(sizes[C / 2]) + ",\"list_size_max\":" + std::to_string(sizes.back()) + "}");
  }
  return 0;
 } catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << std::endl;
  return 1;
 }
}
