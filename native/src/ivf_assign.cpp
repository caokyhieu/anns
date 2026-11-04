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

  // Exact assignment is the reference for every method (and is written if "flat" is requested).
  double t0 = now();
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
    } else {
      require(kind == "hnsw" || kind == "fastpipnn", "--assign entries must be flat, hnsw:B or fastpipnn:B");
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
              ",\"assign_seconds\":" + num(seconds) + ",\"flat_seconds\":" + num(flat_seconds) + ",\"evals_per_row\":" + num(evals) +
              ",\"exact_fraction\":" + num(double(hits.load()) / double(N)) + ",\"empty_lists\":" + std::to_string(empty) +
              ",\"list_size_p50\":" + std::to_string(sizes[C / 2]) + ",\"list_size_max\":" + std::to_string(sizes.back()) + "}");
  }
  return 0;
 } catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << std::endl;
  return 1;
 }
}
