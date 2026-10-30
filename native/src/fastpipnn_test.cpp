#include "fastpipnn.hpp"
#include <iostream>
#include <set>

using namespace fastpipnn;

static void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

template <class T>
static void check_graph(const Index<T>& index) {
  for (Id i = 0; i < index.n; ++i) {
    auto ns = index.neighbors(i);
    require(ns.size() <= index.params.degree, "degree bound violated");
    std::set<Id> unique(ns.begin(), ns.end());
    require(unique.size() == ns.size(), "duplicate edge");
    for (Id q : ns) require(q < index.n && q != i, "invalid edge");
  }
}

template <class T>
static double recall(const Index<T>& index, const std::vector<T>& queries, size_t nq, size_t beam) {
  size_t hits = 0;
  for (size_t qi = 0; qi < nq; ++qi) {
    const T* q = queries.data() + qi * index.dim;
    std::vector<std::pair<float, Id>> exact;
    for (Id p = 0; p < index.n; ++p) exact.push_back({sqdist(index.point(p), q, index.dim), p});
    std::partial_sort(exact.begin(), exact.begin() + 10, exact.end());
    auto found = index.search(q, 10, beam);
    require(found.size() == 10, "search returned fewer than k results");
    for (Id id : found)
      for (size_t j = 0; j < 10; ++j) hits += exact[j].second == id;
  }
  return hits / double(nq * 10);
}

int main() {
  try {
    std::mt19937 rng(7);
    const size_t n = 20000, dim = 32, nq = 100;
    std::vector<float> centers(64 * dim);
    std::normal_distribution<float> normal;
    for (auto& c : centers) c = 4 * normal(rng);
    auto sample = [&](size_t count) {
      std::vector<float> x(count * dim);
      for (size_t i = 0; i < count; ++i) {
        size_t c = rng() % 64;
        for (size_t j = 0; j < dim; ++j) x[i * dim + j] = centers[c * dim + j] + normal(rng);
      }
      return x;
    };
    auto base = sample(n), queries = sample(nq);
    Params p;
    p.leaf_size = 256; p.degree = 32; p.reservoir = 64; p.top_leaders = 200; p.threads = 4;
    Index<float> index;
    index.build(base.data(), n, dim, dim, p);
    check_graph(index);
    require(index.stats.leaf_points >= n, "a point is missing from every leaf");
    double r = recall(index, queries, nq, 64);
    std::cout << "float: n=" << n << " leaves=" << index.stats.leaves << " depth=" << index.stats.depth
              << " seconds=" << index.stats.total_seconds << " recall@10(beam 64)=" << r << std::endl;
    require(r > 0.9, "float recall too low");

    // Determinism: same seed and data give the same graph.
    Index<float> again;
    again.build(base.data(), n, dim, dim, p);
    for (Id i = 0; i < n; ++i) {
      auto a = index.neighbors(i), b = again.neighbors(i);
      require(std::vector<Id>(a.begin(), a.end()) == std::vector<Id>(b.begin(), b.end()), "nondeterministic graph");
    }

    // uint8 path with a padded row stride, as in the benchmark loader.
    const size_t stride = 64;
    std::vector<uint8_t> bytes(n * stride, 0), qbytes(nq * dim);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < dim; ++j) bytes[i * stride + j] = uint8_t(std::clamp(128 + 8 * base[i * dim + j], 0.f, 255.f));
    for (size_t i = 0; i < nq * dim; ++i) qbytes[i] = uint8_t(std::clamp(128 + 8 * queries[i], 0.f, 255.f));
    Index<uint8_t> bytes_index;
    bytes_index.build(bytes.data(), n, dim, stride, p);
    check_graph(bytes_index);
    double rb = recall(bytes_index, qbytes, nq, 64);
    std::cout << "uint8: recall@10(beam 64)=" << rb << std::endl;
    require(rb > 0.85, "uint8 recall too low");

    // Many exact duplicates must terminate through the random-split guard.
    std::vector<float> dup(3000 * 4, 1.0f);
    for (size_t i = 0; i < 100 * 4; ++i) dup[i] = normal(rng);
    Index<float> dup_index;
    Params dp = p; dp.leaf_size = 64; dp.top_leaders = 20;
    dup_index.build(dup.data(), 3000, 4, 4, dp);
    check_graph(dup_index);
    require(dup_index.stats.random_splits > 0, "duplicate guard not exercised");
    std::cout << "duplicates: leaves=" << dup_index.stats.leaves << " random_splits=" << dup_index.stats.random_splits
              << std::endl;
    std::cout << "fastpipnn tests passed" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return 1;
  }
}
