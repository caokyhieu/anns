// Build a fastpipnn graph over float32 vectors and write its adjacency for Python experiments.
// Usage: pipnn_graph <vectors.f32> <n> <dim> <degree> <out.u32> [alpha] [seed] [local_k]
// Output: n uint32 degrees, then n * degree uint32 neighbor IDs (padding = 0xFFFFFFFF).
#include "fastpipnn.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 6) { std::cerr << "usage: pipnn_graph vectors.f32 n dim degree out.u32 [alpha] [seed] [local_k]\n"; return 2; }
  size_t n = std::stoul(argv[2]), dim = std::stoul(argv[3]);
  std::vector<float> data(n * dim);
  std::ifstream in(argv[1], std::ios::binary);
  if (!in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float))) { std::cerr << "short read\n"; return 1; }
  fastpipnn::Params p;
  p.degree = std::stoul(argv[4]);
  p.reservoir = std::max<size_t>(p.reservoir, p.degree);
  p.fanout = {10, 3, 1};
  if (argc > 6) p.alpha = std::stod(argv[6]);
  if (argc > 7) p.seed = std::stoull(argv[7]);
  if (argc > 8) p.local_k = std::stoul(argv[8]);
  fastpipnn::Index<float> index;
  index.build(data.data(), n, dim, dim, p);
  std::vector<uint32_t> degrees(n), adjacency(n * p.degree, fastpipnn::kNone);
  for (size_t i = 0; i < n; ++i) {
    auto ns = index.neighbors(fastpipnn::Id(i));
    degrees[i] = uint32_t(ns.size());
    std::copy(ns.begin(), ns.end(), adjacency.begin() + i * p.degree);
  }
  std::ofstream out(argv[5], std::ios::binary);
  out.write(reinterpret_cast<const char*>(degrees.data()), n * sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(adjacency.data()), adjacency.size() * sizeof(uint32_t));
  std::cout << "pipnn_graph n=" << n << " degree=" << p.degree << " seconds=" << index.stats.total_seconds
            << " mean_degree=" << index.stats.mean_degree << std::endl;
}
