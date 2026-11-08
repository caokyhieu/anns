# V6 sources

| ID | Source | Slides |
|---|---|---|
| RA | lance_index_build/results/ivf_rq_assignment_results.md: runs pilotB-bigann10m-k16384, pilotA-gist-k16384, probeA-gist-assign (2× AMD EPYC 9124, Lance fcd3787076 + patch 3853233ca7e7, REPEATS=1) | 2, 4, 10, 11, A1 |
| RA1K | lance_index_build/results/pilotA-gist-k1000/summary.md | 11 |
| RC | lance_index_build/results/pilotC_hnsw_vs_pipnn_results.md: run pilotC-bigann10m-k5 (patch c86a06454cfd) | 2, 4, 13, A2 |
| MEM | lance_index_build/results/pilotC-mem-1m/builds.jsonl | A4 |
| RG | lance_ivf_rq/README.md, results/results.jsonl (4-core Xeon, GIST1M, 2 repeats) | 2, 4, 6, A3 |
| SVE | lance_arm_sve/README.md, lance-arm-sve.patch (Lance 6e18b21) | 5, 7, 8, 9 |
| NAT | native/results/my-run/summary.json (standalone C++, BIGANN10M uint8, 4 threads) | 13, A4 |
| PLAN | LANCE_INDEX_BUILD_PLAN.md (7 Nov 2026) | 5, 7–9, 14, A4 |
| LSRC | Lance rust/lance/src/dataset.rs L242 (6 GiB index cache); kmeans.rs (HIER_K=16); vector/utils.rs (K×D ≥ 1M helper threshold) | 10, 13, A2 |
| P1 | Rubel et al., PiPNN, KDD 2026, arXiv:2602.21247v3 | 12 |
| ACLE | Linux arm64 SVE ABI; Arm ACLE | 7 |
| ADS | ADS-PLAN.pptx slide 1 (internal) | 3 |
