# Lance index-build experiments: results of record

Only results that hold up are listed; excluded ones are at the end with the reason.
`n` = independent runs per configuration. With n = 1, treat ±1 s build and ±0.005 recall as noise.

## Machine and code (all runs unless stated)

| item | value |
|---|---|
| CPU | 2 × AMD EPYC 9124 (Zen 4), 32 cores, SMT off, 2 NUMA nodes; AVX2, FMA, AVX-512 (F/BW/CD/DQ/VL/VNNI/VPOPCNTDQ/VBMI/BF16); no AMX |
| Cache / RAM | L2 1 MiB per core, L3 128 MiB total; 125.6 GiB RAM |
| Host | shared with other users (1-min load 3–30 during runs) |
| Lance | upstream `04e6199` + patch series in `patches/` (rev per experiment below) |
| Datasets | BIGANN10M: 10M × 128, uint8 converted exactly to f32, L2 · GIST1M: 1M × 960 f32, L2 |
| Recall | recall@10 vs exact top-100, tuning split (BIGANN 1,970 queries, GIST 193) |

---

## E1. Tree assignment: BIGANN10M, IVF_RQ, K = 16,384, fixed centroids

Run `pilotB-bigann10m-k16384` · Lance `fcd3787076` · `MODE=fixed` (centroids trained once, 17.7 s, excluded) · `NUM_BITS=5` · n = 1.
`exact` here already uses the blocked AVX-512 L2 kernel (in `fcd3787076`), so the speedups are against an optimized exact baseline.

| assign | build s | vs exact | assign thread-s | match vs exact | R@10 np16 | np64 | np256 |
|---|---|---|---|---|---|---|---|
| exact | 37.67 | 1.00× | 760.9 | 1.0000 | 0.7724 | 0.9066 | 0.9430 |
| **tree:4** | **15.11** | **2.49×** | **59.4** | 0.9173 | 0.7680 | 0.9062 | 0.9442 |
| tree:16 | 18.97 | 1.99× | 137.3 | 0.9968 | 0.7726 | 0.9083 | 0.9443 |
| tree:1 | 14.96 | 2.52× | 49.5 | 0.5778 | 0.7128 | 0.8808 | 0.9371 |
| pipnn:15:64:256 | 18.43 | 2.04× | 174.2 | 0.9624 | 0.7651 | 0.9013 | 0.9417 |
| hnsw:12:15:32 | 21.77 | 1.73× | 271.4 | 0.9638 | 0.7701 | 0.9060 | 0.9435 |
| hnsw (Lance default params) | 27.07 | 1.39× | 376.0 | 0.9520 | 0.7652 | 0.9026 | 0.9423 |

Match = same centroid as exact on a 100k sample. Full table (12 methods): `results/ivf_rq_assignment_results.md`.
Result: tree:4 cuts the build 2.49× and assignment CPU 12.8×; recall within 0.005 of exact. Centroid graphs (HNSW, PiPNN) are slower than the tree at equal match.

## E2. Tree assignment: GIST1M, IVF_RQ, K = 16,384, centroids retrained per build

Run `pilotA-gist-k16384` · `MODE=complete`, `SAMPLE_RATE=32`, `NUM_BITS=5` · n = 1. Recall also varies with the retrained centroids.

| assign | build s | vs exact | assign thread-s | train_ivf s | R@10 np64 | np256 |
|---|---|---|---|---|---|---|
| exact | 31.38 | 1.00× | 384.7 | 9.33 | 0.7259 | 0.8984 |
| auto (Lance default = centroid HNSW) | 24.25 | 1.29× | 125.6 | 9.72 | 0.7192 | 0.9057 |
| **tree:4** | **22.97** | **1.37×** | 18.9 | 11.79 | 0.7062 | 0.8979 |
| tree:1 | 19.89 | 1.58× | 24.7 | 9.75 | 0.6964 | 0.8964 |

Fixed-model match (`probeA-gist-assign`): tree:1 0.6712, tree:4 0.9635, tree:16 0.9995.

## E3. Tree assignment: GIST1M, K = 1,000 (negative result)

Run `pilotA-gist-k1000` · `MODE=complete`, `NUM_BITS=5` · n = 1.

| assign | exact | auto | tree:4 | tree:16 |
|---|---|---|---|---|
| build s | 11.73 | 11.56 | 11.56 | 11.53 |

Result: no gain at small K; assignment is already cheap.

## E4. PiPNN vs HNSW partition graphs: BIGANN10M, K = 5

Run `pilotC-bigann10m-k5` · Lance `fcd3787076` + patch `c86a06454cfd` · `MODE=complete`, `ASSIGN=exact`, FLAT f32 storage · one process per index · n = 1.
HNSW: M 16, ef_construction 100. PiPNN: degree 32, leaf 1024, local_k 2, reservoir 160, alpha 1.2, 32 threads, partitions built one at a time.

| index | build s | CPU s | peak RSS GB | index MB |
|---|---|---|---|---|
| IVF_HNSW_FLAT | 537.72 | 12,590 | 16.16 | 6,993.5 |
| IVF_PIPNN_FLAT | **109.90** | **1,830** | 17.67 | **5,915.4** |
| PiPNN vs HNSW | **4.9× faster** | **6.9× less** | +1.5 GB | −15% |

Recall@10 (k = 10; the 10 settings measured on both):

| nprobes | ef | HNSW | PiPNN |
|---|---|---|---|
| 1 | 32 / 64 / 128 / 256 / 512 | 0.7444 / 0.8023 / 0.8391 / 0.8545 / 0.8601 | 0.7669 / 0.8214 / 0.8478 / 0.8565 / 0.8593 |
| 2 | 32 / 64 / 128 / 256 / 512 | 0.8452 / 0.9089 / 0.9492 / 0.9655 / 0.9713 | 0.8742 / 0.9346 / 0.9636 / 0.9732 / 0.9762 |

PiPNN is higher at 9 of 10 settings, −0.0008 at nprobes 1, ef 512. The query sweep stopped early (10–11 of 50 settings); no k = 100, no nprobes ≥ 3.

## E5. PiPNN memory pilot: BIGANN 1M prefix, K = 5

Run `pilotC-mem-1m` · same settings as E4 · one process per index · n = 1.

| index | build s | CPU s | peak RSS GB |
|---|---|---|---|
| IVF_HNSW_FLAT | 42.35 | 1,004.7 | 3.20 |
| IVF_PIPNN_FLAT | **8.60** (4.9×) | 152.8 | 5.08 |

## E6. Standalone C++ graph builders: BIGANN10M (cross-check, not Lance)

`native/results/my-run/summary.json` · uint8 squared L2 · 4 build threads, 1 query thread · 200 queries · n = 1.
Graph settings differ from E4 (HNSW M 32, ef_c 200; PiPNN degree 64, leaf 1024).

| builder | build s | peak GB | R@10 beam 32 | beam 64 | beam 128 | QPS beam 64 |
|---|---|---|---|---|---|---|
| HNSW | 1,556.5 | 6.07 | 0.8925 | 0.9575 | 0.9845 | 5,025 |
| PiPNN (original) | 1,128.5 | 20.15 | 0.9000 | 0.9565 | 0.9860 | 3,198 |
| FastPiPNN | **222.3** (7.0×) | 18.85 | 0.8880 | 0.9550 | 0.9860 | 3,787 |

## E7. Constant RaBitQ scale: GIST1M, IVF_RQ, K = 1,000

`lance_ivf_rq/results/results.jsonl` · 4-core Intel Xeon 2.8 GHz (AVX-512, no AMX), not the EPYC · Lance `04e6199` + `lance-ivf-rq-build-opt.patch` · n = 2, means.
Both rows of each pair use the same distance kernel; only `LANCE_RQ_CONST_EX_SCALE` differs.

| bits | scale | build s | shuffle s | speedup | R@10 np20 | np50 | np100 |
|---|---|---|---|---|---|---|---|
| RQ5 | per-vector search | 56.8 | 41.1 | 1.00× | 0.800 | 0.912 | 0.949 |
| RQ5 | constant | 35.0 | 19.4 | **1.62×** | 0.792 | 0.910 | 0.946 |
| RQ9 | per-vector search | 190.3 | 172.6 | 1.00× | 0.799 | 0.917 | 0.956 |
| RQ9 | constant | 38.5 | 21.9 | **4.94×** | 0.800 | 0.919 | 0.956 |

Recall cost: up to −0.008 (RQ5, np20), ≤ 0.003 at np ≥ 50. One dataset only.

## E8. Blocked 4×4 tile vs per-vector L2 at equal SIMD width (microbenchmark)

Run `results/x86_64-coding-agentic/tilewidth-micro-20261109T102520Z` · Lance tree `4881104f` (series + patch 0005).
Lance's public `compute_partitions_arrow_array` (production exact-assignment path) on real base rows; centroids = K base rows at a fixed stride.
Time = median over 2 interleaved rounds of the per-run median of 3 timed calls (1 warm-up). 1-thread runs pinned to CPU 2.

| arm | environment | assignment | tile | L2 batch | L2 pair |
|---|---|---|---|---|---|
| P256 | `LANCE_SIMD_MAX_WIDTH=256 LANCE_DISABLE_BLOCKED_L2=1` | per-vector | – | AVX+FMA | AVX+FMA |
| B256 | `LANCE_SIMD_MAX_WIDTH=256` | blocked | AVX2+FMA | AVX+FMA | AVX+FMA |
| P512 | `LANCE_L2_BATCH_AVX512=1 LANCE_DISABLE_BLOCKED_L2=1` | per-vector | – | AVX-512 | AVX-512 |
| B512 | `LANCE_L2_BATCH_AVX512=1` | blocked | AVX-512 | AVX-512 | AVX-512 |

Speedup = per-vector time ÷ blocked time at the same width.

| data | D | K | table MiB | rows | threads | P256 s | B256 s | **256-bit** | P512 s | B512 s | **512-bit** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BIGANN | 128 | 1,024 | 0.50 | 20k | 1 | 0.233 | 0.112 | **2.07×** | 0.181 | 0.118 | **1.54×** |
| BIGANN | 128 | 4,096 | 2.00 | 20k | 1 | 0.937 | 0.384 | **2.44×** | 0.621 | 0.415 | **1.50×** |
| BIGANN | 128 | 16,384 | 8.00 | 20k | 1 | 3.839 | 1.595 | **2.41×** | 2.956 | 1.705 | **1.73×** |
| BIGANN | 128 | 1,024 | 0.50 | 500k | 32 | 0.209 | 0.111 | **1.88×** | 0.158 | 0.116 | **1.36×** |
| BIGANN | 128 | 4,096 | 2.00 | 500k | 32 | 0.820 | 0.372 | **2.20×** | 0.598 | 0.419 | **1.43×** |
| BIGANN | 128 | 16,384 | 8.00 | 500k | 32 | 3.424 | 1.357 | **2.52×** | 2.638 | 1.858 | **1.42×** |
| GIST | 960 | 64 | 0.23 | 10k | 1 | 0.040 | 0.030 | **1.34×** | 0.040 | 0.028 | **1.43×** |
| GIST | 960 | 128 | 0.47 | 10k | 1 | 0.079 | 0.047 | **1.69×** | 0.078 | 0.043 | **1.79×** |
| GIST | 960 | 256 | 0.94 | 10k | 1 | 0.165 | 0.078 | **2.11×** | 0.156 | 0.073 | **2.14×** |
| GIST | 960 | 64 | 0.23 | 200k | 32 | 0.034 | 0.029 | **1.19×** | 0.033 | 0.030 | **1.10×** |
| GIST | 960 | 128 | 0.47 | 200k | 32 | 0.058 | 0.037 | **1.56×** | 0.051 | 0.041 | **1.24×** |
| GIST | 960 | 256 | 0.94 | 200k | 32 | 0.129 | 0.061 | **2.11×** | 0.114 | 0.068 | **1.69×** |

- Assignments are identical (agreement 1.000000) across all arms and cases.
- 1-thread spread < 1% across rounds; 32-thread spread up to ~±10% (shared host).
- Width alone: per-vector 512 vs 256 is 1.29–1.51× on BIGANN and 1.01–1.15× on GIST. Blocked 512 is not faster than blocked 256 on this Zen 4 (0.73–1.08×).
- "Blocked" = 4×4 register tile + 256 KB centroid cache blocks + ‖x‖²+‖c‖²−2x·c ranking with exact re-score. These three are not separated here.
- Full table, spreads, kernels: `summary.md` in the run folder.

## E9. Blocked vs per-vector at equal width, end-to-end builds

Runs `results/x86_64-coding-agentic/tw-20261109T103054Z-*` · `scripts/tile_width_ablation.sh e2e 3` · same arms and Lance tree as E8 · `ASSIGN=exact`, `NUM_BITS=5`, 32 threads · 3 interleaved rounds (arm order rotated), one process per build. Cells: median (min–max) over rounds.
Summary file: `results/x86_64-coding-agentic/tw-20261109T103054Z-summary.md`.

**BIGANN10M, IVF_RQ, K = 16,384, `MODE=fixed`** (centroids trained per process, ~16 s, excluded from build s)

| arm | build s | assign thread-s | CPU s | R@10 np16 | np64 |
|---|---|---|---|---|---|
| P256 | 79.22 (77.69–84.17) | 1,957 (1,889–2,102) | 2,153 | 0.7720 | 0.9059 |
| B256 | **39.58** (39.09–42.16) | **783** (766–892) | 963 | 0.7721 | 0.9086 |
| P512 | 63.20 (61.13–63.21) | 1,437 (1,417–1,471) | 1,681 | 0.7712 | 0.9072 |
| B512 | **41.67** (41.40–44.57) | **897** (890–972) | 1,072 | 0.7708 | 0.9050 |

| comparison | build | assignment CPU |
|---|---|---|
| **blocked vs per-vector, 256-bit** (P256 ÷ B256) | **2.00×** | **2.50×** |
| **blocked vs per-vector, 512-bit** (P512 ÷ B512) | **1.52×** | **1.60×** |
| width, per-vector (P256 ÷ P512) | 1.25× | 1.36× |
| width, blocked (B256 ÷ B512) | 0.95× | 0.87× |

**GIST1M, IVF_RQ, K = 256, `MODE=complete`** (k-means trained in every build)

| arm | build s | assign thread-s | CPU s | R@10 np4 | np16 |
|---|---|---|---|---|---|
| P256 | 11.83 (11.54–12.52) | 12.4 (12.2–17.9) | 144 | 0.6389 | 0.8922 |
| B256 | 10.65 (10.44–11.22) | 8.6 (7.5–11.8) | 120 | 0.6435 | 0.9057 |
| P512 | 10.81 (10.54–10.92) | 12.3 (11.8–13.0) | 139 | 0.6430 | 0.9005 |
| B512 | 10.74 (10.35–11.24) | 10.6 (9.5–10.7) | 123 | 0.6549 | 0.8964 |

| comparison | build | assignment CPU |
|---|---|---|
| blocked vs per-vector, 256-bit | 1.11× | 1.43× |
| blocked vs per-vector, 512-bit | 1.01× | 1.16× |

- BIGANN K = 16,384: blocking halves the build at equal width (2.00× at 256-bit, 1.52× at 512-bit); recall unchanged (±0.003).
- GIST K = 256: assignment is < 0.5 s of wall time (≈ 12 thread-s over 32 threads), so the build gain is ≤ 1 s and within run-to-run spread. Recall differences here come from k-means retraining, not the kernel (assignments are identical in E8).
- On this Zen 4, the blocked kernel gains nothing from 512-bit (B256 is as fast or faster than B512).

## E10. ARM SVE/SVE2 kernels: correctness only

`lance_arm_sve/` (Lance `6e18b21`): 6 C kernels (f32 dot/L2 pair and batch, u8 dot/L2, 4-bit LUT accumulate), 8 Rust dispatch candidates.
Clang 18 `-Wall -Wextra -Werror`; QEMU 7.2 at 128/256/512/1024/2048-bit SVE, VL changed mid-process, guard pages: all pass. **No ARM timing exists.**

---

## Excluded results

| result | why excluded |
|---|---|
| Blocked L2 2.07× on GIST1M RQ1, K = 1,000 (`lance_ivf_rq`) and the 5.74× "blocked + const" vs baseline | Tile ran AVX-512 against Lance's 256-bit AVX+FMA batch: tiling and width confounded. Replaced by E8/E9. |
| Raw P256 vs B512 or REF vs B512 ratios in E8/E9 | Same confound (different width); shown only as reference. |
| E4 p90/p95/p99 query latency | Harness artefact: Lance's default 6 GiB index cache < index size (5.9–7.0 GB), so partitions reload (~2 s) on both indexes. Only p50 is comparable. |
| Peak RSS in E1–E3 | Builds shared one process; RSS reflects run order. |
| Smoke runs (`smoke-*`, 100k rows) | Functional checks only. |
| Any Kunpeng / ARM performance number | None measured. |
