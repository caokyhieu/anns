# Lance index build experiments

Implementation and benchmark workspace for [`LANCE_INDEX_BUILD_PLAN.md`](../LANCE_INDEX_BUILD_PLAN.md):
tree centroid assignment, PiPNN (centroid assigner and `IVF_PIPNN_FLAT` partition graphs) and
the x86/ARM kernels they use, all inside Lance's IVF build pipeline.

## Reproduce on a new machine (x86_64 or aarch64)

```bash
git clone <this repo> anns && cd anns          # or: git pull
./lance_index_build/bench.sh doctor            # CPU features (NEON/SVE/SVE2), toolchain, data
./lance_index_build/bench.sh setup             # Lance = upstream base + patches/, Eigen & C++ deps
./lance_index_build/bench.sh data              # BIGANN10M + GIST1M, verified by SHA-256
./lance_index_build/bench.sh test              # kernel / tree / PiPNN / k-means correctness
./lance_index_build/bench.sh kernels           # criterion microbenchmarks (+ no-SVE build on ARM)
./lance_index_build/bench.sh suite smoke       # minutes: every backend end to end on 100k
./lance_index_build/bench.sh suite A B C ablation
./lance_index_build/bench.sh report            # summary.md + SVG plots per run
# or everything in order:
./lance_index_build/bench.sh all
```

Choose datasets with `--data` (or `BENCH_DATA`, default `bigann,gist`). `data` then fetches
only those, and suite runs on an unselected or missing dataset are skipped and listed at the
end of the suite, so a BIGANN-only host still runs every BIGANN experiment:

```bash
./lance_index_build/bench.sh --data bigann all     # skips A-gist-k1000 and abl-gist-*
```

`suites/ablation.txt` repeats the assignment ablations on BIGANN10M (K=4096), so the
SVE / NEON / no-tile comparison exists without GIST.

Requirements: git, a C/C++ compiler (Clang ≥ 14 or GCC ≥ 11 for SVE intrinsics), cmake,
python3, uv, curl and rustup (Lance pins Rust 1.98.1 in `rust-toolchain.toml`; `setup`
installs it). Disk: ~13 GB raw data plus ingested Lance datasets and shuffle spill;
Suite C at 10M needs tens of GB of RAM (run the 1M memory pilot first, as `suites/C.txt` does).

What the pieces are:

| Path | Content |
|---|---|
| `bench.sh` | single entry point (subcommands above) |
| `patches/` | the Lance changes as a `git am` series on upstream `LANCE_BASE`; `SERIES.env` pins the base and the resulting tree hash, which `setup` verifies |
| `worktrees/lance/` | generated checkout (git-ignored) |
| `suites/*.txt` | one run per line: `<config> <run-name> [KEY=VALUE ...]` |
| `configs/*.env` | versioned experiment definitions; paths come from `ANNS_DATA` / `DATASETS_DIR` |
| `datasets.lock.json` | dataset sources, sizes and SHA-256 |
| `scripts/run.sh` | builds the runner and executes one config with provenance |
| `scripts/report.py` | summary tables and recall/latency/build-time plots |
| `results/<arch>-<host>/<run>-<stamp>/` | `manifest.json` (git tree, patch hash, CPU features, **selected kernels**), `builds.jsonl`, `queries.csv`, `assignments.jsonl`, `logs/` (git-ignored); `built.jsonl` and `searches.jsonl` are written as each build / search setting finishes, so a killed run keeps them and `report.py` lists it as partial |
| `baseline/` | the pre-existing optimized IVF_RQ patch and the files it touches |

Every manifest records `kernels.{dot_tile, blocked_l2, pipnn}`, so an x86 and an ARM run
are self-describing. Cross-machine timings are context, not speedup ratios (plan §5.5).

### Kernel ablations

| Switch | Effect | Kind |
|---|---|---|
| `LANCE_DISABLE_BLOCKED_L2=1` | exact assignment / k-means use per-vector distances instead of 4x4 tiles | runtime |
| `LANCE_SIMD_MAX_WIDTH=256` | hide AVX-512 from runtime dispatch: every dispatched kernel is ≤ 256-bit (equal-width ablations) | runtime |
| `LANCE_L2_BATCH_AVX512=1` | per-vector f32 L2 batch on 512-bit registers (default dispatch has no AVX-512 batch) | runtime |
| `LANCE_DISABLE_SVE_BUILD=1` | build without SVE objects (NEON on aarch64); `run.sh` uses `target-nosve/` | build |
| `LANCE_PIPNN_SCALAR=1` | scalar PiPNN distance kernel and leaf top-k scan | runtime |
| `LANCE_PIPNN_NATIVE=1` | compile the PiPNN builder with `-march=native` (label such runs) | build |

`suites/ablation.txt` runs the default and each switched-off variant on identical configs.

## ARM / x86 kernels

All kernels sit behind Lance's CPU dispatcher (`lance_core::dispatch!`), select once per
process, and keep a portable fallback; nothing is compiled with a global `-march`.

| Kernel | Candidates | Used by |
|---|---|---|
| `lance_linalg::distance::tile::dot_tile` — 4x4 f32 dot-product tile | AVX-512, AVX2+FMA, **SVE**, **NEON**, scalar | `blocked_l2`: k-means training, hierarchical node k-means, exact IVF assignment |
| f32 L2 batch, one query vs N rows | AVX(+FMA), **SVE 4-row** (bit-identical to the pair kernel), scalar | tree beam expansion (contiguous child blocks), per-vector k-means |
| f32 / u8 pairwise L2 and dot, 4-bit LUT sums | existing AVX*, SVE/SVE2 (from `lance_arm_sve/`) | everything |
| PiPNN `sqdist` | SSE2, **NEON**, **SVE** (HWCAP-detected), scalar | RobustPrune, medoid |
| PiPNN leaf top-k scan | AVX-512 (`-march=native`), **NEON**, scalar | leaf k-NN |

Blocked assignment ranks with `|x|^2 + |c|^2 - 2x.c` and re-scores near-ties exactly, so it
returns the same centroid as the per-vector path except for > 4 candidates tied within f32
rounding. The tree's children are stored contiguously (breadth-first numbering), so each
expansion is one batch-kernel call.

Correctness of the ARM paths was checked here by cross-compiling the Lance test binaries for
`aarch64-unknown-linux-gnu` and running them under QEMU at SVE widths 128/256/512/2048 bits
and with SVE disabled. QEMU proves correctness only; ARM performance needs `bench.sh
kernels` and the suites on real hardware.

## Implementation status

| Plan phase | State | Where |
|---|---|---|
| 0 Baseline | Done: worktree, patch hash, data hashes / GT layout validated | `baseline/`, plan §5.1 |
| 1 Benchmark foundation | Done: runner (GIST + BIGANN, seeds, stage spans, per-query timing, raw outputs, assignment-quality sampling), `report.py`, `bench.sh` | `rust/lance-index-bench`, `bench.sh` |
| 2 Retained hierarchy | Done; greedy/beam/exhaustive; parity tests | `lance-index/src/vector/assignment_tree.rs`, `kmeans.rs` |
| 3 Tree training integration | Done for full-dataset assignment; refinement stays exact and refreshes routing centers | `lance/src/index/vector/ivf.rs`, `IvfBuildParams` |
| 4 PiPNN centroid backend | Done (optional `pipnn` feature, C++ bridge, Rust beam search) | `rust/lance-pipnn`, `lance-index/src/vector/pipnn.rs` |
| 5 PiPNN partition graphs | `IVF_PIPNN_FLAT` builds, persists, reopens and searches (float32 L2); remap / incremental / fresh-segment return `NotSupported`; Python bindings pending | `SubIndexType::Pipnn`, `StageParams::Pipnn`, `VectorIndexParams::ivf_pipnn` |
| 6 ARM kernels | Done for P0/P1: dot tiles (NEON/SVE), SVE L2 batch, tree block routing, PiPNN NEON/SVE distance + NEON leaf scan; QEMU-verified; native ARM measurements pending | `lance-linalg/src/distance/tile.rs`, `simd/sve.c`, `lance-pipnn/cpp/kernels.h` |
| 7 Full evaluation | Not started | `configs/suite*.env` |

### API added

```rust
IvfBuildParams {
    assignment: CentroidAssignment::{Auto, Exact, Hnsw{m, ef_construction, ef},
                                     Pipnn{params, beam} /* feature pipnn */, Tree{beam}},
    assignment_tree: Option<Arc<AssignmentTree>>, // with caller-supplied centroids
    seed, hierarchical_k, refine_iters,
    ..
}
VectorIndexParams::ivf_pipnn(metric, ivf, PipnnBuildParams)       // feature pipnn
lance_index::vector::ivf::PreparedAssignment::{prepare, assign}  // measurement API
```

Explicit backends never silently fall back; `Auto` preserves Lance's existing behaviour
(including its environment switches).

## Running single configs

```bash
cd lance_index_build
scripts/run.sh configs/smoke-bigann100k-assign.env                     # RUN_ID defaults to <config>-<stamp>
scripts/run.sh configs/smoke-bigann100k-graphs.env my-run INDEX=ivf_pipnn_flat PIPNN_DEGREE=48
```

`KEY=VALUE` arguments after the run id override the config and are saved in
`overrides.env`; values in the config override the calling shell environment.
Branch-default PiPNN `leaf_size=1024` must not be used for centroid graphs with
K ≤ ~1024 (see Smoke results). `protoc` is not needed: the runner is a separate
workspace crate depending only on library crates (`cargo test -p lance` does need it).

Runner semantics (see the header of `rust/lance-index-bench/src/main.rs`):

- `MODE=fixed` trains one model per repeat and gives identical centroids (and the retained
  tree) to every backend; assignment quality vs exact is measured on `ASSIGN_SAMPLE`
  seeded base vectors outside build timing. `MODE=complete` trains inside every build.
- Build timing spans `create_index_builder(..).await` (training → commit). Stage spans
  come from Lance's progress callbacks. Peak RSS is reset per build (`/proc/self/clear_refs`).
- Queries: deterministic 20 % tuning / 80 % final split (`QUERY_SET`). Query buffers are
  prepared before timing; recall bookkeeping happens after each timer stops; `WARMUP`
  passes precede each timed pass (results labelled `cache: warm`). Recall uses unique
  returned ids, so missing results reduce recall.
- `N_LIMIT` takes a prefix of the base set and recomputes exact ground truth for it.
- Configuration order is shuffled deterministically per repeat.

## Smoke results (x86, 7 Nov 2026; correctness only, not performance claims)

BIGANN 100k prefix, exact ground truth recomputed, tuning queries (≈2,000), warm cache.

Fixed-model assignment, K=1024, 20k sampled vectors (`results/smoke-pipnn-sweep2`):

| Backend | ID agreement with exact | Notes |
|---|---|---|
| tree:1 (greedy) | 0.739 | |
| tree:8 / tree:16 | 0.996 / 0.9998 | |
| hnsw (Lance default 12/15/15) | 0.994 | existing sequential per-batch search path |
| pipnn:32, leaf 128 | 0.994 | graph fully reachable, mean degree 6.6 |
| pipnn:64, leaf 128 | 0.9998 | |
| pipnn:32, leaf 1024 (branch default) | **rejected** | single leaf → only 2.1 % of centroids reachable |

**Finding:** with the branch defaults, any input of ≤ `leaf_size` points is one leaf and
keeps only its `local_k` edges; at d=128 the 1,024-centroid graph reached 2.1 % of
vertices from the entry point (98 % of partitions could never be chosen). The PiPNN
centroid assigner now fails with an explicit error when the graph is not fully reachable,
and centroid configs use `leaf_size ≪ K`. Graph stats are logged per run
(`centroid_graph` in `assignments.jsonl`).

IVF_RQ5 recall at K=1024 matched exact assignment for auto/hnsw/pipnn-leaf128 within
±0.002 recall@10 at nprobes=64; tree:1 lost ≈0.005.

Partition graphs, K=5, `IVF_HNSW_FLAT` (m=20, efc=150) versus `IVF_PIPNN_FLAT`
(degree 64, leaf 1024), with 32 threads:

| | build | CPU | index | recall@10 nprobes=5 ef=64 / 128 |
|---|---|---|---|---|
| HNSW_FLAT | 2.25 s | 40.6 s | 71.6 MB | 0.996 / 0.999 |
| PIPNN_FLAT | 0.74 s | 7.6 s | 53.9 MB | 0.981 / 0.996 |

Parameters were not matched or tuned: this only demonstrates the full build →
commit → reopen → search path.

## Known gaps / next steps

1. Native ARM runs: `bench.sh all` on the Kunpeng host (kernels, smoke, A/B/C, ablation).
   Nothing here is an ARM performance measurement yet.
2. Full A/B/C evaluation with repeats on x86 (pilots exist under `results/`).
3. PiPNN filtered search: below 10 % selectivity it scans allowed rows exactly; above, it
   walks the graph under the mask. Deletion/filter recall tests on a full Lance dataset pending.
4. Python bindings and index-inspection naming for `IVF_PIPNN_FLAT` (reported as generic
   `IndexType::Vector` today).
5. `SubIndexType::Pipnn` is gated on `lance-index/pipnn` while `lance` matches it under its
   own `pipnn` feature; enabling only the former in a downstream build breaks exhaustive
   matches (enable both, as `lance-index-bench` does).
6. Approximate assignment during global refinement (plan §3.2 step 3) is not implemented —
   refinement is exact; a tree-routed refinement must reject nonzero balance bias.
7. Not yet ARM-specialized: PiPNN leader-routing top-f selection (scalar), uint8 Gram tiles
   for byte input (Lance uses float32), and the leaf/routing GEMMs (Eigen's own NEON
   kernels). HashPrune hashing/lookup already has NEON. SME is out of scope until
   hardware support is confirmed.
