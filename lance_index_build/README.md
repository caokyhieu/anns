# Lance index build experiments

Implementation and benchmark workspace for [`LANCE_INDEX_BUILD_PLAN.md`](../LANCE_INDEX_BUILD_PLAN.md).

- `worktrees/lance/` — isolated Lance checkout (branch `index-build-tree-pipnn`, latest commit
  `fcd378707`; base
  `04e61996311dc9907b8b588cad2503a2b8213a08`) carrying all changes below.
- `baseline/` — the user's pre-existing optimized IVF_RQ patch (`lance-existing.patch`,
  SHA-256 `ffeb70a6…37e2`) and the original files it touches.
- `configs/` — versioned experiment definitions (`KEY=value` env files).
- `scripts/run.sh` — builds the runner and executes one config with provenance.
- `results/<run-id>/` — `manifest.json`, `builds.jsonl`, `queries.csv`,
  `assignments.jsonl`, `config.env`, `logs/`.

## Implementation status

| Plan phase | State | Where |
|---|---|---|
| 0 Baseline | Done: worktree, patch hash, data hashes / GT layout validated | `baseline/`, plan §5.1 |
| 1 Benchmark foundation | Runner done (GIST + BIGANN, seeds, stage spans, per-query timing, raw outputs, assignment-quality sampling); plots/report scripts pending | `rust/lance-index-bench` |
| 2 Retained hierarchy | Done; greedy/beam/exhaustive; parity tests | `lance-index/src/vector/assignment_tree.rs`, `kmeans.rs` |
| 3 Tree training integration | Done for full-dataset assignment; refinement stays exact and refreshes routing centers | `lance/src/index/vector/ivf.rs`, `IvfBuildParams` |
| 4 PiPNN centroid backend | Done (optional `pipnn` feature, C++ bridge, Rust beam search) | `rust/lance-pipnn`, `lance-index/src/vector/pipnn.rs` |
| 5 PiPNN partition graphs | `IVF_PIPNN_FLAT` builds, persists, reopens and searches (float32 L2); remap / incremental / fresh-segment return `NotSupported`; Python bindings pending | `SubIndexType::Pipnn`, `StageParams::Pipnn`, `VectorIndexParams::ivf_pipnn` |
| 6 ARM kernels | Existing SVE/SVE2 pairwise kernels ported with runtime dispatch; new P0 tiled routing / leaf kernels not started; no native ARM run yet | `lance-linalg/src/simd/{sve,sve2}.c` |
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

## Build prerequisites

- Rust 1.98.1 (pinned by Lance).
- Eigen headers for the `pipnn` feature: `EIGEN_INCLUDE_DIR=…/anns/native/third_party/eigen`
  (`scripts/run.sh` sets this).
- The PiPNN C++ builder compiles with portable flags. `LANCE_PIPNN_NATIVE=1` adds
  `-march=native` (enables the reference AVX-512 paths); every manifest records the
  flags actually linked (`pipnn_native_flags`). Label such runs separately.
- `protoc` is **not** installed. Lance's own dev-dependencies need it, so `cargo test -p lance`
  and `cargo bench -p lance` fail. The runner is therefore a separate workspace crate
  that depends only on library crates. `cargo test -p lance-index --features pipnn` works.

## Running

```bash
cd lance_index_build
scripts/run.sh configs/smoke-bigann100k-assign.env           # fixed-model assignment smoke
INDEX=ivf_hnsw_flat scripts/run.sh configs/smoke-bigann100k-graphs.env
INDEX=ivf_pipnn_flat scripts/run.sh configs/smoke-bigann100k-graphs.env
```

Configs are `source`d after the shell environment, so a value set in the config wins;
edit a copy for sweeps. Branch-default PiPNN `leaf_size=1024` must not be used for
centroid graphs with K ≤ ~1024 (see Smoke results).

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

1. Run smoke configs, then Suite A/B pilots on x86 (plan §8 phases 3–4 exit conditions).
2. 1M-scale memory pilot for `IVF_PIPNN_FLAT` before 10M Suite C (plan §6).
3. Filtered search on PiPNN uses the prefilter mask during beam search; deletion/filter
   recall tests in a full Lance dataset are still pending.
4. Python bindings and index-inspection naming for `IVF_PIPNN_FLAT` (reported as generic
   `IndexType::Vector` today).
5. New ARM kernels (P0 tiled L2 + fused argmin/top-k; leaf Gram tiles) and native ARM runs.
6. Approximate assignment during global refinement (plan §3.2 step 3) is not implemented —
   refinement is exact; a tree-routed refinement must reject nonzero balance bias.
7. Report/plot scripts over `builds.jsonl` / `queries.csv`.
