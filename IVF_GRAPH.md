# IVF with graphs for assignment and inside partitions: LanceDB's HNSW vs PiPNN

LanceDB uses HNSW in two places in an IVF index. This note replaces each one with a PiPNN
graph (`fastpipnn`) and measures build time, assignment quality and query cost at matched
recall.

**Answer (synthetic 1M × 128, 4 cores).** PiPNN does better than HNSW in both places, with
one condition.

- **Assignment.** Lance's HNSW over the centroids misassigns more vectors as the partition
  count grows: 6% at 1,024 partitions, 25% at 4,096, 31% at 16,384. No `nprobe` setting
  recovers those vectors, so recall stops at 0.946 at 4,096 partitions. A PiPNN graph over the
  centroids, searched with beam 32, misassigns 0.7% at 4,096 and 2.8% at 16,384, and gives the
  same recall as exact assignment at 1,024 and 4,096. Both graphs take a few seconds, against
  28 s for the exact GEMM at 16,384 partitions.
- **Inside partitions.** A PiPNN graph per partition builds 2.8–4.2× faster than Lance's HNSW
  (m 20, ef_construction 150), at equal or lower query cost at 0.90–0.99 recall.
- **The condition.** In large partitions (64 partitions of ~16k rows) half of every PiPNN graph
  was unreachable from its medoid, and recall stopped at 0.46. Each such partition held two
  well-separated clusters, and the graph had no edges between them. Seeding each search with 64
  sampled rows fixes it (1.5% more distance evaluations), after which PiPNN matches HNSW.
  HNSW's upper layers do this job for HNSW.

| End to end | Partitions | Build (train + assign + sub-index) | R@10 ≥ 0.90 | ≥ 0.95 | ≥ 0.99 |
|---|---:|---:|---:|---:|---:|
| LanceDB: HNSW assign + HNSW per partition | 1,024 | 25.2 s | 0.49 ms | 0.87 ms | not reached (0.989) |
| **PiPNN: PiPNN assign + PiPNN per partition** | 1,024 | **15.8 s** | **0.24 ms** | **0.47 ms** | 1.27 ms |
| LanceDB: HNSW assign + HNSW per partition | 4,096 | 59.4 s | 1.22 ms | not reached (0.946) | – |
| **PiPNN: PiPNN assign + PiPNN per partition** | 4,096 | 57.2 s | **0.52 ms** | **0.52 ms** | 1.29 ms |

Latency is mean single-thread time per query. k-means training is 4.4 s at 1,024 partitions
and 47 s at 4,096, the same for both rows. The LanceDB rows use hnswlib with Lance's exact
parameters. §5 checks them against real Lance.

Code: `native/src/ivf_graph_bench.cpp` (shared helpers in `native/src/bench_common.hpp`),
`native/scripts/run_ivf_graph.sh`, `native/scripts/summarize_ivf_graph.py`,
`lance_ivf_graph_check.py`. Raw results: `native/results/ivf-graph-synthetic1m/`
(`results.jsonl`, `lance.jsonl`) and `native/results/ivf-graph-tuning/`.

## 1. Where LanceDB uses HNSW

Read from `lance-format/lance` main (`74a5d28`, November 2026):

| | 1. HNSW over the centroids | 2. HNSW inside each partition |
|---|---|---|
| Job | Assign every row to its nearest centroid during the build | Search a probed partition at query time |
| Code | `SimpleIndex` in `rust/lance-index/src/vector/utils.rs`, used by `PartitionTransformer` and k-means | `IVF_HNSW_FLAT / _SQ / _PQ` sub-index (`vector/hnsw/builder.rs`) |
| Parameters | `num_edges(12)`, `ef_construction(15)`, top-1 at `ef = 15` | `m = 20` (level 0 keeps 40), `ef_construction = 150`; query `ef` defaults to max(1.5·k·refine, a minimum) |
| When | Automatic when C·d ≥ 10^6 and no AMX-FP16 GEMM; `LANCE_USE_HNSW_SPEEDUP_INDEXING=enabled/disabled` forces it | When the index type is `IVF_HNSW_*` |
| Stored | No, dropped after the build | Yes, one graph per partition |

Queries never use a graph to choose partitions: `kmeans_find_partitions_arrow_array`
(`vector/kmeans.rs`) scores the query against all C centroids and keeps the `nprobes` nearest.
The benchmark does the same.

## 2. Setup

- **Data:** `native/scripts/synthetic.py --n 1000000 --queries 1000`: 1M × 128-d uint8 from a
  128-component Gaussian mixture, 1,000 held-out queries, exact top-100 ground truth. It is the
  same set as `DISTRIBUTED_EXPERIMENTS.md` and `DISK_INDEX.md`. k = 10.
- **Centroids:** Lloyd k-means, 10 iterations, on 256 × C sampled rows, as Lance trains them.
- **Assignment:** `flat` is exact, one GEMM per tile. `hnsw:15` is hnswlib with Lance's
  `SimpleIndex` parameters (M 12, efC 15, top-1 at ef 15). `fastpipnn:B` is a fastpipnn graph
  over the centroids (degree 48, leaf 512, local_k 16, α 1.5; the router settings from
  `DISTRIBUTED_EXPERIMENTS.md`), top-1 at beam B. All are run with 4 threads.
- **Sub-index:** `flat` scans the probed partitions. `hnsw` is hnswlib with Lance's defaults
  (M 20, efC 150). `fastpipnn` is one fastpipnn graph per partition, degree 40 to match HNSW's
  level 0, with the settings from §3. Partitions with fewer than 64 rows get no graph and are
  scanned. Partitions are built in parallel on 4 threads, one partition per thread.
- **Query:** score all C centroids, take the `nprobe` nearest, search each partition for its
  top 10 with `ef` (HNSW) or beam `ef` (fastpipnn), merge. Exact squared L2 throughout (no SQ or
  PQ), so differences come from assignment and graphs only. One thread. Sweeps: `nprobe` up to
  C/16 (or 256), `ef` ∈ {10, 20, 40, 80, 160}.
- **Machine:** 4 vCPUs (Xeon, AVX-512, AMX-BF16/INT8, no AMX-FP16), 15 GB.

## 3. Making PiPNN work inside a partition

fastpipnn's defaults (leaf 1024, 1,000 depth-0 leaders, local_k 2) are tuned for one graph
over 1M points. Partitions hold 250–16,000 rows (78,000 in the largest at C = 64).
Raw: `native/results/ivf-graph-tuning/`.

1. **A leaf as large as the partition gives a k-NN graph.** With the first settings (leaf
   1024, depth-0 leaders = rows / 1000) a 1k-row partition was one leaf. Its graph was the
   2-NN graph: mean degree 3.8, recall 0.29–0.45. The fix sets the leaf size from the
   partition (rows / 8 within [64, 1024]) and the depth-0 leaders to 4 × rows / leaf. At
   1M rows that is fastpipnn's own default.
2. **Separated clusters stay separated.** With many leaders, every row's 10 nearest leaders lie
   in its own Gaussian cluster, so no leaf spans two clusters and the graph splits into
   components:

   | C = 64 (~16k rows / partition), nprobe 2 | Build (wall) | Reachable from medoid | ef 40 | ef 80 | ef 160 |
   |---|---:|---:|---:|---:|---:|
   | HNSW (M 20, efC 150) | 42.4 s | 0.999 | 0.772 | 0.907 | 0.978 |
   | fastpipnn, leaf 1024, depth-0 leaders = rows / leaf (≈ 16) | 12.8 s | 1.000 | 0.669 | 0.809 | 0.889 |
   | fastpipnn, leaf 1024, depth-0 leaders = 4 × rows / leaf (≈ 64) | 15.2 s | 0.500 | 0.388 | 0.436 | 0.461 |
   | same graph, search seeded with 16 rows | 15.2 s | – | 0.801 | 0.905 | 0.955 |
   | same graph, search seeded with 64 rows | 15.2 s | – | **0.821** | **0.927** | **0.977** |

   Raising `local_k`, the degree bound or α did not link the components (`c64-leaders-alpha.jsonl`).
   The final runs therefore search fastpipnn both from the medoid alone and with 64 seeds:
   every (rows / 64)-th row of the partition, counted as distance evaluations. That is the role
   HNSW's upper layers play. HNSW's level 0 alone is also 99.9% reachable here, because
   insertion links every new point to the existing graph.
3. **`local_k` 2 vs 4** made no difference to recall with the size-derived leaf, and 2 builds
   25–35% faster (`auto-leaf.jsonl`).

Final sub-index settings: leaf = rows / 8 within [64, 1024], depth-0 leaders = 4 × rows / leaf,
local_k 2, degree 40, α 1.2, fanout (10, 3).

## 4. Results (`native/scripts/summarize_ivf_graph.py native/results/ivf-graph-synthetic1m/results.jsonl`)

### 4.1 Assignment

| Partitions | Method | Graph build | Assign 1M rows | Distances / row | Exact nearest | Empty lists | Largest list |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1,024 | flat | – | 0.91 s | 1,024 | 1.000 | 0 | 2,824 |
| | hnsw (Lance) | 0.01 s | 1.95 s | 107 | 0.937 | 0 | 4,865 |
| | fastpipnn, beam 15 | 0.03 s | 2.20 s | 146 | 1.000 | 0 | 2,824 |
| | fastpipnn, beam 32 | 0.02 s | 4.30 s | 226 | 1.000 | 0 | 2,824 |
| 4,096 | flat | – | 3.90 s | 4,096 | 1.000 | 0 | 482 |
| | hnsw (Lance) | 0.02 s | 2.05 s | 96 | 0.753 | 25 | 6,712 |
| | fastpipnn, beam 15 | 0.10 s | 3.05 s | 166 | 0.971 | 0 | 2,580 |
| | fastpipnn, beam 32 | 0.11 s | 4.98 s | 209 | 0.993 | 0 | 1,709 |
| 16,384 | flat | – | 27.84 s | 16,384 | 1.000 | 0 | 176 |
| | hnsw (Lance) | 0.06 s | 2.77 s | 136 | 0.687 | 688 | 1,514 |
| | fastpipnn, beam 15 | 0.44 s | 5.14 s | 265 | 0.917 | 178 | 6,440 |
| | fastpipnn, beam 32 | 0.42 s | 7.90 s | 292 | 0.972 | 28 | 1,629 |

Recall@10 at equal `nprobe`, scanning the probed partitions (so assignment is the only
difference):

| Partitions | Method | | | | |
|---:|---|---:|---:|---:|---:|
| 1,024 | | np 4 | np 8 | np 16 | np 32 |
| | flat | 0.712 | 0.959 | 1.000 | 1.000 |
| | hnsw (Lance) | 0.685 | 0.928 | 0.974 | 0.981 |
| | fastpipnn, beam 32 | 0.712 | 0.959 | 1.000 | 1.000 |
| 4,096 | | np 16 | np 32 | np 64 | np 128 |
| | flat | 0.761 | 0.989 | 1.000 | 1.000 |
| | hnsw (Lance) | 0.671 | 0.890 | 0.919 | 0.946 |
| | fastpipnn, beam 15 | 0.742 | 0.965 | 0.980 | 0.989 |
| | fastpipnn, beam 32 | 0.757 | 0.983 | 0.995 | 0.998 |
| 16,384 | | np 32 | np 64 | np 128 | np 256 |
| | flat | 0.697 | 0.927 | 1.000 | 1.000 |
| | hnsw (Lance) | 0.616 | 0.833 | 0.925 | 0.938 |
| | fastpipnn, beam 15 | 0.647 | 0.858 | 0.930 | 0.956 |
| | fastpipnn, beam 32 | 0.680 | 0.903 | 0.976 | 0.986 |

- **Lance's assignment graph is the weak link at many partitions.** With ef 15 and
  ef_construction 15, it sends 25–31% of rows to a partition that is not their nearest at
  4,096–16,384 partitions. It also empties hundreds of partitions and overfills others
  (6,712 rows in one list against a median of 219). Recall then stops below 0.95 at any
  `nprobe`. Lance's own comment reports the same effect at 100M × 768 (0.832 vs 0.964 at 20k
  partitions).
- **A PiPNN graph over the centroids is far more accurate at similar cost:** 1.4–2.2× the
  distances of Lance's HNSW (it searches a wider beam) and a few seconds for 1M rows.
  The router settings from `DISTRIBUTED_EXPERIMENTS.md` were used as they were, with no tuning
  here. Beam 32 is needed from 4,096 partitions on.
- **Exact assignment is cheap up to ~4k partitions on CPU and expensive after.** It takes 28 s
  at 16,384 partitions here (no AMX-FP16), while the graphs take 3–8 s. The tree assignment of
  `LANCE_VS_PIPNN_BUILD.md` is the exact-quality alternative; this note did not rerun it.

### 4.2 Index inside each partition (flat assignment)

Cheapest setting reaching each recall (mean latency; nprobe, ef; distances per query):

| Partitions | Sub-index | Build (wall / CPU) | R@10 ≥ 0.90 | ≥ 0.95 | ≥ 0.99 |
|---:|---|---:|---|---|---|
| 64 | scan | – | 0.55 ms (np 1; 27.7k) | 0.55 ms (np 1) | 0.55 ms (np 1) |
| | HNSW (Lance) | 41.4 / 165.5 s | 0.16 ms (np 1, ef 80; 2.1k) | 0.30 ms (np 1, ef 160; 3.3k) | max 0.974 |
| | fastpipnn | 14.9 / 59.6 s | max 0.462 | – | – |
| | fastpipnn + 64 seeds | 14.9 / 59.6 s | 0.19 ms (np 1, ef 80; 2.1k) | 0.35 ms (np 1, ef 160; 3.1k) | max 0.976 |
| 256 | scan | – | 0.20 ms (np 2; 11.1k) | 0.20 ms (np 2) | 0.42 ms (np 4; 22.9k) |
| | HNSW (Lance) | 34.4 / 137.7 s | 0.48 ms (np 2, ef 160; 5.1k) | 1.01 ms (np 4, ef 160; 9.4k) | max 0.986 |
| | fastpipnn | 11.7 / 46.7 s | 0.47 ms (np 2, ef 160; 4.9k) | 0.94 ms (np 4, ef 160; 9.1k) | max 0.976 |
| | fastpipnn + 64 seeds | 11.7 / 46.7 s | 0.29 ms (np 2, ef 80; 3.5k) | 0.97 ms (np 4, ef 160; 9.2k) | max 0.986 |
| 1,024 | scan | – | 0.20 ms (np 8; 9.4k) | 0.20 ms (np 8) | 0.34 ms (np 16; 18.0k) |
| | HNSW (Lance) | 21.8 / 87.2 s | 0.47 ms (np 8, ef 40; 6.1k) | 0.71 ms (np 8, ef 80; 7.7k) | 1.38 ms (np 16, ef 80; 14.1k) |
| | fastpipnn | 6.9 / 27.6 s | 0.29 ms (np 8, ef 20; 4.6k) | 0.63 ms (np 8, ef 80; 7.5k) | 1.26 ms (np 16, ef 80; 13.8k) |
| 4,096 | scan | – | 0.29 ms (np 32; 12.1k) | 0.29 ms (np 32) | 0.45 ms (np 64; 20.1k) |
| | HNSW (Lance) | 21.5 / 85.9 s | 0.54 ms (np 32, ef 10; 9.7k) | 0.72 ms (np 32, ef 20; 11.1k) | 1.98 ms (np 64, ef 40; 20.0k) |
| | fastpipnn | 5.1 / 20.4 s | 0.54 ms (np 32, ef 10; 9.7k) | 0.54 ms (np 32, ef 10) | 1.30 ms (np 64, ef 20; 17.3k) |

- **PiPNN builds the per-partition graphs 2.8–4.2× faster** (wall and CPU alike) and costs
  the same or less per query at 0.90–0.99 recall, as long as large partitions are searched
  with seeds. Where the graphs are connected (C ≥ 1,024, every graph ≥ 99.8% reachable) seeds
  add ~5% distance evaluations and do not raise recall, so they are only worth it for large
  partitions.
- **Graphs pay only in large partitions.** On this data, scanning wins from 256 partitions on:
  probing a few hundred to a few thousand rows is cheaper than a graph search's
  random access. The scan's distance count is 2–3× the graph's, so the crossover moves
  toward graphs as d grows or when vectors are compressed (SQ, PQ, RaBitQ), which these runs
  do not model.
- **The data is easy for IVF:** neighbors sit in one Gaussian cluster, so one partition holds
  most of them at C = 64. Real embeddings have softer cluster boundaries. That makes PiPNN's
  connectivity problem (§3) milder, but it needs more `nprobe` for a given recall.

### 4.3 End to end

Every combination at 1,024 and 4,096 partitions is in the summarizer output. The table at the
top picks the two pipelines (fastpipnn rows: medoid entry, since all graphs are connected at
these sizes). Swapping one stage at a time:

- At 1,024 partitions, replacing Lance's assignment alone (`fastpipnn:32` + HNSW) brings
  ≥ 0.99 in reach (1.39 ms). Replacing the sub-index alone (`hnsw:15` + fastpipnn) cuts the
  build from 25.2 to 13.3 s but keeps the 0.986–0.989 ceiling, which comes from assignment.
- At 4,096 partitions, assignment decides. Every pipeline with Lance's HNSW assignment stops
  at 0.942–0.946. With `fastpipnn:32` or flat assignment, every sub-index reaches 0.99.

## 5. Real LanceDB check

(pending)

## 6. Caveats

- Synthetic data (128 well-separated Gaussian clusters, uint8, 128-d). It exaggerates both
  the graph connectivity problem (§3) and how well a single partition answers a query.
- The LanceDB baseline is hnswlib with Lance's parameters, not Lance's Rust HNSW. Lance's
  HNSW caps the number of levels at 7 and differs in small details. §5 compares recall with
  real Lance.
- No quantization: Lance's usual `IVF_HNSW_SQ` / `IVF_HNSW_PQ` and `IVF_RQ` would change the
  absolute numbers and the scan-vs-graph crossover, but not the assignment results.
- One build per configuration and 4 cores. Build times are wall clock on a shared VM.
- The PiPNN centroid graph and per-partition graph settings were tuned on this data (§3), and
  HNSW used Lance's defaults. Lance's defaults are what LanceDB users get, so that is the
  comparison that matters, but a tuned HNSW would do better than shown.

## 7. Reproduce

```bash
python3 native/scripts/bootstrap.py
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release && cmake --build native/build -j --target ivf_graph_bench
python3 native/scripts/synthetic.py --n 1000000 --queries 1000 --output native/results/synthetic1m
native/scripts/run_ivf_graph.sh
python3 native/scripts/summarize_ivf_graph.py native/results/ivf-graph-synthetic1m/results.jsonl
uv run --locked --extra lance python lance_ivf_graph_check.py --assign-graph disabled --partitions 64 1024 4096
uv run --locked --extra lance python lance_ivf_graph_check.py --assign-graph enabled --partitions 64 1024 4096
```
