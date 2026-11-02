# Sharded index designs with PiPNN: experiment plan and results

DISTRIBUTED.md designed one global PiPNN graph spread over machines. At the target scale
(10^11–10^12 vectors) the index has to be **sharded** instead: every shard is built and
served independently, and a query is answered by a few or all shards. This note lists
the sharded designs we compare, with the LanceDB 10B architecture as the baseline, and
records the measurements.

Code: `native/src/shard_bench.cpp` (all designs, one binary), `native/scripts/run_shards.sh`
(step 1 runs), `native/scripts/summarize_shards.py` (matched-recall tables),
`lance_shard_check.py` (the same baseline on real Lance).

## 1. Designs

### D6 – baseline: LanceDB at 10B (row segments, shared k-means, HNSW router over centroids)

From [How LanceDB accelerates vector search at 10 billion scale](https://www.lancedb.com/blog/how-lancedb-accelerates-vector-search-at-10-billion-scale)
(Yang Cen, April 2026). The table is split by rows into segments; each segment has its own
IVF index built by its own worker. Centroids are trained once (a fixed cost). The query node
routes with an HNSW over the centroids, and every segment scans the same nprobe partitions.

```
 BUILD
   1. train once (fixed)      sample ─► k-means ─► C centroids  (+ RaBitQ rotation)
   2. split rows              rows 0–1B │ 1B–2B │ … │ 9B–10B
   3. one worker / segment    assign own rows to the shared centroids, RaBitQ-encode
                              ── seg 1 IVF ── seg 2 IVF ── … ── seg S IVF
   4. merge / commit          one logical index (fixed)

 QUERY
   query node: HNSW over C centroids ─► nprobe partition IDs, rotate query once
        │ same partition list to EVERY segment
   executor 1 … executor S    scan own rows of those partitions (RaBitQ binary codes)
        └── top-k·refine each ─► rerank with full vectors ─► merge top-k
```

Reported: 10B random 1536-d float32 vectors, 10 segments × 1B on 10 nodes, nprobe 20,
top-100: p50 18 ms, p95 20 ms, p99 21 ms end to end. Index build about 5× faster with 10
workers; encoding scales near-linearly, centroid training and merge stay fixed. No recall is
reported. Lance only builds the centroid HNSW when C·d ≥ 10^6; below that it scans centroids.

### D1 – row shards, one fastpipnn graph per shard, query all shards

```
 vectors ──hash(id)─► shard 1 ─► fastpipnn      (no communication during build)
                      shard 2 ─► fastpipnn
                      shard S ─► fastpipnn
 query ─► coordinator ─► ALL S shards (beam b) ─► top-k each ─► merge
```

Same shape as D6 (rows split, every query hits every shard), but a graph replaces
IVF + RaBitQ inside the shard. HNSW per shard is run as the graph baseline.

### D5 – D6 with PiPNN partitioning instead of k-means

```
 sample ─► PiPNN ball carving (random leaders, one GEMM pass per level) ─► leaves
        ─► leaf means = C centroids ─► router (exact / HNSW / fastpipnn)
 per segment: assign + RaBitQ-encode own rows (same as D6) ; query as D6
```

Isolates the partition step: is carving cheaper than k-means, and are its lists as good?

### D6c – D6 with a fastpipnn graph inside each IVF partition  *(step 2)*
### D6b – partitions placed on shards by locality, query only owning shards  *(step 2)*

```
 centroids clustered into S groups ─► group g = shard g
 query ─► router ─► nprobe partitions ─► only the shards that own them
```

### D3 – PiPNN level-0 leader groups as shards, route to p shards  *(step 2)*

```
 PiPNN partition tree: level 0 leaders ─► leader groups = SHARDS (1–2 copies at the boundary)
                       levels 1..L     ─► normal PiPNN inside each shard
 router = fastpipnn over level-0 leaders; query ─► top-p shards ─► local beam search ─► merge
```

### D4 – D3 plus a few cross-shard bridge edges  *(step 3)*

```
   shard A graph           shard B graph
   ●──●──◉ ═══ bridge ═══ ◉──●──●     ≤ r_x inter-shard edges per boundary vector
 query starts in the top shard, may cross ≤ h bridges (r_x = 0 is D3; no cap is one global graph)
```

### Map of what each comparison isolates

| Comparison | Changes only | Question |
|---|---|---|
| D6 → D5 | k-means → PiPNN carving | cheaper partitioning at equal recall? |
| D6 → D1 | IVF + RaBitQ → fastpipnn graph per shard | which in-shard index is cheaper per unit recall? |
| D6 / D1 → D6b / D3 | all shards → routed shards | from which S does routing pay off? |
| D3 → D4 | adds bridges | recall regained per cross-shard hop |
| router study | HNSW vs fastpipnn over centroids | does PiPNN's build advantage hold for routers? |

## 2. Setup (step 1)

- Data: `native/scripts/synthetic.py --n 1000000 --queries 1000` – 1M × 128-d uint8,
  128-component Gaussian mixture, 1,000 held-out queries, exact top-100. BIGANN could not
  be downloaded from this environment (dl.fbaipublicfiles.com blocked); rerun with
  `run_shards.sh data/bigann10m/...` where it is reachable.
- Machine: 4 cores, 15 GB. Builds use 4 threads per shard; shards are built one after
  another and timed separately, so "wall (S workers)" = fixed cost + slowest shard and
  "total CPU" = fixed + sum of shards. Queries run on one thread; latency = routing +
  slowest shard + merge; CPU/query = routing + all shards + merge.
- D1: fastpipnn degree 64, leaf 1024, fanout (10, 3), local_k 2, α 1.2; HNSW M 32,
  efConstruction 200. D6/D5: k-means 10 iterations on 256 × C sampled rows; carving on the
  same sample with leaf size 1.6 × sample / C; 1-bit RaBitQ codes with a 4-bit query
  (popcount kernel), refine 10 (k · 10 candidates reranked **per segment**, as in Lance).

## 3. Results, step 1 (synthetic 1M × 128, k = 10)

Raw records: `native/results/shards-synthetic1m/results.jsonl` (all designs),
`segment-rerank.jsonl`, `lance.jsonl`. Full matched-recall tables:
`python3 native/scripts/summarize_shards.py native/results/shards-synthetic1m/results.jsonl`.

### 3.1 Build

| Design | S | Fixed (train + router) | Slowest shard | **Wall, S workers** | Total CPU |
|---|---:|---:|---:|---:|---:|
| D1 fastpipnn per shard | 1 | – | 28.9 s | 28.9 s | 28.9 s |
| D1 fastpipnn per shard | 8 | – | 2.85 s | **2.85 s** | 21.7 s |
| D1 fastpipnn per shard | 32 | – | 0.61 s | **0.61 s** | 18.2 s |
| D1 HNSW per shard | 1 | – | 128.7 s | 128.7 s | 128.7 s |
| D1 HNSW per shard | 8 | – | 6.46 s | 6.46 s | 50.4 s |
| D1 HNSW per shard | 32 | – | 1.43 s | 1.43 s | 41.7 s |
| D6 k-means (ours), C = 1024 | 8 | 5.22 s | 0.32 s | 5.54 s | 7.34 s |
| D6 k-means (ours), C = 1024 | 32 | 5.22 s | 0.08 s | 5.30 s | 7.19 s |
| D6 k-means (ours), C = 4096 | 32 | 61.6 s | 0.23 s | 61.8 s | 67.9 s |
| **Lance IVF_RQ** (real), C = 1024 | 1 | – | 14.7 s | 14.7 s | 14.7 s |
| **Lance IVF_RQ** (real), C = 1024 | 8 | 1.45 s | 1.19 s | 2.68 s | 10.6 s |
| D5 carving, C = 1404 | 8 | 0.33 s | 0.32 s | **0.65 s** | 2.75 s |
| D5 carving, C = 1404 | 32 | 0.33 s | 0.10 s | **0.43 s** | 2.66 s |
| D5 carving, C = 6085 | 32 | 1.33 s | 0.33 s | 1.66 s | 10.5 s |

Fixed = sampling + centroid training + router build. "Wall, S workers" = fixed + slowest shard.

### 3.2 Query cost at matched recall (cheapest setting reaching the target)

| Design | S | Recall ≥ 0.95 | Recall ≥ 0.99 | Latency @0.99 | CPU/query @0.99 |
|---|---:|---|---|---:|---:|
| D1 fastpipnn | 1 | beam 128 (0.988) | beam 256 (0.998) | 0.98 ms | 0.98 ms |
| D1 fastpipnn | 8 | beam 32 (0.983) | beam 64 (0.997) | 0.29 ms | 1.72 ms |
| D1 fastpipnn | 32 | beam 10 (0.970) | beam 16 (0.994) | 0.13 ms | 1.77 ms |
| D1 HNSW | 8 | ef 32 (0.987) | ef 64 (0.999) | 0.19 ms | 1.12 ms |
| D1 HNSW | 32 | ef 16 (0.989) | ef 32 (1.000) | 0.13 ms | 2.07 ms |
| D6 k-means, C 1024, exact router | 8 | nprobe 16, refine 40 (0.967) | nprobe 16, refine 100 (0.998) | 0.69 ms | 1.62 ms |
| D6 k-means, C 1024, exact router | 32 | nprobe 8, refine 100 (0.957) | nprobe 16, refine 100 (0.998) | 1.18 ms | 3.19 ms |
| D5 carving, C 1404, exact router | 8 | nprobe 16, refine 40 (0.967) | nprobe 16, refine 100 (0.997) | 0.71 ms | 1.52 ms |
| D5 carving, C 1404, exact router | 32 | nprobe 16, refine 40 (0.967) | nprobe 16, refine 100 (0.997) | 0.75 ms | 1.73 ms |
| Lance IVF_RQ 1-bit (real) | 1, 8 | nprobe 8, refine 40 (0.962 / 0.966) | not reached at refine 40 | – | – |
| Lance IVF_RQ 5-bit (real, default) | 1 | nprobe 8, refine 10 (0.994) | nprobe 8, refine 10 (0.994) | – | – |

IVF recall does not depend on S (one global rerank at the coordinator); its latency rises with
S here because each shard holds only a few hundred rows per probed list, so per-shard fixed
work (heaps of k·refine, merging S·k·refine candidates) dominates. Lance latencies (7–16 ms)
go through the Python scanner and are not comparable with the C++ numbers.

### 3.3 Router over centroids (router recall of the top-8 centroids)

| Centroids | C | HNSW (M 16, efC 200) | fastpipnn (leaf 512, local_k 16, degree 48, α 1.5) |
|---|---:|---|---|
| carving | 1,404 | 0.95 at ef 64–256 (never higher), build 0.05 s | **1.00** at beam 64, build 0.05 s |
| carving | 6,085 | 0.996 at ef 64, build 0.24 s | 0.99 at beam 64, build 0.23 s |
| carving | 21,543 | 1.00 at ef 128, build 0.80 s | 0.99 at beam 256, build 0.85 s |
| carving | 87,265 | 0.990 at ef 128, build 3.0 s | 0.995 at beam 128, build 4.3 s |

The first fastpipnn router (local_k 4, degree 32) reached only 0.07–0.84: the leaf k-NN
graph over clustered centroids was too sparse to connect clusters. It needs local_k ≈ 16.

### 3.4 What step 1 shows

1. **PiPNN carving removes the fixed cost that limits LanceDB's build speedup.** Training
   1,404 centroids by carving took 0.23 s, against 1.45 s for Lance's own (hierarchical)
   k-means and 5.1 s for our 10-iteration Lloyd k-means; at about 6,000 lists, 1.0 s
   against 61 s for ours. With 8–32 workers the D5 build finishes in 0.43–0.65 s against
   2.68 s for real Lance at S = 8 (4–6× faster wall time), and the per-segment work is the
   same. This is the serial part the LanceDB post names as the reason 10 workers gave only
   5×.
2. **Carved lists are as good as k-means lists.** At the same rerank budget D5 reaches the
   same recall as D6 (0.967 at refine 40, 0.997–0.998 at refine 100). At about 1,000 lists
   it got there scanning half as many codes (8,357 against 16,950), because carving caps
   list size (largest list 2,305 against 2,824 for k-means, median 713 against 901). At about
   6,000 lists the two are within 25% either way.
3. **Inside a shard, a graph answers faster; IVF builds far cheaper.** At recall 0.99 and
   S = 8, D1 latency is 0.19–0.29 ms against 0.69–0.71 ms for D6/D5, with similar CPU per
   query (1.1–1.7 ms against 1.5–1.6 ms). But D1 costs 21.7 s (fastpipnn) or 50.4 s (HNSW)
   of build CPU against 2.75 s for D5, and keeps full vectors plus a degree-64 graph in RAM
   (≈ 384 B per vector here) where IVF needs about 24 B of codes and norms in RAM and full
   vectors only for the rerank.
4. **fastpipnn beats HNSW as the per-shard graph on build, not on queries.** It builds 2.3×
   faster at S = 8–32 (4.4× at S = 1) in wall time and 2.3× less CPU in total. HNSW needs
   slightly fewer distance computations for the same recall.
5. **The router choice hardly matters once it is dense enough.** HNSW and dense fastpipnn
   routers are within ±0.01 router recall and similar build time up to 87k centroids, so
   PiPNN's build advantage does not show here (routers build in seconds anyway). HNSW lost
   about 4% of centroids at C = 1,404 on this clustered data, which capped D5 at 0.952 recall
   with that router.
6. **1-bit RaBitQ needs a large rerank.** With k·10 candidates recall saturates near 0.78
   (ours) and Lance agrees (1-bit, refine 40: 0.971 vs our 0.967). Lance's default 5-bit codes
   reach 0.994 at refine 10. Step 2 should use multi-bit codes in D5/D6 for a fair cost.
7. **Per-segment rerank is a trap for benchmarks.** Reranking k·refine per segment made
   recall climb from 0.78 (S = 1) to 1.00 (S = 32) at the same nprobe, only through 32×
   more reranks (`segment-rerank.jsonl`). Real Lance did not show this (S = 8 recall equal
   to S = 1), so all IVF numbers above use one global rerank.
8. **Sending every query to every shard costs CPU linearly.** D1 CPU per query at ~0.99 goes
   from 0.98 ms (S = 1) to 1.7–2.1 ms (S = 32), and D6 from 0.70 to 3.19 ms. That is the
   motivation for the routed designs (D6b, D3) in step 2.

### 3.5 Limits

- Synthetic data only (128 well-separated clusters, which is hard for graph connectivity and
  easy for IVF). BIGANN was unreachable; rerun `run_shards.sh` on it.
- One machine simulates the shards: no network, stragglers or failures; latency is the
  slowest shard measured on one thread, not a distributed measurement.
- Our D6 k-means is plain Lloyd (slower than Lance's). The D5 vs D6 build claim above is
  therefore stated against real Lance's 1.45 s as well as ours.
- 1-bit RaBitQ with plain 4-bit query rounding, no error bounds; Lance's fast rotation is not
  implemented (dense 128 × 128 rotation, negligible at this d).
- One seed, 1,000 queries.

## 4. Next (step 2)

- Multi-bit RaBitQ (or Lance itself) inside D5/D6, so IVF runs at refine 10.
- D6b and D3: route each query to a few shards; measure CPU/query and shards touched vs S.
- D6c: fastpipnn graph inside IVF partitions.
- Real data (BIGANN 10M or production embeddings) once it can be downloaded here.
