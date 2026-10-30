# Indexing time: Lance's IVF build vs. a PiPNN-style hierarchical build

**Question.** Indexing time is our bottleneck. Can a PiPNN-based design build the IVF
index faster than Lance's (IVF partitions, HNSW over the centroids, RaBitQ codes)
and keep the same recall and QPS?

**Answer.** Yes, by changing one step: how each vector is *assigned* to its partition.
A PiPNN-style assignment (route through a two-level tree of centroids, then one dense
GEMM per coarse cell) assigns up to 33× faster than an exact scan of all centroids
(1.7× at 1k partitions, 7.6× at 4k, 33× at 16k), with the same recall and the same
vectors scanned per query. Lance's other fast path, an HNSW over the centroids, is
about as fast on CPU here, but it misassigns more vectors (up to 18%, leaving some
partitions empty) and loses up to about 1 point of recall at the nprobe that gives
0.94–0.98 recall. It is also not GEMM-shaped. Swapping HNSW for a
PiPNN graph *as the query router* does not matter: routing is already exact with
either, and costs almost nothing. The win is in the build's assignment pass, and it
grows with the partition count, which at our scale (10^12 vectors) is 10^7–10^8.

Code: `ivf_build_experiment.py`. Raw results: `ivf-build-results.json` (1M × 128-d)
and `ivf-build-results-768d.json` (300k × 768-d).

## 1. What Lance does today (read from source)

From `lance-format/lance` at commit `fb465bb` (`rust/lance-index/src/vector/kmeans.rs`,
`vector/utils.rs`), and the same code paths are in `pylance` 14.0.2, which we ran:

| Build step | What Lance does |
| --- | --- |
| Train | k-means on a sample of 256 vectors per partition. **Hierarchical** by default when there are more than 256 partitions (fan-out 16, proportional quotas, sub-trees in parallel). Optional global refinement passes, off by default. |
| Assign | Every vector to its nearest centroid, one of two ways: **flat** (distances to all C centroids; a GEMM only on CPUs with AMX-FP16 and dot product, otherwise a per-vector kernel), or **HNSW over the centroids** (m=12, ef_construction=15, ef=15, top-1). HNSW is used automatically when C × d ≥ 10^6 and the AMX GEMM is unavailable; `LANCE_USE_HNSW_SPEEDUP_INDEXING` forces either one. |
| Encode | RaBitQ (1–5 bits) of residuals, written partition by partition. |
| Query | Nearest `nprobes` centroids, scan their codes, re-rank `k × refine_factor` with full vectors. |

Lance's own comment in `utils.rs`, measured on 100M × 768 fp16:

| Partitions | Flat (AMX GEMM) | HNSW assignment |
| ---: | ---: | ---: |
| 10,000 | 1,965 s, recall 0.980 | 2,011 s, recall 0.976 |
| 20,000 | 2,494 s, recall 0.964 | 2,588 s, recall 0.832 |
| 40,000 | 3,730 s, recall 0.970 | 3,976 s, recall 0.940 |

Two things follow. Build time grows with the partition count, which points to the
assignment pass. And the HNSW path loses recall because some vectors land in a
partition that is not their nearest and no `nprobes` setting recovers them. Without
AMX, Lance says flat is 2.7× slower than HNSW (5,361 s vs 2,011 s at 10k).

So Lance already trains hierarchically and already has a graph over the centroids.
The open question is only the assignment step.

## 2. The experiment

Both designs share data, training, RaBitQ encoder, exact top-`nprobe` routing, scan
and re-rank code (`ivf_build_experiment.py`). Only the assignment changes, so any
difference in recall or vectors scanned comes from assignment alone:

- **flat:** exact; GEMM against all C centroids.
- **hnsw:** hnswlib over the centroids with Lance's parameters (m=12, ef_construction=15, ef=15).
- **tree_t:** each vector picks its t nearest of √C coarse centroids. Then, for each
  coarse centroid, one GEMM scores all vectors that picked it against that
  centroid's leaves. That is about (1 + t)·√C distance evaluations per vector instead of
  C, and it stays matrix-shaped (good for AMX and GPUs). This is PiPNN's build idea
  (partition, then dense products inside each part) applied to assignment, and the
  tree is the one hierarchical training already builds.

We also built the real Lance index (pylance 14.0.2, `IVF_RQ`, 1 bit, same partition
count, refine factor 10) on the same vectors, with HNSW assignment forced off and on,
as a reference for absolute build time and recall.

Data: no real embedding set is reachable from this environment, so vectors are
synthetic but embedding-like: 20k overlapping clusters in a 24-d latent space,
mapped to 128 or 768 dimensions with a decaying spectrum. Recall is recall@10
against exact ground truth. Machine: 4 vCPUs (Xeon, AVX-512, AMX-BF16/INT8 but no
AMX-FP16), 15 GB RAM.

## 3. Results

### 1,000,000 × 128-d: assignment time vs partition count

| Partitions | Train | Encode | flat | hnsw (m=12, ef=15) | tree t=2 | **tree t=4** | tree t=8 | pylance build, flat | pylance build, HNSW assign |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 7.7 s | ~12 s | 4.8 s | 2.7 s | 1.5 s | **2.8 s** | 5.1 s | 11.4 s | 16.8 s |
| 4,096 | 30.2 s | ~12 s | 21.8 s | 3.7 s | 1.8 s | **2.9 s** | 5.9 s | 57.3 s | 42.3 s |
| 16,384 | 32.3 s | ~18 s | 141.7 s | 5.5 s | 2.9 s | **4.4 s** | 7.8 s | 250.6 s | 52.5 s |

Assignment quality (fraction of vectors in their exact nearest partition; partitions
left empty):

| Partitions | hnsw | tree t=2 | tree t=4 | tree t=8 | Distances / vector: flat → tree t=4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.969 | 0.920 | 0.989 | 1.000 | 1,024 → 166 |
| 4,096 | 0.906 (2 empty) | 0.895 | 0.978 | 0.999 | 4,096 → 328 |
| 16,384 | 0.824 (39 empty) | 0.863 | 0.956 | 0.993 | 16,384 → 653 |

### 1,000,000 × 128-d: recall@10 at equal nprobe

Vectors scanned per query are the same for every method at a given `nprobe` (for
example 8.4k at 16k partitions and nprobe 128), so equal recall at equal nprobe means
equal QPS.

| Partitions | Method | np=16 | np=32 | np=64 | np=128 | np=256 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | flat | 0.753 | 0.860 | 0.939 | 0.976 | 0.987 |
| | hnsw | 0.744 | 0.852 | 0.932 | 0.974 | 0.987 |
| | **tree t=4** | 0.752 | 0.860 | 0.938 | 0.978 | 0.990 |
| | pylance, flat | 0.736 | 0.851 | 0.933 | 0.979 | 0.991 |
| | pylance, HNSW assign | 0.724 | 0.849 | 0.920 | 0.968 | 0.988 |
| 16,384 | flat | 0.690 | 0.798 | 0.881 | 0.946 | 0.982 |
| | hnsw | 0.667 | 0.784 | 0.873 | 0.938 | 0.977 |
| | **tree t=4** | 0.685 | 0.795 | 0.881 | 0.944 | 0.981 |
| | pylance, flat | 0.666 | 0.786 | 0.875 | 0.939 | 0.977 |
| | pylance, HNSW assign | 0.670 | 0.778 | 0.875 | 0.941 | 0.976 |

At 1,024 partitions all methods agree within 0.006 at every `nprobe`. Both the
Python HNSW router and the PiPNN router return the exact top-`nprobe` partitions
(agreement ≥ 0.999 at every size).

### 300,000 × 768-d, 4,096 partitions

Training 47.8 s, RaBitQ encoding about 24 s (both shared by all assignment methods).

| Assignment | Time (s) | Evals / vector | Exact nearest | R@10 np=16 | np=32 | np=64 | np=128 | np=256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat | 11.40 | 4,096 | 1.000 | 0.664 | 0.806 | 0.900 | 0.966 | 0.994 |
| hnsw (m=12, ef=15) | 2.85 | — | 0.921 | 0.656 | 0.803 | 0.902 | 0.966 | 0.994 |
| tree, t=2 | 1.13 | 195 | 0.907 | 0.654 | 0.797 | 0.895 | 0.962 | 0.992 |
| **tree, t=4** | **2.44** | 327 | 0.979 | 0.662 | 0.805 | 0.901 | 0.966 | 0.995 |
| tree, t=8 | 4.51 | 593 | 0.999 | 0.664 | 0.806 | 0.900 | 0.966 | 0.995 |
| pylance, flat: build 108.5 s | | | | 0.658 | 0.798 | 0.912 | 0.972 | 0.997 |
| pylance, HNSW assign: build 54.9 s | | | | 0.646 | 0.783 | 0.904 | 0.970 | 0.993 |

## 4. What this shows

- **Assignment is most of Lance's build once there are many partitions.** At 16k
  partitions on 1M vectors, forcing Lance's flat path instead of its HNSW path
  turns a 52 s build into 251 s, so at least 80% of the flat build is assignment. At 768-d with 4k partitions it is 55 s vs 108 s. The
  rest (training, encoding, writing) does not depend on the partition count much.
- **Assignment can be approximate without costing recall.** Tree assignment with t=4
  puts 96–99% of vectors in their exact nearest partition. A misassigned vector
  lands in a neighboring partition, and a query that probes enough partitions probes
  that one too. Recall matches flat within 0.003 at nprobe ≥ 32 (within 0.008 at the
  smallest nprobe), and vectors scanned per query are identical. So a faster build
  does not need more probes, and QPS stays the same.
- **Tree vs flat:** 1.7× (1k partitions), 4.7× (768-d, 4k), 7.6× (128-d, 4k) and
  33× (128-d, 16k) faster. Distances per vector grow as √C instead of C. At our scale's 10^7–10^8 partitions, flat assignment is
  impractical (§5).
- **Tree vs Lance's HNSW path:** about the same CPU time at these sizes. But tree
  keeps flat's recall where HNSW loses some: 0.8–2.3 points at 16k partitions and
  39 empty partitions in our runs, and 1.1–1.3 points for pylance's own HNSW path at
  4k partitions (nprobe 64–128). Lance's 100M benchmark shows the same effect, larger,
  at 20k partitions (0.832 vs 0.964). Tree has a knob, t, that trades time for exactness.
  HNSW at ef=15 does not. And tree is batched dense algebra, so it runs at GEMM speed
  on AMX or GPUs, where Lance found its flat GEMM beats HNSW. It also needs no graph
  build over 10^8 centroids.
- **The PiPNN graph as query router is not where the time is.** Both graphs route
  queries to the exact top-`nprobe` partitions (agreement ≥ 0.9999 at every size), and
  routing is a tiny part of a query. Our Python PiPNN router builds slower than C++
  hnswlib here. The native comparison (`native/results/compare-synthetic1m`) has PiPNN
  building a 1M-node graph 3–5× faster than HNSW. That only starts to matter when
  the centroid graph has 10^7–10^8 nodes and is rebuilt often.
- **The Python reproduction tracks real Lance.** Recall curves agree with pylance
  within 0.01 at nprobe ≥ 64 (ours are up to 0.03 higher at small nprobe), so the
  comparison is about the algorithm, not a weak baseline.

## 5. What it means at 10^12 vectors

With 768-d vectors and partitions of about 10^4 vectors, there are C = 10^8
partitions. Assignment FLOPs (2·d per distance):

| Assignment | Distances / vector | Total FLOPs | Note |
| --- | ---: | ---: | --- |
| flat | 10^8 | 1.5 × 10^23 | Infeasible, even on a GPU cluster |
| tree, 2 levels, t=4 | 10^4 + 4·10^4 = 5 × 10^4 | 7.7 × 10^19 | Already 2,000× less |
| tree, 3 levels (fan-out 464), t=4 | 464·(1 + 4 + 4) ≈ 4.2 × 10^3 | 6.4 × 10^18 | A few GPU-hours of GEMM in total; reading 1.5 PB of vectors (about 50 min at 500 GB/s) becomes the limit |
| HNSW over 10^8 centroids | hundreds to thousands, random access | similar FLOPs, at gather speed | Plus building and storing a 10^8-node graph, and Lance's measured recall loss |

At that scale the assignment pass stops being the bottleneck with a tree, while
with flat it is the whole build. The remaining build time is training (on a sample,
already hierarchical in Lance), RaBitQ encoding (one rotation GEMM per vector) and
I/O.

## 6. Recommendation

1. **Keep Lance's format and query path. Replace only the assignment pass.** Lance
   accepts it from outside: `create_index(..., ivf_centroids=C,
   precomputed_partition_dataset=<row id → partition>)`. So the tree assignment can
   run as a distributed GPU job (Ray or Spark) that writes partition IDs, and Lance
   then only encodes and writes. No fork of Lance is needed. Keep the coarse levels
   of the hierarchical training as the tree.
2. **Use t = 4.** It matched flat recall in every setting we ran. Use t = 8 if real
   embeddings misassign more.
3. **Use 3 levels at 10^8 partitions** and do the per-level GEMMs on GPU, so the
   pass is limited by I/O.
4. **Use a PiPNN graph as the query router over 10^7–10^8 centroids** only if routing
   ever becomes a cost. Its value there is a faster rebuild when partitions change,
   not faster queries.
5. **Validate before committing.** These are 1M-scale synthetic vectors on 4 CPUs.
   Next: (a) real embeddings, about 10^8 of them, at 10^5–10^6 partitions; (b) the
   tree pass on GPU vs Lance's HNSW path, end to end through
   `precomputed_partition_dataset`; (c) check that recall at fixed `nprobe` still matches
   flat, and tune t.

## Caveats

- Synthetic data. Real embeddings may have harder partition boundaries, which would
  push t up. The measurement to repeat is "exact nearest" and recall vs `nprobe`.
- Absolute times are from a 4-vCPU container and NumPy. Compare ratios, not seconds.
  Lance's flat path on this CPU is a per-vector kernel, not a GEMM (no AMX-FP16).
- QPS in the Python scan is noisy (shared CPU). We report vectors scanned per query
  as the stable cost measure. It is identical across assignment methods at each
  `nprobe`.
- The router comparison uses our Python PiPNN, so its build time is not comparable
  with C++ hnswlib. See the native numbers instead.
- Single seed, single run per setting.

## Run it

```bash
uv sync --locked --extra compare --extra lance
uv run --locked python ivf_build_experiment.py --lance                 # 1M x 128-d, 1k/4k/16k partitions
uv run --locked python ivf_build_experiment.py --n 300000 --dim 768 --partitions 4096 \
    --queries 300 --probes 16 32 64 128 256 --lance --output ivf-build-results-768d.json
```
