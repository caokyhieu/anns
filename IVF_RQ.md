# IVF_RQ with a PiPNN assignment, on the datasets of lance discussion #9122

The customer's current index is Lance's `IVF_RQ`. `IVF_GRAPH.md` found that Lance's HNSW over
the centroids, which Lance uses for the build's assignment pass, sends many rows to the wrong
partition once there are many partitions. A PiPNN graph over the same centroids did not.
This experiment checks that on real Lance `IVF_RQ` indexes, with the datasets and settings of
[lance-format/lance discussion #9122](https://github.com/lance-format/lance/discussions/9122)
(PCA pre-transform for IVF + RQ, September 2026).

**Status:** GIST1M is done (§4, tree assignment and 4,096 partitions in §4.1): on the discussion's GIST setup the baseline matches their
numbers, and the assignment method makes no difference, as expected at 256 partitions.
VIBE has not been run. This environment's network policy blocks `huggingface.co`, and full
VIBE needs about 40 GB of RAM. §3 gives the commands for a larger machine.

Code: `lance_ivf_rq_bench.py` (data prep, builds, query sweeps, tables), `native/src/ivf_assign.cpp`
(native assignment against fixed centroids).

## 1. Datasets and settings (from #9122)

| | GIST1M | VIBE `lance-format/vibe-msmarco-qwen-1024` |
|---|---|---|
| Vectors | 1,000,000 × 960, float32 | 8,840,823 × 1024, float32, unit norm |
| Queries / truth | 1,000 official, exact L2 top-10 (top-100 in the texmex files) | 1,000, exact top-100 |
| Metric | Recall@10, L2 | Recall@100. Unit norm, so L2 ranks the same as cosine |
| IVF_RQ | 256 partitions, 2-bit | 1,024 partitions, 1/2/3/5-bit (the discussion ran 2, 3 and 5) |
| Refine | none, 2, 3, 5 (they report 2–5) | none |
| nprobe | 8, 16, 20, 32, 64 (they report 16 and 20) | 8, 16, 32, 64, 96, 128 |
| Runs | median of 3 | median of 3, 10 warm-up queries, 32 GiB index cache |
| Discussion's baseline | 960D, nprobe 16, refine 2: R@10 0.915, p50 6.2 ms, build 9.3 s | 1024×2: R@100 0.909 at nprobe 128, build 67.4 s, peak RSS 36.6 GB |

The discussion used Lance 10.0.0 and did not report its hardware. This harness pins pylance
14.0.2 (`uv.lock`), the version behind the other notes here.

Lance turns its centroid HNSW on automatically when C × d ≥ 10^6 and the CPU has no AMX-FP16
GEMM (`rust/lance-index/src/vector/utils.rs`). That puts the two datasets on different sides:

- **GIST** at 256 × 960 = 2.5 × 10^5 is assigned exactly by default.
- **VIBE** at 1,024 × 1,024 = 1.05 × 10^6 uses the centroid HNSW by default on most CPUs. The
  VIBE numbers in #9122 were built that way unless their CPU had AMX-FP16, which the discussion
  does not report. A customer's index of that shape on such a CPU would be too.

## 2. What is compared

Each row is a real Lance `IVF_RQ` index built and queried by Lance. Rows that share centroids
differ only in which partition each row is stored in.

| Mode | Centroids | Assignment | How |
|---|---|---|---|
| `lance-exact` | Lance k-means | exact | `LANCE_USE_HNSW_SPEEDUP_INDEXING=disabled` |
| `lance-hnsw` | Lance k-means, trained with the HNSW too | Lance's centroid HNSW (m 12, efC 15, ef 15) | `=enabled` |
| `fixed-lance-hnsw` | those of `lance-exact` | Lance's centroid HNSW | `=enabled`, `ivf_centroids=C` |
| `fixed-flat` | those of `lance-exact` | native exact | `precomputed_partition_dataset` |
| `fixed-hnsw15` | those of `lance-exact` | native hnswlib, Lance's settings | same, checks the `IVF_GRAPH.md` stand-in |
| `fixed-fastpipnn16` / `32` | those of `lance-exact` | PiPNN graph over the centroids, beam 16 / 32 | same |

- **The precomputed path works in pylance 14.0.2 for IVF_RQ.** That was checked before building
  the harness: random assignments with the same centroids dropped recall from 0.825 to 0.064,
  and partition sizes matched the file exactly. Lance then skips its own assignment and its
  k-means.
- **The input file:** one-fragment Lance dataset with `row_id: uint64` (Lance `_rowid`) and
  `partition: uint32`. The harness writes it.
- **Build time** of `fixed-*` rows excludes the native assignment. `assign.jsonl` reports that
  time, its distances per row, and the fraction of rows on their exact nearest centroid.
- **RaBitQ draws a new rotation per build,** so two builds with the same partitions differ by
  ±0.005 recall at 200 queries (§4). Compare rows on more queries and repeats than that.

## 3. Running it on a larger machine

Requirements:

- **GIST:** 16 GB RAM and 15 GB disk.
- **VIBE:** 64 GB RAM (the discussion's builds peaked at 30–49 GB) and about 150 GB disk.
  That covers `base.fbin` 36 GB, `base.lance` 36 GB, one index at a time (up to 6 GB), and the
  download.
- **Hosts:** `huggingface.co` for VIBE, and `ann-benchmarks.com` or `ftp.irisa.fr` for GIST.

```bash
git clone -b claude/nifty-einstein-6kpw2k https://github.com/caokyhieu/anns && cd anns
uv sync --locked --extra lance
python3 native/scripts/bootstrap.py
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release && cmake --build native/build -j --target ivf_assign
```

GIST1M, from the texmex files:

```bash
mkdir -p downloads/gist && curl -L ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz | tar xz -C downloads/gist
uv run --locked --extra lance python lance_ivf_rq_bench.py prepare --name gist --base downloads/gist/gist/gist_base.fvecs --query downloads/gist/gist/gist_query.fvecs --gt downloads/gist/gist/gist_groundtruth.ivecs
uv run --locked --extra lance python lance_ivf_rq_bench.py run --data data/gist --preset gist
uv run --locked --extra lance python lance_ivf_rq_bench.py summarize data/gist/results.jsonl
```

Or GIST1M from ann-benchmarks, which needs h5py:

```bash
curl -L -o downloads/gist-960-euclidean.hdf5 https://ann-benchmarks.com/gist-960-euclidean.hdf5
uv run --locked --extra lance --with h5py python lance_ivf_rq_bench.py prepare --name gist --hdf5 downloads/gist-960-euclidean.hdf5
```

VIBE:

```bash
uvx --from huggingface_hub hf download lance-format/vibe-msmarco-qwen-1024 --repo-type dataset --local-dir downloads/vibe
ls -R downloads/vibe | head -50
```

The repository's layout was not visible from here. `prepare` reads `.lance` datasets, Parquet
directories, `.fvecs`, `.fbin` and `.npy`. It takes the first list-typed column unless
`--vector-column` / `--query-column` is given, and computes exact top-100 ground truth if
`--gt` is not given. Point it at the base and query parts of the download, for example:

```bash
uv run --locked --extra lance python lance_ivf_rq_bench.py prepare --name vibe --base downloads/vibe/<base> --query downloads/vibe/<queries> --gt-k 100
uv run --locked --extra lance python lance_ivf_rq_bench.py run --data data/vibe --preset vibe
uv run --locked --extra lance python lance_ivf_rq_bench.py summarize data/vibe/results.jsonl
```

Run time for the full VIBE run:

- **Builds:** 4 bit widths × 7 modes = 28 builds of about 70–120 s each, from the discussion's
  build times.
- **Queries:** 28 × 6 nprobe values × 3 repeats × 1,000 queries at 10–20 ms each, about 2.5 h.

For a first pass, `--bits 2 --modes lance-exact lance-hnsw fixed-lance-hnsw fixed-fastpipnn32
--repeats 1` takes about 15 minutes.

Results go to `data/<name>/results.jsonl` and `data/<name>/assign.jsonl`. Copy them to
`native/results/ivf-rq-<name>/` to commit.

## 4. GIST1M results

ann-benchmarks `gist-960-euclidean.hdf5`: 1M × 960 base, 1,000 queries, its exact top-100
(spot-checked against an exact search: 10/10 on 5 queries). 256 partitions, 2-bit RQ, L2,
k 10, all 1,000 queries, medians of 3 runs, 10 warm-up queries, pylance 14.0.2, 4 vCPUs.
Raw: `native/results/ivf-rq-gist1m/`.

**The baseline reproduces #9122.** `lance-exact` matches the discussion's 960D baseline:

| nprobe / refine | Discussion: Recall@10, p50, p95, QPS | Here (`lance-exact`) |
|---|---|---|
| 16 / 2 | 0.9152, 6.25 ms, 9.92 ms, 149.7 | 0.9181, 6.44 ms, 7.94 ms, 151.8 |
| 16 / 3 | 0.9259, 5.34 ms, 7.72 ms, 175.7 | 0.9293, 6.80 ms, 8.51 ms, 142.4 |
| Index size | 251.39 MiB | 263.7 MB (251.4 MiB) |
| Build | 9.31 s | 36.3 s (4 vCPUs; their hardware is not reported) |

**The assignment method makes no difference here.** Recall@10 (p50 latency):

| Mode | Build | np 16, refine 2 | np 20, refine 5 | np 32, refine 5 | np 64, refine 5 | np 64, no refine |
|---|---:|---:|---:|---:|---:|---:|
| lance-exact | 36.3 s | 0.9181 (6.4 ms) | 0.9517 (7.2 ms) | 0.9833 (7.9 ms) | 0.9978 (10.1 ms) | 0.8217 |
| lance-hnsw | 42.3 s | 0.9161 (6.0 ms) | 0.9528 (6.8 ms) | 0.9856 (7.5 ms) | 0.9974 (10.3 ms) | 0.8244 |
| fixed-lance-hnsw | 29.1 s | 0.9189 (6.3 ms) | 0.9513 (7.0 ms) | 0.9829 (8.2 ms) | 0.9975 (10.3 ms) | 0.8226 |
| fixed-flat | 20.3 s | 0.9182 (5.9 ms) | 0.9515 (7.1 ms) | 0.9831 (8.0 ms) | 0.9976 (10.4 ms) | 0.8240 |
| fixed-hnsw15 | 17.5 s | 0.9181 (6.0 ms) | 0.9504 (7.1 ms) | 0.9824 (7.8 ms) | 0.9971 (10.0 ms) | 0.8231 |
| fixed-fastpipnn16 | 17.8 s | 0.9171 (6.0 ms) | 0.9516 (6.9 ms) | 0.9832 (7.7 ms) | 0.9977 (10.2 ms) | 0.8199 |
| fixed-fastpipnn32 | 18.0 s | 0.9199 (6.3 ms) | 0.9517 (7.3 ms) | 0.9831 (8.1 ms) | 0.9976 (10.5 ms) | 0.8214 |

| Native assignment (1M rows, 4 threads) | Distances / row | Time | Exact nearest | Largest list |
|---|---:|---:|---:|---:|
| flat | 256 | 1.5 s | 1.0000 | 13,104 |
| hnsw, Lance's settings | 87 | 3.9 s | 0.9905 | 13,176 |
| fastpipnn, beam 16 | 116 | 4.0 s | 0.9999 | 13,104 |
| fastpipnn, beam 32 | 146 | 5.5 s | 1.0000 | 13,104 |

- Across all 20 (nprobe, refine) points, the seven modes are within 0.0046 of each other. That
  is the build-to-build spread from RaBitQ's random rotation. The full sweep is in
  `summarize native/results/ivf-rq-gist1m/results.jsonl`.
- With only 256 centroids, even Lance's centroid HNSW puts 99% of rows on their nearest
  centroid, and the 1% it misses lie on partition boundaries, where refine and nprobe 16+ cover
  them. Lance would not turn the HNSW on for this shape anyway (256 × 960 < 10^6).
- At 256 partitions, exact assignment (1.5 s) is cheaper than either graph. The graphs only pay
  off with many partitions.
- **Build times of `fixed-*` rows are not a speed-up.** They skip k-means, because the centroids
  come from `lance-exact`. `lance-hnsw` (42 s) is slower than `lance-exact` (36 s) because k-means
  training also routes through the HNSW, which at 256 centroids is slower than the GEMM.

For the customer's IVF_RQ on GIST-like data at 256 partitions, this means:
- Lance's default assignment is exact.
- Swapping in PiPNN changes neither recall nor latency.
- PCA and bit width, the topics of #9122, are the levers that matter.

The assignment question only arises for indexes where C × d ≥ 10^6, such as VIBE at 1,024
partitions or GIST from about 1,042 partitions. §4.1 tests that on GIST.

### 4.1 Tree assignment, at 256 and 4,096 partitions

This is the `tree_t` assignment of `LANCE_VS_PIPNN_BUILD.md` (`ivf_assign --assign tree:t`). The
leaves must be Lance's own centroids, so the coarse level is k-means over those centroids:
round(√C) coarse cells, each centroid under its nearest coarse centroid. Each row takes its t
nearest coarse cells and is scored against their leaves with one GEMM per cell. The rows enter
Lance through `precomputed_partition_dataset` like the other `fixed-*` modes.

The 4,096-partition run uses the same GIST setup with nprobe scaled ×16 (32–256) and refine none
or 5. 4,096 × 960 = 3.9 × 10^6, so Lance enables its centroid HNSW by default on CPUs without
AMX-FP16.

**Assignment of the 1M rows** (native, 4 threads, base already in memory):

| Partitions | Method | Distances / row | Time | Exact nearest | Empty lists |
|---:|---|---:|---:|---:|---:|
| 256 | exact (GEMM) | 256 | 1.21–1.45 s | 1.0000 | 0 |
| | hnsw, Lance's settings | 87 | 3.93 s | 0.9905 | 0 |
| | fastpipnn, beam 32 | 146 | 5.49 s | 1.0000 | 0 |
| | tree, t = 2 / 4 / 8 | 66 / 110 / 182 | 0.92 / 1.53 / 2.77 s | 0.8927 / 0.9886 / 0.9998 | 0 |
| 4,096 | exact (GEMM) | 4,096 | 17.41 s | 1.0000 | 0 |
| | hnsw, Lance's settings | 137 | 7.52 s | 0.7996 | 26 |
| | fastpipnn, beam 32 | 501 | 29.05 s | 0.9999 | 0 |
| | tree, t = 2 / 4 / 8 | 237 / 412 / 755 | 1.73 / 2.93 / 5.62 s | 0.8239 / 0.9491 / 0.9929 | 0 |

**Recall@10 (p50 latency).** At 256 partitions:

| Mode | np 8, refine 2 | np 16, refine 2 | np 20, refine 5 | np 32, refine 5 | np 64, no refine |
|---|---:|---:|---:|---:|---:|
| lance-exact | 0.8072 | 0.9181 (6.4 ms) | 0.9517 (7.2 ms) | 0.9833 | 0.8217 |
| fixed-tree2 | 0.7918 | 0.9090 (6.2 ms) | 0.9449 (7.1 ms) | 0.9808 | 0.8213 |
| fixed-tree4 | 0.8059 | 0.9176 (6.4 ms) | 0.9519 (6.8 ms) | 0.9837 | 0.8220 |
| fixed-tree8 | 0.8066 | 0.9177 (5.9 ms) | 0.9517 (7.3 ms) | 0.9833 | 0.8205 |

At 4,096 partitions (build includes Lance's k-means only for `lance-*`):

| Mode | Build | np 32, refine 5 | np 64, refine 5 | np 128, refine 5 | np 256, refine 5 | np 128, no refine |
|---|---:|---:|---:|---:|---:|---:|
| lance-exact | 340.7 s | 0.7279 | 0.8448 | 0.9304 (7.3 ms) | 0.9771 (9.1 ms) | 0.8091 |
| lance-hnsw (Lance's default here) | 149.8 s | 0.7262 | 0.8451 | 0.9303 (7.3 ms) | 0.9784 (8.9 ms) | 0.8062 |
| fixed-lance-hnsw | 43.1 s | 0.7264 | 0.8441 | 0.9307 | 0.9777 | 0.8071 |
| fixed-flat | 40.7 s | 0.7279 | 0.8447 | 0.9303 | 0.9770 | 0.8035 |
| fixed-fastpipnn32 | 25.0 s | 0.7278 | 0.8447 | 0.9304 | 0.9771 | 0.8044 |
| fixed-tree2 | 23.0 s | 0.7107 | 0.8303 | 0.9227 | 0.9733 | 0.7970 |
| fixed-tree4 | 23.2 s | 0.7241 | 0.8413 | 0.9290 | 0.9771 | 0.7983 |
| fixed-tree8 | 24.3 s | 0.7278 | 0.8445 | 0.9306 (7.4 ms) | 0.9774 (9.2 ms) | 0.7994 |

Full sweeps: `summarize native/results/ivf-rq-gist1m/results.jsonl`. Raw logs: `run-tree.log`,
`run-4096.log`.

- **Tree t = 8 gives the same recall as exact assignment at both sizes.** At 4,096 partitions it
  is 3.1× faster than the exact GEMM (5.6 s vs 17.4 s) and 5× faster than the PiPNN graph
  (29 s). The graph is slow here because beam search at d = 960 is random access, while the tree
  stays matrix-shaped. Against `fixed-flat` (same centroids, exact assignment) at refine 5:
  - t = 8 is within 0.001 at every nprobe;
  - t = 4 loses up to 0.004;
  - t = 2 loses up to 0.017 at small nprobe.

  The gaps shrink as nprobe grows. Without refine, 2-bit recall varies by up to 0.006 even
  between `lance-exact` and `fixed-flat`, which share centroids and assignment. The no-refine
  columns therefore cannot separate t = 4 from t = 8.
- **At 256 partitions the tree is no faster than the GEMM.** Its 16 cells of 16 leaves make
  GEMMs too small to pay. The tree is for large C.
- **On GIST, Lance's own centroid HNSW does not hurt recall, even at 4,096 partitions.** It puts
  only 80% of rows on their nearest centroid, and Lance's real build left 5 lists empty. Yet
  `lance-hnsw` and `fixed-lance-hnsw` are within ±0.003 of `lance-exact` at every point. Rows that
  miss land in an adjacent partition that a query probing 32+ partitions also reads. This
  differs from the synthetic set of `IVF_GRAPH.md`, whose separated clusters made the same miss
  rate cost 5+ points. Exact-nearest rate alone does not predict the recall loss. Only a recall
  run on the customer's data does.
- **Assignment is a small part of the build.** Lance's end-to-end build at 4,096 partitions
  (340.7 s with exact assignment) is mostly k-means training. With the centroids given, the same
  index builds in 41 s, and exact assignment is 17 s of that natively. With HNSW on, Lance's
  training also routes through the HNSW and the build drops to 149.8 s, at no recall cost here.
  To speed up IVF_RQ builds at this scale, the lever is training: the local leaf training of
  `LANCE_VS_PIPNN_BUILD.md` §7.1, or Lance's HNSW mode. The assignment method is not.
- `fixed-tree2`'s 256-partition build took 264.9 s because it read the dataset from disk after
  the assignment tool had filled the page cache. The other tree builds took 22–24 s.

## 5. Smoke test on this machine

Synthetic 1M × 128 (the set of `IVF_GRAPH.md`), 256 partitions, 2-bit, 200 queries, one repeat.
Raw: `native/results/ivf-rq-smoke-synthetic1m/`.

| Native assignment | Distances / row | Time (4 threads) | Exact nearest |
|---|---:|---:|---:|
| flat | 256 | 0.3 s | 1.0000 |
| hnsw, Lance's settings | 123 | 2.3 s | 0.9977 |
| fastpipnn, beam 16 | 130 | 1.7 s | 1.0000 |
| fastpipnn, beam 32 | 170 | 2.7 s | 1.0000 |

| Mode | Build | R@10 np 8, no refine | np 8, refine 5 |
|---|---:|---:|---:|
| lance-exact | 7.0 s | 0.592 | 0.950 |
| lance-hnsw | 14.0 s | 0.586 | 0.953 |
| fixed-lance-hnsw | 8.4 s | 0.573 | 0.953 |
| fixed-flat | 3.9 s | 0.586 | 0.945 |
| fixed-hnsw15 | 3.6 s | 0.581 | 0.950 |
| fixed-fastpipnn16 | 3.4 s | 0.587 | 0.955 |
| fixed-fastpipnn32 | 3.4 s | 0.584 | 0.955 |

All modes agree within the ±0.005 build-to-build noise. That is expected for this set: 256
centroids are few enough that even Lance's centroid HNSW is 99.8% exact, and Lance would not
enable it here on its own. The test shows that every mode builds and queries.

## 6. What to look for

- **VIBE, `lance-hnsw` and `fixed-lance-hnsw` vs `lance-exact`.** How much recall the default
  assignment of a 1,024 × 1,024 index loses at each nprobe, and whether more nprobe recovers it.
  In `IVF_GRAPH.md`, Lance's centroid HNSW at 1,024 partitions put 6% of rows in the wrong
  partition and capped recall at 0.989. Real Lance at 4,096 partitions stopped at 0.925.
- **`fixed-fastpipnn32` vs `lance-exact`.** Whether a PiPNN assignment matches exact
  assignment at a fraction of its cost. The native assignment also reports how many distances
  each row needs.
- **GIST** at 256 partitions showed no difference (§4), as expected. The next GIST check is
  1,024–4,096 partitions (`run --preset gist --partitions 4096`), where C × d ≥ 10^6 and Lance
  would turn its centroid HNSW on.
