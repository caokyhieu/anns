# IVF_RQ with a PiPNN assignment, on the datasets of lance discussion #9122

The customer's current index is Lance's `IVF_RQ`. `IVF_GRAPH.md` found that Lance's HNSW over
the centroids, which Lance uses for the build's assignment pass, sends many rows to the wrong
partition once there are many partitions. A PiPNN graph over the same centroids did not.
This experiment checks that on real Lance `IVF_RQ` indexes, with the datasets and settings of
[lance-format/lance discussion #9122](https://github.com/lance-format/lance/discussions/9122)
(PCA pre-transform for IVF + RQ, September 2026).

**Status:** the harness is built and checked end to end on the local synthetic set (§4). GIST1M
and VIBE have not been run. This environment's network policy blocks `huggingface.co` and
`ann-benchmarks.com`, and full VIBE needs about 40 GB of RAM. §3 gives the commands for a larger
machine.

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

## 4. Smoke test on this machine

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
enable it here on its own. The test shows that every mode builds and queries. It says nothing
yet about GIST or VIBE.

## 5. What to look for

- **VIBE, `lance-hnsw` and `fixed-lance-hnsw` vs `lance-exact`.** How much recall the default
  assignment of a 1,024 × 1,024 index loses at each nprobe, and whether more nprobe recovers it.
  In `IVF_GRAPH.md`, Lance's centroid HNSW at 1,024 partitions put 6% of rows in the wrong
  partition and capped recall at 0.989. Real Lance at 4,096 partitions stopped at 0.925.
- **`fixed-fastpipnn32` vs `lance-exact`.** Whether a PiPNN assignment matches exact
  assignment at a fraction of its cost. The native assignment also reports how many distances
  each row needs.
- **GIST** should show no difference, since Lance assigns it exactly by default. If it does, the
  cause is elsewhere.
