# Faster Lance IVF_RQ index builds: GIST1M benchmark

> **Superseded.** These changes are part of the Lance patch series in
> [`lance_index_build/patches/`](../lance_index_build/patches/); run experiments with
> [`lance_index_build/bench.sh`](../lance_index_build/bench.sh). Kept for history.

Two build-time optimizations for Lance's `IVF_RQ` index (IVF + RaBitQ), with a
benchmark against unmodified Lance on GIST1M.

- Patch: [`lance-ivf-rq-build-opt.patch`](lance-ivf-rq-build-opt.patch), against
  [`lance-format/lance`](https://github.com/lance-format/lance) commit
  `04e61996311dc9907b8b588cad2503a2b8213a08` (5 Nov 2026).
- Results: [`results/summary.md`](results/summary.md), raw data in
  [`results/results.jsonl`](results/results.jsonl), full log in
  [`results/run_all.log`](results/run_all.log).

## Changes

1. **Blocked f32 L2 nearest-centroid kernel**
   (`rust/lance-index/src/vector/blocked_l2.rs`, wired into `kmeans.rs`).
   Upstream scores one vector against every centroid at a time. At 1,000
   centroids × 960 dimensions that re-reads a 3.8 MB centroid table for every
   vector, so it is memory-bound. The new kernel scores a 4 × 4 tile (4
   vectors × 4 centroids) over L2-sized centroid chunks, using AVX-512 or
   AVX2+FMA. It covers both K-means training and the pass that assigns every
   vector to a partition.
   - Selection matches upstream: the first centroid with the strictly smallest
     distance plus balance bias wins, and rows that are all NaN stay unassigned.
   - The returned distance is recomputed exactly, because RaBitQ uses it as
     ‖x − c‖².
   - On by default. `LANCE_DISABLE_BLOCKED_L2=1` turns it off.
2. **Constant scaling factor for multi-bit codes**
   (`rust/lance-index/src/vector/bq/builder.rs`). Opt-in with
   `LANCE_RQ_CONST_EX_SCALE=1`. Upstream searches for the optimal scaling
   factor separately for every vector. This mode uses one factor per
   (dimension, bit width), averaged over 100 random Gaussian vectors. After the
   random rotation, residuals have nearly the same magnitude distribution, so
   the per-vector optimum varies little. Stored correction factors are still
   computed from the codes actually written, so distance estimates stay
   consistent.
3. **Benchmark harness** (`rust/lance/benches/ivf_rq_gist.rs`). Writes the
   dataset, builds the index through Lance's public API with per-stage timing
   (via `IndexBuildProgress`), and measures Recall@10 and Recall@100 at fixed
   `nprobes`.

## Results

Setup:
- Data: GIST1M, 1M × 960 vectors, 1,000 queries, Euclidean distance.
- Index: 1,000 partitions.
- Machine: 4-core Intel Xeon @ 2.8 GHz with AVX-512 and no AMX, 15 GB RAM.
- Runs: 2 repetitions per configuration; the table shows means.
- Search: fixed `nprobes`, no reranking.

| Code width | Config | Build (s) | IVF training (s) | Shuffle (s) | Speedup | R@10 np=20 | R@10 np=50 | R@10 np=100 |
|---|---|---|---|---|---|---|---|---|
| RQ1 | baseline | 63.6 | 19.0 | 44.0 | 1.00× | 0.610 | 0.656 | 0.668 |
| RQ1 | blocked | 30.8 | 13.3 | 17.0 | **2.07×** | 0.614 | 0.660 | 0.673 |
| RQ5 | baseline | 92.2 | 19.3 | 70.0 | 1.00× | 0.798 | 0.912 | 0.948 |
| RQ5 | blocked | 56.8 | 13.0 | 41.1 | 1.62× | 0.800 | 0.912 | 0.949 |
| RQ5 | blocked + const | 35.0 | 12.7 | 19.4 | **2.63×** | 0.792 | 0.910 | 0.946 |
| RQ9 | baseline | 221.0 | 18.8 | 197.3 | 1.00× | 0.804 | 0.920 | 0.957 |
| RQ9 | blocked | 190.3 | 13.1 | 172.6 | 1.16× | 0.799 | 0.917 | 0.956 |
| RQ9 | blocked + const | 38.5 | 12.4 | 21.9 | **5.74×** | 0.800 | 0.919 | 0.956 |

The shuffle stage covers partition assignment, RaBitQ encoding, and writing.

- The blocked kernel does not change recall beyond noise. K-means is seeded
  from OS randomness, so recall moves about ±0.005 between runs.
- The constant scaling factor costs at most 0.006 Recall@10 (RQ5,
  `nprobes=20`), and the difference is within noise at `nprobes` of 50 or more.
  It removes almost all of the multi-bit encoding cost.
- Unit tests: the kernel's own tests pass, and so do Lance's existing K-means
  and RaBitQ builder tests (86 passed, 0 failed, 1 ignored).

### Caveats

- These results come from one 4-core machine and one dataset. On CPUs with
  AMX, Lance's existing FP16 path may narrow the K-means gap.
- Check the constant scaling factor on other datasets before making it the
  default.

## Reproduce

Needs Rust (the toolchain pinned by Lance's `rust-toolchain.toml`), `protoc`, Python with `h5py` and
`numpy`, and about 15 GB of free disk.

1. Download `gist-960-euclidean.hdf5` from ann-benchmarks into `$DATA_DIR`,
   then convert it to raw files (`base.f32`, `query.f32`, `gt.i32`,
   `meta.txt`):

```bash
python3 scripts/convert_gist.py "$DATA_DIR/gist-960-euclidean.hdf5"
```

2. Check out Lance at the pinned commit and apply the patch:

```bash
git clone https://github.com/lance-format/lance.git "$LANCE_DIR"
```

```bash
git -C "$LANCE_DIR" checkout 04e61996311dc9907b8b588cad2503a2b8213a08
```

```bash
git -C "$LANCE_DIR" apply "$PWD/lance-ivf-rq-build-opt.patch"
```

3. Build the harness (set `PROTOC` if `protoc` is not on your `PATH`):

```bash
cargo build --release --manifest-path "$LANCE_DIR/Cargo.toml" -p lance --bench ivf_rq_gist
```

4. Run the matrix and summarize the results:

```bash
LANCE_DIR="$LANCE_DIR" DATA_DIR="$DATA_DIR" bash scripts/run_all.sh
```

```bash
python3 scripts/summarize.py results/results.jsonl
```
