# KT3 ARM kernels: results for deck v8

Run `armk-20261112T113805Z` on the Kunpeng host (aarch64, SVE 256-bit), 2026-11-12.
Driver: `lance_index_build/kernels_all.sh`; report: `lance_index_build/scripts/kernel_report.py`.
Every step finished `ok`. The full report is reproduced at the end of this file.

## What was measured

Four kernel arms, each in its own process (dispatch is fixed per process):

| arm | build | settings | f32 L2 (pair / batch) | blocked L2 tile |
|---|---|---|---|---|
| legacy | no SVE | `LANCE_DISABLE_BLOCKED_L2=1`, `LANCE_F32_SVE=0` | Lance's own (auto-vectorised "scalar") | off |
| blocked | no SVE | `LANCE_F32_SVE=0` | Lance's own | NEON tile |
| sve | SVE | `LANCE_F32_SVE=1`, `LANCE_DISABLE_BLOCKED_L2=1` | SVE | off |
| new | SVE | `LANCE_F32_SVE=1` | SVE | SVE tile |

Three levels:

1. Criterion micro-benchmarks of each kernel (SVE build vs non-SVE build).
2. Standalone IVF_RQ stages: Lance's own stage functions (k-means training, RQ training,
   assignment, residual + RQ encode, probe) called in memory, without writing an index.
   The non-legacy arms reuse legacy's centroids, so accuracy differences come from the kernels only.
   3 repeats per arm; medians shown.
3. End-to-end Lance IVF_RQ build + search, legacy vs new, 3 builds each.

Datasets: GIST1M (1M x 960 f32, K=1000) and BIGANN10M (10M x 128, stored as f32, K=4096).
IVF_RQ with 5 bits, sample rate 256.

## Headline numbers for the slides

### KT3 first gate: blocked beats pairwise at equal vector width, recall within ±0.005

**Pass on both datasets.**

| stage | dataset | NEON 128-bit (legacy → blocked) | SVE 256-bit (sve → new) |
|---|---|---|---|
| assignment (exact) | GIST1M | 0.989 → 0.247 s, **4.00x** | 1.100 → 0.482 s, **2.28x** |
| assignment (exact) | BIGANN10M | 5.472 → 1.749 s, **3.13x** | 5.987 → 3.184 s, **1.88x** |
| k-means training | GIST1M | 3.591 → 3.057 s, **1.17x** | 3.539 → 3.387 s, **1.05x** |
| k-means training | BIGANN10M | 9.663 → 4.111 s, **2.35x** | 9.112 → 3.975 s, **2.29x** |

Accuracy:

- Assignment with the new kernels picks the same centroid as legacy for 100% of GIST rows
  and 99.9999% of BIGANN10M rows. The mean relative distance difference is about 1e-10.
- **Same-centroid recall:** probe recall@10 equals legacy exactly (Δ = 0.0000) for every arm,
  at every nprobes from 1 to 256, on both datasets.
- **End-to-end search recall:** each arm trains its own k-means here, and float rounding moves
  the centroids a little, so recall moves both up and down.
  - The worst drop is 0.0048 on GIST (k=100, nprobes=16) and 0.0001 on BIGANN10M.
  - The largest change is a gain of +0.0082 (BIGANN10M, k=10, nprobes=1).
  - Slide wording: "recall |Δ| ≤ 0.005".

### End-to-end Lance build time (legacy → new)

| dataset | train_ivf | shuffle (assignment) | total build |
|---|---|---|---|
| GIST1M | 6.20 → 4.57 s (1.36x) | 3.61 → 2.71 s (1.33x) | **10.76 → 8.46 s (1.27x)** |
| BIGANN10M | 15.99 → 8.66 s (1.85x) | 11.05 → 7.78 s (1.42x) | **29.60 → 18.80 s (1.58x)** |

The projection, which adds the standalone stage savings onto the legacy breakdown, predicted
10.03 s (GIST) and 21.68 s (BIGANN10M). The measured builds were faster than that, so the
standalone method is conservative.

### Where the build time goes (legacy, standalone compute)

| dataset | k-means | assignment | RQ encode | residual |
|---|---|---|---|---|
| GIST1M (5.04 s) | 71% | 20% | 8% | 1% |
| BIGANN10M (16.67 s) | 58% | 33% | 5% | 4% |

RQ training takes microseconds and is not a target.

### Kernel micro-benchmarks (SVE build vs non-SVE build, speedup)

| kernel | dim 128 | dim 960 | dim 1536 |
|---|---|---|---|
| l2_f32 (pair) | 1.52x | 1.17x | 0.96x |
| l2_f32_batch | 1.95x | 1.07x | 1.13x |
| dot_f32 (pair) | 1.96x | 1.19x | 1.05x |
| l2_u8 (pair) | 2.40x | 1.80x | 1.92x |
| dot_u8 (pair) | 2.55x | 2.00x | 2.53x |
| blocked l2_block_16 (vs the non-SVE build's tile) | 2.53x | 1.59x | – |
| dot_tile_4x4 (SVE tile vs NEON tile) | 0.95x | **0.74x** | – |

## Corrections and caveats for deck v8

1. **"Blocked L2 in progress" is out of date.** Blocked L2 with argmin is implemented, behind
   `LANCE_DISABLE_BLOCKED_L2`, and passes the first gate. Blocked top-k is not implemented.
2. **The SVE 4x4 tile is slower than the NEON 4x4 tile.** The micro benchmark shows 601 ns vs
   446 ns at dim 960. In the pipeline, NEON blocked assignment (0.247 s on GIST) beats SVE blocked
   assignment (0.482 s), so the best configuration today is SVE pairwise with the NEON tile.
   Next step, Phase 2 (P0 SVE kernels): either dispatch the NEON tile, or write a wider SVE tile
   that uses all 256 bits (for example 4x8).
3. **The GIST k-means gain is small (1.05–1.17x)** even though assignment gains 4x. K=1000 trains
   hierarchically, as many small sub-k-means runs, and distance work is a smaller share of each.
   Larger flat K (BIGANN10M, K=4096) gains 2.3x.
4. **The u8 kernels (2–2.5x) are micro-benchmark results only.** Lance ingests BIGANN as f32,
   so the u8 kernels are not on the build path. Keep them as kernel-level evidence, not
   end-to-end evidence.
5. **Search latency is not a KT3 claim yet.**
   - End-to-end search with the new kernels was 5–22% slower at p50 on BIGANN10M (worst at k=100).
     On GIST it ranged from 9% slower to 9% faster.
   - The two runs searched different indexes, because each trained its own centroids. The
     standalone probe on shared centroids was faster with the new kernels on BIGANN10M
     (87 vs 115 µs at nprobes=1).
   - So the slowdown is not attributed to the kernels. It needs a same-index A/B before it goes
     on a slide.
6. **Not measured here:**
   - The leaf Gram kernel (P0, KT1 + KT2 integration on ARM).
   - The SVE-vs-NEON baseline on a host without SVE.
   - Tree assignment (KT1, `tree:4`) is shown only for context. It is 20x faster than exact
     assignment, with 95–96% of rows assigned to the same centroid as exact, and the new kernels
     change it very little.
   - RQ training and the residual encode stay flat within noise (about ±10%).

## Mapping to the deck roadmap (slide 8)

| phase | status after this run |
|---|---|
| 1. Kunpeng bring-up, SVE vs NEON baseline | done: four arms on one host, SVE 256-bit |
| 2. P0 SVE kernels: blocked L2 + argmin, leaf Gram | blocked L2 + argmin done and gated; SVE tile needs widening; leaf Gram open |
| 3. Integrate KT1 + KT2 on ARM | open (the stage bench shows where it plugs in) |
| 4. Harden / upstream | open |

---

## Full report (`report.md`, as generated)

# ARM kernels: legacy Lance vs new kernels (armk-20261112T113805Z)

## 1. Kernels (criterion)

Mean ns per call. legacy = build without SVE objects (Lance's NEON / scalar kernels and the NEON tile); new = SVE build.

| benchmark | legacy ns | new ns | speedup |
|---|---|---|---|
| arm_distances/dot_f32/128 | 24.359 | 12.41 | 1.963 |
| arm_distances/dot_f32/1536 | 152.559 | 144.754 | 1.054 |
| arm_distances/dot_f32/16 | 7.306 | 3.86 | 1.893 |
| arm_distances/dot_f32/32 | 10.382 | 5.875 | 1.767 |
| arm_distances/dot_f32/4097 | 419.106 | 394.409 | 1.063 |
| arm_distances/dot_f32/768 | 98.075 | 72.215 | 1.358 |
| arm_distances/dot_f32/8 | 5.643 | 3.682 | 1.533 |
| arm_distances/dot_f32/960 | 106.593 | 89.369 | 1.193 |
| arm_distances/dot_f32_batch/128 | 1,430 | 1,006 | 1.421 |
| arm_distances/dot_f32_batch/1536 | 10,139 | 11,809 | 0.859 |
| arm_distances/dot_f32_batch/16 | 428.371 | 247.166 | 1.733 |
| arm_distances/dot_f32_batch/32 | 587.651 | 344.217 | 1.707 |
| arm_distances/dot_f32_batch/4097 | 30,259 | 32,052 | 0.944 |
| arm_distances/dot_f32_batch/768 | 5,765 | 5,639 | 1.022 |
| arm_distances/dot_f32_batch/8 | 253.364 | 149.771 | 1.692 |
| arm_distances/dot_f32_batch/960 | 5,948 | 7,483 | 0.795 |
| arm_distances/dot_u8/128 | 10.982 | 4.309 | 2.549 |
| arm_distances/dot_u8/1536 | 85.679 | 33.871 | 2.53 |
| arm_distances/dot_u8/16 | 3.959 | 2.814 | 1.407 |
| arm_distances/dot_u8/32 | 4.746 | 2.701 | 1.757 |
| arm_distances/dot_u8/4097 | 163.866 | 113.736 | 1.441 |
| arm_distances/dot_u8/768 | 41.917 | 18.646 | 2.248 |
| arm_distances/dot_u8/8 | 3.75 | 2.686 | 1.396 |
| arm_distances/dot_u8/960 | 44.259 | 22.169 | 1.996 |
| arm_distances/l2_f32/128 | 28.878 | 18.974 | 1.522 |
| arm_distances/l2_f32/1536 | 165.565 | 171.715 | 0.964 |
| arm_distances/l2_f32/16 | 8.346 | 4.321 | 1.932 |
| arm_distances/l2_f32/32 | 11.034 | 7.08 | 1.558 |
| arm_distances/l2_f32/4097 | 481.078 | 429.005 | 1.121 |
| arm_distances/l2_f32/768 | 101.008 | 78.805 | 1.282 |
| arm_distances/l2_f32/8 | 4.406 | 3.747 | 1.176 |
| arm_distances/l2_f32/960 | 128.273 | 109.632 | 1.17 |
| arm_distances/l2_f32_batch/128 | 1,689 | 866.895 | 1.948 |
| arm_distances/l2_f32_batch/1536 | 11,245 | 9,952 | 1.13 |
| arm_distances/l2_f32_batch/16 | 486.891 | 168.687 | 2.886 |
| arm_distances/l2_f32_batch/32 | 635.777 | 292.369 | 2.175 |
| arm_distances/l2_f32_batch/4097 | 33,356 | 29,864 | 1.117 |
| arm_distances/l2_f32_batch/768 | 5,724 | 5,388 | 1.062 |
| arm_distances/l2_f32_batch/8 | 247.188 | 188.141 | 1.314 |
| arm_distances/l2_f32_batch/960 | 7,005 | 6,525 | 1.073 |
| arm_distances/l2_u8/128 | 13.261 | 5.525 | 2.4 |
| arm_distances/l2_u8/1536 | 89.863 | 46.846 | 1.918 |
| arm_distances/l2_u8/16 | 3.966 | 2.941 | 1.349 |
| arm_distances/l2_u8/32 | 4.923 | 2.84 | 1.733 |
| arm_distances/l2_u8/4097 | 194.105 | 128.923 | 1.506 |
| arm_distances/l2_u8/768 | 50.238 | 23.202 | 2.165 |
| arm_distances/l2_u8/8 | 3.813 | 2.845 | 1.34 |
| arm_distances/l2_u8/960 | 56.05 | 31.212 | 1.796 |
| arm_lookup/f32_rows/n1024_cols96_selected129 | 13,206 | 13,309 | 0.992 |
| arm_lookup/f32_rows/n16384_cols192_selected1024 | 666,745 | 454,706 | 1.466 |
| arm_lookup/f32_rows/n32_cols16_selected7 | 134.219 | 133.134 | 1.008 |
| arm_lookup/hacc/n1024_cols96_selected129 | 138,229 | 138,429 | 0.999 |
| arm_lookup/hacc/n16384_cols192_selected1024 | 4,408,907 | 4,431,427 | 0.995 |
| arm_lookup/hacc/n32_cols16_selected7 | 733.759 | 734.345 | 0.999 |
| index_tiles/dot_tile_4x4/100 | 53.226 | 77.537 | 0.686 |
| index_tiles/dot_tile_4x4/128 | 67.695 | 71.552 | 0.946 |
| index_tiles/dot_tile_4x4/131 | 103.049 | 102.85 | 1.002 |
| index_tiles/dot_tile_4x4/960 | 445.603 | 600.968 | 0.741 |
| index_tiles/l2_block_16/100 | 547.649 | 173.087 | 3.164 |
| index_tiles/l2_block_16/128 | 437.431 | 172.815 | 2.531 |
| index_tiles/l2_block_16/131 | 456.24 | 236.258 | 1.931 |
| index_tiles/l2_block_16/960 | 1,991 | 1,252 | 1.59 |
| index_tiles/l2_block_32/100 | 1,098 | 351.814 | 3.12 |
| index_tiles/l2_block_32/128 | 874.962 | 365.898 | 2.391 |
| index_tiles/l2_block_32/131 | 914.129 | 496.476 | 1.841 |
| index_tiles/l2_block_32/960 | 3,991 | 2,644 | 1.51 |
| index_tiles/l2_block_4/100 | 138.267 | 49.892 | 2.771 |
| index_tiles/l2_block_4/128 | 110.742 | 49.159 | 2.253 |
| index_tiles/l2_block_4/131 | 116.357 | 61.758 | 1.884 |
| index_tiles/l2_block_4/960 | 495.946 | 304.978 | 1.626 |
| index_tiles/l2_block_6/100 | 207.15 | 75.396 | 2.747 |
| index_tiles/l2_block_6/128 | 165.211 | 84.938 | 1.945 |
| index_tiles/l2_block_6/131 | 171.533 | 104.931 | 1.635 |
| index_tiles/l2_block_6/960 | 739.124 | 535.351 | 1.381 |
| index_tiles/l2_block_64/100 | 2,178 | 680.274 | 3.201 |
| index_tiles/l2_block_64/128 | 1,751 | 673.541 | 2.6 |
| index_tiles/l2_block_64/131 | 1,827 | 913.67 | 1.999 |
| index_tiles/l2_block_64/960 | 8,115 | 5,120 | 1.585 |

## bigann10m

### KT3 first gate (deck slide 7)

Blocked vs pairwise at equal vector width (the deck's first gate). ids = share of rows assigned to the same centroid as legacy, with legacy's centroids.

| stage | width | pairwise → blocked | pairwise (s) | blocked (s) | × | ids = legacy | faster? |
|---|---|---|---|---|---|---|---|
| train / kmeans | NEON (128-bit) | legacy → blocked | 9.663 | 4.111 | 2.35 | – | pass |
| train / kmeans | SVE (256-bit) | sve → new | 9.112 | 3.975 | 2.292 | – | pass |
| assign / exact | NEON (128-bit) | legacy → blocked | 5.472 | 1.749 | 3.129 | 100.0000% | pass |
| assign / exact | SVE (256-bit) | sve → new | 5.987 | 3.184 | 1.88 | 99.9999% | pass |
| assign / auto | NEON (128-bit) | legacy → blocked | 5.492 | 1.698 | 3.234 | – | pass |
| assign / auto | SVE (256-bit) | sve → new | 6.044 | 3.158 | 1.914 | – | pass |

Recall vs legacy (tolerance ±0.005): same centroids (probe) worst |Δ| = 0.0000; end-to-end search (own k-means) worst drop = 0.0001 → **pass**

| where | arm | legacy recall | arm recall | Δ |
|---|---|---|---|---|
| probe nprobes=1 | blocked | 0.329 | 0.329 | +0.0000 |
| probe nprobes=1 | sve | 0.329 | 0.329 | +0.0000 |
| probe nprobes=1 | new | 0.329 | 0.329 | +0.0000 |
| probe nprobes=4 | blocked | 0.638 | 0.638 | +0.0000 |
| probe nprobes=4 | sve | 0.638 | 0.638 | +0.0000 |
| probe nprobes=4 | new | 0.638 | 0.638 | +0.0000 |
| probe nprobes=16 | blocked | 0.881 | 0.881 | +0.0000 |
| probe nprobes=16 | sve | 0.881 | 0.881 | +0.0000 |
| probe nprobes=16 | new | 0.881 | 0.881 | +0.0000 |
| probe nprobes=64 | blocked | 0.981 | 0.981 | +0.0000 |
| probe nprobes=64 | sve | 0.981 | 0.981 | +0.0000 |
| probe nprobes=64 | new | 0.981 | 0.981 | +0.0000 |
| probe nprobes=256 | blocked | 0.999 | 0.999 | +0.0000 |
| probe nprobes=256 | sve | 0.999 | 0.999 | +0.0000 |
| probe nprobes=256 | new | 0.999 | 0.999 | +0.0000 |
| search k=10,nprobes=1 | new | 0.329 | 0.337 | +0.0082 |
| search k=10,nprobes=4 | new | 0.629 | 0.635 | +0.0062 |
| search k=10,nprobes=16 | new | 0.851 | 0.853 | +0.0021 |
| search k=10,nprobes=64 | new | 0.931 | 0.932 | +0.0010 |
| search k=10,nprobes=256 | new | 0.944 | 0.944 | +0.0006 |
| search k=100,nprobes=1 | new | 0.262 | 0.266 | +0.0043 |
| search k=100,nprobes=4 | new | 0.549 | 0.551 | +0.0022 |
| search k=100,nprobes=16 | new | 0.814 | 0.814 | +0.0002 |
| search k=100,nprobes=64 | new | 0.939 | 0.939 | -0.0001 |
| search k=100,nprobes=256 | new | 0.961 | 0.961 | +0.0001 |

### 2. Standalone IVF_RQ stages (Lance's stage functions, in memory, no index written)

Kernels each arm dispatched:

| arm | l2_f32_pair | l2_f32_batch | dot_tile | blocked_l2 |
|---|---|---|---|---|
| legacy | l2_f32_scalar | l2_batch_f32_scalar | dot_tile_neon | None |
| blocked | l2_f32_scalar | l2_batch_f32_scalar | dot_tile_neon | dot_tile_neon |
| sve | l2_f32_sve | l2_batch_f32_sve | dot_tile_sve | None |
| new | l2_f32_sve | l2_batch_f32_sve | dot_tile_sve | dot_tile_sve |

| stage / variant | legacy (s) | blocked (s) | sve (s) | new (s) | ×blocked | ×sve | ×new | accuracy (new arm) |
|---|---|---|---|---|---|---|---|---|
| train / kmeans | 9.663 | 4.111 | 9.112 | 3.975 | 2.35 | 1.06 | 2.431 | – |
| quantizer / rq5 | 3.13e-05 | 2.3e-05 | 2.41e-05 | 1.84e-05 | 1.361 | 1.302 | 1.706 | – |
| assign / auto | 5.492 | 1.698 | 6.044 | 3.158 | 3.234 | 0.909 | 1.739 | ids = exact 100.000% |
| assign / exact | 5.472 | 1.749 | 5.987 | 3.184 | 3.129 | 0.914 | 1.718 | ids = legacy 99.9999%, dist +1.07e-10 |
| assign / tree:4 | 0.279 | 0.264 | 0.245 | 0.224 | 1.06 | 1.139 | 1.249 | ids = exact 95.078% |
| encode / residual | 0.718 | 0.792 | 0.851 | 0.818 | 0.907 | 0.844 | 0.878 | – |
| encode / rq5 | 0.797 | 0.683 | 0.878 | 0.629 | 1.168 | 0.908 | 1.267 | – |
| probe (µs p50) / nprobes=1 | 115.309 | 122.43 | 94.387 | 87.036 | 0.942 | 1.222 | 1.325 | probe recall@10 0.329 |
| probe (µs p50) / nprobes=4 | 118.639 | 125.609 | 93.767 | 92.287 | 0.945 | 1.265 | 1.286 | probe recall@10 0.638 |
| probe (µs p50) / nprobes=16 | 119.089 | 125.749 | 95.487 | 93.117 | 0.947 | 1.247 | 1.279 | probe recall@10 0.881 |
| probe (µs p50) / nprobes=64 | 119.419 | 126.93 | 100.138 | 104.208 | 0.941 | 1.193 | 1.146 | probe recall@10 0.981 |
| probe (µs p50) / nprobes=256 | 122.609 | 130.03 | 98.897 | 110.378 | 0.943 | 1.24 | 1.111 | probe recall@10 0.999 |

Legacy build compute, standalone (16.671 s, Lance's `auto` assignment): train/kmeans 58%, quantizer/rq5 0%, assign/auto 33%, encode/residual 4%, encode/rq5 5%

### 3. End-to-end Lance IVF_RQ build + search

| Lance stage | legacy (s) | new (s) | ×new |
|---|---|---|---|
| merge_partitions | 2.446 | 2.511 | 0.974 |
| shuffle | 11.048 | 7.784 | 1.419 |
| train_ivf | 15.99 | 8.655 | 1.847 |
| train_quantizer | 8.28e-05 | 0.000342 | 0.242 |
| **total build** | 29.603 | 18.801 | 1.575 |

Search (warm, per query):

| search | legacy recall | legacy p50 µs | new recall | new p50 µs | ×p50 |
|---|---|---|---|---|---|
| k=10 nprobes=1 | 0.329 | 2,197 | 0.337 | 2,727 | 0.806 |
| k=10 nprobes=4 | 0.629 | 2,892 | 0.635 | 2,995 | 0.966 |
| k=10 nprobes=16 | 0.851 | 3,694 | 0.853 | 3,714 | 0.995 |
| k=10 nprobes=64 | 0.931 | 5,423 | 0.932 | 5,697 | 0.952 |
| k=10 nprobes=256 | 0.944 | 13,236 | 0.944 | 13,809 | 0.959 |
| k=100 nprobes=1 | 0.262 | 3,329 | 0.266 | 3,896 | 0.854 |
| k=100 nprobes=4 | 0.549 | 3,568 | 0.551 | 4,276 | 0.834 |
| k=100 nprobes=16 | 0.814 | 3,908 | 0.814 | 5,005 | 0.781 |
| k=100 nprobes=64 | 0.939 | 5,973 | 0.939 | 6,898 | 0.866 |
| k=100 nprobes=256 | 0.961 | 10,578 | 0.961 | 12,094 | 0.875 |

Projection: the legacy build's own breakdown minus the time the standalone stages saved (legacy − new). The rest of each stage (I/O, shuffle spill, writing) is assumed unchanged; *measured new* is the actual end-to-end run with the kernels.

| Lance stage | legacy (s) | saved (s) | projected (s) | measured new (s) |
|---|---|---|---|---|
| train_ivf | 15.99 | 5.688 | 10.302 | 8.655 |
| train_quantizer | 8.28e-05 | 1.3e-05 | 6.99e-05 | 0.000342 |
| shuffle | 11.048 | 2.235 | 8.813 | 7.784 |
| **total build** | 29.603 | 7.923 | 21.68 | 18.801 |

## gist

### KT3 first gate (deck slide 7)

Blocked vs pairwise at equal vector width (the deck's first gate). ids = share of rows assigned to the same centroid as legacy, with legacy's centroids.

| stage | width | pairwise → blocked | pairwise (s) | blocked (s) | × | ids = legacy | faster? |
|---|---|---|---|---|---|---|---|
| train / kmeans | NEON (128-bit) | legacy → blocked | 3.591 | 3.057 | 1.175 | – | pass |
| train / kmeans | SVE (256-bit) | sve → new | 3.539 | 3.387 | 1.045 | – | pass |
| assign / exact | NEON (128-bit) | legacy → blocked | 0.989 | 0.247 | 4.001 | 100.0000% | pass |
| assign / exact | SVE (256-bit) | sve → new | 1.1 | 0.482 | 2.282 | 100.0000% | pass |
| assign / auto | NEON (128-bit) | legacy → blocked | 1.001 | 0.246 | 4.076 | – | pass |
| assign / auto | SVE (256-bit) | sve → new | 1.115 | 0.481 | 2.319 | – | pass |

Recall vs legacy (tolerance ±0.005): same centroids (probe) worst |Δ| = 0.0000; end-to-end search (own k-means) worst drop = 0.0048 → **pass**

| where | arm | legacy recall | arm recall | Δ |
|---|---|---|---|---|
| probe nprobes=1 | blocked | 0.201 | 0.201 | +0.0000 |
| probe nprobes=1 | sve | 0.201 | 0.201 | +0.0000 |
| probe nprobes=1 | new | 0.201 | 0.201 | +0.0000 |
| probe nprobes=4 | blocked | 0.465 | 0.465 | +0.0000 |
| probe nprobes=4 | sve | 0.465 | 0.465 | +0.0000 |
| probe nprobes=4 | new | 0.465 | 0.465 | +0.0000 |
| probe nprobes=16 | blocked | 0.768 | 0.768 | +0.0000 |
| probe nprobes=16 | sve | 0.768 | 0.768 | +0.0000 |
| probe nprobes=16 | new | 0.768 | 0.768 | +0.0000 |
| probe nprobes=64 | blocked | 0.959 | 0.959 | +0.0000 |
| probe nprobes=64 | sve | 0.959 | 0.959 | +0.0000 |
| probe nprobes=64 | new | 0.959 | 0.959 | +0.0000 |
| probe nprobes=256 | blocked | 0.999 | 0.999 | +0.0000 |
| probe nprobes=256 | sve | 0.999 | 0.999 | +0.0000 |
| probe nprobes=256 | new | 0.999 | 0.999 | +0.0000 |
| search k=10,nprobes=1 | new | 0.199 | 0.197 | -0.0015 |
| search k=10,nprobes=4 | new | 0.463 | 0.463 | -0.0003 |
| search k=10,nprobes=16 | new | 0.753 | 0.757 | +0.0043 |
| search k=10,nprobes=64 | new | 0.927 | 0.929 | +0.0024 |
| search k=10,nprobes=256 | new | 0.957 | 0.959 | +0.0014 |
| search k=100,nprobes=1 | new | 0.148 | 0.144 | -0.0046 |
| search k=100,nprobes=4 | new | 0.37 | 0.371 | +0.0004 |
| search k=100,nprobes=16 | new | 0.686 | 0.681 | -0.0048 |
| search k=100,nprobes=64 | new | 0.919 | 0.919 | +0.0003 |
| search k=100,nprobes=256 | new | 0.972 | 0.972 | +0.0002 |

### 2. Standalone IVF_RQ stages (Lance's stage functions, in memory, no index written)

Kernels each arm dispatched:

| arm | l2_f32_pair | l2_f32_batch | dot_tile | blocked_l2 |
|---|---|---|---|---|
| legacy | l2_f32_scalar | l2_batch_f32_scalar | dot_tile_neon | None |
| blocked | l2_f32_scalar | l2_batch_f32_scalar | dot_tile_neon | dot_tile_neon |
| sve | l2_f32_sve | l2_batch_f32_sve | dot_tile_sve | None |
| new | l2_f32_sve | l2_batch_f32_sve | dot_tile_sve | dot_tile_sve |

| stage / variant | legacy (s) | blocked (s) | sve (s) | new (s) | ×blocked | ×sve | ×new | accuracy (new arm) |
|---|---|---|---|---|---|---|---|---|
| train / kmeans | 3.591 | 3.057 | 3.539 | 3.387 | 1.175 | 1.015 | 1.06 | – |
| quantizer / rq5 | 1.61e-05 | 1.63e-05 | 1.87e-05 | 1.83e-05 | 0.988 | 0.86 | 0.878 | – |
| assign / auto | 1.001 | 0.246 | 1.115 | 0.481 | 4.076 | 0.898 | 2.082 | ids = exact 100.000% |
| assign / exact | 0.989 | 0.247 | 1.1 | 0.482 | 4.001 | 0.899 | 2.053 | ids = legacy 100.0000%, dist +1.14e-11 |
| assign / tree:4 | 0.114 | 0.115 | 0.113 | 0.119 | 0.998 | 1.012 | 0.959 | ids = exact 96.373% |
| encode / residual | 0.0632 | 0.062 | 0.047 | 0.0624 | 1.018 | 1.343 | 1.012 | – |
| encode / rq5 | 0.387 | 0.386 | 0.499 | 0.404 | 1.003 | 0.775 | 0.959 | – |
| probe (µs p50) / nprobes=1 | 148.282 | 145.721 | 148.322 | 155.812 | 1.018 | 1 | 0.952 | probe recall@10 0.201 |
| probe (µs p50) / nprobes=4 | 148.582 | 126.199 | 142.901 | 165.713 | 1.177 | 1.04 | 0.897 | probe recall@10 0.465 |
| probe (µs p50) / nprobes=16 | 148.681 | 119.709 | 145.971 | 160.393 | 1.242 | 1.019 | 0.927 | probe recall@10 0.768 |
| probe (µs p50) / nprobes=64 | 149.582 | 120.339 | 153.641 | 157.492 | 1.243 | 0.974 | 0.95 | probe recall@10 0.959 |
| probe (µs p50) / nprobes=256 | 152.802 | 123.97 | 142.5 | 160.392 | 1.233 | 1.072 | 0.953 | probe recall@10 0.999 |

Legacy build compute, standalone (5.042 s, Lance's `auto` assignment): train/kmeans 71%, quantizer/rq5 0%, assign/auto 20%, encode/residual 1%, encode/rq5 8%

### 3. End-to-end Lance IVF_RQ build + search

| Lance stage | legacy (s) | new (s) | ×new |
|---|---|---|---|
| merge_partitions | 1.056 | 1.041 | 1.014 |
| shuffle | 3.613 | 2.711 | 1.333 |
| train_ivf | 6.198 | 4.569 | 1.357 |
| train_quantizer | 8.98e-05 | 0.000728 | 0.123 |
| **total build** | 10.759 | 8.458 | 1.272 |

Search (warm, per query):

| search | legacy recall | legacy p50 µs | new recall | new p50 µs | ×p50 |
|---|---|---|---|---|---|
| k=10 nprobes=1 | 0.199 | 2,494 | 0.197 | 2,634 | 0.947 |
| k=10 nprobes=4 | 0.463 | 2,656 | 0.463 | 2,498 | 1.063 |
| k=10 nprobes=16 | 0.753 | 3,202 | 0.757 | 2,947 | 1.086 |
| k=10 nprobes=64 | 0.927 | 4,840 | 0.929 | 4,847 | 0.998 |
| k=10 nprobes=256 | 0.957 | 10,323 | 0.959 | 10,168 | 1.015 |
| k=100 nprobes=1 | 0.148 | 3,243 | 0.144 | 3,423 | 0.948 |
| k=100 nprobes=4 | 0.37 | 3,393 | 0.371 | 3,707 | 0.915 |
| k=100 nprobes=16 | 0.686 | 3,979 | 0.681 | 4,243 | 0.938 |
| k=100 nprobes=64 | 0.919 | 6,205 | 0.919 | 6,457 | 0.961 |
| k=100 nprobes=256 | 0.972 | 10,927 | 0.972 | 11,087 | 0.986 |

Projection: the legacy build's own breakdown minus the time the standalone stages saved (legacy − new). The rest of each stage (I/O, shuffle spill, writing) is assumed unchanged; *measured new* is the actual end-to-end run with the kernels.

| Lance stage | legacy (s) | saved (s) | projected (s) | measured new (s) |
|---|---|---|---|---|
| train_ivf | 6.198 | 0.205 | 5.993 | 4.569 |
| train_quantizer | 8.98e-05 | -2.24e-06 | 9.2e-05 | 0.000728 |
| shuffle | 3.613 | 0.521 | 3.093 | 2.711 |
| **total build** | 10.759 | 0.725 | 10.033 | 8.458 |
