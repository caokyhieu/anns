# V6 brief: Lance index builds on Kunpeng (approved 8 Nov 2026: 20 min, SVE/SVE2, team Vector Index Optimization)

**Mode:** substantial redirect of v5.1 (new deck file; v5.1 preserved).

| Field | Value |
|---|---|
| Working title | Faster Lance Index Builds on Kunpeng: ARM Kernels + Build Algorithms |
| Audience | Technical leadership (inherited from v5) |
| Duration | **20 min proposed** (complete project): 15 timed slides + 5 appendix, 2–3 min Q&A |
| Layout | Research-to-Product, team theme; kicker "ADS · Lance Index Builds on Kunpeng" |
| HQ product / stack | ADS data pipeline, stage (3) vector production → index build (Lance IVF_RQ / IVF_HNSW), slide retained from v5.1 |
| Problem | Index build is the delay before new ADS vectors become searchable; three loops dominate: centroid assignment, multi-bit RQ encoding, per-partition graph build. All are dense-distance loops. On ARM, Lance has only pairwise NEON (and our SVE pair ports), with no tiled kernels |
| Decision ask | Approve a 3-work-package project with **WP1 ARM kernels on Kunpeng as the lead**. Needs Kunpeng server access, owners and a recall tolerance |

## Value thesis per work package

1. **WP1 ARM/Kunpeng kernels (lead).** Today the build-time algorithms only pay off where tiled distance kernels exist. On x86 the 4×4 blocked-L2 tile alone cut GIST1M RQ1 build time 2.07× (63.6 → 30.8 s). Lance on ARM has pairwise kernels only. We have six SVE/SVE2 kernels (QEMU-validated at 128–2048-bit VL); next come the P0 tiled L2+argmin and leaf-Gram kernels that serve WP2 and WP3. **Kunpeng numbers are placeholders** (`[Kunpeng TBD]`) until the runs come in.
2. **WP2 tree assignment.** Lance throws away the hierarchical k-means tree. Reusing it at beam 4 cut BIGANN10M (K=16384) build time from 37.7 to 15.1 s (2.49×) and assignment CPU 12.8×, with recall@10 within 0.005 of exact. HNSW and PiPNN centroid graphs lost to the tree (negative result shown). There is no gain at K=1000 on GIST, where assignment is already cheap.
3. **WP3 PiPNN partition graphs (IVF_PIPNN_FLAT).** Against IVF_HNSW_FLAT on BIGANN10M, K=5: build 537.7 → 109.9 s (4.9×), 6.9× less CPU, a 15% smaller index, and recall higher at 9 of 10 settings measured on both, −0.001 at the 10th. Cost: +1.5 GB peak RSS. The deck reports build time, CPU, peak RSS, index size and recall@10 (nprobes 1–2, all ef), plus p50 parity. Tail latency is omitted: it was a harness cache artefact (6 GiB default index cache < index size) that hit both indexes equally. That explanation goes in the notes and appendix, and there's no rerun.
4. **Carried from v4/v5: blocked L2 + constant RQ scale** (x86, GIST1M): up to 5.74× at RQ9. Constant scale costs at most 0.006 R@10. Shown as the existing optimized baseline and as kernel evidence.

## Evidence labels
[Measured-x86] AMD EPYC 9124 or 4-core Xeon, single runs (REPEATS=1 unless stated) · [Kunpeng TBD] · [Source] Lance code · [Reported] papers · [Proposal].

## Gaps
Kunpeng exact model (SVE/SVE2 confirmed by user), repeats ≥3, Suite C complete query grid, per-process RSS for Suite B, ARM native build (toolchain cert issue), production recall target, owners and schedule.
