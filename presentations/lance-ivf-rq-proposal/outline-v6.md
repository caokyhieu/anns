# V6 outline (as built)

Timed (≈20 min)

| # | Conclusion title | Visual | Key numbers / status |
|---|---|---|---|
| 1 | Faster Lance index builds on Kunpeng | Title | v6 · date · presenters |
| 2 | Three measured wins; Kunpeng is next | 4-stat dashboard + ask strip | 2.5× (tree), 4.9× (PiPNN graphs), 5.7× (blocked L2 + const scale), Kunpeng [TBD] dashed |
| 3 | Target (3): faster index construction | Retained v5.1 ADS pipeline slide | — |
| 4 | Build time sits in three dense-distance loops | Stacked stage bars per workload | BIGANN K16384: shuffle 31.3 of 37.7 s; GIST RQ9: shuffle 197 of 221 s; HNSW_FLAT 538 s graph build |
| 5 | One project, three work packages, one kernel layer | Architecture map: Lance build pipeline with WP2/WP3 integration points over a WP1 kernel dispatch layer (x86 / NEON / SVE) | status tags: done / in progress / planned |
| — | **Proposal 1: ARM kernels on Kunpeng (lead)** | | |
| 6 | Tiling, not ISA, made the x86 win | Before/after: pairwise vs 4×4 tile; IVF train 19.0→13.3 s, shuffle 44→17 s | ARM today: pairwise only |
| 7 | Tiled L2 + fused argmin serves every new loop | Method figure: x-block × c-block register tile, ‖x‖²+‖c‖²−2x·c, bias, argmin; leaf Gram tile; VL-agnostic SVE | consumers: k-means, tree beam, PiPNN routing/leaf kNN |
| 8 | Six SVE kernels ready; P0 tiles next | Prioritized table P0/P1/P2: kernel, consumer, status, Kunpeng metric | 6 kernels QEMU-validated 128–2048-bit; Kunpeng columns [TBD] |
| 9 | Kunpeng evaluation: five ablations, same configs | Ablation ladder (stock → opt → +existing ARM → +algorithm → +new kernels) + placeholder results table | all cells [Kunpeng TBD], dashed |
| — | **Proposal 2: tree assignment** | | |
| 10 | Reuse the trained tree to assign rows | Tree-beam mechanism; cost 16·beam·depth vs K | tree:4 ≈ 256 vs 16,384 distances/row |
| 11 | Tree assignment halves BIGANN10M build time | Bar chart build s by method + match/recall column | 37.7→15.1 s (2.49×), assign CPU 12.8× less, ΔR@10 ≤0.005; GIST K16384 1.37–1.58×; K=1000 no gain; graphs lose to tree |
| — | **Proposal 3: PiPNN partition graphs** | | |
| 12 | PiPNN replaces insertion search with leaf GEMMs | Compact 4-panel method figure (leaves → Gram → HashPrune → RobustPrune) | kernel hooks marked |
| 13 | PiPNN builds partition graphs 4.9× faster | Evidence dashboard: build/CPU/size/RSS bars + recall table | 537.7→109.9 s, 6.9× CPU, −15% size, +1.5 GB, recall higher at 9/10 (−0.001 at 10th); standalone 10M fastpipnn 222 vs 1557 s, 18.8 vs 6.1 GB |
| 14 | Roadmap: Kunpeng baseline first, gates at each step | Phase timeline with exit gates; T-shirt sizes | Phases 0–7 from plan, current status |
| 15 | Ask: Kunpeng access, owners, recall tolerance | Summary: 3 takeaways + risks/mitigations + ask | — |

Appendix

| A1 | Full assignment-method table (BIGANN10M + GIST1M) |
| A2 | Suite C full recall grid (nprobes 1–2 × ef 32–512) + one-line note on why tail latency is excluded (harness cache, both indexes) |
| A3 | Blocked L2 + const scale GIST1M table (RQ1/5/9) |
| A4 | PiPNN memory model and the 1M memory pilot |
| A5 | Runs, sources and references |

## Change log
2026-11-08: v6 built as `ads-lance-kunpeng-index-build-v6.pptx` by `build_deck_v6.py`. It has 15 timed slides and 5 appendix slides. v5.1 and earlier decks are unchanged.
- Restructured from v5.1's two-direction exploration into a complete three-work-package project. WP1 (ARM SVE/SVE2 kernels on Kunpeng) leads the proposal section (slides 6–9).
- Every headline number now comes from our own x86 runs and sits on the slide of the method that produced it: tree 2.49× (RA), PiPNN graphs 4.9× (RC), blocked L2 2.07× / +const 5.74× (RG). v5's "no measured speedup" framing is replaced.
- Kunpeng cells are explicit `TBD` placeholders (slides 2, 8, 9) for the user's native runs.
- Negative results are shown: centroid graphs lose to the tree; no gain at K=1000; the GIST gain over Lance auto is small.
- Suite C tail latency is excluded and explained in A2: a harness index-cache artefact (6 GiB default < index size), identical for both indexes. No rerun, by user decision.
- Recall wording corrected: 10 settings were measured on both indexes. PiPNN is higher at 9 and 0.0008 lower at nprobes 1, ef 512.
- Slide 3 (ADS context) is copied shape for shape from v5.1 slide 2; only the kicker changed.
- Render check: fixed overflowing section labels (2, 13), the ladder/strip overlap (9), the overlapping badge text and table on 15, and empty space (7, 14).
