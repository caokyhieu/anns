# Lance index builds on Kunpeng: v8 (8 slides, key technologies)

Open `ads-lance-kunpeng-index-build-v8.pptx`. Previews: `preview-v8/`.

Changes vs v7: work packages are now key technologies (KT), numbered in pipeline order. Each KT slide has a small diagram and a results table. There is no ask slide.

| # | Slide | Content |
|---|---|---|
| 1 | Title | — |
| 2 | ADS pipeline, stage (3) marked | v5.1 pipeline + callout (KT1–KT3) |
| 3 | Three key technologies in the Lance build | build map + status |
| 4 | PiPNN algorithm | 4-step figure; used by KT1-B and KT2 |
| 5 | KT1 assignment: A tree / B PiPNN centroid graph | 2 mini diagrams + table: build, speedup, assign CPU, recall@10, p50, p90 at nprobes 256 |
| 6 | KT2 PiPNN partition index | diagram + build-cost table + recall@10, p50 and p90 table (note: p90 ≈ 2 s is an index-cache artefact) |
| 7 | KT3 ARM kernels | blocked-kernel figure + kernel status table (Kunpeng TBD) |
| 8 | Summary and roadmap | 3 KT cards + 4 phases |

Data: KT1 from lance_index_build/results/ivf_rq_assignment_results.md (Table 1); KT2 from pilotC_hnsw_vs_pipnn_results.md.
Rebuild: `uv run --no-project --with python-pptx python build_deck_v8.py`. Render: `uv run --no-project --with pillow bash render_v5.sh ads-lance-kunpeng-index-build-v8.pptx preview-v8`.
