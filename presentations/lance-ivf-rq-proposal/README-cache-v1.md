# Track ⑥ · Millisecond data serving from Lance: v1 (8 slides, key technologies)

Open `ads-lance-multimodal-cache-v1.pptx`. Previews: `preview-cache-v1/`. Same form as v8.

| # | Slide | Content |
|---|---|---|
| 1 | Title | — |
| 2 | ADS pipeline, stage (6) marked | v5.1 pipeline, emphasis moved from (3) to (6) + KT callout |
| 3 | Three key technologies in the Lance read path | consumers → reader → L1/L2/L3 → OBS map, status |
| 4 | "ms" needs a close hit and a partial read | [Model] bandwidth table + [Source] what Lance caches today |
| 5 | KT1 tiered byte cache | read through 4 tiers figure + Lance hooks table + prior art + gate |
| 6 | KT2 prefetch from known training order | illustrative timeline + prefetch flow + policy + gate |
| 7 | KT3 cost-aware admission and placement | U = F·C·R/S, object-class table, UB pool placement, baselines + gate |
| 8 | Summary and roadmap | 3 KT cards + 4 phases |

No ADS or UB measurement exists: UB / NVMe / OBS latency and bandwidth are TBD (phase 1).
Rebuild: `uv run --no-project --with python-pptx python build_deck_cache_v1.py`. Render: `uv run --no-project --with pillow bash render_v5.sh ads-lance-multimodal-cache-v1.pptx preview-cache-v1`.
Research basis: `../../lance_indexing_cache_research.md` v2 §3.8, §5.6, §11.
