# Track ④ · Faster vector + tag retrieval in Lance: v1 (8 slides, key technologies)

Open `ads-lance-hybrid-retrieval-v1.pptx`. Previews: `preview-hybrid-v1/`. Same form as v8 (team template, KT slides with diagram + table, summary + roadmap, no ask slide).

| # | Slide | Content |
|---|---|---|
| 1 | Title | — |
| 2 | ADS pipeline, stage (4) marked | v5.1 pipeline, emphasis moved from (3) to (4) + KT callout |
| 3 | Three key technologies in Lance filtered search | query-path map, 3 verified gaps, status |
| 4 | Filtered IVF loads partitions where no row passes | 4-step anatomy + [Model] empty-partition table |
| 5 | KT1 predicate-result cache | lookup flow + design table + evidence/TBD + gate |
| 6 | KT2 filter-aware partition skip | partition × tag summary figure + prepare_partition change + model/prior-art table + gate |
| 7 | KT3 batched mining | queries × partitions figure (18 → 6 loads, illustrative) + prior art + gate |
| 8 | Summary and roadmap | 3 KT cards + 4 phases |

No ADS measurement exists: values are [Source] (Lance code), [Reported] (papers), [Model] or TBD.
Rebuild: `uv run --no-project --with python-pptx python build_deck_hybrid_v1.py`. Render: `uv run --no-project --with pillow bash render_v5.sh ads-lance-hybrid-retrieval-v1.pptx preview-hybrid-v1`.
Shared setup: `ads_track_common.py` (reuses the v6.1 header/helpers like v8). Research basis: `../../lance_indexing_cache_research.md` v2 §3.9, §5.5, §11.
