# Lance index builds on Kunpeng: v6 project proposal

Open `ads-lance-kunpeng-index-build-v6.pptx` (20 slides: 15 timed, about 20 min, plus 5 appendix). Previews: `preview-v6/` (PDF, per-slide PNGs, contact sheet).

Rebuild: `uv run --no-project --with python-pptx --with pillow python build_deck_v6.py` (needs team_deck.py, team-template.pptx, style.json and the v5.1 PPTX for slide 3). Render: `bash render_v5.sh ads-lance-kunpeng-index-build-v6.pptx preview-v6`.

Files: brief-v6.md, outline-v6.md (change log), sources-v6.md, assumptions-and-open-questions-v6.md, diagram-sources-v6.json, slide-manifest-v6.json, layout-warnings-v6.json (empty).

To add Kunpeng results, edit the `TBD` cells in build_deck_v6.py (slide 2 Kunpeng column, slide 8 "Kunpeng result" column, slide 9 results table) and rebuild.
