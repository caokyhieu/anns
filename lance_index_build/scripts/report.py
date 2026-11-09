#!/usr/bin/env python3
"""Summarize one or more run directories into summary.md plus SVG frontier plots.

Usage: scripts/report.py results/<run-id> [results/<run-id> ...] [-o OUT_DIR]

Standard library only. Reads builds.jsonl and assignments.jsonl; per-query rows in
queries.csv are already aggregated into builds.jsonl "search" entries by the runner.
A build whose queries did not finish (run killed) has no builds.jsonl row; it is
recovered from built.jsonl plus its finished settings in searches.jsonl and marked partial.
Matched-recall comparisons pick, per build, the lowest-p95 search setting whose recall
reaches the target; a configuration that never reaches it is reported as unattained.
"""

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

TARGETS = (0.90, 0.95)
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2",
           "#be185d", "#4d7c0f", "#7c3aed", "#b45309", "#0f766e", "#475569"]


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # blank, or the last line of a killed run cut mid-write
    return rows


def config_key(b):
    return (b.get("index"), b.get("mode"), b.get("assign"), b.get("num_partitions"))


def fmt(v, digits=2):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def med_range(values, digits=2):
    values = [v for v in values if v is not None]
    if not values:
        return "–"
    m = statistics.median(values)
    if len(values) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} [{min(values):.{digits}f}–{max(values):.{digits}f}] (n={len(values)})"


def best_at_target(search, k, target):
    ok = [s for s in search if s["k"] == k and s["recall"] >= target]
    if not ok:
        return None
    return min(ok, key=lambda s: s["p95_us"])


def svg_frontier(groups, k, title):
    """groups: {label: [(recall, p95_us), ...]} -> SVG text (log-latency axis)."""
    import math
    pts = [p for g in groups.values() for p in g]
    if not pts:
        return None
    w, h, ml, mr, mt, mb = 760, 440, 70, 230, 40, 50
    xs = [p[0] for p in pts]
    ys = [math.log10(max(p[1], 1.0)) for p in pts]
    x0, x1 = min(xs), max(xs)
    x0 = max(0.0, x0 - 0.02)
    x1 = min(1.0, x1 + 0.02) if x1 > x0 else x0 + 0.1
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 0.1:
        y0, y1 = y0 - 0.1, y1 + 0.1
    px = lambda x: ml + (x - x0) / (x1 - x0) * (w - ml - mr)
    py = lambda y: h - mb - (y - y0) / (y1 - y0) * (h - mt - mb)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<text x="{ml}" y="22" font-size="14" font-weight="bold">{html.escape(title)}</text>']
    # axes
    out.append(f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="#333"/>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="#333"/>')
    for i in range(6):
        x = x0 + (x1 - x0) * i / 5
        out.append(f'<line x1="{px(x):.1f}" y1="{h-mb}" x2="{px(x):.1f}" y2="{h-mb+4}" stroke="#333"/>')
        out.append(f'<text x="{px(x):.1f}" y="{h-mb+18}" text-anchor="middle">{x:.2f}</text>')
    e = math.floor(y0)
    while e <= math.ceil(y1):
        for m in (1, 2, 5):
            y = e + math.log10(m)
            if y0 <= y <= y1:
                out.append(f'<line x1="{ml}" y1="{py(y):.1f}" x2="{w-mr}" y2="{py(y):.1f}" stroke="#eee"/>')
                label = 10 ** y / 1000.0
                out.append(f'<text x="{ml-6}" y="{py(y)+4:.1f}" text-anchor="end">{label:g}</text>')
        e += 1
    out.append(f'<text x="{(ml+w-mr)/2}" y="{h-12}" text-anchor="middle">Recall@{k}</text>')
    out.append(f'<text x="16" y="{(mt+h-mb)/2}" text-anchor="middle" '
               f'transform="rotate(-90 16 {(mt+h-mb)/2})">p95 latency (ms, log)</text>')
    for i, (label, g) in enumerate(sorted(groups.items())):
        c = PALETTE[i % len(PALETTE)]
        g = sorted(g)
        path = " ".join(f"{px(r):.1f},{py(math.log10(max(l,1.0))):.1f}" for r, l in g)
        out.append(f'<polyline points="{path}" fill="none" stroke="{c}" stroke-width="1.5" opacity="0.6"/>')
        for r, l in g:
            out.append(f'<circle cx="{px(r):.1f}" cy="{py(math.log10(max(l,1.0))):.1f}" r="3" fill="{c}"/>')
        ly = mt + 16 * i
        out.append(f'<rect x="{w-mr+12}" y="{ly}" width="10" height="10" fill="{c}"/>')
        out.append(f'<text x="{w-mr+28}" y="{ly+9}">{html.escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.runs[0]
    (out / "figures").mkdir(parents=True, exist_ok=True)

    builds, assigns, manifests = [], [], []
    for run in args.runs:
        done = [dict(b, _run=run.name) for b in load_jsonl(run / "builds.jsonl")]
        done_ids = {b.get("build_id") for b in done}
        searches = defaultdict(list)
        for s in load_jsonl(run / "searches.jsonl"):
            searches[s.pop("build_id", None)].append(s)
        partial = [dict(b, _run=run.name, _partial=True, search=searches.get(b["build_id"], []))
                   for b in load_jsonl(run / "built.jsonl") if b.get("build_id") not in done_ids]
        builds += done + partial
        assigns += [dict(a, _run=run.name) for a in load_jsonl(run / "assignments.jsonl")]
        m = run / "manifest.json"
        if m.exists():
            manifests.append((run.name, json.loads(m.read_text())))

    lines = ["# Index build results", ""]
    for name, m in manifests:
        lines += [f"- `{name}`: {m.get('arch')} · {m.get('logical_cpus')} CPUs · "
                  f"dataset {m.get('dataset')} {m.get('base_rows')}×{m.get('dim')} · "
                  f"PiPNN flags `{m.get('pipnn_native_flags')}` · "
                  f"rev `{m.get('env', {}).get('GIT_REV', '?')[:10]}` · "
                  f"patch `{m.get('env', {}).get('PATCH_SHA256', '?')[:12]}`"]
    lines.append("")

    partial = [b for b in builds if b.get("_partial")]
    if partial:
        lines += ["**Partial builds** (queries interrupted; build stats complete, search "
                  "settings only those finished):", ""]
        lines += [f"- `{b['_run']}` `{b['build_id']}`: {len(b['search'])} settings" for b in partial]
        lines.append("")

    failed = [b for b in builds if b.get("status") != "ok"]
    ok = [b for b in builds if b.get("status") == "ok"]
    groups = defaultdict(list)
    for b in ok:
        groups[config_key(b)].append(b)

    lines += ["## Builds", "",
              "Median [min–max] over repeats. `actual K` is the committed partition count.", "",
              "| index | mode | assign | K | actual K | build s | CPU s | assign thread-s | peak RSS GB | RSS Δ GB | index MB | train s (fixed) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        bs = groups[key]
        lines.append("| " + " | ".join([
            str(key[0]), str(key[1]), f"`{key[2]}`", str(key[3]),
            ",".join(sorted({str(b.get('actual_num_partitions')) for b in bs})),
            med_range([b["build_secs"] for b in bs]),
            med_range([b["cpu_secs"] for b in bs], 1),
            med_range([b.get("assign_thread_secs") for b in bs]),
            med_range([(b.get("peak_rss_bytes") or 0) / 1e9 for b in bs]),
            med_range([(b["peak_rss_delta_bytes"] / 1e9) if b.get("peak_rss_delta_bytes") is not None else None for b in bs]),
            med_range([b.get("index_bytes", 0) / 1e6 for b in bs], 1),
            med_range([b.get("fixed_train_secs") for b in bs]),
        ]) + " |")
    lines.append("")

    # Stage spans (median over repeats).
    stage_names = sorted({s for b in ok for s in b.get("stages", {})})
    if stage_names:
        lines += ["### Stage spans (median seconds)", "",
                  "| index | assign | K | " + " | ".join(stage_names) + " |",
                  "|---|---|---|" + "---|" * len(stage_names)]
        for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
            bs = groups[key]
            cells = []
            for s in stage_names:
                vals = [b["stages"][s] for b in bs if s in b.get("stages", {})]
                cells.append(f"{statistics.median(vals):.2f}" if vals else "–")
            lines.append(f"| {key[0]} | `{key[2]}` | {key[3]} | " + " | ".join(cells) + " |")
        lines.append("")

    if assigns:
        lines += ["## Assignment quality (sampled, outside build timing)", "",
                  "| run | assign | K | sample | ID agree | dist agree | rel excess mean / p99 | prepare s | assign s | exact s | centroid graph |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in assigns:
            if a.get("status") != "ok":
                lines.append(f"| {a['_run']} | `{a['assign']}` | | | **error** | | | | | | {html.escape(a.get('error',''))[:120]} |")
                continue
            g = a.get("centroid_graph")
            gtxt = (f"deg {g['mean_degree']:.1f}, reach {g['reachable_fraction']:.3f}" if g else "")
            lines.append("| " + " | ".join([
                a["_run"], f"`{a['assign']}`", str(a.get("num_partitions")), str(a["sample"]),
                fmt(a["id_agreement"], 4), fmt(a["distance_agreement"], 4),
                f"{a['rel_excess_mean']:.2e} / {a['rel_excess_p99']:.2e}",
                fmt(a["prepare_secs"], 3), fmt(a["assign_secs"], 3), fmt(a["exact_assign_secs"], 3), gtxt,
            ]) + " |")
        lines.append("")

    ks = sorted({s["k"] for b in ok for s in b.get("search", [])})
    for k in ks:
        lines += [f"## Matched recall, k={k}", "",
                  "Lowest-p95 setting reaching each target, per build; median over repeats. "
                  "Tuning-split queries unless noted.", "",
                  "| index | assign | K | " + " | ".join(
                      f"R≥{t:.2f}: p95 ms (setting)" for t in TARGETS) + " | build s |",
                  "|---|---|---|" + "---|" * len(TARGETS) + "---|"]
        for key in sorted(groups, key=lambda kk: tuple(str(x) for x in kk)):
            bs = groups[key]
            cells = []
            for t in TARGETS:
                hits = [best_at_target(b.get("search", []), k, t) for b in bs]
                if not any(hits) or None in hits:
                    reached = sum(h is not None for h in hits)
                    cells.append("unattained" if reached == 0 else f"{reached}/{len(hits)} reached")
                    continue
                p95 = statistics.median(h["p95_us"] / 1000 for h in hits)
                settings = sorted({f"np={h['nprobes']}" + (f",ef={h['ef']}" if h.get("ef") else "") for h in hits})
                cells.append(f"{p95:.2f} ({'; '.join(settings)})")
            lines.append(f"| {key[0]} | `{key[2]}` | {key[3]} | " + " | ".join(cells)
                         + f" | {statistics.median(b['build_secs'] for b in bs):.1f} |")
        lines.append("")
        frontier = defaultdict(list)
        for key, bs in groups.items():
            b = bs[0]  # first repeat; per-repeat curves are in builds.jsonl
            label = f"{key[0]} {key[2]} K={key[3]}"
            frontier[label] = [(s["recall"], s["p95_us"]) for s in b.get("search", []) if s["k"] == k]
        svg = svg_frontier(frontier, k, f"Recall@{k} vs p95 latency (first repeat)")
        if svg:
            name = f"frontier_k{k}.svg"
            (out / "figures" / name).write_text(svg)
            lines += [f"![frontier k={k}](figures/{name})", ""]

    if failed:
        lines += ["## Failed builds", "", "| build | error |", "|---|---|"]
        for b in failed:
            lines.append(f"| {b['build_id']} | {html.escape(str(b.get('error')))[:200]} |")
        lines.append("")

    (out / "summary.md").write_text("\n".join(lines))
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
