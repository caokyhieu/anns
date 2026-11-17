#!/usr/bin/env python3
"""Summarize a kunpeng_ws256.sh session directory as Markdown (stdout and summary.md).

Speedups are against the strongest existing ARM baseline of each operation:
NEON for the scans and ex_dot, the NEON 4x4 tile for assignment (tile-neon).
Arms that failed their correctness check are listed as MISMATCH, never timed.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def micro(out, lines):
    data = rows(out / "micro.jsonl") + rows(out / "micro-assign.jsonl") + rows(out / "micro-assign-svetile.jsonl")
    one = rows(out / "micro-assign-1t.jsonl")
    host = next((r for r in data if r["section"] == "host"), None)
    if host:
        lines += ["## Host", "", f"`{json.dumps({k: host[k] for k in host if k != 'section'})}`", ""]
    for section, keys, base_pred, title in [
        ("hacc", ("dim", "rows"), lambda a: a.endswith("neon"), "WS5 SUM_HACC (accurate scan), ns/row"),
        ("scan", ("dim", "rows"), lambda a: a.endswith("neon"), "WS6A SUM_U16 scan, ns/row"),
        ("ex_dot", ("ex_bits", "dim", "rows"), lambda a: a == "neon", "WS6B ex_dot, ns/row"),
    ]:
        groups = defaultdict(dict)
        for r in data:
            if r["section"] == section:
                groups[tuple(r[k] for k in keys)][r["arm"]] = r
        if not groups:
            continue
        lines += [f"## {title}", "", "| " + " | ".join(keys) + " | arm | ns/row | vs NEON |",
                  "|" + "---|" * (len(keys) + 3)]
        for key, arms in sorted(groups.items()):
            base = next((r["secs"] for a, r in arms.items() if base_pred(a) and r["secs"]), None)
            for arm, r in arms.items():
                n = r["rows"]
                if not r["ok"]:
                    lines.append("| " + " | ".join(map(str, key)) + f" | {arm} | MISMATCH | |")
                    continue
                speed = f"{base / r['secs']:.2f}x" if base else "–"
                lines.append("| " + " | ".join(map(str, key)) + f" | {arm} | {r['secs'] * 1e9 / n:.1f} | {speed} |")
        lines.append("")
    for label, src in [("all threads", data), ("1 thread", one)]:
        groups = defaultdict(dict)
        for r in src:
            if r["section"] == "assign":
                groups[(r["dim"], r["k"], r["rows"])][r["arm"]] = r
        if not groups:
            continue
        lines += [f"## WS2 assignment, complete operation ({label})", "",
                  "| D | K | rows | arm | ms | GFLOP/s | vs tile-neon | agree |", "|---|---|---|---|---|---|---|---|"]
        for key, arms in sorted(groups.items()):
            base = next((r["secs"] for a, r in arms.items() if a == "tile:dot_tile_neon" and r["secs"]), None)
            for arm, r in arms.items():
                if not r["ok"]:
                    lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {arm} | MISMATCH | | | {r['agree_with_tile']} |")
                    continue
                speed = f"{base / r['secs']:.2f}x" if base else "–"
                lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {arm} | {r['secs'] * 1e3:.1f} | "
                             f"{fmt(r['gflops'], 1)} | {speed} | {r['agree_with_tile']}/{key[2]} |")
        lines.append("")


def stages(out, lines):
    base = out / "stages"
    if not base.is_dir():
        return
    table = defaultdict(lambda: defaultdict(list))
    extra = {}
    for arm_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for r in rows(arm_dir / "stages.jsonl"):
            if "secs" not in r or r.get("status") == "error":
                continue
            key = (r["stage"], r.get("variant", ""))
            table[key][arm_dir.name].append(r["secs"])
            agree = {k: v for k, v in r.items() if "agree" in k}
            if agree:
                extra[(key, arm_dir.name)] = agree
    if not table:
        return
    arms = sorted({a for v in table.values() for a in v})
    lines += ["## WS2 build stages (median seconds; GIST1M)", "",
              "| stage | variant | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for key in sorted(table):
        ref = statistics.median(table[key]["tile-neon"]) if table[key].get("tile-neon") else None
        cells = []
        for a in arms:
            if not table[key].get(a):
                cells.append("–")
                continue
            m = statistics.median(table[key][a])
            cell = f"{m:.2f}" + (f" ({ref / m:.2f}x)" if ref and a != "tile-neon" else "")
            if (key, a) in extra:
                cell += " " + ",".join(f"{k}={v}" for k, v in extra[(key, a)].items())
            cells.append(cell)
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    lines.append("")


def search(out, lines):
    base = out / "search"
    if not base.is_dir():
        return
    res = defaultdict(dict)
    for run in sorted(base.glob("ws256-*")):
        # ws256-<case>-<arm>-<mode>-<tag>
        parts = run.name.split("-")
        arm, mode = parts[2], parts[3]
        for r in rows(run / "searches.jsonl"):
            if r.get("split", "test") == "tune":
                continue
            res[(mode, r["k"], r["nprobes"])][arm] = r
    if not res:
        return
    arms = sorted({a for v in res.values() for a in v})
    lines += ["## WS5/6 search (p50 us, recall; same index settings per arm)", "",
              "| mode | k | nprobes | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms)]
    for key in sorted(res):
        ref = res[key].get("neon")
        cells = []
        for a in arms:
            r = res[key].get(a)
            if not r:
                cells.append("–")
                continue
            speed = f" {ref['p50_us'] / r['p50_us']:.2f}x" if ref and a != "neon" else ""
            cells.append(f"{r['p50_us']:.0f}{speed} / {r['recall']:.4f}")
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) + " |")
    lines.append("")


def main():
    out = Path(sys.argv[1])
    lines = [f"# Kunpeng workstreams 2/5/6 — {out.name}", ""]
    micro(out, lines)
    stages(out, lines)
    search(out, lines)
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
