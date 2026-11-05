"""Aggregate results.jsonl into a markdown table (mean over reps, min/max)."""
import json
import sys
from collections import defaultdict
from statistics import mean

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
groups = defaultdict(list)
for r in rows:
    groups[(r["num_bits"], r["label"])].append(r)

order = ["baseline", "blocked", "blocked+const"]
print("| RQ bits | Config | Reps | Build (s) | train_ivf (s) | shuffle (s) | Speedup | R@10 np=20 | R@10 np=50 | R@10 np=100 | R@100 np=100 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for bits in sorted({k[0] for k in groups}):
    base = mean(r["build_secs"] for r in groups[(bits, "baseline")])
    for label in order:
        g = groups.get((bits, label))
        if not g:
            continue
        b = [r["build_secs"] for r in g]
        tr = mean(r["stages"].get("train_ivf", 0) for r in g)
        sh = mean(r["stages"].get("shuffle", 0) for r in g)

        def rec(np, key):
            v = [s[key] for r in g for s in r["search"] if s["nprobes"] == np]
            return f"{mean(v):.3f} ({min(v):.3f}–{max(v):.3f})" if len(v) > 1 else f"{v[0]:.3f}"

        print(
            f"| {bits} | {label} | {len(g)} | {mean(b):.1f} ({min(b):.1f}–{max(b):.1f}) | {tr:.1f} | {sh:.1f} | "
            f"{base / mean(b):.2f}× | {rec(20, 'recall10')} | {rec(50, 'recall10')} | {rec(100, 'recall10')} | {rec(100, 'recall100')} |"
        )
