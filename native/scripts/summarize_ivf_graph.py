"""Tables for ivf_graph_bench results (see IVF_GRAPH.md).

usage: python3 native/scripts/summarize_ivf_graph.py results.jsonl [--targets 0.9,0.95,0.99]

Prints, per partition count: assignment quality, build times, and for every
(assignment, sub-index) pair the cheapest (nprobe, ef) setting that reaches each recall
target, by mean single-thread latency.
"""
import argparse
import json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--targets", default="0.9,0.95,0.99")
a = ap.parse_args()
targets = [float(t) for t in a.targets.split(",")]
rows = [json.loads(line) for line in open(a.results) if line.strip()]


def name(r):
    return r["assign"] + (f":{r['assign_beam']}" if r["assign"] != "flat" else "")


for C in sorted({r["lists"] for r in rows}):
    rs = [r for r in rows if r["lists"] == C]
    print(f"\n## {C} partitions (train {rs[0]['train_seconds']:.1f} s)\n")
    print("| Assignment | Graph build | Assign | Distances / row | Exact nearest | Empty lists | List size p50 / max |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in [r for r in rs if r["kind"] == "assign"]:
        print(f"| {name(r)} | {r['graph_seconds']:.2f} s | {r['assign_seconds']:.2f} s | {r['evals_per_row']:.0f} | "
              f"{r['exact_fraction']:.4f} | {r['empty_lists']} | {r['list_size_p50']:.0f} / {r['list_size_max']:.0f} |")
    print("\n| Assignment | Sub-index | Sub-index build (wall / CPU) | Mean degree | Total build |"
          + "".join(f" R@10 ≥ {t} |" for t in targets))
    print("|---|---|---:|---:|---:|" + "---|" * len(targets))
    queries = defaultdict(list)
    for r in rs:
        if r["kind"] == "query":
            queries[(name(r), r["sub"])].append(r)
    for b in [r for r in rs if r["kind"] == "build"]:
        key = (name(b), b["sub"])
        cells = []
        for t in targets:
            ok = [q for q in queries[key] if q["recall"] >= t]
            if not ok:
                best = max(queries[key], key=lambda q: q["recall"])
                cells.append(f"max {best['recall']:.3f}")
                continue
            q = min(ok, key=lambda q: q["latency_ms_mean"])
            ef = f", ef {q['ef']}" if q["ef"] else ""
            cells.append(f"{q['latency_ms_mean']:.3f} ms (np {q['nprobe']}{ef}; {q['distance_evals']:.0f} dist)")
        print(f"| {key[0]} | {key[1]} | {b['sub_seconds']:.1f} / {b['sub_cpu_seconds']:.1f} s | {b['mean_degree']:.1f} | "
              f"{b['total_seconds']:.1f} s | " + " | ".join(cells) + " |")
