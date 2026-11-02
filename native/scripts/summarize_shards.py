"""Summarize shard_bench / lance_shard_check results at matched recall.

For every configuration (design, index or partition, router, lists, shards) it picks the
cheapest query setting (lowest mean latency) whose mean recall@k reaches each target, and
prints build and query costs side by side as Markdown tables.

Usage: python3 native/scripts/summarize_shards.py native/results/shards-synthetic1m/results.jsonl [lance.jsonl]
"""
import json
import sys
from collections import defaultdict

TARGETS = (0.90, 0.95, 0.99)


def key(r):
    if r["design"] == "D1":
        return ("D1", r["index"], "-", "-", r["shards"], r.get("refine", "-"))
    if r["design"] == "lance":
        return ("lance", "IVF_RQ", "lance-kmeans", r["lists"], r["shards"], r.get("refine", "-"))
    return (r["design"], r["partition"], r["router"], r["lists"], r["shards"], r.get("refine", "-"))


def main(paths):
    builds, queries, routers = {}, defaultdict(list), []
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r["kind"] == "router":
                routers.append(r)
                continue
            if r["kind"] == "build":
                k = key(r)
                for refine in ("-", 10, 40):
                    builds[k[:5] + (refine,)] = r
            else:
                queries[key(r)].append(r)

    print("| Design | Index / partition | Router | Lists | S | Refine | Fixed s | Slowest shard s | Wall s (S workers) | Total CPU s |"
          " Target | Setting | Recall | Latency ms | CPU ms/query | Work/query |")
    print("|" + "---|" * 16)
    for k in sorted(queries, key=lambda k: (k[0], k[1], k[2], str(k[3]), k[4], str(k[5]))):
        b = builds.get(k, {})
        rows = sorted(queries[k], key=lambda r: r["latency_ms_mean"])
        for t in TARGETS:
            ok = [r for r in rows if r["recall"] >= t]
            if not ok:
                continue
            r = ok[0]
            setting = f"beam {r['beam']}" if "beam" in r else f"nprobe {r['nprobe']}"
            if r["design"] == "D1":
                work = f"{r['distance_evals']:.0f} evals"
            elif r["design"] == "lance":
                work = "-"
            else:
                work = f"{r['codes_scanned']:.0f} codes + {r['reranked']:.0f} rerank + {r['router_evals']:.0f} route"
            fixed = b.get("fixed_seconds", 0) + b.get("merge_seconds", 0)
            print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {k[5]} | {fixed:.2f} | {b.get('shard_seconds_max', float('nan')):.2f} |"
                  f" {b.get('wall_seconds', float('nan')):.2f} | {b.get('shard_seconds_sum', float('nan')) + fixed:.2f} |"
                  f" {t:.2f} | {setting} | {r['recall']:.4f} | {r['latency_ms_mean']:.3f} | {r.get('cpu_ms_per_query', float('nan')):.3f} | {work} |")
    if routers:
        print("\n| Centroids from | C | Router | Build s | nprobe | Router beam | Router recall | Evals | ms |")
        print("|" + "---|" * 9)
        for r in routers:
            print(f"| {r['partition']} | {r['lists']} | {r['router']} | {r['router_build_seconds']:.3f} | {r['nprobe']} |"
                  f" {r['router_beam']} | {r['router_recall']:.4f} | {r['router_evals']:.0f} | {r['router_ms']:.4f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
