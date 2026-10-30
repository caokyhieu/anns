"""Seeded build/add/delete/update/churn experiment against an exact oracle."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from pipnn import PiPNN


def run(seed, n, dim, nq, batch, beam):
    rng = np.random.default_rng(seed)
    # Correlated Gaussian vectors, with held-out queries and deliberate update drift.
    transform = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    def sample(count, shift=0):
        return rng.normal(size=(count, dim)) @ transform + shift
    x, queries = sample(n), sample(nq)
    index = PiPNN(seed=seed)
    start = perf_counter()
    index.build(x)
    initial_seconds = perf_counter()-start
    rows = []
    def measure(stage, seconds, operations=0):
        index.validate()
        ids = sorted(index.vectors)
        live = np.array([index.vectors[i] for i in ids])
        # Independent exact ground truth recomputed against current vectors.
        truth = []
        for q in queries:
            ds = np.sum((live-q)**2, axis=1)
            truth.append({ids[j] for j in np.argsort(ds, kind="stable")[:10]})
        scores, evaluations = [], []
        start = perf_counter()
        for q, expected in zip(queries, truth):
            result = index.search(q, k=10, beam=beam)
            scores.append(len(set(result) & expected)/len(expected))
            evaluations.append(index.last_search_evaluations)
        query_seconds = perf_counter()-start
        # A rebuilt index on exactly the same live IDs/vectors is a quality baseline.
        fresh = PiPNN(**index.config)
        start = perf_counter()
        fresh.build(live, ids)
        rebuild_seconds = perf_counter()-start
        fresh_scores = [len(set(fresh.search(q, 10, beam)) & expected)/len(expected)
                        for q, expected in zip(queries, truth)]
        row = dict(stage=stage, n=len(ids), operations=operations,
                   operation_seconds=seconds, mean_mutation_ms=1000*seconds/operations if operations else None,
                   recall_at_10=float(np.mean(scores)), rebuilt_recall_at_10=float(np.mean(fresh_scores)),
                   rebuild_seconds=rebuild_seconds, query_ms=1000*query_seconds/len(queries),
                   mean_graph_distance_evaluations=float(np.mean(evaluations)),
                   max_degree=max(map(len, index.graph.values()), default=0),
                   max_leaf_size=max(map(lambda l: len(l.members), index.leaves)),
                   invariant_checks="passed")
        rows.append(row)
        recall, fresh_recall = row["recall_at_10"], row["rebuilt_recall_at_10"]
        print(f"seed={seed} {stage:8} n={len(ids):4} recall={recall:.3f} "
              f"fresh={fresh_recall:.3f} time={seconds:.3f}s", flush=True)
    measure("build", initial_seconds)
    start = perf_counter()
    for v in sample(batch):
        index.add(v)
    measure("add", perf_counter()-start, batch)
    deleted = rng.choice(sorted(index.vectors), batch, replace=False)
    start = perf_counter()
    for p in deleted:
        index.delete(int(p))
    measure("delete", perf_counter()-start, batch)
    updated = rng.choice(sorted(index.vectors), batch, replace=False)
    replacements = sample(batch, shift=1.5)
    start = perf_counter()
    for p, v in zip(updated, replacements):
        index.update(int(p), v)
    measure("update", perf_counter()-start, batch)
    start = perf_counter()
    for step in range(batch):
        p = int(rng.choice(sorted(index.vectors)))
        if step % 3 == 0:
            index.add(sample(1, shift=2)[0])
        elif step % 3 == 1:
            index.delete(p)
        else:
            index.update(p, sample(1, shift=2)[0])
    measure("churn", perf_counter()-start, batch)
    monitoring = index.monitor(queries, beam=beam, min_recall=.9, rebuild=True)
    index.validate()
    return dict(seed=seed, rows=rows, monitoring=monitoring)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--beam", type=int, default=48)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    if args.n < 10 or not 1 <= args.batch < args.n or args.dim < 1 or args.queries < 1 or args.beam < 10:
        parser.error("require n >= 10, 1 <= batch < n, dim/queries > 0, beam >= 10")
    result = dict(parameters=vars(args), python=platform.python_version(), numpy=np.__version__,
                  runs=[run(s, args.n, args.dim, args.queries, args.batch, args.beam) for s in args.seeds])
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
