"""Side-by-side PiPNN vs HNSW (hnswlib) on identical data, mutations, and ground truth."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
import hnswlib
import numpy as np
from pipnn import PiPNN


class HNSW:
    """Thin hnswlib wrapper exposing the same add/delete/update/search calls as PiPNN.

    Deletion uses hnswlib tombstones (mark_deleted): the node stays in the graph for
    routing but is filtered from results. Update re-adds an existing label, which
    hnswlib handles by replacing the vector and repairing its neighborhood.
    """

    def __init__(self, dim, capacity, m, ef_construction, seed):
        self.index = hnswlib.Index(space="l2", dim=dim)
        self.index.init_index(max_elements=capacity, M=m, ef_construction=ef_construction,
                              random_seed=seed)
        self.index.set_num_threads(1)  # match the single-threaded Python PiPNN

    def build(self, x, ids):
        self.index.add_items(np.asarray(x, dtype=np.float32), np.asarray(ids))
        return self

    def add(self, x, id):
        self.index.add_items(np.asarray(x, dtype=np.float32)[None], [id])

    def delete(self, p):
        self.index.mark_deleted(p)

    def update(self, p, x):
        self.add(x, p)

    def search(self, q, k, beam):
        self.index.set_ef(beam)
        labels, _ = self.index.knn_query(np.asarray(q, dtype=np.float32)[None], k=k)
        return labels[0].tolist()


def recall_sweep(index, queries, truth, beams, counts_evaluations):
    rows = []
    for beam in beams:
        scores, evaluations = [], []
        start = perf_counter()
        for q, expected in zip(queries, truth):
            scores.append(len(set(index.search(q, 10, beam)) & expected)/len(expected))
            if counts_evaluations:
                evaluations.append(index.last_search_evaluations)
        seconds = perf_counter()-start
        rows.append(dict(beam=beam, recall_at_10=float(np.mean(scores)),
                         query_ms=1000*seconds/len(queries),
                         mean_distance_evaluations=float(np.mean(evaluations)) if evaluations else None))
    return rows


def run(seed, args):
    rng = np.random.default_rng(seed)
    dim, batch = args.dim, args.batch
    # Same correlated-Gaussian generator and update drift as experiment.py.
    transform = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    def sample(count, shift=0):
        return rng.normal(size=(count, dim)) @ transform + shift
    x, queries = sample(args.n), sample(args.queries)
    live = {i: v for i, v in enumerate(x)}
    next_id = args.n
    # Pre-generate every operation so both indexes see exactly the same workload.
    plan = []
    for v in sample(batch):
        plan.append(("add", next_id, v)); live[next_id] = v; next_id += 1
    stages = [("add", len(plan))]
    for p in rng.choice(sorted(live), batch, replace=False):
        plan.append(("delete", int(p), None)); del live[int(p)]
    stages.append(("delete", len(plan)))
    for p, v in zip(rng.choice(sorted(live), batch, replace=False), sample(batch, shift=1.5)):
        plan.append(("update", int(p), v)); live[int(p)] = v
    stages.append(("update", len(plan)))
    for step in range(batch):
        p = int(rng.choice(sorted(live)))
        if step % 3 == 0:
            v = sample(1, shift=2)[0]
            plan.append(("add", next_id, v)); live[next_id] = v; next_id += 1
        elif step % 3 == 1:
            plan.append(("delete", p, None)); del live[p]
        else:
            v = sample(1, shift=2)[0]
            plan.append(("update", p, v)); live[p] = v
    stages.append(("churn", len(plan)))

    pipnn_params = dict(leaf_size=args.leaf_size, local_k=args.local_k, degree=args.degree,
                        reservoir_size=args.reservoir, seed=seed)
    indexes = {"pipnn": PiPNN(**pipnn_params),
               "hnsw": HNSW(dim, next_id, args.m, args.ef_construction, seed)}
    build_seconds = {}
    for name, index in indexes.items():
        start = perf_counter()
        index.build(x, list(range(args.n)))
        build_seconds[name] = perf_counter()-start

    vectors = {i: v for i, v in enumerate(x)}
    rows = []
    def measure(stage, seconds, operations):
        ids = sorted(vectors)
        data = np.array([vectors[i] for i in ids])
        # Exact ground truth on the current live set, shared by both indexes.
        truth = [{ids[j] for j in np.argsort(np.sum((data-q)**2, axis=1), kind="stable")[:10]}
                 for q in queries]
        for name, index in indexes.items():
            sweep = recall_sweep(index, queries, truth, args.beams, name == "pipnn")
            rows.append(dict(stage=stage, algorithm=name, n=len(ids), operations=operations,
                             operation_seconds=seconds[name],
                             mean_mutation_ms=1000*seconds[name]/operations if operations else None,
                             queries=sweep))
            best = ", ".join(f"b{r['beam']}={r['recall_at_10']:.3f}" for r in sweep)
            print(f"seed={seed} {stage:7} {name:5} n={len(ids):5} t={seconds[name]:7.3f}s  {best}",
                  flush=True)

    measure("build", build_seconds, 0)
    done = 0
    for stage, end in stages:
        seconds = {}
        for name, index in indexes.items():
            start = perf_counter()
            for op, p, v in plan[done:end]:
                if op == "add":
                    index.add(v, id=p)
                elif op == "delete":
                    index.delete(p)
                else:
                    index.update(p, v)
            seconds[name] = perf_counter()-start
        for op, p, v in plan[done:end]:
            if op == "delete":
                del vectors[p]
            else:
                vectors[p] = v
        measure(stage, seconds, end-done)
        done = end
    return dict(seed=seed, rows=rows)


def summarize(runs, targets):
    """Smallest beam/ef reaching each recall target, per stage, averaged over seeds."""
    lines = ["| Stage | Algorithm | mutation ms/op | " +
             " | ".join(f"beam@{t:.0%}" for t in targets) + " | recall@max beam |",
             "| --- | --- | ---: | " + " | ".join("---:" for _ in targets) + " | ---: |"]
    keys = sorted({(r["stage"], r["algorithm"]) for run in runs for r in run["rows"]},
                  key=lambda k: ([r["stage"] for r in runs[0]["rows"]].index(k[0]), k[1] != "pipnn"))
    for stage, algorithm in keys:
        matching = [r for run in runs for r in run["rows"] if (r["stage"], r["algorithm"]) == (stage, algorithm)]
        beams = [q["beam"] for q in matching[0]["queries"]]
        mean_recall = [np.mean([r["queries"][i]["recall_at_10"] for r in matching]) for i in range(len(beams))]
        cells = []
        for t in targets:
            hit = next((b for b, rec in zip(beams, mean_recall) if rec >= t), None)
            cells.append(str(hit) if hit else "—")
        ms = [r["mean_mutation_ms"] for r in matching if r["mean_mutation_ms"] is not None]
        lines.append(f"| {stage} | {algorithm} | {np.mean(ms):.2f} | " if ms else f"| {stage} | {algorithm} | — | ")
        lines[-1] += " | ".join(cells) + f" | {mean_recall[-1]:.3f} |"
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--beams", type=int, nargs="+", default=[10, 16, 32, 64, 128])
    parser.add_argument("--leaf-size", type=int, default=64)
    parser.add_argument("--local-k", type=int, default=12)
    parser.add_argument("--degree", type=int, default=24)
    parser.add_argument("--reservoir", type=int, default=48)
    # hnswlib allows 2*M neighbors on layer 0, so M=12 matches PiPNN degree 24.
    parser.add_argument("--m", type=int, default=12)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--output", default="compare-results.json")
    args = parser.parse_args()
    if args.n < 10 or not 1 <= args.batch < args.n // 3 or args.dim < 1 or args.queries < 1:
        parser.error("require n >= 10, 1 <= batch < n/3, dim/queries > 0")
    if min(args.beams) < 10:
        parser.error("beams must be >= 10 (k)")
    args.beams = sorted(set(args.beams))
    runs = [run(s, args) for s in args.seeds]
    table = summarize(runs, (.9, .95, .99))
    print("\n" + table)
    result = dict(parameters=vars(args), python=platform.python_version(), numpy=np.__version__,
                  hnswlib=getattr(hnswlib, "__version__", "unknown"), runs=runs, summary_markdown=table)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
