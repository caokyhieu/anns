"""Distributed PiPNN experiment: one global graph vs partition-and-route vs SPANN postings.

All shards are simulated in one process; costs are counted, not timed over a network.
"""
import argparse
import json
import platform
from time import perf_counter
import numpy as np
import dpipnn as D


def data(seed, n, nq, dim, clusters):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))
    scale = rng.uniform(0.5, 1.0, clusters)
    def sample(count):
        c = rng.integers(0, clusters, count)
        return centers[c] + rng.normal(size=(count, dim)) * scale[c, None]
    return sample(n), sample(nq)


def measure(name, queries, truth, run, **settings):
    recall, costs = [], []
    for q, t in zip(queries, truth):
        found, cost = run(q)
        recall.append(len(set(found) & set(t.tolist())) / len(t))
        costs.append(cost)
    row = dict(index=name, **settings, recall_at_10=float(np.mean(recall)))
    row.update({k: float(np.mean([c[k] for c in costs])) for k in costs[0]})
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--clusters", type=int, default=512)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--beams", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="distributed-results.json")
    a = ap.parse_args()

    x, queries = data(a.seed, a.n, a.queries, a.dim, a.clusters)
    truth = D.nearest(queries, x, 10)[0]
    head = D.Head(x, seed=a.seed)
    p = D.Params(seed=a.seed)
    builds, rows = [], []

    def build(name, shards, **kw):
        start = perf_counter()
        c = D.build(x, shards, p, **kw)
        c.stats.update(index=name, build_seconds=perf_counter() - start)
        builds.append(c.stats)
        print(json.dumps(c.stats), flush=True)
        return c

    graphs = [("single machine", build("single machine", 1))]
    routed = build("partitioned", a.shards)
    for eps in (0.05, 0.1, 0.2):
        graphs.append((f"global eps={eps}", build(f"global eps={eps}", a.shards, eps=eps, max_replicas=4)))
    graphs.append(("global top-2", build("global top-2", a.shards, fixed=2)))
    graphs.append(("global no replicas", routed))
    # Same build without the HashPrune combiner: identical graph, more shuffle.
    raw = build("global eps=0.1 no combiner", a.shards, eps=0.1, max_replicas=4, combine=False)
    same = np.array_equal(raw.indices, graphs[2][1].indices) and np.array_equal(raw.indptr, graphs[2][1].indptr)
    builds[-1]["identical_to_combined"] = bool(same)

    for name, c in graphs:
        for beam in a.beams:
            rows.append(measure(name, queries, truth, lambda q: D.search_global(x, c, head, q, beam), beam=beam))
    for probes in (1, 2, 4, a.shards):
        for beam in a.beams[:3]:
            rows.append(measure("partition-and-route", queries, truth,
                                lambda q: D.search_routed(x, routed, head, q, beam, probes),
                                beam=beam, probes=probes))
    start = perf_counter()
    spann = D.Spann(x, posting=64, eps1=0.15, seed=a.seed)
    builds.append(dict(index="spann", build_seconds=perf_counter() - start,
                       replication=spann.replication, lists=len(spann.centers)))
    print(json.dumps(builds[-1]), flush=True)
    for probes in (8, 16, 32, 64, 128):
        rows.append(measure("spann", queries, truth, lambda q: spann.search(q, probes, eps2=0.6), probes=probes))

    with open(a.output, "w") as f:
        json.dump(dict(config=vars(a), params=vars(p), python=platform.python_version(),
                       numpy=np.__version__, builds=builds, searches=rows), f, indent=1, default=str)


if __name__ == "__main__":
    main()
