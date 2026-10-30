"""Training-data retrieval experiment: build time, large-k recall, and batch amortization.

Compares a PiPNN graph with SPANN-style posting lists (brute-force or PiPNN-graph
router over centroids) when each query asks for many results, and measures how
much of the data a batch of queries must read when postings are scanned once per batch.
"""
import argparse
import json
import platform
from time import perf_counter
import numpy as np
import dpipnn as D
from distributed_experiment import data


def timed(f):
    start = perf_counter()
    out = f()
    return out, perf_counter() - start


def recall(found, truth):
    return len(set(found) & set(truth.tolist())) / len(truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--clusters", type=int, default=512)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--graph-queries", type=int, default=100)
    ap.add_argument("--posting", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="retrieval-results.json")
    a = ap.parse_args()

    x, queries = data(a.seed, a.n, a.queries, a.dim, a.clusters)
    kmax = 1000
    truth = np.concatenate([D.nearest(queries[s:s+64], x, kmax, chunk=64)[0]
                            for s in range(0, len(queries), 64)])
    gq = slice(0, a.graph_queries)
    head = D.Head(x, seed=a.seed)
    rows, builds = [], []

    def log(row, store):
        store.append(row)
        print(json.dumps(row), flush=True)

    # --- Build time ----------------------------------------------------------------
    p = D.Params(seed=a.seed)
    fast = D.Params(seed=a.seed, fanout=(2, 1), reservoir=32)
    graph, t = timed(lambda: D.build(x, 1, p))
    log(dict(index="pipnn graph", fanout=p.fanout, reservoir=p.reservoir, build_seconds=t,
             mean_degree=graph.stats["mean_degree"]), builds)
    fgraph, t = timed(lambda: D.build(x, 1, fast))
    log(dict(index="pipnn graph (fast build)", fanout=fast.fanout, reservoir=fast.reservoir,
             build_seconds=t, mean_degree=fgraph.stats["mean_degree"]), builds)
    ivf, t = timed(lambda: D.Spann(x, posting=a.posting, eps1=0.15, seed=a.seed))
    log(dict(index="postings (brute router)", build_seconds=t, lists=len(ivf.centers),
             replication=ivf.replication), builds)
    givf, t = timed(lambda: D.Spann(x, posting=a.posting, eps1=0.15, seed=a.seed, router="graph"))
    log(dict(index="postings (PiPNN router)", build_seconds=t, lists=len(givf.centers),
             replication=givf.replication, router_build_seconds=givf.router_seconds), builds)

    # --- Recall at growing k --------------------------------------------------------
    for k in (10, 100, 1000):
        t = truth[:, :k]
        for name, g in (("pipnn graph", graph), ("pipnn graph (fast build)", fgraph)):
            for mult in (1, 2, 4):
                beam = max(16, k) * mult
                res = [D.search_global(x, g, head, q, beam, k) for q in queries[gq]]
                log(dict(index=name, k=k, beam=beam,
                         recall=float(np.mean([recall(f, tt) for (f, _), tt in zip(res, t[gq])])),
                         random_reads=float(np.mean([c["reads"] for _, c in res])),
                         vectors_touched=float(np.mean([c["distance_evals"] for _, c in res]))), rows)
        for name, index in (("postings (brute router)", ivf), ("postings (PiPNN router)", givf)):
            for probes in (8, 16, 32, 64, 128, 256):
                res = [index.search(q, probes, eps2=1e9, k=k) for q in queries[gq]]
                log(dict(index=name, k=k, probes=probes,
                         recall=float(np.mean([recall(f, tt) for (f, _), tt in zip(res, t[gq])])),
                         random_reads=float(np.mean([c["lists"] for _, c in res])),
                         vectors_touched=float(np.mean([c["vectors_scanned"] for _, c in res]))), rows)

    # --- Batch amortization (k = 1000) ----------------------------------------------
    batch = []
    for probes in (32, 64, 128):
        lists = [ivf.probe(q, probes, eps2=1e9) for q in queries]
        rec = float(np.mean([recall(ivf.search(q, probes, eps2=1e9, k=kmax)[0], t)
                             for q, t in zip(queries[gq], truth[gq])]))
        for size in (1, 10, 100, 1000):
            per = []
            for s in range(0, len(queries), size):
                vectors, _ = ivf.batch_io(lists[s:s+size])
                per.append(vectors / len(lists[s:s+size]))
            log(dict(probes=probes, recall_at_1000=rec, batch=size,
                     vectors_read_per_query=float(np.mean(per)),
                     fraction_of_data_per_query=float(np.mean(per)) / a.n,
                     full_scan_vectors_per_query=a.n / size), batch)

    with open(a.output, "w") as f:
        json.dump(dict(config=vars(a), params=vars(p), fast_params=vars(fast),
                       python=platform.python_version(), numpy=np.__version__,
                       builds=builds, searches=rows, batching=batch), f, indent=1, default=str)


if __name__ == "__main__":
    main()
