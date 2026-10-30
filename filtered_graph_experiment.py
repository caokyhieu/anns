"""Filtered search inside a graph index (HNSW / DiskANN style) on the same data and filters
as filtered_search_experiment.py.

hnswlib builds the graph. Four in-graph filter policies are compared, all scored as recall
against exact filtered search and as work = distance computations per query:
  hnswlib     the library's own filter callback (C++ search; reports recall, filter-callback
              calls and ms, not distance computations)
  traverse    best-first search on HNSW's base layer that walks through failing nodes but
              only keeps passing ones as results, and stops only once it holds `beam`
              passing nodes (hnswlib's and FAISS IDSelector's rule), re-implemented so the
              work is counted in distance computations
  strict      walks only through passing nodes (a graph built without the filter in mind,
              searched on the predicate subgraph)
  acorn       strict, plus 2-hop expansion: neighbors of failing neighbors are also tried
              (ACORN-1-style compensation for a disconnected predicate subgraph)
All graph searches start from the best of `--entries` sampled nodes (passing nodes for
strict and acorn, like Filtered-DiskANN's per-label entry points); those distances count.
Reference columns give the work of exact search over the passing rows (flat) and of IVF
with nprobes scaled by 1 / selectivity (the planner in FILTERED_SEARCH.md).
"""
import argparse
import heapq
import json
from time import perf_counter
import numpy as np
import hnswlib
from filtered_search_experiment import TAGS, make_table, make_queries


def base_layer(index, n):
    """Neighbor lists of HNSW layer 0, read from hnswlib's serialized state."""
    s = index.__getstate__()[0]
    size, m0 = s["size_data_per_element"], s["max_M0"]
    raw = np.frombuffer(s["data_level0"], np.uint8)[: n * size].reshape(n, size)
    count = raw[:, :4].copy().view(np.uint32)[:, 0] & 0xFFFF
    links = raw[:, 4 : 4 + 4 * m0].copy().view(np.uint32)
    label = raw[:, s["label_offset"] : s["label_offset"] + 8].copy().view(np.uint64)[:, 0].astype(np.int64)
    graph = [None] * n
    for i in range(n):
        graph[label[i]] = label[links[i, : count[i]]]
    return graph


def search(graph, emb, q, mask, k, beam, policy, entries):
    """Best-first search; returns (ids, distance computations)."""
    dist = lambda ids: ((emb[ids] - q) ** 2).sum(1)
    work = len(entries)
    d0 = dist(entries)
    visited = set(entries.tolist())
    cand = [(d, i) for d, i in zip(d0.tolist(), entries.tolist())]
    heapq.heapify(cand)
    top = []  # max-heap of the best `beam` passing nodes: (-d, i)
    for d, i in cand:
        if mask[i]:
            heapq.heappush(top, (-d, i))
    top = heapq.nsmallest(beam, top)
    heapq.heapify(top)
    while cand:
        d, i = heapq.heappop(cand)
        if len(top) >= beam and d > -top[0][0]:
            break
        nbrs = graph[i]
        if policy == "acorn":
            fail = nbrs[~mask[nbrs]]
            nbrs = np.concatenate([nbrs[mask[nbrs]]] + [graph[j] for j in fail])
        nbrs = [j for j in dict.fromkeys(nbrs.tolist()) if j not in visited]
        visited.update(nbrs)
        nbrs = np.array(nbrs, dtype=np.int64)
        if policy != "traverse" and len(nbrs):
            nbrs = nbrs[mask[nbrs]]
        if not len(nbrs):
            continue
        work += len(nbrs)
        for dj, j in zip(dist(nbrs).tolist(), nbrs.tolist()):
            if len(top) < beam or dj < -top[0][0]:
                heapq.heappush(cand, (dj, j))
                if mask[j]:
                    heapq.heappush(top, (-dj, j))
                    if len(top) > beam:
                        heapq.heappop(top)
    return [i for _, i in sorted((-d, i) for d, i in top)][:k], work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--clusters", type=int, default=400)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--beams", type=int, nargs="+", default=[200, 800])
    ap.add_argument("--entries", type=int, default=8)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--partitions", type=int, default=256, help="IVF reference")
    ap.add_argument("--nprobes", type=int, default=16, help="IVF reference")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="filtered-graph-results.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    table, centers, cid, emb, tunnel_clusters = make_table(rng, a.n, a.dim, a.clusters)
    t = perf_counter()
    index = hnswlib.Index("l2", a.dim)
    index.init_index(a.n, M=a.M, ef_construction=a.ef_construction, random_seed=a.seed)
    index.add_items(emb, np.arange(a.n))
    graph = base_layer(index, a.n)
    print(json.dumps({"stage": "hnsw build", "s": round(perf_counter() - t, 1)}), flush=True)

    weather = table.column("weather").to_numpy(zero_copy_only=False)
    tags = table.column("tags").to_pylist()
    has = {tag: np.array([tag in ts for ts in tags]) for tag in TAGS}
    others = np.setdiff1d(np.arange(a.clusters), tunnel_clusters)
    q_all = make_queries(rng, centers, np.arange(a.clusters), a.queries)
    q_in = make_queries(rng, centers, tunnel_clusters, a.queries)
    q_out = make_queries(rng, centers, others, a.queries)
    cases = [
        ("none", np.ones(a.n, bool), q_all, "uncorrelated"),
        ("weather = 'rain'", weather == "rain", q_all, "uncorrelated"),
        ("pedestrian (10%)", has["pedestrian"], q_all, "uncorrelated"),
        ("cut_in (1%)", has["cut_in"], q_all, "uncorrelated"),
        ("rain AND cut_in", (weather == "rain") & has["cut_in"], q_all, "uncorrelated"),
        ("emergency_vehicle (0.1%)", has["emergency_vehicle"], q_all, "uncorrelated"),
        ("tunnel, query inside", has["tunnel"], q_in, "correlated"),
        ("tunnel, query outside", has["tunnel"], q_out, "anti-correlated"),
    ]
    results = []
    for name, mask, queries, relation in cases:
        rows = np.flatnonzero(mask)
        s = len(rows) / a.n
        probes = min(a.partitions, int(np.ceil(a.nprobes / s)))
        row = {"filter": name, "relation": relation, "selectivity": round(s, 5), "passing_rows": int(len(rows)),
               "flat_work": int(len(rows)), "ivf_planner_work": int(probes * a.n / a.partitions)}
        truths = [set(rows[np.argsort(((emb[rows] - q) ** 2).sum(1))[: a.k]].tolist()) for q in queries]
        for beam in a.beams:
            index.set_ef(beam)
            recalls, works, times = [], [], []
            for q, truth in zip(queries, truths):
                calls = [0]
                def keep(i):
                    calls[0] += 1
                    return bool(mask[i])
                t = perf_counter()
                labels, _ = index.knn_query(q, k=min(a.k, len(rows)), filter=keep)
                times.append(perf_counter() - t)
                recalls.append(len(truth & set(labels[0].tolist())) / len(truth))
                works.append(calls[0])
            row[f"hnswlib@{beam}"] = {"recall": round(float(np.mean(recalls)), 3), "filter_calls": int(np.mean(works)),
                                      "ms": round(1000 * float(np.median(times)), 1)}
            for policy in ("traverse", "strict", "acorn"):
                recalls, works = [], []
                for q, truth in zip(queries, truths):
                    pool = rows if policy != "traverse" else np.arange(a.n)
                    entries = rng.choice(pool, min(a.entries, len(pool)), replace=False)
                    got, work = search(graph, emb, q, mask, a.k, beam, policy, entries)
                    recalls.append(len(truth & set(got)) / len(truth))
                    works.append(work)
                row[f"{policy}@{beam}"] = {"recall": round(float(np.mean(recalls)), 3), "work": int(np.mean(works))}
        print(json.dumps(row), flush=True)
        results.append(row)
    with open(a.output, "w") as f:
        json.dump({"config": vars(a), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
