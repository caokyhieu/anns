"""IVF indexing time: Lance's design vs. a PiPNN-style hierarchical design.

Both designs share everything except the steps under test, so recall and QPS
differences come only from those steps:

* training: two-level balanced k-means on a sample (Lance trains hierarchically too)
* encoding: 1-bit RaBitQ codes of residuals, re-ranked with full vectors (IVF_RQ)
* search: exact top-nprobe routing, RaBitQ scan of the probed partitions, re-rank

The steps under test:

* assignment of every vector to a partition
    - flat:  GEMM against all C centroids (Lance's default path)
    - hnsw:  top-1 search in an HNSW over the centroids, m=12, ef=15
             (Lance's LANCE_USE_HNSW_SPEEDUP_INDEXING path)
    - tree_t: each vector picks its t nearest of ~sqrt(C) coarse centroids, then one
             GEMM per coarse centroid scores the vectors that picked it against that
             centroid's leaves (PiPNN-style: partition, then dense products inside
             each part); about (1 + t) * sqrt(C) evaluations per vector
* router: the graph over the C centroids that queries use (HNSW vs PiPNN),
  reported as build time and agreement with exact top-nprobe routing

The real Lance (pylance) build of the same data is timed as a reference.
"""
import argparse
import json
import os
import platform
import shutil
import tempfile
from time import perf_counter
import numpy as np
import dpipnn as D
from distributed_experiment import data


def timed(f):
    start = perf_counter()
    out = f()
    return out, perf_counter() - start


def embedding_like(seed, n, nq, dim, clusters, latent=24):
    """Synthetic stand-in for real embeddings: a mixture of many overlapping clusters
    in a low-dimensional latent space, mapped to `dim` with a decaying spectrum, plus
    small isotropic noise. Queries come from the same distribution."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, latent)) * 1.5
    scale = rng.uniform(0.5, 1.5, clusters)
    q, _ = np.linalg.qr(rng.normal(size=(dim, latent)))
    basis = (q * (np.arange(1, latent + 1) ** -0.5)).T
    def sample(count):
        c = rng.integers(0, clusters, count)
        z = centers[c] + rng.normal(size=(count, latent)) * scale[c, None]
        return (z @ basis + rng.normal(size=(count, dim)) * 0.02).astype(np.float32)
    return sample(n), sample(nq)


def nearest(x, c, k=1, chunk=16384):
    """Exact k nearest rows of c for each row of x, by chunked GEMM."""
    cn = (c * c).sum(1)
    ids = np.empty((len(x), k), np.int64)
    for s in range(0, len(x), chunk):
        d = cn[None] - 2 * (x[s:s+chunk] @ c.T)
        if k == 1:
            ids[s:s+chunk, 0] = d.argmin(1)
        else:
            p = np.argpartition(d, k - 1, 1)[:, :k]
            ids[s:s+chunk] = np.take_along_axis(p, np.take_along_axis(d, p, 1).argsort(1), 1)
    return ids


def lloyd(x, k, rng, iters):
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        a = nearest(x, c)[:, 0]
        sums = np.zeros_like(c)
        np.add.at(sums, a, x)
        counts = np.bincount(a, minlength=k)
        live = counts > 0
        c[live] = sums[live] / counts[live, None]
        dead = np.flatnonzero(~live)
        c[dead] = x[rng.choice(len(x), len(dead), replace=False)]
    return c, nearest(x, c)[:, 0]


def train(x, parts, rng, iters, sample_rate=256):
    """Two-level k-means with proportional quotas (as Lance's hierarchical trainer).

    Returns leaf centroids, coarse centroids and each leaf's coarse parent."""
    s = x[rng.choice(len(x), min(len(x), sample_rate * parts), replace=False)]
    top = int(round(np.sqrt(parts)))
    coarse, a = lloyd(s, top, rng, iters)
    sizes = np.bincount(a, minlength=top)
    quota = np.maximum(1, np.round(sizes / sizes.sum() * parts).astype(int))
    quota[np.argmax(quota)] += parts - quota.sum()
    leaves, parent = [], []
    for j in range(top):
        sub = s[a == j]
        q = min(quota[j], len(sub))
        leaves.append(lloyd(sub, q, rng, iters)[0])
        parent += [j] * q
    return np.concatenate(leaves).astype(np.float32), coarse.astype(np.float32), np.array(parent)


def assign_tree(x, coarse, leaves, parent, t):
    """Each vector takes its t nearest coarse centroids; then, per coarse centroid, one
    GEMM scores every vector that chose it against that centroid's leaves. Exact over
    the chosen subtrees, GEMM-shaped, about t*C/sqrt(C) evaluations per vector."""
    top = nearest(x, coarse, t)
    best = np.full(len(x), np.inf, np.float32)
    out = np.zeros(len(x), np.int64)
    flat = top.ravel()
    owner = np.repeat(np.arange(len(x)), t)
    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(len(coarse) + 1))
    kids = [np.flatnonzero(parent == j) for j in range(len(coarse))]
    ln = (leaves * leaves).sum(1)
    evals = len(coarse) * len(x)
    for j in range(len(coarse)):
        rows = owner[order[bounds[j]:bounds[j+1]]]
        if len(rows) == 0:
            continue
        c = kids[j]
        for s in range(0, len(rows), 16384):
            r = rows[s:s+16384]
            d = ln[c][None] - 2 * (x[r] @ leaves[c].T)
            i = d.argmin(1)
            v = d[np.arange(len(r)), i]
            better = v < best[r]
            best[r[better]] = v[better]
            out[r[better]] = c[i[better]]
        evals += len(rows) * len(c)
    return out, evals / len(x)


class RaBitQ:
    """1-bit RaBitQ of residuals, rows stored grouped by partition."""

    def __init__(self, x, assign, centroids, rng):
        self.d = x.shape[1]
        self.rot = np.linalg.qr(rng.normal(size=(self.d, self.d)))[0].astype(np.float32)
        self.c = centroids
        self.crot = centroids @ self.rot
        order = np.argsort(assign, kind="stable")
        self.ids = order
        self.offsets = np.searchsorted(assign[order], np.arange(len(centroids) + 1))
        self.x = x[order]
        r = self.x - centroids[assign[order]]
        self.nr = np.linalg.norm(r, axis=1).astype(np.float32)
        o = (r / np.maximum(self.nr, 1e-12)[:, None]) @ self.rot
        self.bits = np.where(o >= 0, 1, -1).astype(np.int8)
        self.xo = np.maximum(np.abs(o).sum(1) / np.sqrt(self.d), 1e-6).astype(np.float32)
        # <b, rot^T c_p> per row: lets one GEMV score rows of many partitions at once.
        self.bc = (self.bits * self.crot[assign[order]]).sum(1).astype(np.float32)
        self.part = assign[order]

    def search(self, q, probes, k, refine):
        rows = np.concatenate([np.arange(self.offsets[p], self.offsets[p+1]) for p in probes])
        qrot = q @ self.rot
        nq = np.linalg.norm(q[None] - self.c[self.part[rows]], axis=1)
        ip = (self.bits[rows] @ qrot - self.bc[rows]) / (np.sqrt(self.d) * self.xo[rows])
        est = self.nr[rows] ** 2 + nq ** 2 - 2 * self.nr[rows] * ip
        r = min(len(rows), k * refine)
        cand = rows[np.argpartition(est, r - 1)[:r]]
        exact = ((self.x[cand] - q) ** 2).sum(1)
        best = cand[np.argsort(exact)[:k]]
        return self.ids[best], len(rows)


def evaluate(index, queries, truth, routes, probes_list, k, refine):
    rows = []
    for probes in probes_list:
        hits, scanned = 0, 0
        start = perf_counter()
        for q, t, r in zip(queries, truth, routes):
            found, n = index.search(q, r[:probes], k, refine)
            hits += len(np.intersect1d(found, t))
            scanned += n
        sec = perf_counter() - start
        rows.append(dict(nprobe=probes, recall_at_10=hits / truth.size,
                         qps=len(queries) / sec, vectors_scanned=scanned / len(queries)))
    return rows


def lance_reference(path, parts, probes_list, k, refine, hnsw_assign):
    """Build Lance IVF_RQ (1 bit) on the same vectors and time it. Runs in its own
    process: Lance reads LANCE_USE_HNSW_SPEEDUP_INDEXING once per process."""
    import subprocess
    import sys
    env = dict(os.environ, LANCE_USE_HNSW_SPEEDUP_INDEXING="enabled" if hnsw_assign else "disabled")
    args = json.dumps(dict(path=path, parts=parts, probes=probes_list, k=k, refine=refine))
    out = subprocess.run([sys.executable, __file__, "--lance-worker", args], env=env,
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def lance_worker(args):
    import lance
    import pyarrow as pa
    a = json.loads(args)
    z = np.load(a["path"])
    x, queries, truth = z["x"], z["queries"], z["truth"]
    tmp = tempfile.mkdtemp(prefix="lance-ivf-")
    try:
        vec = pa.FixedSizeListArray.from_arrays(pa.array(x.ravel()), x.shape[1])
        ds = lance.write_dataset(pa.table({"id": np.arange(len(x)), "v": vec}), tmp,
                                 max_rows_per_file=1 << 20)
        _, sec = timed(lambda: ds.create_index("v", "IVF_RQ", num_partitions=a["parts"], num_bits=1))
        ds = lance.dataset(tmp)
        rows = []
        for probes in a["probes"]:
            hits = 0
            start = perf_counter()
            for q, t in zip(queries, truth):
                got = ds.to_table(columns=["id"], nearest=dict(
                    column="v", q=q, k=a["k"], nprobes=probes,
                    refine_factor=a["refine"])).column(0).to_numpy()
                hits += len(np.intersect1d(got, t))
            rows.append(dict(nprobe=probes, recall_at_10=hits / truth.size,
                             qps=len(queries) / (perf_counter() - start)))
        print(json.dumps(dict(build_seconds=sec, search=rows)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--clusters", type=int, default=20000, help="mixture components")
    ap.add_argument("--data", choices=["embed", "blobs"], default="embed")
    ap.add_argument("--partitions", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tree-t", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--probes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--refine", type=int, default=10)
    ap.add_argument("--lance", action="store_true", help="also time pylance IVF_RQ")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="ivf-build-results.json")
    a = ap.parse_args()

    k = 10
    gen = embedding_like if a.data == "embed" else data
    x, queries = gen(a.seed, a.n, a.queries, a.dim, a.clusters)
    x, queries = x.astype(np.float32), queries.astype(np.float32)
    truth = nearest(queries, x, k)
    report = dict(parameters=vars(a), platform=platform.platform(), cpus=os.cpu_count(),
                  numpy=np.__version__, runs=[])

    for parts in a.partitions:
        rng = np.random.default_rng(a.seed)
        run = dict(partitions=parts, phases={}, assignment={}, router={}, search={})
        (leaves, coarse, parent), t = timed(lambda: train(x, parts, rng, a.iters))
        run["phases"]["train_seconds"] = t
        truth_assign, t = timed(lambda: nearest(x, leaves)[:, 0])
        assigns = {"flat": (truth_assign, t, float(parts))}

        import hnswlib
        h = hnswlib.Index("l2", a.dim)
        _, hb = timed(lambda: (h.init_index(parts, M=12, ef_construction=15), h.add_items(leaves)))
        h.set_ef(15)
        (lab, _), hq = timed(lambda: h.knn_query(x, k=1))
        assigns["hnsw"] = (lab[:, 0].astype(np.int64), hb + hq, None)

        for tt in a.tree_t:
            (tr, ev), t = timed(lambda: assign_tree(x, coarse, leaves, parent, tt))
            assigns[f"tree_t{tt}"] = (tr, t, ev)

        # Router graphs over the leaf centroids (queries use them; tree/graph reuse PiPNN).
        hr = hnswlib.Index("l2", a.dim)
        _, t = timed(lambda: (hr.init_index(parts, M=16, ef_construction=200), hr.add_items(leaves)))
        run["router"]["hnsw_build_seconds"] = t
        graph, t = timed(lambda: D.build(leaves.astype(np.float64), 1,
                                         D.Params(seed=a.seed, degree=24)))
        run["router"]["pipnn_build_seconds"] = t

        exact_routes = nearest(queries, leaves, max(a.probes))
        hr.set_ef(max(a.probes) * 2)
        hl, _ = hr.knn_query(queries, k=max(a.probes))
        run["router"]["hnsw_route_agreement"] = float(np.mean(
            [len(np.intersect1d(u, v)) / len(u) for u, v in zip(hl, exact_routes)]))
        head = D.Head(leaves.astype(np.float64), seed=a.seed)
        gl = [D.search_global(leaves.astype(np.float64), graph, head, q.astype(np.float64),
                              max(a.probes) * 2, max(a.probes))[0] for q in queries[:100]]
        run["router"]["pipnn_route_agreement"] = float(np.mean(
            [len(np.intersect1d(u, v)) / len(u) for u, v in zip(gl, exact_routes)]))

        for name, (asg, sec, ev) in assigns.items():
            sizes = np.bincount(asg, minlength=parts)
            run["assignment"][name] = dict(
                seconds=sec, evals_per_vector=ev,
                exact_fraction=float(np.mean(asg == truth_assign)),
                max_partition=int(sizes.max()), empty_partitions=int((sizes == 0).sum()))
            print(json.dumps(dict(partitions=parts, assign=name, **run["assignment"][name])), flush=True)

        enc_rng = np.random.default_rng(a.seed + 1)
        for name, (asg, _, _) in assigns.items():
            index, t = timed(lambda: RaBitQ(x, asg, leaves, enc_rng))
            run["phases"][f"encode_seconds_{name}"] = t
            run["search"][name] = evaluate(index, queries, truth, exact_routes, a.probes, k, a.refine)
            for row in run["search"][name]:
                print(json.dumps(dict(partitions=parts, assign=name, **row)), flush=True)
            del index

        if a.lance:
            path = os.path.join(tempfile.gettempdir(), f"ivf-data-{os.getpid()}.npz")
            np.savez(path, x=x, queries=queries[:200], truth=truth[:200])
            try:
                for hnsw_assign in (False, True):
                    key = "lance_hnsw_assign" if hnsw_assign else "lance"
                    run[key] = lance_reference(path, parts, a.probes, k, a.refine, hnsw_assign)
                    print(json.dumps(dict(partitions=parts, lance=key, **run[key])), flush=True)
            finally:
                os.remove(path)

        print(json.dumps(dict(partitions=parts, phases=run["phases"], router=run["router"])), flush=True)
        report["runs"].append(run)
        with open(a.output, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--lance-worker":
        lance_worker(sys.argv[2])
    else:
        main()
