"""Two follow-ups on the hierarchical IVF build (see LANCE_VS_PIPNN_BUILD.md).

1. Local leaf training. Lance trains every centroid on one sample of 256 vectors per
   partition, collected on one node: 2.6e10 vectors at 1e8 partitions. Instead, train
   only the coarse level centrally (256 per coarse centroid), route all vectors to
   their coarse cell, and let each cell train its own leaves from the vectors routed
   to it. Cells are independent, so that part runs in parallel where the data lives.

2. Spilling with fewer partitions. The tree assignment already scores each vector
   against nearby leaves, so a second-nearest leaf is free. Put a vector into it too
   when it is almost as close (SPANN's rule: d2 <= (1 + eps) * d1), then check whether
   fewer partitions reach the same recall at the same vectors scanned per query.
   Fewer partitions shrink the training sample, assignment and router.

Training, encoding (1-bit RaBitQ + re-rank), routing and scan come from
ivf_build_experiment.py, so only the step under test changes.
"""
import argparse
import json
import os
import platform
from time import perf_counter
import numpy as np
from ivf_build_experiment import (RaBitQ, embedding_like, lloyd, nearest, timed, train,
                                  assign_tree)


def train_local(x, parts, rng, iters, sample_rate=256):
    """Coarse level on a small central sample; leaves per coarse cell from its own
    routed vectors. Returns leaves, coarse, parent, routed cell per vector, timings."""
    top = int(round(np.sqrt(parts)))
    s = x[rng.choice(len(x), min(len(x), sample_rate * top), replace=False)]
    (coarse, _), central = timed(lambda: lloyd(s, top, rng, iters))
    cell, route = timed(lambda: nearest(x, coarse)[:, 0])
    sizes = np.bincount(cell, minlength=top)
    quota = np.maximum(1, np.round(sizes / sizes.sum() * parts).astype(int))
    quota[np.argmax(quota)] += parts - quota.sum()
    leaves, parent, cell_seconds = [], [], []
    for j in range(top):
        start = perf_counter()
        members = x[cell == j]
        q = min(quota[j], len(members))
        sub = members[rng.choice(len(members), min(len(members), sample_rate * q), replace=False)]
        leaves.append(lloyd(sub, q, rng, iters)[0])
        parent += [j] * q
        cell_seconds.append(perf_counter() - start)
    timing = dict(central_sample=len(s), central_seconds=central, route_seconds=route,
                  local_seconds_sum=float(sum(cell_seconds)),
                  local_seconds_max_cell=float(max(cell_seconds)))
    return (np.concatenate(leaves).astype(np.float32), coarse.astype(np.float32),
            np.array(parent), cell, timing)


def assign_local(x, leaves, parent, cell):
    """Fused with local training: each vector only searches its own cell's leaves."""
    out = np.empty(len(x), np.int64)
    for j in np.unique(cell):
        rows = np.flatnonzero(cell == j)
        kids = np.flatnonzero(parent == j)
        out[rows] = kids[nearest(x[rows], leaves[kids])[:, 0]]
    return out


def assign_tree2(x, coarse, leaves, parent, t):
    """assign_tree that also keeps the second-nearest leaf and both distances."""
    top = nearest(x, coarse, t)
    best = np.full((len(x), 2), np.inf, np.float32)
    ids = np.zeros((len(x), 2), np.int64)
    flat = top.ravel()
    owner = np.repeat(np.arange(len(x)), t)
    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(len(coarse) + 1))
    ln = (leaves * leaves).sum(1)
    xn = (x * x).sum(1)
    for j in range(len(coarse)):
        rows = owner[order[bounds[j]:bounds[j+1]]]
        c = np.flatnonzero(parent == j)
        if len(rows) == 0 or len(c) == 0:
            continue
        for s in range(0, len(rows), 16384):
            r = rows[s:s+16384]
            d = xn[r, None] + ln[c][None] - 2 * (x[r] @ leaves[c].T)
            m = min(2, len(c))
            p = np.argpartition(d, m - 1, 1)[:, :m]
            dv = np.take_along_axis(d, p, 1)
            # merge this cell's two best with the running two best
            cd = np.concatenate([best[r], dv], 1)
            ci = np.concatenate([ids[r], c[p]], 1)
            o = np.argsort(cd, 1)[:, :2]
            best[r] = np.take_along_axis(cd, o, 1)
            ids[r] = np.take_along_axis(ci, o, 1)
    return ids, np.sqrt(np.maximum(best, 0))


class SpillIndex(RaBitQ):
    """RaBitQ over replicated rows; search maps rows back to vector IDs and dedups."""

    def __init__(self, x, rows, assign, centroids, rng):
        super().__init__(x[rows], assign, centroids, rng)
        self.orig = rows

    def search(self, q, probes, k, refine):
        ids, scanned = super().search(q, probes, 2 * k, refine)
        ids = self.orig[ids]
        _, first = np.unique(ids, return_index=True)
        return ids[np.sort(first)][:k], scanned


def evaluate(index, queries, truth, routes, probes_list, k, refine):
    rows = []
    for probes in probes_list:
        if probes > routes.shape[1]:
            break
        hits, scanned = 0, 0
        start = perf_counter()
        for q, t, r in zip(queries, truth, routes):
            found, n = index.search(q, r[:probes], k, refine)
            hits += len(np.intersect1d(found, t))
            scanned += n
        sec = perf_counter() - start
        rows.append(dict(nprobe=probes, recall_at_10=hits / truth.size,
                         qps=len(queries) / sec, vectors_scanned=scanned / len(queries)))
        if rows[-1]["recall_at_10"] >= 0.99:
            break
    return rows


def scanned_at(rows, target):
    """Vectors scanned per query to reach `target` recall, interpolated linearly."""
    for a, b in zip(rows, rows[1:]):
        if a["recall_at_10"] < target <= b["recall_at_10"]:
            f = (target - a["recall_at_10"]) / (b["recall_at_10"] - a["recall_at_10"])
            return a["vectors_scanned"] + f * (b["vectors_scanned"] - a["vectors_scanned"])
    return rows[0]["vectors_scanned"] if rows and rows[0]["recall_at_10"] >= target else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--clusters", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tree-t", type=int, default=4)
    ap.add_argument("--refine", type=int, default=10)
    ap.add_argument("--local-partitions", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--spill-partitions", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1])
    ap.add_argument("--probes", type=int, nargs="+",
                    default=[2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    ap.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.98])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip", choices=["local", "spill"], default=None)
    ap.add_argument("--output", default="hier-build-results.json")
    a = ap.parse_args()

    k = 10
    x, queries = embedding_like(a.seed, a.n, a.queries, a.dim, a.clusters)
    truth = nearest(queries, x, k)
    report = dict(parameters=vars(a), platform=platform.platform(), cpus=os.cpu_count(),
                  numpy=np.__version__, local=[], spill=[])

    def save():
        with open(a.output, "w") as f:
            json.dump(report, f, indent=1)

    def search_rows(index, leaves):
        routes = nearest(queries, leaves, min(max(a.probes), len(leaves)))
        rows = evaluate(index, queries, truth, routes, a.probes, k, a.refine)
        return rows, {str(tg): scanned_at(rows, tg) for tg in a.targets}

    # --- 1. Local leaf training ------------------------------------------------------
    for parts in ([] if a.skip == "local" else a.local_partitions):
        rng = np.random.default_rng(a.seed)
        (leaves, coarse, parent), t = timed(lambda: train(x, parts, rng, a.iters))
        asg, ta = timed(lambda: assign_tree(x, coarse, leaves, parent, a.tree_t)[0])
        variants = {"global_sample": (leaves, asg, dict(
            central_sample=min(len(x), 256 * parts), central_seconds=t, assign_seconds=ta))}
        rng = np.random.default_rng(a.seed)
        (lleaves, lcoarse, lparent, cell, timing), _ = timed(
            lambda: train_local(x, parts, rng, a.iters))
        la, tl = timed(lambda: assign_local(x, lleaves, lparent, cell))
        variants["local_fused"] = (lleaves, la, dict(timing, assign_seconds=tl))
        lt, tt = timed(lambda: assign_tree(x, lcoarse, lleaves, lparent, a.tree_t)[0])
        variants["local_then_tree"] = (lleaves, lt, dict(timing, assign_seconds=tt))
        # One global Lloyd step over all vectors repairs leaves at cell boundaries.
        # Per-leaf sums and counts merge across workers, so it distributes as one reduce.
        def refine():
            sums = np.zeros_like(lleaves, dtype=np.float64)
            np.add.at(sums, lt, x)
            counts = np.bincount(lt, minlength=len(lleaves))
            r = lleaves.copy()
            r[counts > 0] = (sums[counts > 0] / counts[counts > 0, None]).astype(np.float32)
            return r, assign_tree(x, lcoarse, r, lparent, a.tree_t)[0]
        (rleaves, ra), tr = timed(refine)
        variants["local_tree_refine1"] = (rleaves, ra, dict(timing, assign_seconds=tt + tr))
        refs = {}
        for name, (lv, asg, info) in variants.items():
            key = id(lv)
            if key not in refs:
                refs[key] = nearest(x, lv)[:, 0]
            ref = refs[key]
            index = RaBitQ(x, asg, lv, np.random.default_rng(a.seed + 1))
            rows, at = search_rows(index, lv)
            row = dict(partitions=parts, variant=name, **info,
                       exact_fraction=float(np.mean(asg == ref)),
                       max_partition=int(np.bincount(asg, minlength=parts).max()),
                       search=rows, scanned_at=at)
            report["local"].append(row)
            print(json.dumps({kk: v for kk, v in row.items() if kk != "search"}), flush=True)
            save()

    # --- 2. Spilling with fewer partitions -------------------------------------------
    for parts in ([] if a.skip == "spill" else a.spill_partitions):
        rng = np.random.default_rng(a.seed)
        (leaves, coarse, parent), t_train = timed(lambda: train(x, parts, rng, a.iters))
        (ids, dist), t_assign = timed(lambda: assign_tree2(x, coarse, leaves, parent, a.tree_t))
        for eps in a.eps:
            spill = (dist[:, 1] <= (1 + eps) * dist[:, 0]) if eps > 0 else np.zeros(len(x), bool)
            rows_ = np.concatenate([np.arange(len(x)), np.flatnonzero(spill)])
            asg = np.concatenate([ids[:, 0], ids[spill, 1]])
            index, t_enc = timed(lambda: SpillIndex(x, rows_, asg, leaves,
                                                    np.random.default_rng(a.seed + 1)))
            rows, at = search_rows(index, leaves)
            row = dict(partitions=parts, eps=eps, replication=len(rows_) / len(x),
                       train_seconds=t_train, train_sample=min(len(x), 256 * parts),
                       assign_seconds=t_assign, encode_seconds=t_enc,
                       search=rows, scanned_at=at)
            report["spill"].append(row)
            print(json.dumps({kk: v for kk, v in row.items() if kk != "search"}), flush=True)
            save()
            del index


if __name__ == "__main__":
    main()
