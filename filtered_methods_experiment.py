"""Methods from the filtered-search literature (FILTERED_SEARCH_LITERATURE.md) run on the
indexes we built: HNSW (hnswlib), our from-scratch fastpipnn graph (native/), an unpruned
k-NN graph, and IVF. Same synthetic AV table and filters as filtered_search_experiment.py,
plus AND / OR filters.

Every method is scored as recall@k against exact filtered search and as work: distance
computations per query (graphs, IVF, brute force), and for IVF also rows read.

Graphs (base layer for HNSW; degree = max out-degree):
  hnsw-d32, hnsw-d64   hnswlib M = 16 / 32 (RNG-style heuristic pruning)
  pipnn-d64           fastpipnn defaults (local_k 2, alpha 1.2, HashPrune + RobustPrune)
  pipnn-d64-dense     fastpipnn with local_k 8 and alpha 2.0 (fewer edges pruned)
  knn-d32, knn-d64     plain nearest neighbors, no pruning (KGraph style, Li et al. claim 4)

Graph search policies (all start from `entries` randomly sampled passing rows, whose
distances count as work, unless noted):
  acorn       walk passing nodes, plus all passing 2-hop neighbors through failing ones
              (NaviX `blind` without a cap; ACORN-1 on a normal graph)
  navix       NaviX per-candidate adaptive expansion from local selectivity: onehop-s
              above 50%, `directed` 2-hop (1-hop sorted by distance) while the expected
              passing 2-hop count is at least 3 x degree, `blind` below that
  milvus      dual-pool traversal (walk every node, keep only passing ones; hnswlib's rule)
              from one random entry, a brute-force switch when > 93% of rows fail, and a
              safety net that re-runs exactly when fewer than k rows come back
IVF plans (numpy IVF, 256 k-means partitions, base nprobe 16):
  ivf-fixed   probe 16 nearest partitions, score passing rows (Lance prefilter)
  ivf-scaled  nprobe = 16 / s (our planner without the flat switch)
  ivf-skip    partition x label counts: rank by centroid distance, skip partitions with no
              passing row, probe 16 non-empty ones (FILTERED_SEARCH.md 4.2)
  caps-hN     CAPS attribute-frequency tree of height N inside each partition, nprobe 16:
              read only sub-partitions that can match (fewer rows read, same partitions)
  caps-hN+skip  CAPS with the non-empty partition rule
Collections of indexes:
  sieve       SIEVE: greedy choice of HNSW sub-indexes for the workload's filters by
              benefit per byte under a memory budget, then per query the cheaper of the
              best subsuming sub-index (dual-pool search) and brute force, by SIEVE's cost
              model
  pathfinder  PathFinder: one HNSW per attribute value (weather values, tags) plus the
              root graph; per DNF clause pick the covering graph with the highest
              U = card(R, p) / (sum card(g) * |G|^0.4), union the clauses, dual-pool search
              each chosen graph with the full predicate and merge
Also reported per filter: mean GLS correlation rho (Amanbayev et al.) and its rank
correlation with per-query recall of ivf-fixed and navix on hnsw-d32.
"""
import argparse
import heapq
import json
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
import hnswlib
from filtered_search_experiment import TAGS, make_table, make_queries
from filtered_graph_experiment import base_layer

PIPNN_GRAPH = Path(__file__).parent / "native" / "build" / "pipnn_graph"


def log(**kv):
    print(json.dumps(kv), flush=True)


# ---------------------------------------------------------------- graphs

def hnsw_graph(emb, M, seed):
    index = hnswlib.Index("l2", emb.shape[1])
    index.init_index(len(emb), M=M, ef_construction=200, random_seed=seed)
    index.add_items(emb, np.arange(len(emb)))
    return index, base_layer(index, len(emb))


def pipnn_graph(emb, degree, seed, alpha=1.2, local_k=2):
    with tempfile.TemporaryDirectory() as tmp:
        vec, out = Path(tmp) / "v.f32", Path(tmp) / "g.u32"
        np.ascontiguousarray(emb, np.float32).tofile(vec)
        subprocess.run([str(PIPNN_GRAPH), str(vec), str(len(emb)), str(emb.shape[1]), str(degree), str(out), str(alpha),
                        str(seed), str(local_k)], check=True, stdout=subprocess.DEVNULL)
        raw = np.fromfile(out, np.uint32)
    n = len(emb)
    deg, adj = raw[:n], raw[n:].reshape(n, degree).astype(np.int64)
    return [adj[i, : deg[i]] for i in range(n)]


def knn_graph(index, emb, degree):
    index.set_ef(max(200, 2 * degree))
    labels, _ = index.knn_query(emb, k=degree + 1)
    return [row[row != i][:degree].astype(np.int64) for i, row in enumerate(labels)]


# ---------------------------------------------------------------- graph search

def graph_search(graph, emb, q, mask, k, beam, policy, entries):
    """Best-first search on passing nodes with acorn / navix expansion; returns (ids, work)."""
    def dist(ids):
        return ((emb[ids] - q) ** 2).sum(1)
    work = len(entries)
    visited = set(entries.tolist())
    cand = list(zip(dist(entries).tolist(), entries.tolist()))
    heapq.heapify(cand)
    top = [(-d, i) for d, i in cand]
    top = heapq.nlargest(beam, top)  # (-d, i): the nearest `beam`
    heapq.heapify(top)
    while cand:
        d, i = heapq.heappop(cand)
        if len(top) >= beam and d > -top[0][0]:
            break
        nb = graph[i]
        nb = nb[[j not in visited for j in nb.tolist()]] if len(nb) else nb
        if not len(nb):
            continue
        cap = len(graph[i])
        passing = mask[nb]
        pre = {}  # distances already computed for 1-hop nodes by `directed`
        if policy == "acorn":
            new = np.concatenate([nb[passing]] + [graph[j][mask[graph[j]]] for j in nb[~passing]])
        else:
            local = passing.mean()
            if local >= 0.5:  # onehop-s
                new = nb[passing]
            elif local * cap >= 3:  # directed: expected passing 2-hop >= 3 x degree
                d1 = dist(nb)
                work += len(nb)
                visited.update(nb.tolist())
                pre = dict(zip(nb.tolist(), d1.tolist()))
                new, got = [], 0
                for j in nb[np.argsort(d1)].tolist():
                    add = np.array([j]) if mask[j] else graph[j][mask[graph[j]]]
                    new.append(add)
                    got += len(add)
                    if got >= cap:
                        break
                new = np.concatenate(new)
            else:  # blind: 2-hop in scan order, capped at the degree
                new, got = [], 0
                for j in nb.tolist():
                    add = np.array([j]) if mask[j] else graph[j][mask[graph[j]]]
                    new.append(add)
                    got += len(add)
                    if got >= cap:
                        break
                new = np.concatenate(new)
        ids = [j for j in dict.fromkeys(new.tolist()) if j in pre or j not in visited]
        if not ids:
            continue
        ids = np.array(ids, np.int64)
        fresh = np.array([j not in pre for j in ids.tolist()])
        dj = np.empty(len(ids))
        if fresh.any():
            dj[fresh] = dist(ids[fresh])
        dj[~fresh] = [pre[j] for j in ids[~fresh].tolist()]
        work += int(fresh.sum())
        visited.update(ids.tolist())
        for dv, j in zip(dj.tolist(), ids.tolist()):
            if len(top) < beam or dv < -top[0][0]:
                heapq.heappush(cand, (dv, j))
                heapq.heappush(top, (-dv, j))
                if len(top) > beam:
                    heapq.heappop(top)
    return [i for _, i in sorted((-d, i) for d, i in top)][:k], work


def dual_pool(graph, emb, q, mask, k, beam, entries, allowed=None):
    """Walk every node (or every node in `allowed`), keep only passing ones as results."""
    def dist(ids):
        return ((emb[ids] - q) ** 2).sum(1)
    work = len(entries)
    visited = set(entries.tolist())
    cand = list(zip(dist(entries).tolist(), entries.tolist()))
    heapq.heapify(cand)
    top = [(-d, i) for d, i in cand if mask[i]]
    heapq.heapify(top)
    while cand:
        d, i = heapq.heappop(cand)
        if len(top) >= beam and d > -top[0][0]:
            break
        ids = [j for j in graph[i].tolist() if j not in visited]
        if not ids:
            continue
        visited.update(ids)
        ids = np.array(ids, np.int64)
        work += len(ids)
        for dv, j in zip(dist(ids).tolist(), ids.tolist()):
            if len(top) < beam or dv < -top[0][0]:
                heapq.heappush(cand, (dv, j))
                if mask[j]:
                    heapq.heappush(top, (-dv, j))
                    if len(top) > beam:
                        heapq.heappop(top)
    return [i for _, i in sorted((-d, i) for d, i in top)][:k], work


def brute(emb, rows, q, k):
    d = ((emb[rows] - q) ** 2).sum(1)
    return rows[np.argsort(d)[:k]].tolist(), len(rows)


def milvus(graph, emb, q, mask, rows, k, beam, rng):
    if len(rows) < 0.07 * len(mask):
        return brute(emb, rows, q, k)
    got, work = dual_pool(graph, emb, q, mask, k, beam, rng.choice(len(mask), 1))
    if len(got) < min(k, len(rows)):  # safety net
        exact, w = brute(emb, rows, q, k)
        return exact, work + w
    return got, work


# ---------------------------------------------------------------- IVF and CAPS

def kmeans(rng, x, parts, iters=12, sample=50_000):
    s = x[rng.choice(len(x), min(sample, len(x)), replace=False)]
    c = s[rng.choice(len(s), parts, replace=False)].copy()
    for _ in range(iters):
        a = assign(s, c)
        for p in range(parts):
            m = a == p
            if m.any():
                c[p] = s[m].mean(0)
    return c, assign(x, c)


def assign(x, c):
    out = np.empty(len(x), np.int64)
    cc = (c ** 2).sum(1)
    for b in range(0, len(x), 20_000):
        out[b : b + 20_000] = np.argmin(cc - 2 * x[b : b + 20_000] @ c.T, 1)
    return out


class IVF:
    def __init__(self, rng, emb, attrs, parts, heights):
        self.emb, self.attrs = emb, attrs  # attrs: name -> bool mask
        self.centroids, self.part = kmeans(rng, emb, parts)
        self.lists = [np.flatnonzero(self.part == p) for p in range(parts)]
        names = list(attrs)
        mat = np.stack([attrs[a] for a in names], 1)
        # CAPS attribute-frequency tree per partition: buckets [(label or None, rows)]
        self.caps = {}
        for h in heights:
            trees = []
            for rows in self.lists:
                rest, buckets = rows, []
                for _ in range(h):
                    if not len(rest):
                        break
                    j = int(mat[rest].sum(0).argmax())
                    hit = mat[rest, j]
                    buckets.append((names[j], rest[hit]))
                    rest = rest[~hit]
                buckets.append((None, rest))
                trees.append(buckets)
            self.caps[h] = trees

    def order(self, q):
        return np.argsort(((self.centroids - q) ** 2).sum(1))

    def search(self, q, mask, labels, k, nprobe, skip=False, caps=None):
        """labels: the conjunction's label names when CAPS applies (None: CAPS reads all)."""
        order = self.order(q)
        work, read, cands, probed = len(self.centroids), 0, [], 0
        for p in order:
            if probed >= nprobe:
                break
            rows = self.lists[p]
            if skip and not mask[rows].any():
                continue
            probed += 1
            if caps is not None and labels is not None:
                parts, seen = [], set()
                for name, bucket in self.caps[caps][p]:
                    if name is None:
                        if not seen & labels:
                            parts.append(bucket)
                    elif not seen & labels:
                        parts.append(bucket)
                    if name is not None:
                        seen.add(name)
                rows = np.concatenate(parts) if parts else rows[:0]
            read += len(rows)
            rows = rows[mask[rows]]
            work += len(rows)
            cands.append(rows)
        rows = np.concatenate(cands) if cands else np.zeros(0, np.int64)
        got = rows[np.argsort(((self.emb[rows] - q) ** 2).sum(1))[:k]].tolist()
        return got, work, read


# ---------------------------------------------------------------- SIEVE and PathFinder

def sieve_fit(filters, freq, n, M, budget, sef, cor=0.5):
    """Greedy SIEVE: add sub-indexes (one per workload filter) by benefit per byte."""
    def cost(f_card, h_card):
        return np.log2(max(h_card, 2)) * sef * (h_card / f_card) ** cor

    def best(chosen, f):
        card = filters[f].sum()
        opts = [card]  # brute force, gamma = 1 distance per passing row
        for h in chosen:
            if not (filters[f] & ~filters[h]).any():
                opts.append(cost(card, filters[h].sum()))
        return min(opts)

    chosen, used = ["__all__"], 0.0
    filters = dict(filters, __all__=np.ones(n, bool))
    while True:
        base = sum(freq[f] * best(chosen, f) for f in freq)
        pick, gain = None, 0.0
        for h in freq:
            if h in chosen:
                continue
            size = M * filters[h].sum()
            if used + size > budget:
                continue
            g = (base - sum(freq[f] * best(chosen + [h], f) for f in freq)) / size
            if g > gain:
                pick, gain = h, g
        if pick is None:
            return chosen
        chosen.append(pick)
        used += M * filters[pick].sum()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--clusters", type=int, default=400)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--entries", type=int, default=64)
    ap.add_argument("--beam-sweep", type=int, nargs="+", default=[100, 400])
    ap.add_argument("--entry-sweep", type=int, nargs="+", default=[1, 8, 64, 512])
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--partitions", type=int, default=256)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--caps-heights", type=int, nargs="+", default=[3, 8])
    ap.add_argument("--sieve-budget", type=float, default=0.5, help="extra index memory, x the base graph")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="filtered-methods-results.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    table, centers, cid, emb, tunnel_clusters = make_table(rng, a.n, a.dim, a.clusters)
    weather = table.column("weather").to_numpy(zero_copy_only=False)
    tags = table.column("tags").to_pylist()
    attrs = {f"weather={w}": weather == w for w in np.unique(weather)}
    attrs.update({t: np.array([t in ts for ts in tags]) for t in TAGS})

    t = perf_counter()
    hnsw16, g = hnsw_graph(emb, 16, a.seed)
    graphs = {"hnsw-d32": g}
    _, graphs["hnsw-d64"] = hnsw_graph(emb, 32, a.seed)
    log(stage="hnsw build", s=round(perf_counter() - t, 1))
    for name, alpha, local_k in (("pipnn-d64", 1.2, 2), ("pipnn-d64-dense", 2.0, 8)):
        t = perf_counter()
        graphs[name] = pipnn_graph(emb, 64, a.seed, alpha, local_k)
        log(stage=f"fastpipnn {name} build", s=round(perf_counter() - t, 1))
    t = perf_counter()
    graphs["knn-d64"] = knn_graph(hnsw16, emb, 64)
    graphs["knn-d32"] = [r[:32] for r in graphs["knn-d64"]]
    log(stage="knn graph", s=round(perf_counter() - t, 1))
    for name, gr in graphs.items():
        log(graph=name, mean_degree=round(float(np.mean([len(r) for r in gr])), 1))

    t = perf_counter()
    ivf = IVF(rng, emb, attrs, a.partitions, a.caps_heights)
    log(stage="ivf + caps build", s=round(perf_counter() - t, 1))

    A = attrs
    others = np.setdiff1d(np.arange(a.clusters), tunnel_clusters)
    q_all = make_queries(rng, centers, np.arange(a.clusters), a.queries)
    q_in = make_queries(rng, centers, tunnel_clusters, a.queries)
    q_out = make_queries(rng, centers, others, a.queries)
    # (name, DNF as list of AND-clauses of label names, queries, relation)
    cases = [
        ("weather = 'rain'", [["weather=rain"]], q_all, "uncorrelated"),
        ("pedestrian (10%)", [["pedestrian"]], q_all, "uncorrelated"),
        ("cut_in (1%)", [["cut_in"]], q_all, "uncorrelated"),
        ("emergency_vehicle (0.1%)", [["emergency_vehicle"]], q_all, "uncorrelated"),
        ("tunnel, query inside", [["tunnel"]], q_in, "correlated"),
        ("tunnel, query outside", [["tunnel"]], q_out, "anti-correlated"),
        ("rain AND cut_in", [["weather=rain", "cut_in"]], q_all, "uncorrelated"),
        ("night AND pedestrian", [["weather=night", "pedestrian"]], q_all, "uncorrelated"),
        ("pedestrian OR tunnel", [["pedestrian"], ["tunnel"]], q_all, "mixed"),
        ("cut_in OR emergency_vehicle", [["cut_in"], ["emergency_vehicle"]], q_all, "uncorrelated"),
        ("(rain AND pedestrian) OR tunnel", [["weather=rain", "pedestrian"], ["tunnel"]], q_out, "anti-correlated"),
    ]

    def mask_of(dnf):
        m = np.zeros(a.n, bool)
        for clause in dnf:
            c = np.ones(a.n, bool)
            for lab in clause:
                c &= A[lab]
            m |= c
        return m

    # SIEVE: the workload is the case list, equal frequency; sub-indexes are hnswlib M = 16
    masks = {name: mask_of(dnf) for name, dnf, _, _ in cases}
    uniq = {}
    for name, m in masks.items():
        uniq.setdefault(m.tobytes(), name)
    wl = {name: masks[name] for name in uniq.values()}
    budget = a.sieve_budget * 32 * a.n
    chosen = sieve_fit(wl, {f: 1.0 for f in wl}, a.n, 32, budget, a.beam)
    sub = {}
    for f in chosen[1:]:
        rows = np.flatnonzero(wl[f])
        idx = hnswlib.Index("l2", a.dim)
        idx.init_index(len(rows), M=16, ef_construction=200, random_seed=a.seed)
        idx.add_items(emb[rows], np.arange(len(rows)))
        sub[f] = (rows, base_layer(idx, len(rows)))
    log(stage="sieve fit", chosen=chosen[1:], extra_edges_x_base=round(sum(32 * len(r) for r, _ in sub.values()) / (32 * a.n), 3))

    # PathFinder: one hnswlib graph per attribute value, plus the root graph (hnsw-d32)
    t = perf_counter()
    pf = {}
    for lab, m in A.items():
        rows = np.flatnonzero(m)
        idx = hnswlib.Index("l2", a.dim)
        idx.init_index(len(rows), M=16, ef_construction=200, random_seed=a.seed)
        idx.add_items(emb[rows], np.arange(len(rows)))
        pf[lab] = (rows, base_layer(idx, len(rows)))
    log(stage="pathfinder per-value graphs", s=round(perf_counter() - t, 1),
        extra_rows_x_base=round(sum(len(r) for r, _ in pf.values()) / a.n, 3))

    def pathfinder_plan(dnf, m):
        plan = []
        for clause in dnf:
            cm = np.ones(a.n, bool)
            for lab in clause:
                cm &= A[lab]
            # candidate single graphs covering the clause: each label's graph, or the root
            opts = [(int(cm.sum()) / len(pf[lab][0]), lab) for lab in clause] + [(int(cm.sum()) / a.n, "__root__")]
            plan.append(max(opts)[1])
        plan = list(dict.fromkeys(plan))
        u = lambda gs: int(m.sum()) / (sum(a.n if x == "__root__" else len(pf[x][0]) for x in gs) * len(gs) ** 0.4)
        if "__root__" not in plan and u(["__root__"]) > u(plan):
            plan = ["__root__"]
        return plan

    def sub_search(rows, g, q, m, k, beam):
        local = m[rows]
        if not local.any():
            return [], 0
        entries = rng.choice(np.flatnonzero(local), min(a.entries, int(local.sum())), replace=False)
        got, w = dual_pool(g, emb[rows], q, local, k, beam, entries)
        return rows[got].tolist(), w

    results = []
    rho_rows = []
    for name, dnf, queries, relation in cases:
        m = masks[name]
        rows = np.flatnonzero(m)
        s = len(rows) / a.n
        k = min(a.k, len(rows))
        row = {"filter": name, "relation": relation, "selectivity": round(s, 5), "passing_rows": int(len(rows))}
        truths = [set(brute(emb, rows, q, k)[0]) for q in queries]
        # GLS correlation: pass rate among the unfiltered top-k over global pass rate
        hnsw16.set_ef(max(200, a.k))
        top, _ = hnsw16.knn_query(queries, k=a.k)
        r = np.maximum(m[top].mean(1), 1e-9) / s
        rho = (r - 1) / (r + 1)
        row["gls_rho"] = round(float(rho.mean()), 3)
        rec = {}

        def score(key, fn):
            rs, ws, rd = [], [], []
            for q, truth in zip(queries, truths):
                out = fn(q)
                got, w = out[0], out[1]
                rs.append(len(truth & set(got)) / len(truth))
                ws.append(w)
                if len(out) > 2:
                    rd.append(out[2])
            cell = {"recall": round(float(np.mean(rs)), 3), "work": int(np.mean(ws))}
            if rd:
                cell["rows_read"] = int(np.mean(rd))
            row[key] = cell
            rec[key] = rs

        def entries(cnt):
            return rng.choice(rows, min(cnt, len(rows)), replace=False)

        for gname, gr in graphs.items():
            for policy in ("acorn", "navix"):
                score(f"{policy}/{gname}", lambda q: graph_search(gr, emb, q, m, k, a.beam, policy, entries(a.entries)))
        for gname in ("hnsw-d64", "pipnn-d64-dense"):
            for beam in a.beam_sweep:
                for policy in ("acorn", "navix"):
                    score(f"{policy}/{gname}/beam={beam}", lambda q: graph_search(graphs[gname], emb, q, m, k, beam, policy, entries(a.entries)))
            for cnt in a.entry_sweep:
                score(f"navix/{gname}/entries={cnt}", lambda q: graph_search(graphs[gname], emb, q, m, k, a.beam, "navix", entries(cnt)))
        for gname in ("hnsw-d32", "pipnn-d64"):
            score(f"milvus/{gname}", lambda q: milvus(graphs[gname], emb, q, m, rows, k, a.beam, rng))

        conj = set(dnf[0]) if len(dnf) == 1 else None  # CAPS handles conjunctions only
        probes = min(a.partitions, int(np.ceil(a.nprobe / s)))
        score("ivf-fixed", lambda q: ivf.search(q, m, None, k, a.nprobe))
        score("ivf-scaled", lambda q: ivf.search(q, m, None, k, probes))
        score("ivf-skip", lambda q: ivf.search(q, m, None, k, a.nprobe, skip=True))
        for h in a.caps_heights:
            score(f"caps-h{h}", lambda q: ivf.search(q, m, conj, k, a.nprobe, caps=h))
            score(f"caps-h{h}+skip", lambda q: ivf.search(q, m, conj, k, a.nprobe, skip=True, caps=h))

        # SIEVE query-time choice: cheapest of brute force and subsuming sub-indexes
        def sieve(q):
            best, plan = len(rows), None
            for f, (srows, sg) in sub.items():
                if not (m & ~wl[f]).any():
                    c = np.log2(max(len(srows), 2)) * a.beam * (len(srows) / len(rows)) ** 0.5
                    if c < best:
                        best, plan = c, f
            if plan is None:
                if len(rows) < 0.07 * a.n:
                    return brute(emb, rows, q, k)
                return dual_pool(graphs["hnsw-d32"], emb, q, m, k, a.beam, rng.choice(a.n, 1))
            srows, sg = sub[plan]
            return sub_search(srows, sg, q, m, k, a.beam)
        score("sieve", sieve)

        plan = pathfinder_plan(dnf, m)
        row["pathfinder_plan"] = plan

        def pathfinder(q):
            got, work = [], 0
            for gname in plan:
                if gname == "__root__":
                    ids, w = dual_pool(graphs["hnsw-d32"], emb, q, m, k, a.beam, entries(a.entries))
                else:
                    ids, w = sub_search(*pf[gname], q, m, k, a.beam)
                got += ids
                work += w
            got = np.array(list(dict.fromkeys(got)), np.int64)
            return got[np.argsort(((emb[got] - q) ** 2).sum(1))[:k]].tolist(), work
        score("pathfinder", pathfinder)
        row["brute"] = {"recall": 1.0, "work": int(len(rows))}

        for key in ("ivf-fixed", "navix/hnsw-d32"):
            rho_rows += [(key, float(x), float(y)) for x, y in zip(rho, rec[key])]
        log(**row)
        results.append(row)

    def spearman(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    gls = {}
    for key in ("ivf-fixed", "navix/hnsw-d32"):
        pts = np.array([(x, y) for k2, x, y in rho_rows if k2 == key])
        gls[key] = round(spearman(pts[:, 0], pts[:, 1]), 3)
    log(gls_rho_vs_recall_spearman=gls)
    with open(a.output, "w") as f:
        json.dump({"config": vars(a), "sieve_chosen": chosen[1:], "gls_spearman": gls, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
