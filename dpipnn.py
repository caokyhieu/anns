"""Simulated distributed PiPNN: sharded construction plus serving cost accounting.

One process plays every shard; each shard sees only its own vectors and the edge
records sent to it, and every cross-shard byte is counted. See DISTRIBUTED.md.

Build (MapReduce shaped, one global graph):
  1. route:   sample centroids, assign each vector a balanced primary shard and
              closure replicas (SPANN-style rule) on nearby shards;
  2. map:     each shard runs local PiPNN partitioning and leaf k-NN on its
              primaries plus replicas, emitting candidate edges (p, q);
  3. combine: each shard HashPrunes its edges per source (a partial reservoir);
  4. shuffle: partial reservoirs go to the shard owning the source;
  5. reduce:  the owner merges partial reservoirs and RobustPrunes to `degree`.
HashPrune keeps, per source, the nearest candidate per hash bucket and then the
`reservoir` nearest buckets under a total (distance, ID) order, so it is exactly
mergeable: hashprune(A + B) == hashprune(hashprune(A) + hashprune(B)).
"""
from dataclasses import dataclass, field
import numpy as np


def sqdist(a, b):
    """Squared L2 between rows of a and b via one matrix product."""
    return np.maximum(0., (a*a).sum(1)[:, None] + (b*b).sum(1)[None] - 2*a@b.T)


def nearest(x, centers, count, chunk=8192):
    """Indices and squared distances of the `count` nearest centers per row, sorted."""
    count = min(count, len(centers))
    idx = np.empty((len(x), count), np.int64)
    dist = np.empty((len(x), count))
    for s in range(0, len(x), chunk):
        d = sqdist(x[s:s+chunk], centers)
        part = np.argpartition(d, count-1, axis=1)[:, :count] if count < len(centers) \
            else np.broadcast_to(np.arange(len(centers)), d.shape)
        pd = np.take_along_axis(d, part, 1)
        order = np.argsort(pd, axis=1, kind="stable")
        idx[s:s+chunk] = np.take_along_axis(part, order, 1)
        dist[s:s+chunk] = np.take_along_axis(pd, order, 1)
    return idx, dist


def kmeans(x, k, rng, iters=8, sample=50000):
    """Plain Lloyd iterations on a sample; stands in for distributed k-means."""
    pts = x[rng.choice(len(x), min(sample, len(x)), replace=False)]
    centers = pts[rng.choice(len(pts), k, replace=False)].copy()
    for _ in range(iters):
        assign = nearest(pts, centers, 1)[0][:, 0]
        sums = np.zeros_like(centers)
        np.add.at(sums, assign, pts)
        counts = np.bincount(assign, minlength=k)
        moved = counts > 0
        centers[moved] = sums[moved] / counts[moved, None]
    return centers


def closure(order, dist, eps, max_replicas, centers=None, rng_rule=False):
    """Boolean (n, c) mask of extra assignments within (1+eps) of the nearest distance.

    `order`/`dist` are sorted nearest-center lists (squared distances). Column 0 is
    always kept. With rng_rule, skip center j if some kept center i is closer to j
    than the vector is (SPANN's RNG rule, which avoids near-duplicate postings).
    """
    root = np.sqrt(dist)
    keep = root <= (1 + eps) * root[:, :1]
    keep[:, 0] = True
    keep &= np.cumsum(keep, axis=1) <= max_replicas
    if rng_rule and centers is not None:
        for j in range(1, order.shape[1]):
            rows = np.flatnonzero(keep[:, j])
            for i in range(j):
                cc = sqdist_rows(centers[order[rows, i]], centers[order[rows, j]])
                bad = keep[rows, i] & (cc < dist[rows, j])
                keep[rows[bad], j] = False
    return keep


def sqdist_rows(a, b):
    d = a - b
    return np.einsum("ij,ij->i", d, d)


# ---- HashPrune as an exactly mergeable reduction ------------------------------------

def hash_codes(sketch, src, dst, chunk=1 << 20):
    """Bucket of q - p under sign random projections (sketches are linear)."""
    weights = (2. ** np.arange(sketch.shape[1])).astype(np.float32)
    out = np.empty(len(src), np.int64)
    for s in range(0, len(src), chunk):
        bits = np.greater_equal(np.take(sketch, dst[s:s+chunk], 0), np.take(sketch, src[s:s+chunk], 0))
        bits = bits.astype(np.float32)
        out[s:s+chunk] = bits @ weights
    return out


def hashprune(src, dst, dist, sketch, capacity):
    """Exact HashPrune reservoirs for a batch of candidate edges (any order, duplicates ok)."""
    if len(src) == 0:
        return src, dst, dist
    h = hash_codes(sketch, src, dst)
    order = np.lexsort((dst, dist, src))
    src, dst, dist, h = src[order], dst[order], dist[order], h[order]
    # First (nearest) candidate of each (source, bucket); sources stay contiguous.
    _, first = np.unique((src << sketch.shape[1]) | h, return_index=True)
    first.sort()
    src, dst, dist = src[first], dst[first], dist[first]
    starts = np.r_[0, np.flatnonzero(src[1:] != src[:-1]) + 1]
    rank = np.arange(len(src)) - np.repeat(starts, np.diff(np.r_[starts, len(src)]))
    keep = rank < capacity
    return src[keep], dst[keep], dist[keep]


# ---- Local (single-shard) PiPNN ----------------------------------------------------

@dataclass
class Params:
    leaf_size: int = 256
    local_k: int = 4
    fanout: tuple = (4, 2)        # intra-shard fanout per depth; deeper levels use 1
    leaders_per_leaf: float = 2.  # leaders per level ~ leaders_per_leaf * size / leaf_size
    max_leaders: int = 1000
    max_depth: int = 12
    hash_bits: int = 12
    reservoir: int = 64
    degree: int = 32
    alpha: float = 1.2            # applied to squared distances, like fastpipnn
    seed: int = 7


def partition(x, ids, rng, p, depth=0):
    """Randomized ball carving: route to `fanout` nearest of sampled leaders, recurse."""
    if len(ids) <= p.leaf_size:
        return [ids]
    if depth >= p.max_depth:
        return np.array_split(rng.permutation(ids), -(-len(ids) // p.leaf_size))
    f = p.fanout[depth] if depth < len(p.fanout) else 1
    m = int(min(p.max_leaders, len(ids), max(f + 1, np.ceil(p.leaders_per_leaf * len(ids) / p.leaf_size))))
    f = min(f, m - 1)
    leaders = x[rng.choice(ids, m, replace=False)]
    route = nearest(x[ids], leaders, f)[0]
    member = np.repeat(ids, f)
    group = route.ravel()
    order = np.argsort(group, kind="stable")
    member, group = member[order], group[order]
    cuts = np.flatnonzero(np.diff(group)) + 1
    leaves = []
    for g in np.split(member, cuts):
        if len(g) == len(ids):  # duplicates: split randomly instead of recursing forever
            leaves += np.array_split(rng.permutation(g), -(-len(g) // p.leaf_size))
        else:
            leaves += partition(x, g, rng, p, depth + 1)
    return leaves


def leaf_edges(x, leaves, k):
    """Bidirected exact k-NN candidate edges (src, dst, dist) inside each leaf, one GEMM per leaf.

    Duplicate pairs from overlapping leaves are kept; HashPrune collapses them.
    """
    src, dst, dist = [], [], []
    for leaf in leaves:
        if len(leaf) < 2:
            continue
        d = sqdist(x[leaf], x[leaf])
        np.fill_diagonal(d, np.inf)
        kk = min(k, len(leaf) - 1)
        nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        a, b = np.repeat(leaf, kk), leaf[nn.ravel()]
        w = np.take_along_axis(d, nn, 1).ravel()
        src += [a, b]
        dst += [b, a]
        dist += [w, w]
    if not src:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    return np.concatenate(src), np.concatenate(dst), np.concatenate(dist)


def robust_prune(x, src, dst, dist, n, degree, alpha, chunk=4096):
    """RobustPrune every source's candidates (vectorized across sources). Returns CSR."""
    order = np.lexsort((dst, dist, src))
    src, dst, dist = src[order], dst[order], dist[order]
    counts = np.bincount(src, minlength=n)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    width = int(counts.max(initial=0))
    out = np.full((n, degree), -1, np.int64)
    if width:
        col = np.arange(len(src)) - starts[src]
        cand = np.full((n, width), -1, np.int64)
        cd = np.full((n, width), np.inf)
        cand[src, col], cd[src, col] = dst, dist
        for s in range(0, n, chunk):
            c, d = cand[s:s+chunk], cd[s:s+chunk]
            v = x[np.maximum(c, 0)]
            pair = np.maximum(0., (v*v).sum(2)[:, :, None] + (v*v).sum(2)[:, None]
                              - 2*np.einsum("bid,bjd->bij", v, v))
            chosen = np.zeros(c.shape, bool)
            taken = np.zeros(len(c), np.int64)
            for i in range(width):
                blocked = (chosen & (alpha * pair[:, i, :] <= d[:, i, None])).any(1)
                ok = (c[:, i] >= 0) & ~blocked & (taken < degree)
                chosen[ok, i] = True
                out[s + np.flatnonzero(ok), taken[ok]] = c[ok, i]
                taken += ok
    deg = (out >= 0).sum(1)
    indptr = np.r_[0, np.cumsum(deg)]
    return indptr, out[out >= 0]


# ---- Distributed build -------------------------------------------------------------

EDGE_BYTES = 8 + 8 + 2    # source ID, target ID, bfloat16 distance


@dataclass
class Cluster:
    """Sharded index: primary owner per vector plus build accounting."""
    owner: np.ndarray
    centers: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    stats: dict = field(default_factory=dict)

    def neighbors(self, p):
        return self.indices[self.indptr[p]:self.indptr[p+1]]


def assign_shards(x, shards, rng, slack=1.05, eps=0.0, max_replicas=1, fixed=None, candidates=8):
    """Balanced primary shard per vector plus boundary replicas.

    Primary: nearest centroid with spare capacity (cap = slack * n / shards).
    Replicas: other shards within (1+eps) of the nearest centroid distance, at most
    max_replicas memberships in total, or exactly the `fixed` nearest shards.
    """
    n = len(x)
    centers = kmeans(x, shards, rng) if shards > 1 else x.mean(0, keepdims=True)
    order, dist = nearest(x, centers, min(candidates, shards))
    cap = int(np.ceil(slack * n / shards))
    owner = np.full(n, -1, np.int64)
    load = np.zeros(shards, np.int64)
    for r in range(order.shape[1]):
        free = np.flatnonzero(owner < 0)
        if not len(free):
            break
        want = order[free, r]
        by = np.lexsort((dist[free, r], want))
        free, want = free[by], want[by]
        starts = np.r_[0, np.flatnonzero(np.diff(want)) + 1]
        rank = np.arange(len(want)) - np.repeat(starts, np.diff(np.r_[starts, len(want)]))
        ok = rank < cap - load[want]
        owner[free[ok]] = want[ok]
        load += np.bincount(want[ok], minlength=shards)
    left = np.flatnonzero(owner < 0)  # only possible if candidates < shards
    owner[left] = order[left, 0]
    extra = np.zeros(order.shape, bool)
    if fixed is not None:
        extra[:, :fixed] = True
    elif max_replicas > 1:
        extra = closure(order, dist, eps, max_replicas)
    rows, cols = np.nonzero(extra)
    shard_of = order[rows, cols]
    members = [np.unique(np.r_[np.flatnonzero(owner == s), rows[shard_of == s]]) for s in range(shards)]
    return owner, centers, members


def build(x, shards=8, p=None, eps=0.0, max_replicas=1, fixed=None, combine=True, seed=None):
    """Distributed PiPNN build over `shards` simulated machines; shards=1 is plain PiPNN."""
    p = p or Params()
    seed = p.seed if seed is None else seed
    n = len(x)
    rng = np.random.default_rng(seed)
    owner, centers, members = assign_shards(x, shards, rng, eps=eps, max_replicas=max_replicas, fixed=fixed)
    planes = rng.normal(size=(p.hash_bits, x.shape[1]))
    sketch = (x @ planes.T).astype(np.float32)  # every shard derives this from the shared seed
    stats = dict(shards=shards, eps=eps, max_replicas=max_replicas, fixed=fixed, combine=combine,
                 replication=sum(map(len, members)) / n,
                 shard_load_max_over_mean=float(np.bincount(owner, minlength=shards).max() * shards / n),
                 replica_vector_bytes=int((sum(map(len, members)) - n) * x.shape[1] * x.dtype.itemsize),
                 raw_edges=0, shuffled_edges=0, local_edges=0, leaves=0, leaf_points=0)
    parts = []
    for s in range(shards):
        srng = np.random.default_rng([seed, s])
        leaves = partition(x, members[s], srng, p)
        stats["leaves"] += len(leaves)
        stats["leaf_points"] += sum(map(len, leaves))
        a, b, d = leaf_edges(x, leaves, p.local_k)
        stats["raw_edges"] += len(a)
        if combine:
            a, b, d = hashprune(a, b, d, sketch, p.reservoir)
        remote = owner[a] != s
        stats["shuffled_edges"] += int(remote.sum())
        stats["local_edges"] += int((~remote).sum())
        parts.append((a, b, d))
    a, b, d = (np.concatenate(z) for z in zip(*parts))
    a, b, d = hashprune(a, b, d, sketch, p.reservoir)
    # RobustPrune needs every candidate's full vector at the source's owner.
    stats["prune_vector_fetches"] = int((owner[a] != owner[b]).sum())
    stats["shuffle_bytes"] = stats["shuffled_edges"] * EDGE_BYTES
    stats["mean_reservoir"] = len(a) / n
    indptr, indices = robust_prune(x, a, b, d, n, p.degree, p.alpha)
    stats["mean_degree"] = len(indices) / n
    stats["cross_shard_edge_fraction"] = float((owner[np.repeat(np.arange(n), np.diff(indptr))]
                                                != owner[indices]).mean()) if len(indices) else 0.
    return Cluster(owner, centers, indptr, indices, stats)


def build_partitioned(x, shards=8, p=None, seed=None):
    """Baseline: independent PiPNN graph per shard, no cross-shard edges or replicas."""
    return build(x, shards, p, eps=0.0, max_replicas=1, seed=seed)


# ---- Serving -----------------------------------------------------------------------

def beam_search(x, cluster, q, seeds, beam, k):
    """Best-first beam search; returns top-k IDs and the expansion (node read) sequence."""
    seeds = np.unique(seeds)
    known = dict(zip(seeds.tolist(), sqdist(q[None], x[seeds])[0].tolist()))
    pool = sorted((d, i) for i, d in known.items())[:beam]
    done, reads = set(), []
    while True:
        nxt = next((i for _, i in pool if i not in done), None)
        if nxt is None:
            break
        done.add(nxt)
        reads.append(nxt)
        nb = [j for j in cluster.neighbors(nxt).tolist() if j not in known]
        if nb:
            ds = sqdist(q[None], x[nb])[0]
            known.update(zip(nb, ds.tolist()))
            pool = sorted(pool + list(zip(ds.tolist(), nb)))[:beam]
    return [i for _, i in pool[:k]], reads, len(known)


class Head:
    """In-memory head index: a uniform sample searched exhaustively for entry points."""
    def __init__(self, x, fraction=0.01, seeds=4, seed=0):
        rng = np.random.default_rng(seed)
        self.ids = np.sort(rng.choice(len(x), max(64, int(fraction * len(x))), replace=False))
        self.x, self.seeds = x, seeds

    def entry(self, q, allowed=None):
        ids = self.ids if allowed is None else self.ids[allowed[self.ids]]
        d = sqdist(q[None], self.x[ids])[0]
        return ids[np.argsort(d, kind="stable")[:self.seeds]]


def search_global(x, cluster, head, q, beam, k=10):
    """DistributedANN-style: one global graph, entry from the head, reads go to owners."""
    found, reads, evals = beam_search(x, cluster, q, head.entry(q), beam, k)
    owners = cluster.owner[reads]
    return found, dict(reads=len(reads), distance_evals=evals, shards=len(set(owners.tolist())),
                       shard_switches=int((owners[1:] != owners[:-1]).sum()))


def search_routed(x, cluster, head, q, beam, probes, k=10):
    """Partition-and-route: search the `probes` nearest shards independently, merge."""
    order = np.argsort(sqdist(q[None], cluster.centers)[0], kind="stable")[:probes]
    found, reads, evals = [], 0, 0
    for s in order:
        f, r, e = beam_search(x, cluster, q, head.entry(q, cluster.owner == s), beam, k)
        found += f
        reads += len(r)
        evals += e
    found = sorted(set(found), key=lambda i: float(sqdist(q[None], x[[i]])[0, 0]))[:k]
    return found, dict(reads=reads, distance_evals=evals, shards=len(order), shard_switches=len(order) - 1)


# ---- SPANN-style inverted index (comparison) ---------------------------------------

class Spann:
    """Posting lists with closure assignment; centroids live in memory, postings on disk."""
    def __init__(self, x, posting=64, eps1=0.15, max_replicas=8, rng_rule=True, seed=0):
        rng = np.random.default_rng(seed)
        self.x = x
        self.centers = kmeans(x, max(2, len(x) // posting), rng, iters=6)
        order, dist = nearest(x, self.centers, max_replicas)
        keep = closure(order, dist, eps1, max_replicas, self.centers, rng_rule)
        rows, cols = np.nonzero(keep)
        lists = order[rows, cols]
        by = np.argsort(lists, kind="stable")
        self.members = rows[by]
        self.offsets = np.r_[0, np.cumsum(np.bincount(lists, minlength=len(self.centers)))]
        self.replication = len(rows) / len(x)

    def search(self, q, probes, eps2=0.6, k=10, page_vectors=32):
        d = sqdist(q[None], self.centers)[0]
        order = np.argsort(d, kind="stable")[:probes]
        root = np.sqrt(d[order])
        order = order[root <= (1 + eps2) * root[0]]
        ids = np.concatenate([self.members[self.offsets[c]:self.offsets[c+1]] for c in order])
        scanned = len(ids)
        ids = np.unique(ids)
        ds = sqdist(q[None], self.x[ids])[0]
        found = ids[np.argsort(ds, kind="stable")[:k]].tolist()
        sizes = np.diff(self.offsets)[order]
        return found, dict(lists=len(order), vectors_scanned=scanned,
                           pages=int(np.ceil(sizes / page_vectors).sum()))
