"""Readable mutable PiPNN prototype: L2 distance, stable integer IDs."""
from collections import Counter
from dataclasses import dataclass, field
import numpy as np


def distances(a, b):
    """Squared L2 via dense matrix multiplication."""
    return np.maximum(0., (a*a).sum(1)[:, None] + (b*b).sum(1) - 2*a@b.T)


@dataclass(eq=False)
class Node:
    leaders: object = None
    children: list = field(default_factory=list)
    fanout: int = 1
    members: set = field(default_factory=set)
    local: dict = field(default_factory=dict)
    edges: set = field(default_factory=set)


class PiPNN:
    def __init__(self, leaf_size=64, local_k=12, degree=24, reservoir_size=48,
                 hash_bits=12, fanout=(3, 2), leader_fraction=.08,
                 max_leaders=64, alpha=1.2, seed=7):
        for name, value in dict(leaf_size=leaf_size, local_k=local_k, degree=degree,
                                reservoir_size=reservoir_size, hash_bits=hash_bits,
                                max_leaders=max_leaders).items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if leaf_size < 2 or max_leaders < 2 or not 0 < leader_fraction <= 1:
            raise ValueError("invalid partition parameters")
        if not fanout or any(not isinstance(f, int) or f < 1 for f in fanout):
            raise ValueError("fanout must contain positive integers")
        if not np.isfinite(alpha) or alpha < 1:
            raise ValueError("alpha must be finite and >= 1")
        self.config = dict(leaf_size=leaf_size, local_k=local_k, degree=degree,
                           reservoir_size=reservoir_size, hash_bits=hash_bits,
                           fanout=tuple(fanout), leader_fraction=leader_fraction,
                           max_leaders=max_leaders, alpha=alpha, seed=seed)
        self.__dict__.update(self.config)
        self.vectors, self.dim, self.root = {}, None, None
        self.next_id = self.mutations = 0

    def _vector(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("expected a nonempty finite vector")
        if self.dim is not None and len(x) != self.dim:
            raise ValueError("dimension mismatch")
        return x.copy()

    def build(self, vectors, ids=None):
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError("expected a finite (n, dimension) array")
        ids = list(range(len(x))) if ids is None else list(ids)
        if len(ids) != len(x) or len(set(ids)) != len(ids) or any(
                not isinstance(i, (int, np.integer)) or i < 0 for i in ids):
            raise ValueError("IDs must be unique nonnegative integers")
        self.dim = x.shape[1]
        self.vectors = {int(i): v.copy() for i, v in zip(ids, x)}
        self.next_id = max(ids, default=-1) + 1
        self.rng = np.random.default_rng(self.seed)
        self.planes = self.rng.normal(size=(self.hash_bits, self.dim))
        self.sketches = {i: self.planes@v for i, v in self.vectors.items()}
        self.leaves = []
        self.memberships = {i: [] for i in ids}
        self.candidates = {i: Counter() for i in ids}
        self.reservoirs = {i: {} for i in ids}
        self.base = {i: set() for i in ids}
        self.incoming = {i: set() for i in ids}
        self.graph = {i: set() for i in ids}
        self.root = self._partition(sorted(ids), 0)
        for leaf in self.leaves:
            self._local_rebuild(leaf)
        self._sync(self.leaves)
        self.mutations = 0
        return self

    def _partition(self, ids, depth):
        node = Node(members=set(ids))
        if len(ids) > self.leaf_size and depth < 32:
            f = self.fanout[depth] if depth < len(self.fanout) else 1
            count = min(len(ids), self.max_leaders,
                        max(f+1, int(np.ceil(len(ids)*self.leader_fraction))))
            leaders = self.rng.choice(ids, size=count, replace=False)
            anchors = np.array([self.vectors[i] for i in leaders])
            x = np.array([self.vectors[i] for i in ids])
            f = min(f, count-1)
            routes = np.argsort(distances(x, anchors), axis=1, kind="stable")[:, :f]
            groups = [[i for i, row in zip(ids, routes) if j in row] for j in range(count)]
            # Unsplittable duplicates form an oversized leaf rather than infinite recursion.
            if max(map(len, groups)) < len(ids):
                node.members.clear()
                node.leaders, node.fanout = anchors.copy(), f
                node.children = [self._partition(g, depth+1) for g in groups]
                return node
        self.leaves.append(node)
        for i in ids:
            self.memberships[i].append(node)
        return node

    def _route(self, x, node=None):
        node = self.root if node is None else node
        if node.leaders is None:
            return [node]
        order = np.argsort(distances(x[None], node.leaders)[0], kind="stable")
        return [leaf for j in order[:node.fanout]
                for leaf in self._route(x, node.children[j])]

    def _d(self, p, q):
        delta = self.vectors[p] - self.vectors[q]
        return float(delta@delta)

    def _rank(self, p, ids):
        return sorted(ids, key=lambda q: (self._d(p, q), q))

    def _local_rebuild(self, leaf):
        ids = sorted(leaf.members)
        leaf.local = {}
        if not ids:
            return
        x = np.array([self.vectors[i] for i in ids])
        d = distances(x, x)
        np.fill_diagonal(d, np.inf)
        for row, p in enumerate(ids):
            order = np.argsort(d[row], kind="stable")[:min(self.local_k, len(ids)-1)]
            leaf.local[p] = {ids[j] for j in order}

    def _local_insert(self, leaf, p):
        ids = sorted(leaf.members)
        if ids:
            d = distances(self.vectors[p][None], np.array([self.vectors[q] for q in ids]))[0]
            order = np.argsort(d, kind="stable")[:self.local_k]
            leaf.local[p] = {ids[j] for j in order}
            for q in ids:
                leaf.local[q] = set(self._rank(q, leaf.local[q] | {p})[:self.local_k])
        else:
            leaf.local[p] = set()
        leaf.members.add(p)

    def _hash(self, p, q):
        return np.packbits(self.sketches[q] >= self.sketches[p]).tobytes()

    def _stream(self, p, q):
        reservoir = self.reservoirs[p]
        h = self._hash(p, q)
        key = lambda i: (self._d(p, i), i)
        if h in reservoir:
            if key(q) < key(reservoir[h]):
                reservoir[h] = q
        elif len(reservoir) < self.reservoir_size:
            reservoir[h] = q
        else:
            worst = max(reservoir, key=lambda bucket: key(reservoir[bucket]))
            if key(q) < key(reservoir[worst]):
                del reservoir[worst]
                reservoir[h] = q

    def _prune(self, p, candidates):
        selected = []
        for q in self._rank(p, candidates):
            # alpha squared implements Euclidean RobustPrune with squared distances.
            if all(self.alpha**2 * self._d(q, r) >= self._d(p, q) for r in selected):
                selected.append(q)
                if len(selected) == self.degree:
                    break
        return set(selected)

    def _sync(self, leaves, force=()):
        """Apply reference-counted leaf edge deltas and repair dependent nodes."""
        delta = Counter()
        for leaf in leaves:
            edges = {(p, q) for p, ns in leaf.local.items() for q in ns}
            edges |= {(q, p) for p, q in edges}
            delta.update(edges - leaf.edges)
            delta.subtract(leaf.edges - edges)
            leaf.edges = edges
        changed, removed, added = set(force), set(force), {}
        for (p, q), amount in delta.items():
            if not amount:
                continue
            old = self.candidates[p][q]
            new = old + amount
            assert new >= 0
            if new:
                self.candidates[p][q] = new
            else:
                self.candidates[p].pop(q, None)
            if old == 0 and new:
                added.setdefault(p, set()).add(q)
                changed.add(p)
            if old and new == 0:
                removed.add(p)
                changed.add(p)
        for p in changed:
            if p in removed:
                self.reservoirs[p] = {}
                stream = self.candidates[p]
            else:
                stream = added.get(p, ())
            for q in stream:
                self._stream(p, q)
        affected = set(changed)
        for p in changed:
            new, old = self._prune(p, self.reservoirs[p].values()), self.base[p]
            for q in old - new:
                self.incoming[q].discard(p)
            for q in new - old:
                self.incoming[q].add(p)
            affected.update(old ^ new)
            self.base[p] = new
        for p in affected:
            self.graph[p] = self._prune(p, self.base[p] | self.incoming[p])
        self.last_repair = dict(leaves=len(leaves), reservoirs=len(changed), graph_nodes=len(affected))

    def add(self, vector, id=None):
        x = self._vector(vector)
        if self.root is None:
            self.build(np.empty((0, len(x))))
        p = self.next_id if id is None else id
        if not isinstance(p, (int, np.integer)) or p < 0 or p in self.vectors:
            raise ValueError("ID must be a new nonnegative integer")
        p = int(p)
        leaves = self._route(x)
        self.vectors[p], self.sketches[p] = x, self.planes@x
        self.memberships[p] = leaves
        self.candidates[p], self.reservoirs[p] = Counter(), {}
        self.base[p], self.incoming[p], self.graph[p] = set(), set(), set()
        for leaf in leaves:
            self._local_insert(leaf, p)
        self._sync(leaves, force=(p,))
        self.next_id = max(self.next_id, p+1)
        self.mutations += 1
        return p

    def delete(self, p):
        if p not in self.vectors:
            raise KeyError(p)
        leaves = self.memberships[p]
        for leaf in leaves:
            leaf.members.remove(p)
            self._local_rebuild(leaf)
        # Retain the vector until all incoming references have been repaired.
        self._sync(leaves, force=(p,))
        for mapping in (self.vectors, self.sketches, self.memberships, self.candidates,
                        self.reservoirs, self.base, self.incoming, self.graph):
            del mapping[p]
        self.mutations += 1

    def update(self, p, vector):
        if p not in self.vectors:
            raise KeyError(p)
        x = self._vector(vector)
        self.delete(p)
        self.add(x, id=p)
        self.mutations -= 1

    def rebuild(self):
        if self.dim is None:
            return self
        next_id, ids = self.next_id, sorted(self.vectors)
        x = np.array([self.vectors[i] for i in ids]).reshape(-1, self.dim)
        self.build(x, ids)
        self.next_id = next_id
        return self

    def search(self, query, k=10, beam=64):
        x = self._vector(query)
        if k < 1 or beam < k:
            raise ValueError("require beam >= k >= 1")
        if not self.vectors:
            self.last_search_evaluations = 0
            return []
        # One live representative per routed leaf; no leaf vector scans.
        seeds = {min(leaf.members) for leaf in self._route(x) if leaf.members}
        seeds.add(min(self.vectors))
        cache = {}
        def key(p):
            if p not in cache:
                d = self.vectors[p]-x
                cache[p] = float(d@d)
            return cache[p], p
        current = set(sorted(seeds, key=key)[:beam])
        visited = set()
        while current - visited:
            p = min(current - visited, key=key)
            visited.add(p)
            current = set(sorted(current | self.graph[p], key=key)[:beam])
        self.last_search_evaluations = len(cache)
        return sorted(current, key=key)[:k]

    def exact(self, query, k=10):
        x = self._vector(query)
        return sorted(self.vectors, key=lambda p: (float(np.sum((self.vectors[p]-x)**2)), p))[:k]

    def monitor(self, queries, k=10, beam=64, min_recall=.9, rebuild=False,
                max_mutations=None):
        queries = list(queries)
        if not queries or not self.vectors:
            raise ValueError("monitor requires queries and live vectors")
        def measure():
            scores, evaluations = [], []
            for q in queries:
                truth = set(self.exact(q, k))
                found = set(self.search(q, k, beam))
                scores.append(len(truth & found)/len(truth))
                evaluations.append(self.last_search_evaluations)
            return float(np.mean(scores)), float(np.mean(evaluations))
        recall, evaluations = measure()
        due = recall < min_recall or (max_mutations is not None and self.mutations >= max_mutations)
        report = dict(recall=recall, mean_distance_evaluations=evaluations,
                      rebuild_recommended=bool(due), rebuilt=False)
        if due and rebuild:
            self.rebuild()
            report["rebuilt"] = True
            report["recall_after_rebuild"], _ = measure()
        return report

    def validate(self):
        """Expensive reconstruction of state invariants for tests/experiments."""
        live = set(self.vectors)
        expected = {p: Counter() for p in live}
        for leaf in self.leaves:
            assert leaf.members <= live
            assert set(leaf.local) == leaf.members
            for p in leaf.members:
                assert leaf.local[p] == set(self._rank(p, leaf.members-{p})[:self.local_k])
                assert leaf in self.memberships[p]
            edges = {(p, q) for p, ns in leaf.local.items() for q in ns}
            edges |= {(q, p) for p, q in edges}
            assert leaf.edges == edges
            for p, q in edges:
                expected[p][q] += 1
        for p in live:
            assert self.candidates[p] == expected[p]
            winners = {}
            for q in self._rank(p, expected[p]):
                winners.setdefault(self._hash(p, q), q)
            best = set(self._rank(p, winners.values())[:self.reservoir_size])
            assert set(self.reservoirs[p].values()) == best
            assert self.base[p] == self._prune(p, best)
            assert self.incoming[p] == {q for q in live if p in self.base[q]}
            assert self.graph[p] == self._prune(p, self.base[p] | self.incoming[p])
            assert self.graph[p] <= live-{p} and len(self.graph[p]) <= self.degree
        return True
