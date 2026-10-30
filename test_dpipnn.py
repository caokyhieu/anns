import unittest
import numpy as np
import dpipnn as D


def mixture(rng, n, d=8, clusters=12):
    centers = rng.normal(size=(clusters, d)) * 2
    return centers[rng.integers(0, clusters, n)] + rng.normal(size=(n, d))


class DistributedTests(unittest.TestCase):
    def test_hashprune_is_exactly_mergeable(self):
        rng = np.random.default_rng(3)
        n, edges = 300, 20000
        sketch = rng.normal(size=(n, 6)).astype(np.float32)
        src, dst = rng.integers(0, n, edges), rng.integers(0, n, edges)
        dist = rng.integers(0, 50, edges).astype(float)  # many ties exercise the ID order
        whole = D.hashprune(src, dst, dist, sketch, 7)
        part = rng.integers(0, 5, edges)
        pieces = [D.hashprune(src[part == s], dst[part == s], dist[part == s], sketch, 7) for s in range(5)]
        merged = D.hashprune(*(np.concatenate(z) for z in zip(*pieces)), sketch, 7)
        for a, b in zip(whole, merged):
            np.testing.assert_array_equal(a, b)
        self.assertLessEqual(np.bincount(whole[0]).max(), 7)

    def test_combiner_does_not_change_graph(self):
        x = mixture(np.random.default_rng(5), 3000)
        p = D.Params(leaf_size=64, degree=12, reservoir=24)
        a = D.build(x, 4, p, eps=0.1, max_replicas=3, combine=True)
        b = D.build(x, 4, p, eps=0.1, max_replicas=3, combine=False)
        np.testing.assert_array_equal(a.indptr, b.indptr)
        np.testing.assert_array_equal(a.indices, b.indices)
        self.assertLess(a.stats["shuffled_edges"], b.stats["shuffled_edges"])

    def test_graph_and_assignment_invariants(self):
        x = mixture(np.random.default_rng(9), 4000)
        p = D.Params(leaf_size=64, degree=10, reservoir=20)
        c = D.build(x, 5, p, eps=0.0, max_replicas=1)
        self.assertEqual(c.stats["replication"], 1.0)
        self.assertLessEqual(np.bincount(c.owner).max(), int(np.ceil(1.05 * len(x) / 5)))
        deg = np.diff(c.indptr)
        self.assertLessEqual(deg.max(), 10)
        src = np.repeat(np.arange(len(x)), deg)
        self.assertFalse((src == c.indices).any())
        # No replicas means no cross-shard edges: shards are independent graphs.
        self.assertEqual(c.stats["cross_shard_edge_fraction"], 0.0)
        r = D.build(x, 5, p, eps=0.2, max_replicas=3)
        self.assertGreater(r.stats["replication"], 1.0)
        self.assertGreater(r.stats["cross_shard_edge_fraction"], 0.0)

    def test_search_finds_neighbors(self):
        rng = np.random.default_rng(11)
        x = mixture(rng, 4000)
        q = mixture(rng, 30)
        truth = D.nearest(q, x, 10)[0]
        c = D.build(x, 4, D.Params(leaf_size=64, degree=16, reservoir=32), eps=0.1, max_replicas=3)
        head = D.Head(x)
        spann = D.Spann(x, posting=32)
        recall = lambda found, t: len(set(found) & set(t.tolist())) / 10
        g = np.mean([recall(D.search_global(x, c, head, qi, 64)[0], t) for qi, t in zip(q, truth)])
        r = np.mean([recall(D.search_routed(x, c, head, qi, 64, 4)[0], t) for qi, t in zip(q, truth)])
        s = np.mean([recall(spann.search(qi, 64, eps2=10)[0], t) for qi, t in zip(q, truth)])
        self.assertGreater(g, 0.9)
        self.assertGreater(r, 0.9)
        self.assertGreater(s, 0.9)


if __name__ == "__main__":
    unittest.main()
