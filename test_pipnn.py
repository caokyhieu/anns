import unittest
import numpy as np
from pipnn import PiPNN


class MutationTests(unittest.TestCase):
    def test_random_mutations_and_reconstruction(self):
        rng = np.random.default_rng(12)
        index = PiPNN(leaf_size=16, local_k=4, degree=8, reservoir_size=12).build(rng.normal(size=(70, 6)))
        self.assertTrue(index.validate())
        root = index.root
        expected = {p: v.copy() for p, v in index.vectors.items()}
        for step in range(36):
            p = int(rng.choice(sorted(expected)))
            x = rng.normal(size=6) + (4 if step % 5 == 0 else 0)
            if step % 3 == 0:
                new = index.add(x)
                expected[new] = x
            elif step % 3 == 1:
                index.update(p, x)
                expected[p] = x
            else:
                index.delete(p)
                del expected[p]
            self.assertIs(index.root, root)  # Mutations do not rebuild the tree.
            self.assertEqual(set(index.vectors), set(expected))
            for q, v in expected.items():
                np.testing.assert_array_equal(index.vectors[q], v)
            self.assertTrue(index.validate())
            self.assertTrue(set(index.search(x, k=5, beam=20)) <= set(expected))

    def test_duplicate_ties_empty_and_stable_ids(self):
        index = PiPNN(leaf_size=4, local_k=2, degree=3).build(np.zeros((12, 3)))
        self.assertTrue(index.validate())
        for p in range(12):
            index.delete(p)
            self.assertTrue(index.validate())
        self.assertEqual(index.search([0, 0, 0]), [])
        index.rebuild()
        self.assertEqual(index.add([0, 0, 0]), 12)
        self.assertEqual(index.search([0, 0, 0]), [12])
        index.update(12, [1, 1, 1])
        self.assertTrue(index.validate())

    def test_hashprune_order_and_deleted_winner(self):
        index = PiPNN(local_k=8, hash_bits=1, reservoir_size=2).build(
            np.array([[0., 0], [1, 0], [2, 0], [3, 0], [-1, 0], [-2, 0]]))
        expected = dict(index.reservoirs[0])
        rng = np.random.default_rng(2)
        for _ in range(15):
            index.reservoirs[0] = {}
            for q in rng.permutation([1, 2, 3, 4, 5]):
                index._stream(0, int(q))
            self.assertEqual(index.reservoirs[0], expected)
        index.delete(1)
        self.assertIn(2, index.reservoirs[0].values())
        self.assertTrue(index.validate())

    def test_recall_trigger_restores_damaged_graph(self):
        rng = np.random.default_rng(91)
        index = PiPNN().build(rng.normal(size=(40, 4)))
        for neighbors in index.graph.values():
            neighbors.clear()
        report = index.monitor(rng.normal(size=(10, 4)), min_recall=.99,
                               rebuild=True)
        self.assertLess(report["recall"], .99)
        self.assertTrue(report["rebuilt"])
        self.assertGreater(report["recall_after_rebuild"], report["recall"])
        self.assertTrue(index.validate())

    def test_invalid_update_does_not_remove_vector(self):
        index = PiPNN().build(np.eye(3))
        for x in ([1, 2], [np.nan, 0, 0]):
            with self.assertRaises(ValueError):
                index.update(0, x)
        with self.assertRaises(ValueError):
            index.add([0, 0, 0], id=0)
        with self.assertRaises(KeyError):
            index.delete(99)
        np.testing.assert_array_equal(index.vectors[0], [1, 0, 0])
        self.assertTrue(index.validate())

    def test_monitor_rebuild_and_query_oracle(self):
        rng = np.random.default_rng(1)
        index = PiPNN().build(rng.normal(size=(30, 4)))
        index.add(np.zeros(4))
        queries = rng.normal(size=(5, 4))
        report = index.monitor(queries, beam=64, max_mutations=1, rebuild=True)
        self.assertTrue(report["rebuilt"])
        self.assertEqual(index.mutations, 0)
        self.assertTrue(index.validate())
        for q in queries:
            self.assertEqual(index.search(q, k=5, beam=64), index.exact(q, 5))


if __name__ == "__main__":
    unittest.main()
