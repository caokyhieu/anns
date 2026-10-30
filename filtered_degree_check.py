"""Does a denser base graph rescue 2-hop (ACORN-style) filtered search at 1% selectivity?

Same data as filtered_graph_experiment.py; HNSW with M = 16 and M = 32 (base-layer degree
32 and 64), 20 queries, k = 100, 8 passing entry points. Checks the NaviX / ACORN claim
that 2-hop expansion needs a dense enough graph.
"""
import numpy as np
import hnswlib
from filtered_search_experiment import TAGS, make_table, make_queries
from filtered_graph_experiment import base_layer, search
rng = np.random.default_rng(7)
n, dim = 200_000, 64
table, centers, cid, emb, tc = make_table(rng, n, dim, 400)
tags = table.column("tags").to_pylist()
masks = {t: np.array([t in ts for ts in tags]) for t in ("pedestrian", "cut_in")}
q = make_queries(rng, centers, np.arange(400), 20)
for M in (16, 32):
    idx = hnswlib.Index("l2", dim); idx.init_index(n, M=M, ef_construction=200, random_seed=7)
    idx.add_items(emb, np.arange(n)); g = base_layer(idx, n)
    for t, mask in masks.items():
        rows = np.flatnonzero(mask)
        truths = [set(rows[np.argsort(((emb[rows]-x)**2).sum(1))[:100]].tolist()) for x in q]
        for beam in (200, 800, 2000):
            r, w = [], []
            for x, tr in zip(q, truths):
                got, work = search(g, emb, x, mask, 100, beam, "acorn", rng.choice(rows, 8, replace=False))
                r.append(len(tr & set(got)) / 100); w.append(work)
            print(f"M={M} (base degree {2*M}) {t} beam={beam}: acorn recall {np.mean(r):.3f} work {int(np.mean(w))}", flush=True)
