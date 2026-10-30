"""Hybrid (label filter + vector) search on Lance: how recall and latency depend on the filter.

Synthetic AV-style table: clustered 64-d embeddings, a weather column (BITMAP) and a
scenario-tag list (LABEL_LIST). Tags are either independent of the embedding
("uncorrelated") or confined to a few embedding clusters ("correlated", the realistic
case: rows tagged `tunnel` look alike). Each filter is run against queries drawn from
the whole distribution; for the correlated tag we also run queries drawn from inside its
clusters (filter agrees with the query) and outside them (filter disagrees: the probed
IVF partitions hold no passing rows).

Strategies, all measured against exact filtered search computed in NumPy:
  post        filter after ANN (prefilter=False), fixed nprobes
  pre_fixed   prefilter, fixed nprobes
  pre_expand  prefilter, minimum_nprobes = base, maximum_nprobes = all partitions
  flat        prefilter + exact scan over the passing rows (use_index=False)
  planner     count passing rows with the scalar indexes; flat below a threshold,
              otherwise pre_expand with nprobes scaled by 1 / selectivity
"""
import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
import pyarrow as pa
import lance

WEATHER = ["clear", "rain", "fog", "snow", "night"]
# tag -> fraction of rows carrying it; None marks the cluster-correlated tag
TAGS = {"pedestrian": 0.10, "cut_in": 0.01, "emergency_vehicle": 0.001, "tunnel": None}


def make_table(rng, n, dim, clusters):
    centers = rng.normal(size=(clusters, dim)).astype(np.float32) * 4
    cid = rng.integers(0, clusters, n)
    emb = centers[cid] + rng.normal(size=(n, dim)).astype(np.float32)
    weather = rng.choice(WEATHER, n)
    tunnel_clusters = rng.choice(clusters, max(1, clusters // 100), replace=False)
    tags = [[] for _ in range(n)]
    for tag, frac in TAGS.items():
        rows = np.flatnonzero(np.isin(cid, tunnel_clusters)) if frac is None else np.flatnonzero(rng.random(n) < frac)
        for r in rows:
            tags[r].append(tag)
    table = pa.table({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "weather": pa.array(weather),
        "tags": pa.array(tags, type=pa.list_(pa.string())),
        "emb": pa.FixedSizeListArray.from_arrays(pa.array(emb.ravel()), dim),
    })
    return table, centers, cid, emb, tunnel_clusters


def make_queries(rng, centers, from_clusters, count):
    pick = rng.choice(from_clusters, count)
    return (centers[pick] + rng.normal(size=(count, centers.shape[1])).astype(np.float32)).astype(np.float32)


def run(ds, q, flt, k, strategy, base, partitions, flat_rows, n):
    near = {"column": "emb", "q": q, "k": k}
    prefilter = True
    if strategy == "post":
        near.update(minimum_nprobes=base, maximum_nprobes=base)
        prefilter = False
    elif strategy == "pre_fixed":
        near.update(minimum_nprobes=base, maximum_nprobes=base)
    elif strategy == "pre_expand":
        near.update(minimum_nprobes=base, maximum_nprobes=partitions)
    elif strategy == "flat":
        near["use_index"] = False
    elif strategy == "planner":
        passing = ds.count_rows(filter=flt)
        if passing <= flat_rows:
            near["use_index"] = False
        else:
            probes = min(partitions, int(np.ceil(base * n / passing)))
            near.update(minimum_nprobes=probes, maximum_nprobes=partitions)
    return ds.to_table(columns=["id", "_distance"], nearest=near, filter=flt, prefilter=prefilter).column("id").to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="lance-filtered")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--clusters", type=int, default=400)
    ap.add_argument("--partitions", type=int, default=256)
    ap.add_argument("--nprobes", type=int, default=16)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--flat-rows", type=int, default=10_000, help="planner: exact scan at or below this many passing rows")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="filtered-search-results.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    root = Path(a.root)
    shutil.rmtree(root, ignore_errors=True)
    table, centers, cid, emb, tunnel_clusters = make_table(rng, a.n, a.dim, a.clusters)
    ds = lance.write_dataset(table, str(root / "segments.lance"), max_rows_per_file=a.n // 8)
    t = perf_counter()
    ds.create_index("emb", "IVF_FLAT", num_partitions=a.partitions)
    ds.create_scalar_index("weather", "BITMAP")
    ds.create_scalar_index("tags", "LABEL_LIST")
    print(json.dumps({"stage": "index", "s": round(perf_counter() - t, 1)}), flush=True)

    weather = table.column("weather").to_numpy(zero_copy_only=False)
    tags = table.column("tags").to_pylist()
    has = {tag: np.array([tag in ts for ts in tags]) for tag in TAGS}
    others = np.setdiff1d(np.arange(a.clusters), tunnel_clusters)
    q_all = make_queries(rng, centers, np.arange(a.clusters), a.queries)
    q_in = make_queries(rng, centers, tunnel_clusters, a.queries)
    q_out = make_queries(rng, centers, others, a.queries)

    cases = [
        ("none", None, np.ones(a.n, bool), q_all, "uncorrelated"),
        ("weather = 'rain'", "weather = 'rain'", weather == "rain", q_all, "uncorrelated"),
        ("pedestrian (10%)", "array_has_any(tags, ['pedestrian'])", has["pedestrian"], q_all, "uncorrelated"),
        ("cut_in (1%)", "array_has_any(tags, ['cut_in'])", has["cut_in"], q_all, "uncorrelated"),
        ("rain AND cut_in", "weather = 'rain' AND array_has_any(tags, ['cut_in'])", (weather == "rain") & has["cut_in"], q_all, "uncorrelated"),
        ("emergency_vehicle (0.1%)", "array_has_any(tags, ['emergency_vehicle'])", has["emergency_vehicle"], q_all, "uncorrelated"),
        ("tunnel, query inside", "array_has_any(tags, ['tunnel'])", has["tunnel"], q_in, "correlated"),
        ("tunnel, query outside", "array_has_any(tags, ['tunnel'])", has["tunnel"], q_out, "anti-correlated"),
    ]
    strategies = ["post", "pre_fixed", "pre_expand", "flat", "planner"]
    results = []
    for name, flt, mask, queries, relation in cases:
        rows = np.flatnonzero(mask)
        row = {"filter": name, "relation": relation, "selectivity": round(len(rows) / a.n, 5), "passing_rows": int(len(rows))}
        for strategy in strategies:
            if flt is None and strategy in ("post", "planner"):
                continue
            recalls, counts, times = [], [], []
            for q in queries:
                d = ((emb[rows] - q) ** 2).sum(1)
                truth = set(rows[np.argsort(d)[:a.k]].tolist())
                t = perf_counter()
                got = run(ds, q, flt, a.k, strategy, a.nprobes, a.partitions, a.flat_rows, a.n)
                times.append(perf_counter() - t)
                counts.append(len(got))
                recalls.append(len(truth & set(got.tolist())) / len(truth))
            row[strategy] = {"recall": round(float(np.mean(recalls)), 3), "returned": round(float(np.mean(counts)), 1),
                             "ms": round(1000 * float(np.median(times)), 1)}
        print(json.dumps(row), flush=True)
        results.append(row)
    report = {"config": vars(a), "results": results}
    Path(a.output).write_text(json.dumps(report, indent=1))
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
