"""Real LanceDB reference for native/src/ivf_graph_bench.cpp (IVF_GRAPH.md).

Builds Lance's IVF_HNSW_FLAT (k-means partitions, one HNSW per partition, m 20 and
ef_construction 150 as Lance's defaults) on the same vectors, with Lance's HNSW over the
centroids for assignment either forced on or off. The switch is the process-wide environment
variable LANCE_USE_HNSW_SPEEDUP_INDEXING, read once, so each mode runs in its own process:

  uv run --locked --extra lance python lance_ivf_graph_check.py --assign-graph disabled
  uv run --locked --extra lance python lance_ivf_graph_check.py --assign-graph enabled

Latency here includes Python and Lance's query planning, so only recall and build time
compare directly with the native numbers.
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="native/results/synthetic1m")
ap.add_argument("--work", default="data/lance-ivf-graph")
ap.add_argument("--assign-graph", choices=["enabled", "disabled"], required=True)
ap.add_argument("--partitions", type=int, nargs="+", default=[64, 1024])
ap.add_argument("--nprobes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
ap.add_argument("--efs", type=int, nargs="+", default=[10, 20, 40, 80, 160])
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--queries", type=int, default=300)
ap.add_argument("--output", default="native/results/ivf-graph-synthetic1m/lance.jsonl")
a = ap.parse_args()
os.environ["LANCE_USE_HNSW_SPEEDUP_INDEXING"] = a.assign_graph  # before lance is imported

import json  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import lance  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402


def read_bin(path, dtype):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d)


base = read_bin(Path(a.data) / "base.u8bin", np.uint8).astype(np.float32)
queries = read_bin(Path(a.data) / "query.u8bin", np.uint8).astype(np.float32)[: a.queries]
gt = read_bin(Path(a.data) / "gt.ibin", np.uint32)[: a.queries, : a.k]
n, d = base.shape
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
out = open(a.output, "a")


def record(row):
    row.update(dataset=Path(a.data).name, n=n, dim=d, k=a.k, queries=len(queries), index="IVF_HNSW_FLAT",
               m=20, ef_construction=150, assign_graph=a.assign_graph, lance=lance.__version__)
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n")
    out.flush()


for parts in a.partitions:
    uri = Path(a.work) / f"p{parts}-{a.assign_graph}.lance"
    shutil.rmtree(uri, ignore_errors=True)
    table = pa.table({"id": pa.array(np.arange(n, dtype=np.uint32)),
                      "vec": pa.FixedSizeListArray.from_arrays(pa.array(base.ravel()), d)})
    ds = lance.write_dataset(table, uri)
    t = time.perf_counter()
    ds.create_index("vec", "IVF_HNSW_FLAT", num_partitions=parts, m=20, ef_construction=150)
    record({"kind": "build", "partitions": parts, "build_seconds": time.perf_counter() - t})
    ds = lance.dataset(uri)
    for nprobes in [p for p in a.nprobes if p <= parts]:
        for ef in a.efs:
            nearest = {"column": "vec", "k": a.k, "ef": ef, "nprobes": nprobes}
            ds.to_table(columns=["id"], nearest={**nearest, "q": queries[0]})  # warm-up
            hits, lat = 0, []
            for qi, q in enumerate(queries):
                t = time.perf_counter()
                res = ds.to_table(columns=["id"], nearest={**nearest, "q": q})
                lat.append((time.perf_counter() - t) * 1e3)
                hits += len(set(res.column("id").to_numpy().tolist()) & set(gt[qi].tolist()))
            record({"kind": "query", "partitions": parts, "nprobe": nprobes, "ef": ef,
                    "recall": hits / (len(queries) * a.k), "latency_ms": float(np.mean(lat))})
    shutil.rmtree(uri, ignore_errors=True)
sys.exit(0)
