"""LanceDB HNSW on the same data as native/src/disk_bench.cpp.

Lance has no stand-alone paged HNSW: its HNSW indexes are IVF_HNSW_{SQ,PQ,FLAT}, i.e. IVF
partitions on disk, each holding an HNSW sub-graph that is loaded into the index cache when a
query first touches the partition. This script measures both regimes:

  warm  dataset opened once; partitions stay in the index cache after first use;
  cold  dataset reopened per query with a tiny index cache, so every query loads the
        partitions it probes from disk (bytes read via /proc/self/io rchar).

Usage:
  uv run --locked --extra lance python lance_hnsw_check.py --data native/results/synthetic1m \
      --output native/results/disk-synthetic1m/lance-hnsw.jsonl
"""
import argparse
import json
import shutil
import struct
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


def read_bin(path, dtype):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d)


def rchar():
    for line in open("/proc/self/io"):
        if line.startswith("rchar:"):
            return int(line.split()[1])
    return 0


def du(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="native/results/synthetic1m")
    ap.add_argument("--work", default="data/lance-hnsw")
    ap.add_argument("--index", default="IVF_HNSW_SQ", choices=["IVF_HNSW_SQ", "IVF_HNSW_PQ", "IVF_HNSW_FLAT"])
    ap.add_argument("--partitions", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--efs", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--nprobes", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--cold-queries", type=int, default=50)
    ap.add_argument("--output", default="native/results/disk-synthetic1m/lance-hnsw.jsonl")
    a = ap.parse_args()

    base = read_bin(Path(a.data) / "base.u8bin", np.uint8).astype(np.float32)
    queries = read_bin(Path(a.data) / "query.u8bin", np.uint8).astype(np.float32)[: a.queries]
    gt = read_bin(Path(a.data) / "gt.ibin", np.uint32)[: a.queries, : a.k]
    n, d = base.shape
    out = open(a.output, "a")

    def record(row):
        row.update(dataset=Path(a.data).name, n=n, dim=d, k=a.k, index=a.index, m=a.m, ef_construction=a.ef_construction)
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()

    def run(ds, q, ef, nprobes):
        return ds.to_table(columns=["id"], nearest={"column": "vec", "q": q, "k": a.k, "ef": ef, "nprobes": nprobes})

    for parts in a.partitions:
        uri = Path(a.work) / f"p{parts}.lance"
        shutil.rmtree(uri, ignore_errors=True)
        table = pa.table({"id": pa.array(np.arange(n, dtype=np.uint32)),
                          "vec": pa.FixedSizeListArray.from_arrays(pa.array(base.ravel()), d)})
        ds = lance.write_dataset(table, uri)
        data_bytes = du(uri)
        t = time.perf_counter()
        ds.create_index("vec", a.index, num_partitions=parts, m=a.m, ef_construction=a.ef_construction)
        build = time.perf_counter() - t
        index_bytes = du(uri) - data_bytes
        record({"kind": "build", "design": "lance-hnsw", "partitions": parts, "build_seconds": build,
                "index_bytes": index_bytes, "data_bytes": data_bytes})
        probes = [1] if parts == 1 else [p for p in a.nprobes if p <= parts]
        ds = lance.dataset(uri)
        for nprobes in probes:
            for ef in a.efs:
                run(ds, queries[0], ef, nprobes)  # warm-up: load the probed partitions
                hits, lat = 0, []
                for qi, q in enumerate(queries):
                    t = time.perf_counter()
                    res = run(ds, q, ef, nprobes)
                    lat.append((time.perf_counter() - t) * 1e3)
                    hits += len(set(res.column("id").to_numpy().tolist()) & set(gt[qi].tolist()))
                lat.sort()
                record({"kind": "query", "mode": "warm", "design": "lance-hnsw", "partitions": parts,
                        "nprobe": nprobes, "ef": ef, "recall": hits / (len(queries) * a.k),
                        "latency_ms": float(np.mean(lat)), "latency_ms_p99": lat[int(len(lat) * 0.99)]})
        # Cold: reopen with a tiny index cache before every query.
        for nprobes in probes:
            ef = 64
            hits, lat, read = 0, [], []
            for qi, q in enumerate(queries[: a.cold_queries]):
                cold = lance.dataset(uri, index_cache_size_bytes=1)
                r0, t = rchar(), time.perf_counter()
                res = run(cold, q, ef, nprobes)
                lat.append((time.perf_counter() - t) * 1e3)
                read.append(rchar() - r0)
                hits += len(set(res.column("id").to_numpy().tolist()) & set(gt[qi].tolist()))
            record({"kind": "query", "mode": "cold", "design": "lance-hnsw", "partitions": parts, "nprobe": nprobes,
                    "ef": ef, "recall": hits / (a.cold_queries * a.k), "latency_ms": float(np.mean(lat)),
                    "bytes_read": float(np.mean(read))})


if __name__ == "__main__":
    main()
