"""Lance IVF_RQ with different row -> partition assignments, on the datasets and settings of
lance-format/lance discussion #9122 (GIST1M, VIBE msmarco-qwen-1024). See IVF_RQ.md.

Only the assignment changes between builds that share centroids. Everything else is Lance's
IVF_RQ: RaBitQ codes, query path, refine.

  lance-exact    Lance end to end, centroid HNSW forced off (exact assignment)
  lance-hnsw     Lance end to end, centroid HNSW forced on (k-means training uses it too)
  fixed-*        the centroids of lance-exact, given to Lance as ivf_centroids, with rows assigned by
    fixed-lance-hnsw  Lance's own centroid HNSW (forced on)
    fixed-flat        native exact assignment (sanity check: should match lance-exact)
    fixed-hnsw15      native hnswlib with Lance's SimpleIndex settings (stand-in check)
    fixed-fastpipnnB  a PiPNN (fastpipnn) graph over the centroids, beam B
    fixed-treeT       the tree assignment of LANCE_VS_PIPNN_BUILD.md: sqrt(C) coarse cells built
                      over the centroids, each row searches the leaves of its T nearest cells
  The native assignments come from native/build/ivf_assign and enter through
  precomputed_partition_dataset.

Steps (each Lance build runs in its own process: LANCE_USE_HNSW_SPEEDUP_INDEXING is read once):

  # 1. data -> data/<name>/{base.fbin, query.fbin, gt.ibin, base.lance}
  python lance_ivf_rq_bench.py prepare --name gist --base gist_base.fvecs --query gist_query.fvecs \
      --gt gist_groundtruth.ivecs
  python lance_ivf_rq_bench.py prepare --name gist --hdf5 gist-960-euclidean.hdf5
  python lance_ivf_rq_bench.py prepare --name vibe --base <vibe base .lance/.parquet dir> \
      --query <vibe query .lance/.parquet> [--gt <gt>] [--vector-column embedding] [--gt-k 100]
  # 2. builds and query sweeps
  python lance_ivf_rq_bench.py run --data data/gist --preset gist
  python lance_ivf_rq_bench.py run --data data/vibe --preset vibe
  # 3. tables
  python lance_ivf_rq_bench.py summarize data/gist/results.jsonl
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ASSIGN_BIN = ROOT / "native/build/ivf_assign"
# Settings of discussion #9122 (GIST: 256 partitions, 2-bit, refine 2/3/5, nprobe 16/20, k 10;
# VIBE: 1024 partitions, 1/2/3/5-bit, no refine, nprobe 8..128, Recall@100, medians of 3).
PRESETS = {
    "gist": dict(partitions=256, bits=[2], nprobes=[8, 16, 20, 32, 64], refines=[0, 2, 3, 5], k=10, metric="l2"),
    "vibe": dict(partitions=1024, bits=[1, 2, 3, 5], nprobes=[8, 16, 32, 64, 96, 128], refines=[0], k=100, metric="l2"),
}
DEFAULT_MODES = ["lance-exact", "lance-hnsw", "fixed-lance-hnsw", "fixed-flat", "fixed-hnsw15",
                 "fixed-fastpipnn16", "fixed-fastpipnn32"]
ALL_MODES = DEFAULT_MODES + ["fixed-tree1", "fixed-tree2", "fixed-tree4", "fixed-tree8"]


# ----------------------------------------------------------------------------- binary files
def write_bin(path, x):
    with open(path, "wb") as f:
        np.array(x.shape, dtype="<u4").tofile(f)
        x.tofile(f)


def read_bin(path, dtype, mmap=False):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    if mmap:
        return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))
    return np.fromfile(path, dtype=dtype, offset=8, count=n * d).reshape(n, d)


def read_vecs(path, dtype):
    """.fvecs / .ivecs / .bvecs: every row is an int32 dimension then the values."""
    raw = np.fromfile(path, dtype=np.int32 if dtype != np.uint8 else np.uint8)
    if dtype == np.uint8:
        d = int(raw[:4].view(np.int32)[0])
        return raw.reshape(-1, d + 4)[:, 4:]
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].view(dtype)


def iter_vectors(path, column, batch_rows=262144):
    """Yield float32 blocks from .fvecs/.bvecs/.fbin/.u8bin/.npy, a Lance dataset or Parquet."""
    p = str(path)
    if p.endswith(".fvecs"):
        yield read_vecs(p, np.float32).astype(np.float32, copy=False)
    elif p.endswith(".bvecs"):
        yield read_vecs(p, np.uint8).astype(np.float32)
    elif p.endswith(".fbin"):
        yield read_bin(p, np.float32)
    elif p.endswith(".u8bin"):
        yield read_bin(p, np.uint8).astype(np.float32)
    elif p.endswith(".npy"):
        yield np.load(p, mmap_mode="r").astype(np.float32)
    else:
        import pyarrow.dataset as pads
        if p.rstrip("/").endswith(".lance") or (Path(p) / "_versions").exists():
            import lance
            src = lance.dataset(p)
        else:
            src = pads.dataset(p, format="parquet")
        names = src.schema.names
        col = column or next(f.name for f in src.schema if str(f.type).startswith(("fixed_size_list", "list")))
        if col not in names:
            raise SystemExit(f"{p}: no column {col!r}; columns are {names}")
        for batch in src.to_batches(columns=[col], batch_size=batch_rows):
            arr = batch.column(0)
            d = len(arr[0])
            yield np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, d)


def count_rows(path, column):
    p = str(path)
    if p.endswith((".fvecs", ".bvecs", ".fbin", ".u8bin", ".npy")):
        return None
    import pyarrow.dataset as pads
    if p.rstrip("/").endswith(".lance") or (Path(p) / "_versions").exists():
        import lance
        return lance.dataset(p).count_rows()
    return pads.dataset(p, format="parquet").count_rows()


def to_fbin(src, dst, column):
    """Stream any supported source into a .fbin file without holding it in memory."""
    n = count_rows(src, column)
    if n is None:
        x = np.concatenate(list(iter_vectors(src, column)))
        write_bin(dst, x)
        return x.shape
    written, d = 0, None
    with open(dst, "wb") as f:
        f.write(b"\0" * 8)
        for block in iter_vectors(src, column):
            d = block.shape[1]
            np.ascontiguousarray(block, dtype=np.float32).tofile(f)
            written += len(block)
            print(f"  {dst.name}: {written:,} / {n:,} rows", end="\r", file=sys.stderr)
        f.seek(0)
        np.array([written, d], dtype="<u4").tofile(f)
    print(file=sys.stderr)
    return written, d


def exact_gt(base, queries, k, block=131072):
    """Exact squared-L2 top-k by blocked GEMM over a memory-mapped base."""
    qn = (queries * queries).sum(1)
    best_d = np.full((len(queries), k), np.inf, np.float32)
    best_i = np.zeros((len(queries), k), np.int64)
    for s in range(0, len(base), block):
        b = np.asarray(base[s:s + block], dtype=np.float32)
        dist = qn[:, None] + (b * b).sum(1)[None] - 2 * queries @ b.T
        cat_d = np.concatenate([best_d, dist], 1)
        cat_i = np.concatenate([best_i, np.arange(s, s + len(b))[None].repeat(len(queries), 0)], 1)
        top = np.argpartition(cat_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(cat_d, top, 1)
        best_i = np.take_along_axis(cat_i, top, 1)
        print(f"  ground truth: {min(s + block, len(base)):,} / {len(base):,} rows", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    order = np.argsort(best_d, axis=1)
    return np.take_along_axis(best_i, order, 1).astype(np.uint32)


def write_lance_base(fbin, uri):
    import lance
    import pyarrow as pa
    x = read_bin(fbin, np.float32, mmap=True)
    n, d = x.shape
    schema = pa.schema([("id", pa.uint64()), ("vec", pa.list_(pa.float32(), d))])

    def batches(step=131072):
        for s in range(0, n, step):
            block = np.ascontiguousarray(x[s:s + step])
            yield pa.record_batch([pa.array(np.arange(s, s + len(block), dtype=np.uint64)),
                                   pa.FixedSizeListArray.from_arrays(pa.array(block.ravel()), d)], schema=schema)

    shutil.rmtree(uri, ignore_errors=True)
    lance.write_dataset(pa.RecordBatchReader.from_batches(schema, batches()), uri)


def cmd_prepare(a):
    out = ROOT / "data" / a.name if not a.out else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.hdf5:
        import h5py
        with h5py.File(a.hdf5, "r") as h:
            write_bin(out / "base.fbin", np.asarray(h["train"], dtype=np.float32))
            write_bin(out / "query.fbin", np.asarray(h["test"], dtype=np.float32))
            if "neighbors" in h:
                write_bin(out / "gt.ibin", np.asarray(h["neighbors"], dtype=np.uint32))
    else:
        if not (a.base and a.query):
            raise SystemExit("prepare needs --hdf5, or --base and --query")
        print("base:", to_fbin(a.base, out / "base.fbin", a.vector_column), file=sys.stderr)
        print("query:", to_fbin(a.query, out / "query.fbin", a.query_column or a.vector_column), file=sys.stderr)
        if a.gt:
            gt = read_vecs(a.gt, np.int32) if a.gt.endswith(".ivecs") else read_bin(a.gt, np.uint32)
            write_bin(out / "gt.ibin", np.ascontiguousarray(gt, dtype=np.uint32))
    q = read_bin(out / "query.fbin", np.float32)
    if a.max_queries:
        q = q[: a.max_queries]
        write_bin(out / "query.fbin", np.ascontiguousarray(q))
    if not (out / "gt.ibin").exists() or a.recompute_gt:
        base = read_bin(out / "base.fbin", np.float32, mmap=True)
        write_bin(out / "gt.ibin", exact_gt(base, q, a.gt_k))
    gt = read_bin(out / "gt.ibin", np.uint32)
    if len(gt) > len(q):
        write_bin(out / "gt.ibin", np.ascontiguousarray(gt[: len(q)]))
    print("writing", out / "base.lance", file=sys.stderr)
    write_lance_base(out / "base.fbin", str(out / "base.lance"))
    base = read_bin(out / "base.fbin", np.float32, mmap=True)
    print(json.dumps(dict(name=a.name, n=len(base), dim=base.shape[1], queries=len(q), gt_k=gt.shape[1])))


# ----------------------------------------------------------------------------- one build + sweep
def cmd_child(a):
    """One Lance IVF_RQ build and its query sweep. Runs in its own process."""
    import lance
    import pyarrow as pa

    data = Path(a.data)
    uri = str(data / "base.lance")
    ds = lance.dataset(uri)
    kwargs = dict(num_partitions=a.partitions, num_bits=a.bits, metric=a.metric, name="vec_idx", replace=True)
    if a.centroids:
        kwargs["ivf_centroids"] = read_bin(a.centroids, np.float32)
    if a.assignment:
        parts = np.fromfile(a.assignment, dtype=np.uint32)
        rowids = ds.to_table(columns=[], with_row_id=True).column("_rowid").to_numpy()
        if len(rowids) != len(parts):
            raise SystemExit(f"{a.assignment}: {len(parts)} rows, dataset has {len(rowids)}")
        puri = str(data / "partitions.lance")
        shutil.rmtree(puri, ignore_errors=True)
        lance.write_dataset(pa.table({"row_id": pa.array(rowids.astype(np.uint64)), "partition": pa.array(parts)}),
                            puri, max_rows_per_file=1 << 31)
        kwargs["precomputed_partition_dataset"] = puri
    t = time.perf_counter()
    ds.create_index("vec", "IVF_RQ", **kwargs)
    build = time.perf_counter() - t
    ds = lance.dataset(uri)
    ds.cleanup_old_versions(older_than=timedelta(seconds=0), delete_unverified=True)
    if a.save_centroids:
        os.environ["LANCE_INCLUDE_VECTOR_CENTROIDS"] = "true"
        st = ds.stats.index_stats("vec_idx")["indices"][0]
        write_bin(a.save_centroids, np.asarray(st["centroids"], dtype=np.float32))
    st = lance.dataset(uri).stats.index_stats("vec_idx")["indices"][0]
    sizes = np.array([p["size"] for p in st.get("partitions", [])])
    index_bytes = sum(f.stat().st_size for f in (data / "base.lance" / "_indices").rglob("*") if f.is_file())
    common = dict(dataset=data.name, mode=a.mode, partitions=a.partitions, bits=a.bits, metric=a.metric,
                  lance=lance.__version__, hnsw_env=os.environ.get("LANCE_USE_HNSW_SPEEDUP_INDEXING", "auto"))
    rows = [dict(kind="build", **common, build_seconds=build, index_bytes=index_bytes,
                 empty_lists=int((sizes == 0).sum()) if len(sizes) else None,
                 list_size_max=int(sizes.max()) if len(sizes) else None)]
    print(json.dumps(rows[-1]), flush=True)

    queries = read_bin(data / "query.fbin", np.float32)[: a.queries or None]
    gt = read_bin(data / "gt.ibin", np.uint32)[: len(queries), : a.k]
    ds = lance.dataset(uri, index_cache_size_bytes=a.cache_gib << 30)

    def search(q, nprobes, refine):
        nearest = {"column": "vec", "q": q, "k": a.k, "nprobes": nprobes}
        if refine:
            nearest["refine_factor"] = refine
        return ds.to_table(columns=["id"], nearest=nearest)

    for q in queries[: a.warmup]:
        search(q, max(a.nprobes), 0)
    for nprobes in a.nprobes:
        for refine in a.refines:
            runs = []
            for _ in range(a.repeats):
                hits, lat = 0, []
                t0 = time.perf_counter()
                for i, q in enumerate(queries):
                    t = time.perf_counter()
                    ids = search(q, nprobes, refine).column("id").to_numpy()
                    lat.append((time.perf_counter() - t) * 1e3)
                    hits += len(np.intersect1d(ids, gt[i]))
                wall = time.perf_counter() - t0
                runs.append((hits / gt.size, float(np.percentile(lat, 50)), float(np.percentile(lat, 95)), len(queries) / wall))
            med = [float(np.median([r[j] for r in runs])) for j in range(4)]
            rows.append(dict(kind="query", **common, k=a.k, queries=len(queries), nprobe=nprobes, refine=refine or None,
                             repeats=a.repeats, recall=med[0], p50_ms=med[1], p95_ms=med[2], qps=med[3]))
            print(json.dumps(rows[-1]), flush=True)
    with open(a.output, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------------------- orchestration
def cmd_run(a):
    p = dict(PRESETS[a.preset])
    for key in ("partitions", "bits", "nprobes", "refines", "k", "metric"):
        if getattr(a, key) is not None:
            p[key] = getattr(a, key)
    data = Path(a.data)
    out = Path(a.output or data / "results.jsonl")
    modes = a.modes or DEFAULT_MODES
    C = p["partitions"]
    cent = data / f"centroids-{C}.fbin"

    def child(mode, bits, env, extra):
        cmd = [sys.executable, __file__, "child", "--data", str(data), "--mode", mode, "--partitions", str(C),
               "--bits", str(bits), "--metric", p["metric"], "--k", str(p["k"]), "--queries", str(a.queries),
               "--repeats", str(a.repeats), "--warmup", str(a.warmup), "--cache-gib", str(a.cache_gib),
               "--output", str(out), "--nprobes", *map(str, p["nprobes"]), "--refines", *map(str, p["refines"]), *extra]
        e = dict(os.environ, LANCE_INCLUDE_VECTOR_CENTROIDS="true")
        if env:
            e["LANCE_USE_HNSW_SPEEDUP_INDEXING"] = env
        print(f"== {mode}, {bits}-bit", file=sys.stderr, flush=True)
        subprocess.run(cmd, env=e, check=True)

    for bits in p["bits"]:
        # Centroids come from the exact build at the first bit width and are reused afterwards.
        need_centroids = not cent.exists()
        if "lance-exact" in modes or need_centroids:
            child("lance-exact", bits, "disabled", ["--save-centroids", str(cent)] if need_centroids else [])
        if "lance-hnsw" in modes:
            child("lance-hnsw", bits, "enabled", [])
        if "fixed-lance-hnsw" in modes:
            child("fixed-lance-hnsw", bits, "enabled", ["--centroids", str(cent)])
        native = [m[len("fixed-"):] for m in modes if m.startswith("fixed-") and m != "fixed-lance-hnsw"]
        if native:
            prefix = data / f"assign-{C}"
            specs = ",".join("flat" if m == "flat" else m.replace("hnsw", "hnsw:").replace("fastpipnn", "fastpipnn:").replace("tree", "tree:") for m in native)
            if not all((data / f"assign-{C}.{m}.u32").exists() for m in native):
                subprocess.run([str(ASSIGN_BIN), "--base", str(data / "base.fbin"), "--centroids", str(cent),
                                "--assign", specs, "--out-prefix", str(prefix), "--output", str(data / "assign.jsonl"),
                                "--threads", str(a.threads or os.cpu_count())], check=True)
            for m in native:
                child(f"fixed-{m}", bits, "disabled", ["--centroids", str(cent), "--assignment", f"{prefix}.{m}.u32"])


# ----------------------------------------------------------------------------- tables
def cmd_summarize(a):
    rows = [json.loads(line) for line in open(a.results) if line.strip()]
    assign_file = Path(a.results).parent / "assign.jsonl"
    if assign_file.exists():
        print("\n| Native assignment | Distances / row | Assign time | Exact nearest | Empty lists | Largest list |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in map(json.loads, open(assign_file)):
            name = r["assign"] + (f":{r['assign_beam']}" if r["assign_beam"] else "")
            print(f"| {name} ({r['lists']} lists) | {r['evals_per_row']:.0f} | {r['assign_seconds']:.2f} s | "
                  f"{r['exact_fraction']:.4f} | {r['empty_lists']} | {r['list_size_max']} |")
    for parts, bits in sorted({(r["partitions"], r["bits"]) for r in rows}):
        rs = [r for r in rows if r["bits"] == bits and r["partitions"] == parts]
        probes = sorted({r["nprobe"] for r in rs if r["kind"] == "query"})
        refines = sorted({r["refine"] or 0 for r in rs if r["kind"] == "query"})
        k = next(r["k"] for r in rs if r["kind"] == "query")
        print(f"\n## {bits}-bit RQ, {rs[0]['partitions']} partitions, Recall@{k}\n")
        for refine in refines:
            print(f"refine {refine or 'none'}:\n")
            print("| Mode | Build | Empty lists | " + " | ".join(f"np {n}" for n in probes) + " |")
            print("|---|---:|---:|" + "---:|" * len(probes))
            for b in [r for r in rs if r["kind"] == "build"]:
                q = {r["nprobe"]: r for r in rs if r["kind"] == "query" and r["mode"] == b["mode"] and (r["refine"] or 0) == refine}
                cells = [f"{q[n]['recall']:.4f} ({q[n]['p50_ms']:.1f} ms)" if n in q else "–" for n in probes]
                print(f"| {b['mode']} | {b['build_seconds']:.1f} s | {b['empty_lists']} | " + " | ".join(cells) + " |")
            print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--name", required=True)
    pr.add_argument("--out")
    pr.add_argument("--hdf5")
    pr.add_argument("--base")
    pr.add_argument("--query")
    pr.add_argument("--gt")
    pr.add_argument("--vector-column")
    pr.add_argument("--query-column")
    pr.add_argument("--gt-k", type=int, default=100)
    pr.add_argument("--max-queries", type=int, default=0)
    pr.add_argument("--recompute-gt", action="store_true")
    ch = sub.add_parser("child")
    ru = sub.add_parser("run")
    for s in (ch, ru):
        s.add_argument("--data", required=True)
        s.add_argument("--queries", type=int, default=0, help="0 = all")
        s.add_argument("--repeats", type=int, default=3)
        s.add_argument("--warmup", type=int, default=10)
        s.add_argument("--cache-gib", type=int, default=32)
        s.add_argument("--output")
        s.add_argument("--nprobes", type=int, nargs="+")
        s.add_argument("--refines", type=int, nargs="+", help="0 = no refine")
        s.add_argument("--k", type=int)
        s.add_argument("--metric")
        s.add_argument("--partitions", type=int)
    ch.add_argument("--mode", required=True)
    ch.add_argument("--bits", type=int, required=True)
    ch.add_argument("--centroids")
    ch.add_argument("--assignment")
    ch.add_argument("--save-centroids")
    ru.add_argument("--preset", choices=PRESETS, required=True)
    ru.add_argument("--bits", type=int, nargs="+")
    ru.add_argument("--modes", nargs="+", choices=ALL_MODES)
    ru.add_argument("--threads", type=int, default=0)
    sm = sub.add_parser("summarize")
    sm.add_argument("results")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "child": cmd_child, "run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
