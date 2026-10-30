"""Sequential process-isolated comparison with raw logs and matched-recall summaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument("--base",default="data/bigann10m/base.10M.u8bin")
p.add_argument("--queries",default="data/bigann10m/query.u8bin")
p.add_argument("--gt",default="data/bigann10m/gt.10M.ibin")
p.add_argument("--threads",type=int,default=32)
p.add_argument("--query-threads",type=int,default=1)
p.add_argument("--query-count",type=int,default=10000)
p.add_argument("--repeats",type=int,default=1)
p.add_argument("--beams",default="16,32,64,128,256,512")
p.add_argument("--output",default="native/results/bigann10m")
p.add_argument("--m",type=int,default=32)
p.add_argument("--ef-construction",type=int,default=200)
p.add_argument("--alpha",type=float,default=1.2)
p.add_argument("--degree",type=int,default=64)
p.add_argument("--allow-small",action="store_true",help="explicitly label a smoke dataset; never claim it is BIGANN 10M")
a=p.parse_args()
if a.repeats<1:p.error("repeats must be positive")
paths=[Path(name).resolve() for name in (a.base,a.queries,a.gt)]
for path in paths:
    if not path.is_file():p.error(f"missing input: {path}; use prepare_bigann.py or supply local paths")
import struct
with paths[0].open("rb") as f:n,d=struct.unpack("<II",f.read(8))
if not a.allow_small and (n,d)!=(10_000_000,128):p.error("BIGANN 10M requires exactly 10,000,000 x 128 uint8 vectors")
out=Path(a.output).resolve()
out.mkdir(parents=True,exist_ok=True)
if (out/"run.json").exists():p.error("output directory already contains a run; choose a new directory")
binary=ROOT/"native/build/ann_bench"
if not binary.is_file():p.error("build native/build/ann_bench first")
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()
metadata={"parameters":vars(a),"dataset_label":"custom/smoke" if a.allow_small else "BIGANN first 10M",
          "platform":platform.platform(),"cpu":platform.processor(),
          "cpu_affinity":sorted(os.sched_getaffinity(0)),"start_load_average":os.getloadavg(),
          "sources":json.loads((ROOT/"native/sources.lock.json").read_text()),
          "inputs":[{"path":str(path),"bytes":path.stat().st_size,"sha256":digest(path)} for path in paths],
          "binary_sha256":digest(binary),"commands":[],"status":"running"}
(out/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
results=[]
try:
    for repeat in range(a.repeats):
        order=["pipnn","hnsw"] if repeat%2==0 else ["hnsw","pipnn"]
        for algorithm in order:
            target=out/f"{algorithm}-{repeat}.json"
            command=[str(binary),"--algorithm",algorithm,"--base",str(paths[0]),
                     "--queries",str(paths[1]),"--gt",str(paths[2]),
                     "--threads",str(a.threads),"--query-threads",str(a.query_threads),
                     "--query-count",str(a.query_count),"--beams",a.beams,
                     "--m",str(a.m),"--ef-construction",str(a.ef_construction),
                     "--alpha",str(a.alpha),"--degree",str(a.degree),
                     "--reservoir",str(max(160,int(a.degree*2.5))),"--output",str(target)]
            metadata["commands"].append(command)
            (out/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
            print("Running",algorithm,"repeat",repeat,flush=True)
            start=time.monotonic()
            with (out/f"{algorithm}-{repeat}.log").open("w") as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads(target.read_text())
            result["repeat"]=repeat
            result["process_wall_seconds"]=time.monotonic()-start
            results.append(result)
            print(algorithm,"build_seconds",round(result["build_seconds"],3),
                  "best_recall",max(row["recall_at_10"] for row in result["queries"]),flush=True)
    targets=[]
    for target in (.9,.95,.99):
        rows=[]
        for result in results:
            eligible=[q for q in result["queries"] if q["recall_at_10"]>=target]
            if eligible:
                best=max(eligible,key=lambda q:q["qps"])
                rows.append({"algorithm":result["algorithm"],"repeat":result["repeat"],
                             "build_seconds":result["build_seconds"],**best})
            else:
                rows.append({"algorithm":result["algorithm"],"repeat":result["repeat"],"reached":False})
        targets.append({"target_recall":target,"measurements":rows})
    (out/"summary.json").write_text(json.dumps({"results":results,"matched_recall":targets},indent=2)+"\n")
    metadata["status"]="complete"
except Exception:
    metadata["status"]="failed"
    raise
finally:
    metadata["end_load_average"]=os.getloadavg()
    (out/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
print(out/"summary.json")
