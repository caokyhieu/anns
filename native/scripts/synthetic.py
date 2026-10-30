"""Create uint8 smoke-test vectors and an independent exact top-100 oracle."""
import argparse
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--n",type=int,default=20000)
p.add_argument("--queries",type=int,default=200)
p.add_argument("--dim",type=int,default=128)
p.add_argument("--output",default="native/results/synthetic")
a=p.parse_args()
assert a.n>=100 and a.queries>0 and a.dim>0
rng=np.random.default_rng(7)
# Clustered descriptors avoid a trivial exact self-query workload.
centers=rng.uniform(30,220,size=(128,a.dim))
def sample(n):
    return np.clip(centers[rng.integers(128,size=n)] + rng.normal(0,25,size=(n,a.dim)),0,255).astype(np.uint8)
base,queries=sample(a.n),sample(a.queries)
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
for name,x in [("base.u8bin",base),("query.u8bin",queries)]:
    with (out/name).open("wb") as f:
        np.array(x.shape,dtype="<u4").tofile(f);x.tofile(f)
truth=np.empty((a.queries,100),dtype="<u4")
# Integer arithmetic gives exact distances for uint8 vectors; no GEMM cancellation.
for i,q in enumerate(queries):
    delta=base.astype(np.int32)-q.astype(np.int32)
    ds=np.einsum("ij,ij->i",delta,delta)
    truth[i]=np.argsort(ds,kind="stable")[:100]
with (out/"gt.ibin").open("wb") as f:
    np.array(truth.shape,dtype="<u4").tofile(f);truth.tofile(f)
print(out)
