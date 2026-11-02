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
# A float32 GEMM shortlists 400 candidates per query; exact integer arithmetic then ranks them,
# so GEMM rounding can only matter if a true top-100 neighbour falls outside the shortlist.
shortlist=min(400,a.n)
bf=base.astype(np.float32); bn=np.einsum("ij,ij->i",bf,bf)
for s in range(0,a.queries,256):
    qs=queries[s:s+256]
    approx=bn[None,:]-2*(qs.astype(np.float32)@bf.T)
    cand=np.argpartition(approx,shortlist-1,axis=1)[:,:shortlist]
    for r,q in enumerate(qs):
        delta=base[cand[r]].astype(np.int32)-q.astype(np.int32)
        ds=np.einsum("ij,ij->i",delta,delta)
        order=np.lexsort((cand[r],ds))
        truth[s+r]=cand[r][order[:100]]
with (out/"gt.ibin").open("wb") as f:
    np.array(truth.shape,dtype="<u4").tofile(f);truth.tofile(f)
print(out)
