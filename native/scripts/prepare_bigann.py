"""Fetch only the first 10M BIGANN vectors; keep TLS verification enabled."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import struct
import urllib.request

ROOT="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/"
def fetch(url,path,expected,context,prefix=False):
    if path.exists():
        if path.stat().st_size!=expected:
            raise ValueError(f"existing file has wrong size: {path}")
        return
    temp=path.with_suffix(path.suffix+".partial")
    req=urllib.request.Request(url,headers={"Range":f"bytes=0-{expected-1}"} if prefix else {})
    written=0
    with urllib.request.urlopen(req,context=context,timeout=90) as src, temp.open("wb") as dst:
        if prefix and src.status==206:
            if not src.headers.get("Content-Range","").startswith(f"bytes 0-{expected-1}/"):
                raise ValueError("server returned a different byte range")
        while written<expected:
            block=src.read(min(8*1024*1024,expected-written))
            if not block: raise ValueError("download truncated")
            dst.write(block);written+=len(block)
            if prefix and written%(128*1024*1024)<8*1024*1024:
                print(f"{path.name}: {written/expected:.0%}",flush=True)
        if not prefix and src.read(1):
            raise ValueError("unexpected bytes after dataset")
    if prefix:
        with temp.open("r+b") as f:
            n,d=struct.unpack("<II",f.read(8))
            if (n,d)!=(1_000_000_000,128):raise ValueError("unexpected BIGANN source header")
            f.seek(0);f.write(struct.pack("<I",10_000_000))
    os.replace(temp,path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="data/bigann10m")
    p.add_argument("--ca-file",help="additional trusted CA bundle for a managed network")
    a=p.parse_args()
    context=ssl.create_default_context(cafile=a.ca_file)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    specs=[
        ("bigann/base.1B.u8bin","base.10M.u8bin",8+10_000_000*128,True),
        ("bigann/query.public.10K.u8bin","query.u8bin",8+10_000*128,False),
        ("GT_10M/bigann-10M","gt.10M.ibin",8+10_000*100*8,False)]
    manifest={"dataset":"BIGANN first 10M","source":"https://big-ann-benchmarks.com/neurips21.html","files":[]}
    for remote,name,size,prefix in specs:
        path=root/name
        fetch(ROOT+remote,path,size,context,prefix)
        with path.open("rb") as f:
            header=struct.unpack("<II",f.read(8))
        expected=(10_000_000,128) if prefix else ((10_000,100) if name.startswith("gt") else (10_000,128))
        if header!=expected:raise ValueError(f"bad header for {name}: {header}")
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
        manifest["files"].append({"path":name,"bytes":size,"sha256":h.hexdigest(),"url":ROOT+remote,"header":header})
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(root)
if __name__=="__main__":
    main()
