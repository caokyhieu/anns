"""Fetch exact source revisions. No system packages or global environment changes."""
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
lock=json.loads((root/"sources.lock.json").read_text())
for name,item in lock.items():
    destination=root/"third_party"/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not (destination/".git").exists():
        subprocess.run(["git","init",str(destination)],check=True)
        subprocess.run(["git","-C",str(destination),"remote","add","origin",item["url"]],check=True)
    actual=subprocess.run(["git","-C",str(destination),"rev-parse","HEAD"],capture_output=True,text=True)
    if actual.returncode or actual.stdout.strip()!=item["commit"]:
        status=subprocess.run(["git","-C",str(destination),"status","--porcelain"],capture_output=True,text=True,check=True)
        if status.stdout.strip():raise RuntimeError(f"refusing to replace modified dependency: {destination}")
        subprocess.run(["git","-C",str(destination),"fetch","--depth","1","origin",item["commit"]],check=True)
        subprocess.run(["git","-C",str(destination),"checkout","--detach",item["commit"]],check=True)
    print(name,item["commit"])
