#!/usr/bin/env python3
"""Validate direct H0 event sinks against buffered F32 streaming and in-memory H0."""
from __future__ import annotations
import argparse,csv,os,re,subprocess,tempfile
from collections import Counter
from pathlib import Path
from f32_fixture import stage_f32_subset

CONFIG_RE=re.compile(r"PROFILE_CONFIG scalar_h0_stream .*?h0_birth_buffer=(?P<birth>\S+).*?h0_event_storage=(?P<events>\S+)")

def args():
    p=argparse.ArgumentParser(); p.add_argument("input",type=Path)
    p.add_argument("--binary",type=Path,default=Path(os.environ.get("BETTI_CURVES_BINARY","target/release/betti_curves")))
    p.add_argument("--slice-limit",type=int,default=4); p.add_argument("--slab-depth",type=int,default=2)
    p.add_argument("--foreground-connectivity",choices=(6,26),type=int,default=26); return p.parse_args()
def read(p):
    with p.open(newline="",encoding="utf-8") as h:
        r=csv.reader(h); head=next(r,None)
        if head != ["birth","death"]: raise RuntimeError(f"bad header {head}")
        return Counter(tuple(row) for row in r if row)
def run(binary,inp,work,slab,fg,mode,extra=()):
    work.mkdir(parents=True,exist_ok=True); out=work/"intervals.csv"
    cmd=[str(binary.resolve()),str(inp.resolve()),str(slab),str(fg),mode,str(out),*extra]
    cp=subprocess.run(cmd,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if cp.returncode: raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
    return read(out),cp.stdout
def eq(label,a,b):
    if a!=b: raise SystemExit(f"FAIL {label}\nmissing={(a-b).most_common(10)}\nextra={(b-a).most_common(10)}")
    print(f"PASS {label}: {sum(a.values())} intervals agree exactly")
def main():
    a=args()
    if not a.binary.is_file(): raise SystemExit(f"binary not found: {a.binary}")
    with tempfile.TemporaryDirectory(prefix="betti_h0_direct_") as td:
        root=Path(td); f32=stage_f32_subset(a.input,root/"f32_input",a.slice_limit)
        oracle,_=run(a.binary,f32,root/"oracle",a.slab_depth,a.foreground_connectivity,"h0-scalar")
        configs={
            "buffered-copy":("copy","buffered"),
            "buffered-reuse":("reuse-input","buffered"),
            "direct-reuse":("reuse-input","direct"),
        }
        results={}
        for label,(birth,events) in configs.items():
            got,log=run(a.binary,f32,root/label,a.slab_depth,a.foreground_connectivity,"h0-scalar-stream",(
                "--f32-key-mode","native32","--h0-birth-buffer",birth,"--h0-event-storage",events,"--event-order","verify"))
            eq(f"H0 {label} vs in-memory",oracle,got)
            m=CONFIG_RE.search(log)
            if not m or m.group("birth")!=birth or m.group("events")!=events:
                raise SystemExit(f"FAIL {label}: configuration profile mismatch")
            results[label]=got
        eq("H0 direct-reuse vs buffered-reuse",results["buffered-reuse"],results["direct-reuse"])
    print("PASS: H0 direct event sinks preserve exact persistence")
if __name__=="__main__": main()
