#!/usr/bin/env python3
"""Validate native-F32 H2 summaries/disk/global state and packed global UF."""
from __future__ import annotations

import argparse,csv,os,re,subprocess,tempfile
from collections import Counter
from pathlib import Path
from f32_fixture import stage_f32_subset

STORAGE_RE=re.compile(r"PROFILE_H2_STORAGE scalar_h2_stream pipeline=(?P<pipeline>\S+) disk_key_bytes=(?P<key>\d+) .*?global_h2_birth_state=(?P<birth>\S+) global_h2_uf_layout=(?P<layout>\S+)")
STATE_NATIVE_RE=re.compile(r"PROFILE_GLOBAL_H2_STATE strategy=compact uf_layout=(?P<layout>\S+) nodes=(?P<nodes>\d+) key_bytes=(?P<key>\d+) parent_bytes=(?P<parent>\d+) rank_bytes=(?P<rank>\d+) birth_bytes=(?P<birth>\d+) total_bytes=(?P<total>\d+)")


def args():
    p=argparse.ArgumentParser(); p.add_argument("input",type=Path)
    p.add_argument("--binary",type=Path,default=Path(os.environ.get("BETTI_CURVES_BINARY","target/release/betti_curves")))
    p.add_argument("--slice-limit",type=int,default=4); p.add_argument("--slab-depth",type=int,default=2)
    p.add_argument("--foreground-connectivity",choices=(6,26),type=int,default=26)
    return p.parse_args()

def read(path):
    with path.open(newline="",encoding="utf-8") as h:
        r=csv.reader(h); head=next(r,None)
        if head != ["birth","death"]: raise RuntimeError(f"bad header {head}")
        return Counter(tuple(row) for row in r if row)

def run(binary,inp,work,slab,fg,mode,extra=()):
    work.mkdir(parents=True,exist_ok=True); out=work/"intervals.csv"
    cmd=[str(binary.resolve()),str(inp.resolve()),str(slab),str(fg),mode,str(out),*extra]
    cp=subprocess.run(cmd,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if cp.returncode: raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
    return read(out),cp.stdout

def equal(label,a,b):
    if a!=b: raise SystemExit(f"FAIL {label}\nmissing={(a-b).most_common(10)}\nextra={(b-a).most_common(10)}")
    print(f"PASS {label}: {sum(a.values())} intervals agree exactly")

def native_state(label,log,layout):
    sm=STORAGE_RE.search(log); st=STATE_NATIVE_RE.search(log)
    if not sm or sm.group("pipeline")!="native32-end-to-end" or int(sm.group("key"))!=4 or sm.group("birth")!="compact" or sm.group("layout")!=layout:
        raise SystemExit(f"FAIL {label}: H2 native storage profile mismatch")
    if not st or st.group("layout")!=layout or int(st.group("key"))!=4:
        raise SystemExit(f"FAIL {label}: H2 native global-state profile mismatch")
    vals={k:int(st.group(k)) for k in ("nodes","parent","rank","birth","total")}
    if vals["total"] != vals["parent"]+vals["rank"]+vals["birth"]: raise SystemExit(f"FAIL {label}: byte accounting")
    print(f"PASS {label}: global state {vals['total']/2**20:.2f} MiB, rank {vals['rank']/2**20:.2f} MiB")
    return vals

def main():
    a=args()
    if not a.binary.is_file(): raise SystemExit(f"binary not found: {a.binary}")
    with tempfile.TemporaryDirectory(prefix="betti_h2_f32_e2e_") as td:
        root=Path(td); f32=stage_f32_subset(a.input,root/"f32_input",a.slice_limit)
        oracle,_=run(a.binary,f32,root/"oracle",a.slab_depth,a.foreground_connectivity,"h2-scalar")
        legacy,_=run(a.binary,f32,root/"legacy",a.slab_depth,a.foreground_connectivity,"h2-scalar-stream",(
            "--f32-key-mode","legacy64","--global-h2-birth-state","compact","--global-h2-uf-layout","parent-rank"))
        equal("H2 legacy64 vs in-memory",oracle,legacy)
        states={}; results={}
        for layout in ("parent-rank","packed"):
            got,log=run(a.binary,f32,root/layout,a.slab_depth,a.foreground_connectivity,"h2-scalar-stream",(
                "--f32-key-mode","native32","--global-h2-birth-state","compact","--global-h2-uf-layout",layout))
            equal(f"H2 native32/{layout} vs in-memory",oracle,got)
            states[layout]=native_state(f"H2 native32/{layout}",log,layout); results[layout]=got
        equal("H2 native32 packed vs parent-rank",results["parent-rank"],results["packed"])
        ref,packed=states["parent-rank"],states["packed"]
        if packed["nodes"]!=ref["nodes"] or packed["birth"]!=ref["birth"] or packed["parent"]!=ref["parent"]:
            raise SystemExit("FAIL: H2 packed changed nodes/parent/birth capacity")
        if ref["rank"]<=0 or packed["rank"]!=0: raise SystemExit("FAIL: H2 packed did not remove rank array")
        print(f"PASS: H2 packed saves {(ref['total']-packed['total'])/2**20:.2f} MiB beyond native F32")
    print("PASS: H2 F32 end-to-end + packed global UF preserves exact persistence")

if __name__=="__main__": main()
