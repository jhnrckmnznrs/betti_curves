#!/usr/bin/env python3
"""Validate H0 F32 input-buffer reuse against the copy path and in-memory oracle."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from f32_fixture import stage_f32_subset

CONFIG_RE = re.compile(r"PROFILE_CONFIG scalar_h0_stream .*?h0_birth_buffer=(?P<buffer>\S+)")
STORAGE_RE = re.compile(r"PROFILE_H0_STORAGE scalar_h0_stream pipeline=(?P<pipeline>\S+) disk_key_bytes=(?P<key>\d+).*?h0_birth_buffer=(?P<buffer>\S+)")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("BETTI_CURVES_BINARY","target/release/betti_curves")))
    p.add_argument("--slice-limit", type=int, default=4)
    p.add_argument("--slab-depth", type=int, default=2)
    p.add_argument("--foreground-connectivity", choices=(6,26), type=int, default=26)
    return p.parse_args()


def read(path):
    with path.open(newline="",encoding="utf-8") as h:
        r=csv.reader(h); header=next(r,None)
        if header != ["birth","death"]: raise RuntimeError(f"bad header {header} in {path}")
        return Counter(tuple(row) for row in r if row)


def run(binary,input_path,work,slab,fg,mode,extra=()):
    work.mkdir(parents=True,exist_ok=True); out=work/"intervals.csv"
    cmd=[str(binary.resolve()),str(input_path.resolve()),str(slab),str(fg),mode,str(out),*extra]
    cp=subprocess.run(cmd,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if cp.returncode: raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
    return read(out),cp.stdout


def equal(label,a,b):
    if a!=b:
        raise SystemExit(f"FAIL {label}\nmissing={(a-b).most_common(10)}\nextra={(b-a).most_common(10)}")
    print(f"PASS {label}: {sum(a.values())} intervals agree exactly")


def main():
    a=parse_args()
    if not a.binary.is_file(): raise SystemExit(f"binary not found: {a.binary}")
    with tempfile.TemporaryDirectory(prefix="betti_h0_birth_buffer_") as td:
        root=Path(td); f32=stage_f32_subset(a.input,root/"f32_input",a.slice_limit)
        oracle,_=run(a.binary,f32,root/"oracle",a.slab_depth,a.foreground_connectivity,"h0-scalar")
        results={}
        for strategy in ("copy","reuse-input"):
            result,log=run(a.binary,f32,root/strategy,a.slab_depth,a.foreground_connectivity,"h0-scalar-stream",(
                "--f32-key-mode","native32","--h0-birth-buffer",strategy,
            ))
            equal(f"H0 {strategy} vs in-memory",oracle,result)
            m=CONFIG_RE.search(log); sm=STORAGE_RE.search(log)
            if not m or m.group("buffer")!=strategy: raise SystemExit(f"FAIL {strategy}: PROFILE_CONFIG mismatch")
            if not sm or sm.group("pipeline")!="native32-end-to-end" or int(sm.group("key"))!=4 or sm.group("buffer")!=strategy:
                raise SystemExit(f"FAIL {strategy}: native32 storage profile mismatch")
            results[strategy]=result
        equal("H0 reuse-input vs copy",results["copy"],results["reuse-input"])
    print("PASS: H0 F32 birth-buffer reuse preserves exact persistence")

if __name__=="__main__": main()
