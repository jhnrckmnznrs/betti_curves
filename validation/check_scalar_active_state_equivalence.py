#!/usr/bin/env python3
"""Exact scalar persistence check for separate versus parent-sentinel active state."""
from __future__ import annotations
import argparse, csv, os, subprocess, tempfile
from collections import Counter
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path)
    p.add_argument('--binary',type=Path,default=Path(os.environ.get('BETTI_CURVES_BINARY','target/release/betti_curves')))
    p.add_argument('--slab-depth',type=int,default=16)
    p.add_argument('--foreground-connectivity',choices=(6,26),type=int,default=26)
    p.add_argument('--modes',nargs='+',choices=('h0','h2'),default=('h0','h2'))
    p.add_argument('--f32-key-mode',choices=('legacy64','native32'),default='native32')
    return p.parse_args()

def read_intervals(path):
    with path.open(newline='',encoding='utf-8') as h:
        r=csv.reader(h); header=next(r,None)
        if header != ['birth','death']: raise RuntimeError(f'unexpected header {header!r} in {path}')
        return Counter(tuple(row) for row in r if row)

def run(binary,input_path,work,slab,fg,mode,extra=None):
    work.mkdir(parents=True,exist_ok=True); out=work/'intervals.csv'
    cmd=[str(binary.resolve()),str(input_path.resolve()),str(slab),str(fg),mode,str(out)]
    if extra: cmd.extend(extra)
    cp=subprocess.run(cmd,cwd=work,text=True,capture_output=True)
    if cp.returncode: raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}{cp.stderr}")
    return read_intervals(out), cp.stdout+cp.stderr

def check(label,a,b):
    if a!=b:
        raise SystemExit(f'FAIL {label}: persistence multisets differ\nmissing={(a-b).most_common(10)}\nextra={(b-a).most_common(10)}')
    print(f'PASS {label}: {sum(a.values())} intervals agree exactly')

def main():
    a=parse_args()
    if not a.binary.is_file(): raise SystemExit(f'binary not found: {a.binary}')
    fixed=['--merge-strategy','scan','--interface-order','radix','--event-order','verify',
           '--f32-key-mode',a.f32_key_mode,'--neighbor-kernel','interior-fast',
           '--representative-active-check','recheck','--union-kernel','root-carrying',
           '--h0-pruning-cache','64k','--neighbor-root-check','parent-shortcut']
    with tempfile.TemporaryDirectory(prefix='betti_active_state_equiv_') as td:
        root=Path(td)
        for dim in a.modes:
            oracle,_=run(a.binary,a.input,root/f'{dim}_oracle',a.slab_depth,a.foreground_connectivity,f'{dim}-scalar')
            separate,_=run(a.binary,a.input,root/f'{dim}_separate',a.slab_depth,a.foreground_connectivity,f'{dim}-scalar-stream',fixed+['--active-state','separate'])
            sentinel,_=run(a.binary,a.input,root/f'{dim}_sentinel',a.slab_depth,a.foreground_connectivity,f'{dim}-scalar-stream',fixed+['--active-state','parent-sentinel'])
            check(f'{dim} separate vs in-memory',oracle,separate)
            check(f'{dim} parent-sentinel vs in-memory',oracle,sentinel)
            check(f'{dim} parent-sentinel vs separate',separate,sentinel)
    print('PASS: parent-sentinel active state preserves exact scalar persistence')
if __name__=='__main__': main()
