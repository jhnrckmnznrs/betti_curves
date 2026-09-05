#!/usr/bin/env python3
"""Compare local-sweep operation counters for separate vs parent-sentinel active state."""
from __future__ import annotations
import argparse, csv, hashlib, subprocess, tempfile
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path)
    p.add_argument('--binary',type=Path,default=Path('target/release/betti_curves'))
    p.add_argument('--slab-depth',type=int,default=16)
    p.add_argument('--foreground-connectivity',choices=(6,26),type=int,default=26)
    p.add_argument('--f32-key-mode',choices=('legacy64','native32'),default='native32')
    p.add_argument('--output',type=Path,default=Path('scalar_active_state_diagnostics.csv'))
    return p.parse_args()

def canonical_hash(path: Path):
    with path.open(newline='',encoding='utf-8') as h:
        r=csv.reader(h); header=next(r,None)
        if header != ['birth','death']: raise RuntimeError(f'unexpected header {header!r}')
        rows=sorted(tuple(row) for row in r if row)
    d=hashlib.sha256()
    for b,e in rows: d.update(f'{b},{e}\n'.encode())
    return d.hexdigest(),len(rows)

def parse_kv_line(log: str, prefix: str):
    for line in log.splitlines():
        if line.startswith(prefix):
            out={}
            for token in line.split()[2:]:
                if '=' in token:
                    k,v=token.split('=',1); out[k]=v
            return out
    raise RuntimeError(f'missing {prefix}')

def main():
    a=parse_args(); rows=[]
    if not a.binary.is_file(): raise SystemExit(f'binary not found: {a.binary}')
    with tempfile.TemporaryDirectory(prefix='betti_active_state_diag_') as td:
        root=Path(td)
        for mode in ('h0-scalar-stream','h2-scalar-stream'):
            for state in ('separate','parent-sentinel'):
                out=root/f'{mode}_{state}.csv'
                cmd=[str(a.binary.resolve()),str(a.input.resolve()),str(a.slab_depth),str(a.foreground_connectivity),mode,str(out),
                     '--merge-strategy','scan','--interface-order','radix','--event-order','verify',
                     '--f32-key-mode',a.f32_key_mode,'--neighbor-kernel','interior-fast',
                     '--representative-active-check','recheck','--union-kernel','root-carrying',
                     '--h0-pruning-cache','64k','--neighbor-root-check','parent-shortcut',
                     '--active-state',state,'--sweep-diagnostics']
                cp=subprocess.run(cmd,text=True,capture_output=True)
                log=cp.stdout+cp.stderr
                if cp.returncode: raise RuntimeError(f'{mode} {state} failed\n{log}')
                sweep=parse_kv_line(log,'PROFILE_SWEEP ')
                prep=parse_kv_line(log,'PROFILE_PREP ')
                digest,count=canonical_hash(out)
                row={'mode':mode,'active_state':state,'interval_rows':count,'canonical_sha256':digest}
                for k,v in sweep.items():
                    if v.isdigit(): row[k]=int(v)
                for k in ('local_sweep_seconds','total_voxels','interior_fast_voxels'):
                    if k in prep: row[k]=float(prep[k]) if 'seconds' in k else int(prep[k])
                checks=int(row.get('active_state_checks',0)); hits=int(row.get('active_neighbor_hits',0))
                finds=int(row.get('find_calls',0)); steps=int(row.get('find_parent_steps',0))
                row['active_hit_rate']=hits/checks if checks else 0.0
                row['parent_steps_per_find']=steps/finds if finds else 0.0
                rows.append(row)
    for mode in ('h0-scalar-stream','h2-scalar-stream'):
        vals={(r['canonical_sha256'],r['interval_rows']) for r in rows if r['mode']==mode}
        if len(vals)!=1: raise RuntimeError(f'{mode} persistence differs across active-state modes: {vals}')
        pair=[r for r in rows if r['mode']==mode]
        if pair[0].get('active_state_checks') != pair[1].get('active_state_checks'):
            raise RuntimeError(f'{mode} active-state check counts differ')
        if pair[0].get('active_neighbor_hits') != pair[1].get('active_neighbor_hits'):
            raise RuntimeError(f'{mode} active-neighbor results differ')
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['mode']} {r['active_state']}: checks={r.get('active_state_checks',0)} active_hits={r.get('active_neighbor_hits',0)} "
              f"hit_rate={r['active_hit_rate']:.2%} finds={r.get('find_calls',0)} parent_steps/find={r['parent_steps_per_find']:.3f} "
              f"sweep={r.get('local_sweep_seconds',float('nan')):.6f}s")
    print(f'PASS: active-state diagnostics agree structurally; wrote {a.output}')
if __name__=='__main__': main()
