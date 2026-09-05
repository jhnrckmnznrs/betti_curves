#!/usr/bin/env python3
"""Compare interface-representation diagnostics for vector vs root-invariant state."""
from __future__ import annotations
import argparse,csv,hashlib,subprocess,tempfile
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('--binary',type=Path,default=Path('target/release/betti_curves'))
    p.add_argument('--slab-depth',type=int,default=16); p.add_argument('--foreground-connectivity',choices=(6,26),type=int,default=26)
    p.add_argument('--f32-key-mode',choices=('legacy64','native32'),default='native32'); p.add_argument('--output',type=Path,default=Path('scalar_interface_state_diagnostics.csv')); return p.parse_args()

def hsh(path):
    with path.open(newline='',encoding='utf-8') as h:
        r=csv.reader(h); header=next(r,None)
        if header!=['birth','death']: raise RuntimeError(f'unexpected header {header!r}')
        rows=sorted(tuple(x) for x in r if x)
    d=hashlib.sha256(); [d.update(f'{b},{e}\n'.encode()) for b,e in rows]; return d.hexdigest(),len(rows)

def kv(log,prefix):
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
    with tempfile.TemporaryDirectory(prefix='betti_interface_state_diag_') as td:
        root=Path(td)
        for mode in ('h0-scalar-stream','h2-scalar-stream'):
            active='separate' if mode.startswith('h0') else 'parent-sentinel'
            for state in ('vector','root-invariant'):
                out=root/f'{mode}_{state}.csv'
                cmd=[str(a.binary.resolve()),str(a.input.resolve()),str(a.slab_depth),str(a.foreground_connectivity),mode,str(out),'--merge-strategy','scan','--interface-order','radix','--event-order','verify','--f32-key-mode',a.f32_key_mode,'--neighbor-kernel','interior-fast','--representative-active-check','recheck','--union-kernel','root-carrying','--h0-pruning-cache','64k','--neighbor-root-check','parent-shortcut','--active-state',active,'--interface-state',state,'--sweep-diagnostics']
                cp=subprocess.run(cmd,text=True,capture_output=True); log=cp.stdout+cp.stderr
                if cp.returncode: raise RuntimeError(f'{mode} {state} failed\n{log}')
                sweep=kv(log,'PROFILE_SWEEP '); prep=kv(log,'PROFILE_PREP '); digest,count=hsh(out)
                row={'mode':mode,'interface_state':state,'active_state':active,'interval_rows':count,'canonical_sha256':digest}
                for k,v in sweep.items():
                    try: row[k]=int(v)
                    except ValueError: pass
                for k in ('local_sweep_seconds','total_voxels','interior_fast_voxels'):
                    if k in prep: row[k]=float(prep[k]) if 'seconds' in k else int(prep[k])
                finds=int(row.get('find_calls',0)); steps=int(row.get('find_parent_steps',0)); row['parent_steps_per_find']=steps/finds if finds else 0.0
                unions=int(row.get('successful_unions',0)); forced=int(row.get('interface_forced_root_unions',0)); row['forced_interface_root_fraction']=forced/unions if unions else 0.0
                rows.append(row)
    for mode in ('h0-scalar-stream','h2-scalar-stream'):
        pair=[r for r in rows if r['mode']==mode]; vals={(r['canonical_sha256'],r['interval_rows']) for r in pair}
        if len(vals)!=1: raise RuntimeError(f'{mode} persistence differs: {vals}')
        root=next(r for r in pair if r['interface_state']=='root-invariant')
        vector=next(r for r in pair if r['interface_state']=='vector')
        if root.get('interface_state_bytes') != 0: raise RuntimeError(f'{mode} root-invariant still reports interface-state allocation')
        if root.get('interface_rep_writes') != 0: raise RuntimeError(f'{mode} root-invariant performed interface_rep writes')
        if int(vector.get('interface_state_bytes',0)) <= 0: raise RuntimeError(f'{mode} vector reports no interface-state allocation')
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['mode']} {r['interface_state']}: state_bytes={r.get('interface_state_bytes',0)} queries={r.get('interface_rep_queries',0)} writes={r.get('interface_rep_writes',0)} forced_roots={r.get('interface_forced_root_unions',0)} interface-interface={r.get('interface_interface_unions',0)} max_rank={r.get('max_rank_observed',0)} parent_steps/find={r['parent_steps_per_find']:.3f} sweep={r.get('local_sweep_seconds',float('nan')):.6f}s")
    print(f'PASS: interface-state diagnostics preserve exact outputs and root-invariant allocates no interface_rep vector; wrote {a.output}')
if __name__=='__main__': main()
