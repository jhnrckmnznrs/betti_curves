#!/usr/bin/env python3
"""Single-pass operation diagnostics for parent-rank versus packed local UF layout."""
from __future__ import annotations
import argparse,csv,hashlib,subprocess,tempfile
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('--binary',type=Path,default=Path('target/release/betti_curves')); p.add_argument('--slab-depth',type=int,default=16); p.add_argument('--foreground-connectivity',choices=(6,26),type=int,default=26); p.add_argument('--f32-key-mode',choices=('legacy64','native32'),default='native32'); p.add_argument('--output',type=Path,default=Path('scalar_uf_layout_diagnostics.csv')); return p.parse_args()


def hsh(path):
    with path.open(newline='',encoding='utf-8') as h:
        r=csv.reader(h); header=next(r,None)
        if header!=['birth','death']: raise RuntimeError(f'unexpected header {header!r}')
        rows=sorted(tuple(x) for x in r if x)
    d=hashlib.sha256()
    for b,e in rows: d.update(f'{b},{e}\n'.encode())
    return d.hexdigest(),len(rows)


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
    with tempfile.TemporaryDirectory(prefix='betti_uf_layout_diag_') as td:
        root=Path(td)
        for mode in ('h0-scalar-stream','h2-scalar-stream'):
            active='separate' if mode.startswith('h0') else 'parent-sentinel'
            for layout in ('parent-rank','packed'):
                out=root/f'{mode}_{layout}.csv'
                cmd=[str(a.binary.resolve()),str(a.input.resolve()),str(a.slab_depth),str(a.foreground_connectivity),mode,str(out),'--merge-strategy','scan','--interface-order','radix','--event-order','verify','--f32-key-mode',a.f32_key_mode,'--neighbor-kernel','interior-fast','--representative-active-check','recheck','--union-kernel','root-carrying','--h0-pruning-cache','64k','--neighbor-root-check','parent-shortcut','--active-state',active,'--interface-state','root-invariant','--uf-layout',layout,'--sweep-diagnostics']
                cp=subprocess.run(cmd,text=True,capture_output=True)
                if cp.returncode: raise RuntimeError(f"failed: {' '.join(cmd)}\n{cp.stdout}{cp.stderr}")
                digest,count=hsh(out); d=kv(cp.stdout+cp.stderr,'PROFILE_SWEEP')
                row={'mode':mode,'uf_layout':layout,'active_state':active,'interval_rows':count,'canonical_sha256':digest,**d}
                rows.append(row)
    for mode in ('h0-scalar-stream','h2-scalar-stream'):
        g=[r for r in rows if r['mode']==mode]
        if len({(r['canonical_sha256'],r['interval_rows']) for r in g})!=1: raise RuntimeError(f'{mode} persistence differs across layouts')
        ref=next(r for r in g if r['uf_layout']=='parent-rank'); packed=next(r for r in g if r['uf_layout']=='packed')
        if int(packed.get('uf_rank_state_bytes','-1'))!=0: raise RuntimeError(f'{mode} packed layout still reports rank-state bytes')
        if int(ref.get('uf_rank_state_bytes','0'))<=0: raise RuntimeError(f'{mode} parent-rank layout did not report rank-state bytes')
        for field in ('union_attempts','successful_unions','same_root_unions'):
            if ref.get(field)!=packed.get(field): raise RuntimeError(f'{mode} {field} differs: {ref.get(field)} vs {packed.get(field)}')
    fields=['mode','uf_layout','active_state','interval_rows','canonical_sha256','uf_parent_state_bytes','uf_rank_state_bytes','max_rank_observed','find_calls','find_parent_steps','union_attempts','successful_unions','same_root_unions','interface_forced_root_unions']
    with a.output.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); [w.writerow({f:r.get(f,'') for f in fields}) for r in rows]
    print('mode\tlayout\trank_MiB\tfinds\tparent_steps\tsteps/find\tmax_rank')
    for r in rows:
        finds=int(r.get('find_calls',0)); steps=int(r.get('find_parent_steps',0)); rankb=int(r.get('uf_rank_state_bytes',0)); ratio=steps/finds if finds else 0
        print(f"{r['mode']}\t{r['uf_layout']}\t{rankb/1048576:.3f}\t{finds}\t{steps}\t{ratio:.6f}\t{r.get('max_rank_observed','')}")
    print(f'PASS: packed layout removes rank vector and preserves exact persistence; wrote {a.output}')

if __name__=='__main__': main()
