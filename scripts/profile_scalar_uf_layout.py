#!/usr/bin/env python3
"""Balanced wall/RSS profiler for parent-rank versus packed local UF layout."""
from __future__ import annotations
import argparse, csv, hashlib, re, shutil, statistics, subprocess, tempfile, time
from collections import defaultdict
from pathlib import Path

PROFILE_RE = re.compile(r"PROFILE\s+(?P<name>\S+)\s+prepare_seconds=(?P<prepare>[0-9.]+)\s+reduce_seconds=(?P<reduce>[0-9.]+)\s+cleanup_seconds=(?P<cleanup>[0-9.]+)\s+total_seconds=(?P<total>[0-9.]+)")
PREP_RE = re.compile(r"PROFILE_PREP\s+(?P<name>\S+)\s+decode_seconds=(?P<decode>[0-9.]+)\s+key_conversion_seconds=(?P<key_conversion>[0-9.]+)\s+slab_copy_seconds=(?P<slab_copy>[0-9.]+)\s+scalar_order_seconds=(?P<scalar_order>[0-9.]+)\s+local_sweep_seconds=(?P<local_sweep>[0-9.]+)\s+local_run_write_seconds=(?P<local_run_write>[0-9.]+)\s+cross_interface_seconds=(?P<cross_interface>[0-9.]+)\s+unaccounted_seconds=(?P<unaccounted>[0-9.]+)\s+total_voxels=(?P<total_voxels>[0-9]+)\s+interior_fast_voxels=(?P<interior_fast_voxels>[0-9]+)")
TIME_FIELDS={'User time (seconds)':'user_seconds','System time (seconds)':'system_seconds','Maximum resident set size (kbytes)':'max_rss_kb','File system inputs':'fs_inputs','File system outputs':'fs_outputs','Major (requiring I/O) page faults':'major_faults','Minor (reclaiming a frame) page faults':'minor_faults'}


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('--binary',type=Path,default=Path('target/release/betti_curves'))
    p.add_argument('--slab-depth',type=int,default=16); p.add_argument('--foreground-connectivity',choices=(6,26),type=int,default=26)
    p.add_argument('--repeats',type=int,default=5); p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--modes',nargs='+',choices=('h0-scalar-stream','h2-scalar-stream'),default=('h0-scalar-stream','h2-scalar-stream'))
    p.add_argument('--f32-key-mode',choices=('legacy64','native32'),default='native32'); p.add_argument('--output',type=Path,default=Path('scalar_uf_layout_profile.csv')); p.add_argument('--keep-logs',action='store_true'); return p.parse_args()


def canonical_hash(path):
    with path.open(newline='',encoding='utf-8') as h:
        r=csv.reader(h); header=next(r,None)
        if header!=['birth','death']: raise RuntimeError(f'unexpected persistence header {header!r}')
        rows=sorted(tuple(row) for row in r if row)
    d=hashlib.sha256()
    for b,e in rows: d.update(f'{b},{e}\n'.encode())
    return d.hexdigest(),len(rows)


def parse_time(path):
    out={}
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        if ':' not in line: continue
        k,v=line.strip().split(':',1); f=TIME_FIELDS.get(k)
        if f:
            try: out[f]=float(v.strip())
            except ValueError: pass
    return out


def parse_internal(log):
    out={}; m=PROFILE_RE.search(log)
    if m: out.update(prepare_seconds=float(m.group('prepare')),reduce_seconds=float(m.group('reduce')),cleanup_seconds=float(m.group('cleanup')),internal_total_seconds=float(m.group('total')))
    m=PREP_RE.search(log)
    if m:
        for n in ('decode','key_conversion','slab_copy','scalar_order','local_sweep','local_run_write','cross_interface','unaccounted'): out[f'{n}_seconds']=float(m.group(n))
    return out


def run_once(a,layout,mode,repeat,sequence,root,record):
    tag='warmup' if not record else f'r{repeat:03d}'; rd=root/f'{sequence:04d}_{mode}_{layout}_{tag}'; rd.mkdir(parents=True)
    intervals=rd/'intervals.csv'; timing=rd/'time.txt'; log=rd/'run.log'; active='separate' if mode.startswith('h0') else 'parent-sentinel'
    cmd=['/usr/bin/time','-v','-o',str(timing),str(a.binary.resolve()),str(a.input.resolve()),str(a.slab_depth),str(a.foreground_connectivity),mode,str(intervals),
         '--merge-strategy','scan','--interface-order','radix','--event-order','verify','--f32-key-mode',a.f32_key_mode,'--neighbor-kernel','interior-fast','--representative-active-check','recheck','--union-kernel','root-carrying','--h0-pruning-cache','64k','--neighbor-root-check','parent-shortcut','--active-state',active,'--interface-state','root-invariant','--uf-layout',layout]
    t=time.perf_counter(); cp=subprocess.run(cmd,cwd=rd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); wall=time.perf_counter()-t; log.write_text(cp.stdout,encoding='utf-8')
    if cp.returncode: raise RuntimeError(f'{mode} {layout} failed; see {log}')
    if not record:
        if not a.keep_logs: shutil.rmtree(rd,ignore_errors=True)
        return None
    digest,count=canonical_hash(intervals); row={'mode':mode,'uf_layout':layout,'active_state':active,'repeat':repeat,'run_sequence':sequence,'wall_seconds':wall,'interval_rows':count,'canonical_sha256':digest,**parse_time(timing),**parse_internal(cp.stdout)}
    if not a.keep_logs: shutil.rmtree(rd,ignore_errors=True)
    return row


def med(rows,f):
    vals=[float(r[f]) for r in rows if r.get(f,'')!='']; return statistics.median(vals) if vals else float('nan')


def main():
    a=parse_args(); rows=[]; seq=0
    if not a.binary.is_file(): raise SystemExit(f'binary not found: {a.binary}')
    with tempfile.TemporaryDirectory(prefix='betti_uf_layout_profile_') as td:
        root=Path(td)
        for mode in a.modes:
            for _ in range(a.warmup):
                for layout in ('parent-rank','packed'): run_once(a,layout,mode,-1,seq,root,False); seq+=1
            for repeat in range(a.repeats):
                order=('parent-rank','packed') if repeat%2==0 else ('packed','parent-rank')
                for layout in order: rows.append(run_once(a,layout,mode,repeat,seq,root,True)); seq+=1
    by=defaultdict(set)
    for r in rows: by[r['mode']].add((r['canonical_sha256'],r['interval_rows']))
    bad={k:v for k,v in by.items() if len(v)!=1}
    if bad: raise RuntimeError(f'UF-layout outputs disagree: {bad}')
    fields=['mode','uf_layout','active_state','repeat','run_sequence','wall_seconds','user_seconds','system_seconds','max_rss_kb','prepare_seconds','local_sweep_seconds','scalar_order_seconds','local_run_write_seconds','cross_interface_seconds','reduce_seconds','interval_rows','canonical_sha256']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); [w.writerow({f:r.get(f,'') for f in fields}) for r in rows]
    groups=defaultdict(list)
    for r in rows: groups[(r['mode'],r['uf_layout'])].append(r)
    print('\nMedian UF-layout ablation'); print('mode\tlayout\twall\tprepare\tsweep\treduce\tRSS_MiB')
    for key in sorted(groups):
        g=groups[key]; print(f"{key[0]}\t{key[1]}\t{med(g,'wall_seconds'):.6f}\t{med(g,'prepare_seconds'):.6f}\t{med(g,'local_sweep_seconds'):.6f}\t{med(g,'reduce_seconds'):.6f}\t{med(g,'max_rss_kb')/1024:.2f}")
    for mode in a.modes:
        ref=groups[(mode,'parent-rank')]; packed=groups[(mode,'packed')]
        print(f"speedup {mode} wall: {med(ref,'wall_seconds')/med(packed,'wall_seconds'):.4f}x; sweep: {med(ref,'local_sweep_seconds')/med(packed,'local_sweep_seconds'):.4f}x; RSS change: {(med(packed,'max_rss_kb')/med(ref,'max_rss_kb')-1)*100:.2f}%")
    print(f'PASS: exact persistence agreement across UF layouts; wrote {a.output}')

if __name__=='__main__': main()
