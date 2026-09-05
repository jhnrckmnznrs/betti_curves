#!/usr/bin/env python3
"""Exactness gate for v1.26 outside-dominated structural pruning in hierarchical H2."""
from __future__ import annotations
import argparse,csv,os,re,subprocess,tempfile
from collections import Counter
from pathlib import Path
from f32_fixture import stage_f32_subset

PROFILE_RE=re.compile(
 r"PROFILE_H2_HIER_STREAM\s+leaf_slabs=(?P<leaf>\d+)\s+combines=(?P<combines>\d+)\s+"
 r"max_live_summaries=(?P<live>\d+)\s+max_pair_nodes=(?P<pair>\d+)\s+"
 r"max_pair_state_bytes=(?P<state>\d+)\s+final_interface_nodes=(?P<final>\d+)\s+"
 r"finalized_pair_bytes=(?P<pair_bytes>\d+)\s+root_attach_bytes=(?P<root_attach>\d+)\s+"
 r"root_outside_bytes=(?P<root_outside>\d+)\s+root_interface_bytes=(?P<root_interface>\d+)\s+"
 r"root_materialized=(?P<root>\S+)\s+disk_key_bytes=(?P<key>\d+).*?"
 r"global_h2_birth_state=(?P<birth>\S+)\s+global_h2_uf_layout=(?P<layout>\S+).*?"
 r"attach_finalized_early=(?P<finalized>\d+)\s+attach_propagated=(?P<propagated>\d+)\s+"
 r"outside_propagated=(?P<outside>\d+)\s+outside_structural_elided=(?P<elided>\d+).*?"
 r"h2_hier_cross_storage=(?P<cross>\S+)\s+h2_hier_outside_structural_pruning=(?P<pruning>\S+)"
)

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('input',type=Path)
 p.add_argument('--binary',type=Path,default=Path(os.environ.get('BETTI_CURVES_BINARY','target/release/betti_curves')))
 p.add_argument('--slice-limit',type=int,default=64); p.add_argument('--slab-depths',type=int,nargs='+',default=(4,8,16))
 p.add_argument('--foreground-connectivity',choices=(6,26),type=int,default=26); return p.parse_args()

def read_intervals(path):
 with path.open(newline='',encoding='utf-8') as h:
  r=csv.reader(h)
  if next(r,None)!=['birth','death']: raise RuntimeError(f'bad header: {path}')
  return Counter(tuple(row) for row in r if row)

def run(binary,inp,work,slab,fg,mode,extra=()):
 work.mkdir(parents=True,exist_ok=True); out=work/'intervals.csv'
 cmd=[str(binary.resolve()),str(inp.resolve()),str(slab),str(fg),mode,str(out),*extra]
 cp=subprocess.run(cmd,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 if cp.returncode: raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
 return read_intervals(out),cp.stdout

def equal(label,a,b):
 if a!=b:
  raise SystemExit(f"FAIL {label}: persistence multisets differ\nmissing={(a-b).most_common(12)}\nextra={(b-a).most_common(12)}")
 print(f"PASS {label}: {sum(a.values())} intervals agree exactly")

def profile(log,width,height,slab,expected):
 m=PROFILE_RE.search(log)
 if not m: raise SystemExit(f'FAIL d{slab} {expected}: missing/old PROFILE_H2_HIER_STREAM')
 if m.group('cross')!='direct': raise SystemExit(f"FAIL d{slab}: expected direct cross, got {m.group('cross')}")
 if m.group('pruning')!=expected: raise SystemExit(f"FAIL d{slab}: expected pruning={expected}, got {m.group('pruning')}")
 vals={k:int(m.group(k)) for k in ('pair','state','final','root_attach','root_outside','root_interface','key','finalized','propagated','outside','elided')}
 if m.group('root')!='false' or any(vals[k] for k in ('final','root_attach','root_outside','root_interface')):
  raise SystemExit(f'FAIL d{slab}: terminal-free root invariant failed')
 if vals['key']!=4 or m.group('birth')!='compact' or m.group('layout')!='packed':
  raise SystemExit(f'FAIL d{slab}: expected native32 compact/packed storage')
 face=width*height
 if vals['pair']>4*face: raise SystemExit(f"FAIL d{slab}: pair frontier {vals['pair']} > 4A={4*face}")
 if vals['state']>8*vals['pair']+4: raise SystemExit(f'FAIL d{slab}: pair state exceeds packed/native32 bound')
 if expected=='off' and vals['elided']!=0: raise SystemExit(f"FAIL d{slab}: off path reports {vals['elided']} elisions")
 print(f"PASS d{slab} {expected}: pair={vals['pair']} outside={vals['outside']:,} structural_elided={vals['elided']:,}")
 return vals

def main():
 a=parse_args()
 if not a.binary.is_file(): raise SystemExit(f'binary not found: {a.binary}')
 hp=subprocess.run([str(a.binary.resolve()),'--help'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout or ''
 if '--h2-hier-outside-structural-pruning' not in hp:
  raise SystemExit('FAIL: selected binary lacks --h2-hier-outside-structural-pruning; rebuild v1.26')
 with tempfile.TemporaryDirectory(prefix='betti_v126_h2_outside_') as td:
  root=Path(td); f32=stage_f32_subset(a.input,root/'f32_input',a.slice_limit)
  import tifffile
  first=tifffile.imread(sorted(f32.glob('*.tif*'))[0]); height,width=first.shape
  oracle,_=run(a.binary,f32,root/'oracle',max(a.slab_depths),a.foreground_connectivity,'h2-scalar')
  common=('--f32-key-mode','native32','--local-h2-birth-state','compact','--global-h2-birth-state','compact','--global-h2-uf-layout','packed','--h2-hier-cross-storage','direct')
  total_elided=0
  for slab in a.slab_depths:
   off,lo=run(a.binary,f32,root/f'd{slab}_off',slab,a.foreground_connectivity,'h2-scalar-hierarchical-stream',(*common,'--h2-hier-outside-structural-pruning','off'))
   pruned,lp=run(a.binary,f32,root/f'd{slab}_pruned',slab,a.foreground_connectivity,'h2-scalar-hierarchical-stream',(*common,'--h2-hier-outside-structural-pruning','outside-dominated'))
   equal(f'd{slab} off vs in-memory',oracle,off); equal(f'd{slab} pruned vs in-memory',oracle,pruned); equal(f'd{slab} pruned vs off',off,pruned)
   profile(lo,width,height,slab,'off'); total_elided+=profile(lp,width,height,slab,'outside-dominated')['elided']
  if total_elided==0: raise SystemExit('FAIL: outside-dominated path did not elide any structural events')
 print(f'PASS: v1.26 outside-dominated H2 structural pruning preserves exact persistence; elided={total_elided:,}')
if __name__=='__main__': main()
