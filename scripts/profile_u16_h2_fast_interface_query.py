#!/usr/bin/env python3
"""Paired A/B benchmark for the U16 H2 fast root-invariant interface query.

Reference: BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=0
Candidate: BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=1

Both use the production d16-era kernel settings, including root-invariant
interface state, parent-shortcut neighbor root checks, and root dedup off.
"""
from __future__ import annotations
import argparse, csv, math, os, re, statistics, subprocess, tempfile, time
from pathlib import Path

RSS_RE=re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
USER_RE=re.compile(r"User time \(seconds\):\s*([0-9.]+)")
SYS_RE=re.compile(r"System time \(seconds\):\s*([0-9.]+)")
LEAF_PREFIX="PROFILE_U16_PERSIST_LEAF dimension=h2 "
COUNTERS=("zero_persistence_pairs_elided","union_attempts","successful_unions","same_root_unions",
          "neighbor_find_calls","neighbor_find_parent_steps")
INVARIANTS=COUNTERS

def command(binary:Path, stack:Path, depth:int, conn:int, out:Path, diagnostics:bool=False)->list[str]:
    cmd=[str(binary),str(stack),str(depth),str(conn),"h2-scalar-hierarchical-stream",str(out),
         "--local-h2-birth-state","compact",
         "--global-h2-birth-state","compact",
         "--global-h2-uf-layout","packed",
         "--h2-hier-cross-storage","direct",
         "--h2-hier-outside-structural-pruning","off",
         "--neighbor-root-check","parent-shortcut",
         "--interface-state","root-invariant"]
    if diagnostics: cmd.append("--sweep-diagnostics")
    return cmd

def parse_leaf(stdout:str)->dict[str,str]:
    line=next((x for x in stdout.splitlines() if x.startswith(LEAF_PREFIX)),None)
    if line is None: raise RuntimeError("missing PROFILE_U16_PERSIST_LEAF")
    out={}
    for token in line.split():
        if "=" in token:
            k,v=token.split("=",1); out[k]=v
    return out

def req(pat,text,label):
    m=pat.search(text)
    if m is None: raise RuntimeError(f"missing {label}")
    return m.group(1)

def run_once(cmd:list[str], timing_path:Path, fast:bool)->dict[str,object]:
    env=os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"]="1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"]="1"
    env["BETTI_PERSIST_H2_ROOT_DEDUP"]="0"
    env["BETTI_PERSIST_H2_FAST_INTERFACE_QUERY"]="1" if fast else "0"
    start=time.perf_counter()
    proc=subprocess.run(["/usr/bin/time","-v","-o",str(timing_path),*cmd],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    wall=time.perf_counter()-start
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    expected="on" if fast else "off"
    if f"h2_fast_interface_query={expected}" not in proc.stdout:
        raise RuntimeError(f"expected h2_fast_interface_query={expected}")
    if "h2_root_dedup=off" not in proc.stdout:
        raise RuntimeError("benchmark requires root dedup off")
    leaf=parse_leaf(proc.stdout); t=timing_path.read_text()
    out={"wall_seconds":wall,
         "user_seconds":float(req(USER_RE,t,"user time")),
         "system_seconds":float(req(SYS_RE,t,"system time")),
         "max_rss_mib":float(req(RSS_RE,t,"RSS"))/1024.0,
         "scalar_order_seconds":float(leaf["scalar_order_seconds"]),
         "local_sweep_seconds":float(leaf["local_sweep_seconds"])}
    for k in COUNTERS: out[k]=int(leaf[k])
    return out

def median(xs): return float(statistics.median(float(x) for x in xs))
def mad(xs):
    vals=[float(x) for x in xs]; c=median(vals); return median(abs(x-c) for x in vals)
def iqr(xs):
    vals=[float(x) for x in xs]
    if len(vals)<2:return 0.0
    q1,_,q3=statistics.quantiles(vals,n=4,method="inclusive"); return float(q3-q1)
def pct(a,b): return 100.0*(b-a)/a if a else 0.0

def sign_test(deltas):
    vals=[x for x in deltas if x!=0.0]; n=len(vals)
    if not n:return 0,0,1.0
    wins=sum(x<0 for x in vals); k=min(wins,n-wins)
    tail=sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return wins,n,min(1.0,2.0*tail)

def pair_order(pair_id,start_with):
    first=start_with if pair_id%2 else ("candidate" if start_with=="reference" else "reference")
    return first,("candidate" if first=="reference" else "reference")

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("stack",type=Path)
    ap.add_argument("--binary",type=Path,default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths",nargs="+",type=int,default=[16,32,64])
    ap.add_argument("--foreground-connectivity",type=int,default=26,choices=(6,18,26))
    ap.add_argument("--warmup",type=int,default=1)
    ap.add_argument("--repeats",type=int,default=11)
    ap.add_argument("--start-with",choices=("reference","candidate"),default="reference")
    ap.add_argument("--output",type=Path,default=Path("profiles/v23_5_u16_h2_fast_interface_query_paired.csv"))
    args=ap.parse_args()
    if args.warmup<0 or args.repeats<2: ap.error("require warmup>=0 and repeats>=2")
    if any(d<=0 for d in args.slab_depths) or len(set(args.slab_depths))!=len(args.slab_depths): ap.error("slab depths must be positive and unique")
    stack=args.stack.resolve(); binary=args.binary.resolve()
    if not stack.exists(): ap.error(f"stack does not exist: {stack}")
    if not binary.exists(): ap.error(f"binary does not exist: {binary}")
    args.output.parent.mkdir(parents=True,exist_ok=True)

    rows=[]; pairs=[]; seq=0
    with tempfile.TemporaryDirectory(prefix="u16_h2_fast_iface_") as td_raw:
        td=Path(td_raw)
        for depth in args.slab_depths:
            for warm in range(1,args.warmup+1):
                for backend in pair_order(warm,args.start_with):
                    seq+=1; fast=backend=="candidate"
                    print(f"warmup={warm} {backend} d={depth}")
                    run_once(command(binary,stack,depth,args.foreground_connectivity,td/f"w{seq}.csv"),td/f"w{seq}.time",fast)
            for pair_id in range(1,args.repeats+1):
                order=pair_order(pair_id,args.start_with); current={}
                for pos,backend in enumerate(order,1):
                    seq+=1; fast=backend=="candidate"
                    r=run_once(command(binary,stack,depth,args.foreground_connectivity,td/f"r{seq}.csv"),td/f"r{seq}.time",fast)
                    row={"backend":backend,"slab_depth":depth,"pair_id":pair_id,"run_position":pos,
                         "pair_order":"-".join(order),"sequence":seq,**r}
                    rows.append(row); current[backend]=row
                    print(f"pair={pair_id:02d} {'/'.join(order)} pos={pos} {backend} d={depth}: wall={r['wall_seconds']:.3f}s sweep={r['local_sweep_seconds']:.3f}s")
                ref,cand=current["reference"],current["candidate"]
                for m in INVARIANTS:
                    if int(ref[m])!=int(cand[m]):
                        raise RuntimeError(f"invariant failed d={depth} pair={pair_id}: {m} {ref[m]} != {cand[m]}")
                pr={"slab_depth":depth,"pair_id":pair_id,"pair_order":"-".join(order)}
                for m in ("wall_seconds","local_sweep_seconds","user_seconds","system_seconds","max_rss_mib"):
                    a=float(ref[m]); b=float(cand[m])
                    pr[f"reference_{m}"]=a; pr[f"candidate_{m}"]=b; pr[f"delta_{m}"]=b-a; pr[f"pct_delta_{m}"]=pct(a,b)
                pairs.append(pr)

    with args.output.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary=args.output.with_name(args.output.stem+"_summary.csv")
    fields=["backend","slab_depth","repeats"]
    metrics=("wall_seconds","local_sweep_seconds","user_seconds","system_seconds","max_rss_mib")
    for m in metrics: fields += [f"median_{m}",f"mad_{m}",f"iqr_{m}"]
    with summary.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
        for backend in ("reference","candidate"):
            for depth in args.slab_depths:
                rs=[r for r in rows if r["backend"]==backend and int(r["slab_depth"])==depth]
                out={"backend":backend,"slab_depth":depth,"repeats":len(rs)}
                for m in metrics:
                    vals=[float(r[m]) for r in rs]
                    out[f"median_{m}"]=median(vals); out[f"mad_{m}"]=mad(vals); out[f"iqr_{m}"]=iqr(vals)
                w.writerow(out)

    pairs_path=args.output.with_name(args.output.stem+"_pairs.csv")
    with pairs_path.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)

    paired=args.output.with_name(args.output.stem+"_paired_summary.csv")
    pfields=["slab_depth","pairs","median_pct_delta_wall_seconds","mad_pct_delta_wall_seconds","iqr_pct_delta_wall_seconds",
             "median_pct_delta_local_sweep_seconds","mad_pct_delta_local_sweep_seconds","iqr_pct_delta_local_sweep_seconds",
             "candidate_faster_wall_pairs","non_tied_wall_pairs","candidate_faster_wall_fraction","two_sided_sign_test_p_wall",
             "candidate_faster_local_sweep_pairs","non_tied_local_sweep_pairs","candidate_faster_local_sweep_fraction","two_sided_sign_test_p_local_sweep",
             "median_pct_delta_wall_reference_first","median_pct_delta_wall_candidate_first","wall_order_effect_pp"]
    with paired.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=pfields); w.writeheader()
        for depth in args.slab_depths:
            rs=[r for r in pairs if int(r["slab_depth"])==depth]
            wall=[float(r["pct_delta_wall_seconds"]) for r in rs]; sweep=[float(r["pct_delta_local_sweep_seconds"]) for r in rs]
            ww,wn,wp=sign_test([float(r["delta_wall_seconds"]) for r in rs]); sw,sn,sp=sign_test([float(r["delta_local_sweep_seconds"]) for r in rs])
            rf=[float(r["pct_delta_wall_seconds"]) for r in rs if r["pair_order"]=="reference-candidate"]
            cf=[float(r["pct_delta_wall_seconds"]) for r in rs if r["pair_order"]=="candidate-reference"]
            rfm=median(rf) if rf else 0.0; cfm=median(cf) if cf else 0.0
            out={"slab_depth":depth,"pairs":len(rs),
                 "median_pct_delta_wall_seconds":median(wall),"mad_pct_delta_wall_seconds":mad(wall),"iqr_pct_delta_wall_seconds":iqr(wall),
                 "median_pct_delta_local_sweep_seconds":median(sweep),"mad_pct_delta_local_sweep_seconds":mad(sweep),"iqr_pct_delta_local_sweep_seconds":iqr(sweep),
                 "candidate_faster_wall_pairs":ww,"non_tied_wall_pairs":wn,"candidate_faster_wall_fraction":ww/wn if wn else 0.0,"two_sided_sign_test_p_wall":wp,
                 "candidate_faster_local_sweep_pairs":sw,"non_tied_local_sweep_pairs":sn,"candidate_faster_local_sweep_fraction":sw/sn if sn else 0.0,"two_sided_sign_test_p_local_sweep":sp,
                 "median_pct_delta_wall_reference_first":rfm,"median_pct_delta_wall_candidate_first":cfm,"wall_order_effect_pp":cfm-rfm}
            w.writerow(out)
            print(f"d{depth}: median wall {out['median_pct_delta_wall_seconds']:+.3f}% | sweep {out['median_pct_delta_local_sweep_seconds']:+.3f}% | wall wins {ww}/{wn} p={wp:.5f}")
    print(f"wrote {args.output}\nwrote {summary}\nwrote {pairs_path}\nwrote {paired}")
    return 0

if __name__=="__main__": raise SystemExit(main())
