#!/usr/bin/env python3
"""Profile flat H2 and hierarchical disk-vs-direct cross-interface storage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median

import sys
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "validation")))
from f32_fixture import stage_f32_subset

TIME_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(?P<v>\d+)")
USER_RE = re.compile(r"User time \(seconds\):\s*(?P<v>[0-9.]+)")
SYS_RE = re.compile(r"System time \(seconds\):\s*(?P<v>[0-9.]+)")
FS_IN_RE = re.compile(r"File system inputs:\s*(?P<v>\d+)")
FS_OUT_RE = re.compile(r"File system outputs:\s*(?P<v>\d+)")
WALL_RE = re.compile(
    r"^\s*Elapsed \(wall clock\) time .*?\):\s*"
    r"(?P<v>\d+(?::\d+){0,2}(?:\.\d+)?)\s*$", re.MULTILINE,
)
HIER_RE = re.compile(
    r"PROFILE_H2_HIER_STREAM\s+leaf_slabs=(?P<leaf>\d+)\s+combines=(?P<combines>\d+)\s+"
    r"max_live_summaries=(?P<live>\d+)\s+max_pair_nodes=(?P<pair>\d+)\s+"
    r"max_pair_state_bytes=(?P<state>\d+)\s+final_interface_nodes=(?P<final>\d+)\s+"
    r"finalized_pair_bytes=(?P<pair_bytes>\d+)\s+root_attach_bytes=(?P<root_attach>\d+)\s+"
    r"root_outside_bytes=(?P<root_outside>\d+)\s+root_interface_bytes=(?P<root_interface>\d+)\s+"
    r"root_materialized=(?P<root>\S+)\s+disk_key_bytes=(?P<key>\d+).*?"
    r"leaf_attach_events=(?P<leaf_attach>\d+)\s+leaf_outside_events=(?P<leaf_outside>\d+)\s+"
    r"attach_finalized_early=(?P<finalized>\d+)\s+attach_propagated=(?P<propagated>\d+)\s+"
    r"outside_propagated=(?P<outside_prop>\d+).*?h2_hier_cross_storage=(?P<cross_storage>\S+)"
)
CROSS_LINE_RE = re.compile(
    r"^PROFILE_H2_HIER_STREAM_(?:COMBINE|FINAL)\b.*$", re.MULTILINE
)
CROSS_RETAINED_RE = re.compile(r"\bcross_retained=(?P<retained>\d+)\b")
CROSS_STORAGE_RE = re.compile(r"\bcross_storage=(?P<storage>\S+)")
FLAT_STORAGE_RE = re.compile(
    r"PROFILE_H2_STORAGE scalar_h2_stream pipeline=(?P<pipeline>\S+) disk_key_bytes=(?P<key>\d+) "
    r"interface_nodes=(?P<nodes>\d+) interface_birth_bytes=(?P<birth_bytes>\d+) "
    r"local_pair_bytes=(?P<pair_bytes>\d+) attach_bytes=(?P<attach>\d+) outside_bytes=(?P<outside>\d+) "
    r"interface_bytes=(?P<interface>\d+) cross_bytes=(?P<cross>\d+) total_run_bytes=(?P<total>\d+)"
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("BETTI_CURVES_BINARY","target/release/betti_curves")))
    p.add_argument("--slice-limit", type=int, default=0)
    p.add_argument("--slab-depths", type=int, nargs="+", default=(8,16,32))
    p.add_argument("--foreground-connectivity", choices=(6,26), type=int, default=26)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("cx09t1_v1_25_h2_cross_storage.csv"))
    return p.parse_args()


def wall_seconds(text:str)->float:
    parts=text.strip().split(":")
    if len(parts)==1: return float(parts[0])
    if len(parts)==2: return float(parts[0])*60+float(parts[1])
    if len(parts)==3: return float(parts[0])*3600+float(parts[1])*60+float(parts[2])
    raise ValueError(text)


def counter(path:Path):
    with path.open(newline="",encoding="utf-8") as h:
        r=csv.reader(h)
        if next(r,None)!=["birth","death"]: raise RuntimeError(f"bad header: {path}")
        return Counter(tuple(row) for row in r if row)


def digest(c:Counter):
    h=hashlib.sha256()
    for (b,d),n in sorted(c.items()):
        h.update(f"{b},{d},{n}\n".encode())
    return h.hexdigest()


def run_one(time_bin,binary,inp,work,slab,fg,variant):
    work.mkdir(parents=True,exist_ok=True)
    out=work/"intervals.csv"
    common=("--f32-key-mode","native32","--local-h2-birth-state","compact",
            "--global-h2-birth-state","compact","--global-h2-uf-layout","packed")
    if variant=="flat":
        mode="h2-scalar-stream"; extra=common
    elif variant in ("hier-disk","hier-direct"):
        mode="h2-scalar-hierarchical-stream"
        storage=variant.split("-",1)[1]
        extra=(*common,"--h2-hier-cross-storage",storage)
    else: raise ValueError(variant)
    cmd=[time_bin,"-v",str(binary.resolve()),str(inp.resolve()),str(slab),str(fg),mode,str(out),*extra]
    cp=subprocess.run(cmd,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if cp.returncode: raise RuntimeError(f"failed: {' '.join(cmd)}\n{cp.stdout}")
    log=cp.stdout
    def one(rx):
        m=rx.search(log)
        if not m: raise RuntimeError(f"missing timing field {rx.pattern}")
        return m.group("v")
    c=counter(out)
    wm=WALL_RE.search(log)
    if not wm: raise RuntimeError("missing GNU-time elapsed line")
    row=dict(
        variant=variant, slab_depth=slab,
        wall_seconds=wall_seconds(wm.group("v")), user_seconds=float(one(USER_RE)),
        system_seconds=float(one(SYS_RE)), peak_rss_kib=int(one(TIME_RE)),
        fs_inputs=int(one(FS_IN_RE)), fs_outputs=int(one(FS_OUT_RE)),
        interval_count=sum(c.values()), canonical_sha256=digest(c),
    )
    if variant=="flat":
        m=FLAT_STORAGE_RE.search(log)
        if not m: raise RuntimeError("missing flat H2 storage profile")
        row.update(flat_total_run_bytes=int(m.group("total")), flat_cross_bytes=int(m.group("cross")))
    else:
        m=HIER_RE.search(log)
        if not m: raise RuntimeError("missing hierarchical H2 profile")
        for k in ("leaf","combines","live","pair","state","pair_bytes","leaf_attach","leaf_outside","finalized","propagated","outside_prop"):
            row[k]=int(m.group(k))
        row["root_materialized"]=m.group("root")
        row["cross_storage"]=m.group("cross_storage")
        retained=[]
        cross_line_storage=[]
        for line in CROSS_LINE_RE.findall(log):
            retained_match=CROSS_RETAINED_RE.search(line)
            if not retained_match:
                continue
            storage_match=CROSS_STORAGE_RE.search(line)
            retained.append(int(retained_match.group("retained")))
            cross_line_storage.append(storage_match.group("storage") if storage_match else "disk")
        row["hier_cross_retained_total"]=sum(retained)
        row["hier_cross_run_equiv_bytes"]=sum(retained)*12  # native32 pair event: 4-byte key + 2*u32
        row["cross_line_storage_consistent"]=all(x==row["cross_storage"] for x in cross_line_storage)
    return row,c


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r})
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    a=parse_args()
    time_bin=shutil.which("/usr/bin/time") or shutil.which("time")
    if not a.binary.is_file(): raise SystemExit(f"binary not found: {a.binary}")
    help_cp=subprocess.run([str(a.binary.resolve()),"--help"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if "--h2-hier-cross-storage" not in (help_cp.stdout or ""):
        raise SystemExit("binary does not support --h2-hier-cross-storage; rebuild v1.25")
    with tempfile.TemporaryDirectory(prefix="betti_v125_h2_cross_profile_") as td:
        root=Path(td); f32=stage_f32_subset(a.input,root/"f32_input",a.slice_limit)
        rows=[]
        variants=("flat","hier-disk","hier-direct")
        for slab in a.slab_depths:
            reference=None
            for rep in range(1,a.repeats+1):
                shift=(rep-1)%len(variants)
                order=variants[shift:]+variants[:shift]
                results={}
                for variant in order:
                    print(f"RUN d{slab} repeat={rep} {variant}",flush=True)
                    row,c=run_one(time_bin,a.binary,f32,root/f"d{slab}_r{rep}_{variant}",slab,a.foreground_connectivity,variant)
                    row["repeat"]=rep; rows.append(row); results[variant]=c; write_csv(a.output,rows)
                if not (results["flat"]==results["hier-disk"]==results["hier-direct"]):
                    ref=results["flat"]
                    for variant in ("hier-disk","hier-direct"):
                        if results[variant]!=ref:
                            raise SystemExit(
                                f"FAIL d{slab} repeat={rep} {variant} persistence differs\n"
                                f"missing={(ref-results[variant]).most_common(10)}\n"
                                f"extra={(results[variant]-ref).most_common(10)}")
                if reference is None: reference=results["flat"]
                elif reference!=results["flat"]: raise SystemExit(f"FAIL d{slab}: repeat persistence changed")
        write_csv(a.output,rows)
        summary=[]
        for slab in a.slab_depths:
            for variant in variants:
                rr=[r for r in rows if r["slab_depth"]==slab and r["variant"]==variant]
                r0=rr[0]
                summary.append(dict(
                    slab_depth=slab,variant=variant,repeats=len(rr),
                    median_wall_seconds=median(r["wall_seconds"] for r in rr),
                    median_user_seconds=median(r["user_seconds"] for r in rr),
                    median_system_seconds=median(r["system_seconds"] for r in rr),
                    median_peak_rss_mib=median(r["peak_rss_kib"] for r in rr)/1024,
                    median_fs_inputs=median(r["fs_inputs"] for r in rr),
                    median_fs_outputs=median(r["fs_outputs"] for r in rr),
                    interval_count=r0["interval_count"],canonical_sha256=r0["canonical_sha256"],
                    max_pair_nodes=r0.get("pair",""),
                    attach_finalized_early=r0.get("finalized",""),attach_propagated=r0.get("propagated",""),
                    outside_propagated=r0.get("outside_prop",""),
                    hierarchical_cross_events=r0.get("hier_cross_retained_total",""),
                    hierarchical_cross_run_equiv_mib=(r0.get("hier_cross_run_equiv_bytes",0)/2**20) if variant!="flat" else "",
                    flat_total_run_mib=(r0.get("flat_total_run_bytes",0)/2**20) if variant=="flat" else "",
                    flat_cross_mib=(r0.get("flat_cross_bytes",0)/2**20) if variant=="flat" else "",
                ))
        sp=a.output.with_name(a.output.stem+"_summary.csv")
        write_csv(sp,summary)
        print(f"Wrote {a.output}")
        print(f"Wrote {sp}")

if __name__=="__main__": main()
