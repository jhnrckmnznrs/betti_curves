#!/usr/bin/env python3
"""Profile flat vs hierarchical scalar persistence on an integer TIFF stack."""
from __future__ import annotations

import argparse, csv, os, re, statistics, subprocess, tempfile, time
from pathlib import Path

RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
USER_RE = re.compile(r"User time \(seconds\):\s*([0-9.]+)")
SYS_RE = re.compile(r"System time \(seconds\):\s*([0-9.]+)")


def command(binary: Path, stack: Path, depth: int, conn: int, dim: str, backend: str, out: Path) -> list[str]:
    mode = f"{dim}-scalar-{'hierarchical-' if backend == 'hierarchical' else ''}stream"
    cmd = [str(binary), str(stack), str(depth), str(conn), mode, str(out)]
    if backend == "hierarchical":
        if dim == "h0":
            cmd += ["--h0-birth-buffer", "reuse-input", "--h0-event-storage", "direct",
                    "--global-h0-uf-layout", "packed", "--h0-hier-attach-pruning", "elder-dominated"]
        else:
            cmd += ["--local-h2-birth-state", "compact", "--global-h2-birth-state", "compact",
                    "--global-h2-uf-layout", "packed", "--h2-hier-cross-storage", "direct",
                    "--h2-hier-outside-structural-pruning", "off"]
    return cmd


def run_once(cmd: list[str], timing_path: Path) -> tuple[float, float, float, float]:
    start = time.perf_counter()
    p = subprocess.run(["/usr/bin/time", "-v", "-o", str(timing_path), *cmd],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    wall = time.perf_counter() - start
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")
    text = timing_path.read_text()
    rss = float(RSS_RE.search(text).group(1)) / 1024.0
    user = float(USER_RE.search(text).group(1))
    system = float(SYS_RE.search(text).group(1))
    return wall, rss, user, system


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[8,16,32])
    ap.add_argument("--foreground-connectivity", type=int, default=26)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", type=Path, default=Path("profiles/u16_persistence_hierarchy.csv"))
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory(prefix="u16_persist_prof_") as td:
        td=Path(td)
        seq=0
        for dim in ("h0","h2"):
            for backend in ("flat","hierarchical"):
                for depth in args.slab_depths:
                    for w in range(args.warmup):
                        seq += 1
                        out=td/f"{seq}_{dim}_{backend}_d{depth}_warm.csv"; tim=td/f"{seq}.time"
                        print(f"warmup {backend}/{dim} d={depth}")
                        run_once(command(args.binary.resolve(), args.stack.resolve(), depth, args.foreground_connectivity, dim, backend, out), tim)
                    for rep in range(1,args.repeats+1):
                        seq += 1
                        out=td/f"{seq}_{dim}_{backend}_d{depth}_r{rep}.csv"; tim=td/f"{seq}.time"
                        wall,rss,user,system=run_once(command(args.binary.resolve(), args.stack.resolve(), depth, args.foreground_connectivity, dim, backend, out), tim)
                        print(f"{backend}/{dim} d={depth} r={rep}: {wall:.3f}s {rss:.1f} MiB")
                        rows.append(dict(dimension=dim,backend=backend,slab_depth=depth,repeat=rep,wall_seconds=wall,max_rss_mib=rss,user_seconds=user,system_seconds=system))
    fields=list(rows[0])
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary=args.output.with_name(args.output.stem+'_summary.csv')
    groups={}
    for r in rows: groups.setdefault((r['dimension'],r['backend'],r['slab_depth']),[]).append(r)
    with summary.open('w',newline='') as f:
        fields2=['dimension','backend','slab_depth','repeats','median_wall_seconds','median_max_rss_mib','median_user_seconds','median_system_seconds']
        w=csv.DictWriter(f,fieldnames=fields2); w.writeheader()
        for (dim,b,d), rs in sorted(groups.items()):
            w.writerow(dict(dimension=dim,backend=b,slab_depth=d,repeats=len(rs),
                median_wall_seconds=statistics.median(x['wall_seconds'] for x in rs),
                median_max_rss_mib=statistics.median(x['max_rss_mib'] for x in rs),
                median_user_seconds=statistics.median(x['user_seconds'] for x in rs),
                median_system_seconds=statistics.median(x['system_seconds'] for x in rs)))
    print(f"wrote {args.output} and {summary}")
    return 0

if __name__ == '__main__': raise SystemExit(main())
