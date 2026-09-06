#!/usr/bin/env python3
"""Exact U16 H2 persistence equivalence: division/modulo reference vs fast interface query."""
from __future__ import annotations
import argparse, collections, csv, hashlib, os, subprocess, tempfile
from pathlib import Path

Interval = tuple[str, str]

def rows(path: Path) -> list[Interval]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence header in {path}: {reader.fieldnames}")
        return [(r["birth"], r["death"]) for r in reader]

def digest(intervals: list[Interval]) -> str:
    counts = collections.Counter(intervals); h = hashlib.sha256()
    for (birth, death), count in sorted(counts.items()):
        h.update(birth.encode()); h.update(b"\0"); h.update(death.encode()); h.update(b"\0")
        h.update(str(count).encode()); h.update(b"\n")
    return h.hexdigest()

def run(cmd: list[str], fast: bool) -> str:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"] = "1"
    env["BETTI_PERSIST_H2_ROOT_DEDUP"] = "0"
    env["BETTI_PERSIST_H2_FAST_INTERFACE_QUERY"] = "1" if fast else "0"
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}):\n{' '.join(cmd)}\n{proc.stdout}")
    expected = "on" if fast else "off"
    if f"h2_fast_interface_query={expected}" not in proc.stdout:
        raise RuntimeError(f"run did not report h2_fast_interface_query={expected}")
    if "h2_root_dedup=off" not in proc.stdout:
        raise RuntimeError("equivalence gate requires root dedup off")
    return proc.stdout

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6,18,26))
    ap.add_argument("--allow-order-difference", action="store_true")
    args = ap.parse_args()
    stack=args.stack.resolve(); binary=args.binary.resolve()
    if not stack.exists(): ap.error(f"stack does not exist: {stack}")
    if not binary.exists(): ap.error(f"binary does not exist: {binary}")
    if any(d <= 0 for d in args.slab_depths) or len(set(args.slab_depths)) != len(args.slab_depths):
        ap.error("slab depths must be positive and unique")

    extra = [
        "--local-h2-birth-state","compact",
        "--global-h2-birth-state","compact",
        "--global-h2-uf-layout","packed",
        "--h2-hier-cross-storage","direct",
        "--h2-hier-outside-structural-pruning","off",
        "--neighbor-root-check","parent-shortcut",
        "--interface-state","root-invariant",
    ]
    baseline = None
    with tempfile.TemporaryDirectory(prefix="u16_h2_fast_iface_equiv_") as td_raw:
        td=Path(td_raw)
        for depth in args.slab_depths:
            ref=td/f"d{depth}_reference.csv"; cand=td/f"d{depth}_fast.csv"
            base=[str(binary),str(stack),str(depth),str(args.foreground_connectivity),
                  "h2-scalar-hierarchical-stream"]
            run(base+[str(ref)]+extra, False)
            run(base+[str(cand)]+extra, True)
            a=rows(ref); b=rows(cand)
            ca,cb=collections.Counter(a),collections.Counter(b)
            if ca != cb:
                raise RuntimeError(f"FAIL H2 d={depth}: interval multisets differ")
            if not args.allow_order_difference and a != b:
                first=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
                raise RuntimeError(f"FAIL H2 d={depth}: row order differs at {first}: {a[first]} vs {b[first]}")
            if baseline is None:
                baseline=ca
            elif ca != baseline:
                raise RuntimeError(f"FAIL H2 d={depth}: slab-depth interval multiset differs from d={args.slab_depths[0]}")
            print(f"PASS H2 d={depth}: {len(a)} exact intervals; fast interface query matches reference; sha256={digest(a)}")
    print("PASS H2 fast-interface-query equivalence gate")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
