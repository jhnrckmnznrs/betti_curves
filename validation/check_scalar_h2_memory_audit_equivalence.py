#!/usr/bin/env python3
"""Verify that enabling the H2 memory audit does not change exact persistence."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slab-depth", type=int, default=16)
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    return p.parse_args()


def read_intervals(path: Path):
    with path.open(newline="", encoding="utf-8") as h:
        reader = csv.reader(h)
        header = next(reader, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected header {header!r}")
        return Counter(tuple(row) for row in reader if row)


def run(binary: Path, input_path: Path, work: Path, slab: int, fg: int, mode: str, extra=()):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(out), *extra]
    cp = subprocess.run(cmd, cwd=work, text=True, capture_output=True)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}{cp.stderr}")
    return read_intervals(out), cp.stdout + cp.stderr


def check(label: str, expected, actual):
    if expected != actual:
        raise SystemExit(f"FAIL {label}: missing={(expected-actual).most_common(10)} extra={(actual-expected).most_common(10)}")
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def main():
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    common = [
        "--merge-strategy", "scan", "--interface-order", "radix", "--event-order", "verify",
        "--f32-key-mode", args.f32_key_mode, "--neighbor-kernel", "interior-fast",
        "--representative-active-check", "recheck", "--union-kernel", "root-carrying",
        "--neighbor-root-check", "parent-shortcut", "--active-state", "parent-sentinel",
        "--interface-state", "root-invariant",
    ]
    with tempfile.TemporaryDirectory(prefix="betti_h2_mem_equiv_") as td:
        root = Path(td)
        oracle, _ = run(args.binary, args.input, root / "oracle", args.slab_depth,
                        args.foreground_connectivity, "h2-scalar")
        for layout in ("parent-rank", "packed"):
            baseline, _ = run(args.binary, args.input, root / f"{layout}_base", args.slab_depth,
                              args.foreground_connectivity, "h2-scalar-stream",
                              common + ["--uf-layout", layout])
            audited, log = run(args.binary, args.input, root / f"{layout}_audit", args.slab_depth,
                               args.foreground_connectivity, "h2-scalar-stream",
                               common + ["--uf-layout", layout, "--h2-memory-audit"])
            if "PROFILE_MEM " not in log or "PROFILE_MEMVEC " not in log:
                raise SystemExit(f"FAIL {layout}: audit output is missing memory records")
            check(f"{layout} baseline vs in-memory", oracle, baseline)
            check(f"{layout} audited vs in-memory", oracle, audited)
            check(f"{layout} audit flag vs baseline", baseline, audited)
    print("PASS: H2 memory audit is observational and both UF layouts preserve exact persistence")


if __name__ == "__main__":
    main()
