#!/usr/bin/env python3
"""Exact U16 H0 persistence equivalence across leaf-worker counts.

The candidate uses the same native32 hierarchical persistence binary with
BETTI_PERSIST_H0_LEAF_WORKERS=1 and a requested parallel worker count. CSV
interval rows are compared as multisets, so concurrency may not alter the
mathematical output or multiplicities.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import subprocess
import tempfile
from pathlib import Path


def rows(path: Path) -> collections.Counter[tuple[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence header in {path}: {reader.fieldnames}")
        return collections.Counter((row["birth"], row["death"]) for row in reader)


def run(binary: Path, stack: Path, depth: int, conn: int, out: Path, workers: int) -> str:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H0_LEAF_WORKERS"] = str(workers)
    cmd = [
        str(binary), str(stack), str(depth), str(conn),
        "h0-scalar-hierarchical-stream", str(out),
        "--h0-birth-buffer", "reuse-input",
        "--h0-event-storage", "direct",
        "--global-h0-uf-layout", "packed",
        "--h0-hier-attach-pruning", "elder-dominated",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if proc.returncode:
        raise RuntimeError(
            f"command failed with code {proc.returncode}:\n{' '.join(cmd)}\n{proc.stdout}"
        )
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depth", type=int, default=16)
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    ap.add_argument("--parallel-workers", type=int, default=4)
    args = ap.parse_args()
    if args.parallel_workers < 1:
        raise SystemExit("--parallel-workers must be positive")

    stack = args.stack.resolve()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="u16_h0_parallel_leaf_equiv_") as td_raw:
        td = Path(td_raw)
        serial_out = td / "h0_workers1.csv"
        parallel_out = td / f"h0_workers{args.parallel_workers}.csv"
        serial_log = run(
            binary, stack, args.slab_depth, args.foreground_connectivity, serial_out, 1
        )
        parallel_log = run(
            binary, stack, args.slab_depth, args.foreground_connectivity,
            parallel_out, args.parallel_workers,
        )
        a = rows(serial_out)
        b = rows(parallel_out)
        if a != b:
            missing = list((a - b).items())[:10]
            extra = list((b - a).items())[:10]
            raise RuntimeError(
                "FAIL H0: worker-count interval multisets differ\n"
                f"missing from parallel: {missing}\nextra in parallel: {extra}"
            )
        if "PROFILE_U16_PERSIST_H0_PARALLEL" not in serial_log:
            raise RuntimeError("serial run did not report PROFILE_U16_PERSIST_H0_PARALLEL")
        if "PROFILE_U16_PERSIST_H0_PARALLEL" not in parallel_log:
            raise RuntimeError("parallel run did not report PROFILE_U16_PERSIST_H0_PARALLEL")
        print(
            f"PASS H0: {sum(a.values())} exact intervals; workers=1 matches "
            f"workers={args.parallel_workers}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
