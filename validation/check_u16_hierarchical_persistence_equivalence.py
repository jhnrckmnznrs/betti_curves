#!/usr/bin/env python3
"""Compare flat and hierarchical exact persistence on the same scalar TIFF stack.

The reference is the long-standing disk-backed flat scalar stream.  The candidate
is the bounded hierarchical stream.  CSV rows are compared as multisets so the
validator is insensitive to emission order while remaining exact in birth/death
text values.
"""
from __future__ import annotations

import argparse
import collections
import csv
import subprocess
import tempfile
from pathlib import Path


def rows(path: Path) -> collections.Counter[tuple[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence header in {path}: {reader.fieldnames}")
        return collections.Counter((r["birth"], r["death"]) for r in reader)


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"command failed with code {proc.returncode}:\n{' '.join(cmd)}\n{proc.stdout}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depth", type=int, default=16)
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    ap.add_argument("--dimensions", nargs="+", choices=("h0", "h2"), default=("h0", "h2"))
    args = ap.parse_args()

    stack = args.stack.resolve()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="u16_hier_persistence_equiv_") as td:
        root = Path(td)
        for dim in args.dimensions:
            ref = root / f"{dim}_flat.csv"
            cand = root / f"{dim}_hier.csv"
            flat_mode = f"{dim}-scalar-stream"
            hier_mode = f"{dim}-scalar-hierarchical-stream"
            base = [str(binary), str(stack), str(args.slab_depth), str(args.foreground_connectivity)]
            run(base + [flat_mode, str(ref)])
            extra: list[str]
            if dim == "h0":
                extra = [
                    "--h0-birth-buffer", "reuse-input",
                    "--h0-event-storage", "direct",
                    "--global-h0-uf-layout", "packed",
                    "--h0-hier-attach-pruning", "elder-dominated",
                ]
            else:
                extra = [
                    "--local-h2-birth-state", "compact",
                    "--global-h2-birth-state", "compact",
                    "--global-h2-uf-layout", "packed",
                    "--h2-hier-cross-storage", "direct",
                    "--h2-hier-outside-structural-pruning", "off",
                ]
            run(base + [hier_mode, str(cand)] + extra)
            a, b = rows(ref), rows(cand)
            if a != b:
                missing = list((a - b).items())[:10]
                extra_rows = list((b - a).items())[:10]
                raise RuntimeError(
                    f"FAIL {dim.upper()}: flat/hierarchical interval multisets differ\n"
                    f"missing from hierarchical: {missing}\nextra in hierarchical: {extra_rows}"
                )
            print(f"PASS {dim.upper()}: {sum(a.values())} exact intervals; hierarchical matches flat reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
