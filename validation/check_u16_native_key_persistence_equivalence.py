#!/usr/bin/env python3
"""Exact same-binary equivalence for compact U16 persistence keys.

Runs the optimized hierarchical persistence mode twice: once with the v20
native32 U16 key path enabled and once with the v19 wide64 fallback forced via
BETTI_PERSIST_U16_NATIVE_KEYS=0. CSV interval rows are compared as multisets.
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


def extra_args(dim: str) -> list[str]:
    if dim == "h0":
        return [
            "--h0-birth-buffer", "reuse-input",
            "--h0-event-storage", "direct",
            "--global-h0-uf-layout", "packed",
            "--h0-hier-attach-pruning", "elder-dominated",
        ]
    return [
        "--local-h2-birth-state", "compact",
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", "packed",
        "--h2-hier-cross-storage", "direct",
        "--h2-hier-outside-structural-pruning", "off",
    ]


def run(cmd: list[str], native: bool) -> str:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1" if native else "0"
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
    ap.add_argument("--dimensions", nargs="+", choices=("h0", "h2"), default=("h0", "h2"))
    args = ap.parse_args()

    stack = args.stack.resolve()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="u16_native_key_equiv_") as td_raw:
        td = Path(td_raw)
        for dim in args.dimensions:
            wide = td / f"{dim}_wide64.csv"
            native = td / f"{dim}_native32.csv"
            mode = f"{dim}-scalar-hierarchical-stream"
            base = [
                str(binary), str(stack), str(args.slab_depth),
                str(args.foreground_connectivity), mode,
            ]
            run(base + [str(wide)] + extra_args(dim), native=False)
            native_log = run(base + [str(native)] + extra_args(dim), native=True)

            a = rows(wide)
            b = rows(native)
            if a != b:
                missing = list((a - b).items())[:10]
                extra = list((b - a).items())[:10]
                raise RuntimeError(
                    f"FAIL {dim.upper()}: wide64/native32 interval multisets differ\n"
                    f"missing from native32: {missing}\nextra in native32: {extra}"
                )
            if "PROFILE_U16_PERSIST_LEAF" not in native_log:
                raise RuntimeError(
                    f"FAIL {dim.upper()}: native32 run did not report PROFILE_U16_PERSIST_LEAF"
                )
            print(
                f"PASS {dim.upper()}: {sum(a.values())} exact intervals; "
                "native32 U16 keys match v19 wide64 fallback"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
