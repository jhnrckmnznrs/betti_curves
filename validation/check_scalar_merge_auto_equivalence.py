#!/usr/bin/env python3
"""Validate the scale-aware auto merge selector against scan/heap and in-memory H0/H2."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

SELECT_RE = re.compile(
    r"PROFILE_MERGE_SELECT scalar_(?P<dim>h[02])_stream requested=auto effective=(?P<effective>scan|heap) readers=(?P<readers>\d+) threshold=(?P<threshold>\d+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slab-depth", type=int, default=16)
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    return p.parse_args()


def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        if next(r, None) != ["birth", "death"]:
            raise RuntimeError(f"unexpected interval header in {path}")
        return Counter(tuple(row) for row in r if row)


def run(binary: Path, inp: Path, work: Path, slab: int, fg: int, mode: str, extra=()):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(inp.resolve()), str(slab), str(fg), mode, str(out), *extra]
    cp = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
    return read_intervals(out), cp.stdout


def equal(label: str, expected, observed) -> None:
    if expected != observed:
        raise SystemExit(
            f"FAIL {label}\nmissing={(expected-observed).most_common(10)}\nextra={(observed-expected).most_common(10)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def main() -> None:
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    for dim in ("h0", "h2"):
        with tempfile.TemporaryDirectory(prefix=f"betti_{dim}_merge_auto_") as td:
            root = Path(td)
            oracle, _ = run(a.binary, a.input, root / "oracle", a.slab_depth, a.foreground_connectivity, f"{dim}-scalar")
            results = {}
            for merge in ("scan", "heap", "auto"):
                got, log = run(
                    a.binary,
                    a.input,
                    root / merge,
                    a.slab_depth,
                    a.foreground_connectivity,
                    f"{dim}-scalar-stream",
                    ("--merge-strategy", merge),
                )
                equal(f"{dim.upper()} {merge} vs in-memory", oracle, got)
                results[merge] = got
                if merge == "auto":
                    m = SELECT_RE.search(log)
                    if not m or m.group("dim") != dim:
                        raise SystemExit(f"FAIL {dim}: auto merge selection profile missing")
                    readers = int(m.group("readers")); threshold = int(m.group("threshold"))
                    expected = "heap" if readers > threshold else "scan"
                    if m.group("effective") != expected:
                        raise SystemExit(
                            f"FAIL {dim}: auto selected {m.group('effective')} for {readers} readers; expected {expected} at threshold {threshold}"
                        )
                    print(f"PASS {dim.upper()} auto selector: readers={readers}, effective={expected}")
            equal(f"{dim.upper()} auto vs scan", results["scan"], results["auto"])
            equal(f"{dim.upper()} auto vs heap", results["heap"], results["auto"])
    print("PASS: scale-aware auto merge selection preserves exact H0/H2 persistence")


if __name__ == "__main__":
    main()
