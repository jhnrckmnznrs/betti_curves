#!/usr/bin/env python3
"""Validate pairwise hierarchical H0 summaries against flat exact H0 implementations."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="TIFF-stack directory (CX09T1 is the intended first target)")
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slab-depths", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    return p.parse_args()


def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        if next(r, None) != ["birth", "death"]:
            raise RuntimeError(f"unexpected header in {path}")
        return Counter(tuple(row) for row in r if row)


def run(binary: Path, input_path: Path, work: Path, slab: int, fg: int, mode: str):
    work.mkdir(parents=True, exist_ok=True)
    output = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(output)]
    cp = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
    return read_intervals(output), cp.stdout


def check(label: str, expected: Counter[tuple[str, str]], observed: Counter[tuple[str, str]]) -> None:
    if expected != observed:
        raise SystemExit(
            f"FAIL {label}: persistence differs\n"
            f"missing={(expected-observed).most_common(10)}\n"
            f"extra={(observed-expected).most_common(10)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def main() -> None:
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    help_cp = subprocess.run([str(a.binary.resolve()), "--help"], text=True, capture_output=True)
    if "h0-scalar-hierarchical" not in help_cp.stdout + help_cp.stderr:
        raise SystemExit("stale binary: rebuild the architectural candidate; hierarchical H0 mode is missing")

    with tempfile.TemporaryDirectory(prefix="betti_h0_hier_equiv_") as td:
        root = Path(td)
        reference = None
        for slab in a.slab_depths:
            flat, _ = run(a.binary, a.input, root / f"flat_d{slab}", slab, a.foreground_connectivity, "h0-scalar")
            hier, log = run(a.binary, a.input, root / f"hier_d{slab}", slab, a.foreground_connectivity, "h0-scalar-hierarchical")
            stream, _ = run(a.binary, a.input, root / f"stream_d{slab}", slab, a.foreground_connectivity, "h0-scalar-stream")
            check(f"hierarchical d{slab} vs flat in-memory", flat, hier)
            check(f"hierarchical d{slab} vs production stream", stream, hier)
            if "PROFILE_H0_HIER_COMBINE" not in log and len(list(a.input.glob("*.tif*"))) > slab:
                raise SystemExit(f"FAIL d{slab}: no hierarchical combine profile was emitted")
            if reference is None:
                reference = hier
            else:
                check(f"hierarchical d{slab} vs first slab depth", reference, hier)

    print("PASS: pairwise hierarchical H0 composition preserves exact persistence across slab depths")


if __name__ == "__main__":
    main()
