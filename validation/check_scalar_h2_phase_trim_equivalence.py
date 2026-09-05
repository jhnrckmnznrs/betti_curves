#!/usr/bin/env python3
"""Exact H2 persistence check for the targeted before-reduce glibc trim."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

TRIM_RE = re.compile(
    r"PROFILE_TRIM\s+scalar_h2_stream\s+strategy=(?P<strategy>\S+)\s+"
    r"attempted=(?P<attempted>true|false)\s+released=(?P<released>true|false)\s+"
    r"seconds=(?P<seconds>[0-9.]+)"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument(
        "--binary",
        type=Path,
        default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")),
    )
    p.add_argument("--slab-depth", type=int, default=16)
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    p.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    return p.parse_args()


def read_intervals(path: Path):
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        header = next(r, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected header {header!r} in {path}")
        return Counter(tuple(row) for row in r if row)


def run(binary: Path, input_path: Path, work: Path, slab: int, fg: int, mode: str, extra=None):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(out)]
    if extra:
        cmd.extend(extra)
    cp = subprocess.run(cmd, cwd=work, text=True, capture_output=True)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}{cp.stderr}")
    return read_intervals(out), cp.stdout + cp.stderr


def check(label: str, expected, actual):
    if expected != actual:
        raise SystemExit(
            f"FAIL {label}: persistence multisets differ\n"
            f"missing={(expected-actual).most_common(10)}\n"
            f"extra={(actual-expected).most_common(10)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def validate_trim_line(label: str, log: str, expected_strategy: str):
    m = TRIM_RE.search(log)
    if not m:
        raise SystemExit(f"FAIL {label}: missing PROFILE_TRIM line")
    if m.group("strategy") != expected_strategy:
        raise SystemExit(
            f"FAIL {label}: expected strategy {expected_strategy}, got {m.group('strategy')}"
        )
    attempted = m.group("attempted") == "true"
    if attempted != (expected_strategy == "before-reduce"):
        raise SystemExit(
            f"FAIL {label}: attempted={attempted} inconsistent with strategy {expected_strategy}"
        )
    print(
        f"PASS {label}: trim profile strategy={m.group('strategy')} "
        f"attempted={m.group('attempted')} released={m.group('released')} "
        f"seconds={m.group('seconds')}"
    )


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")

    common = [
        "--merge-strategy", "scan",
        "--interface-order", "radix",
        "--event-order", "verify",
        "--f32-key-mode", a.f32_key_mode,
        "--neighbor-kernel", "interior-fast",
        "--representative-active-check", "recheck",
        "--union-kernel", "root-carrying",
        "--h0-pruning-cache", "64k",
        "--neighbor-root-check", "parent-shortcut",
        "--active-state", "parent-sentinel",
        "--interface-state", "root-invariant",
    ]

    with tempfile.TemporaryDirectory(prefix="betti_h2_phase_trim_equiv_") as td:
        root = Path(td)
        oracle, _ = run(
            a.binary,
            a.input,
            root / "oracle",
            a.slab_depth,
            a.foreground_connectivity,
            "h2-scalar",
        )

        reference = None
        for layout in ("parent-rank", "packed"):
            for trim in ("off", "before-reduce"):
                label = f"{layout}/{trim}"
                intervals, log = run(
                    a.binary,
                    a.input,
                    root / f"{layout}_{trim}",
                    a.slab_depth,
                    a.foreground_connectivity,
                    "h2-scalar-stream",
                    common + ["--uf-layout", layout, "--phase-trim", trim],
                )
                check(f"H2 {label} vs in-memory", oracle, intervals)
                if reference is None:
                    reference = intervals
                else:
                    check(f"H2 {label} vs first stream configuration", reference, intervals)
                validate_trim_line(f"H2 {label}", log, trim)

    print("PASS: targeted phase trim preserves exact scalar H2 persistence")


if __name__ == "__main__":
    main()
