#!/usr/bin/env python3
"""Verify trusting pruner representatives against recheck streaming and in-memory persistence."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

SWEEP_RE = re.compile(r"PROFILE_SWEEP\s+\S+\s+.*active_rechecks=(?P<rechecks>\d+)\s+active_recheck_failures=(?P<failures>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="scalar TIFF-stack directory")
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")),
    )
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--modes", nargs="+", choices=("h0", "h2"), default=("h0", "h2"))
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    return parser.parse_args()


def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected header in {path}: {header!r}")
        return Counter(tuple(row) for row in reader if row)


def run(
    binary: Path,
    input_path: Path,
    work: Path,
    slab: int,
    fg: int,
    mode: str,
    extra: list[str] | None = None,
) -> tuple[Counter[tuple[str, str]], str]:
    work.mkdir(parents=True, exist_ok=True)
    output = work / "intervals.csv"
    command = [
        str(binary.resolve()),
        str(input_path.resolve()),
        str(slab),
        str(fg),
        mode,
        str(output),
    ]
    if extra:
        command.extend(extra)
    completed = subprocess.run(command, cwd=work, text=True, capture_output=True)
    stdout = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{stdout}")
    return read_intervals(output), stdout


def assert_equal(
    dimension: str,
    label: str,
    expected: Counter[tuple[str, str]],
    observed: Counter[tuple[str, str]],
) -> None:
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise SystemExit(
            f"FAIL {dimension} {label}: persistence multisets differ\n"
            f"missing: {missing.most_common(10)}\nextra: {extra.most_common(10)}"
        )
    print(f"PASS {dimension} {label}: {sum(expected.values())} intervals agree exactly")


def main() -> None:
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    if not args.input.is_dir():
        raise SystemExit(f"input stack not found: {args.input}")

    fixed = [
        "--merge-strategy", "scan",
        "--interface-order", "radix",
        "--event-order", "verify",
        "--f32-key-mode", args.f32_key_mode,
        "--neighbor-kernel", "interior-fast",
    ]

    with tempfile.TemporaryDirectory(prefix="betti_representative_active_equivalence_") as temp:
        root = Path(temp)
        for dimension in args.modes:
            oracle, _ = run(
                args.binary, args.input, root / f"{dimension}_oracle",
                args.slab_depth, args.foreground_connectivity, f"{dimension}-scalar",
            )
            recheck, recheck_log = run(
                args.binary, args.input, root / f"{dimension}_recheck",
                args.slab_depth, args.foreground_connectivity, f"{dimension}-scalar-stream",
                [*fixed, "--representative-active-check", "recheck", "--sweep-diagnostics"],
            )
            trust, _ = run(
                args.binary, args.input, root / f"{dimension}_trust",
                args.slab_depth, args.foreground_connectivity, f"{dimension}-scalar-stream",
                [*fixed, "--representative-active-check", "trust-pruner"],
            )
            match = SWEEP_RE.search(recheck_log)
            if match is None:
                raise SystemExit(f"FAIL {dimension}: PROFILE_SWEEP diagnostics line not found")
            rechecks = int(match.group("rechecks"))
            failures = int(match.group("failures"))
            if rechecks == 0:
                raise SystemExit(f"FAIL {dimension}: diagnostics recorded no representative rechecks")
            if failures != 0:
                raise SystemExit(
                    f"FAIL {dimension}: pruner returned {failures} inactive representatives"
                )
            print(f"PASS {dimension} invariant: {rechecks} representative rechecks, 0 failures")
            assert_equal(dimension, "recheck vs in-memory", oracle, recheck)
            assert_equal(dimension, "trust-pruner vs in-memory", oracle, trust)
            assert_equal(dimension, "trust-pruner vs recheck", recheck, trust)

    print("PASS: trusting representative activity preserves exact scalar persistence")


if __name__ == "__main__":
    main()
