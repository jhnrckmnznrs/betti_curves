#!/usr/bin/env python3
"""Verify native 32-bit F32 stream keys against legacy-wide and in-memory persistence."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="F32 TIFF-stack directory")
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")),
    )
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--modes", nargs="+", choices=("h0", "h2"), default=("h0", "h2"))
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
) -> Counter[tuple[str, str]]:
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
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return read_intervals(output)


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
        "--merge-strategy",
        "scan",
        "--interface-order",
        "radix",
        "--event-order",
        "verify",
        "--neighbor-kernel",
        "generic",
    ]

    with tempfile.TemporaryDirectory(prefix="betti_f32_key_equivalence_") as temp:
        root = Path(temp)
        for dimension in args.modes:
            oracle = run(
                args.binary,
                args.input,
                root / f"{dimension}_oracle",
                args.slab_depth,
                args.foreground_connectivity,
                f"{dimension}-scalar",
            )
            legacy = run(
                args.binary,
                args.input,
                root / f"{dimension}_legacy64",
                args.slab_depth,
                args.foreground_connectivity,
                f"{dimension}-scalar-stream",
                [*fixed, "--f32-key-mode", "legacy64"],
            )
            native = run(
                args.binary,
                args.input,
                root / f"{dimension}_native32",
                args.slab_depth,
                args.foreground_connectivity,
                f"{dimension}-scalar-stream",
                [*fixed, "--f32-key-mode", "native32"],
            )
            assert_equal(dimension, "legacy64 vs in-memory", oracle, legacy)
            assert_equal(dimension, "native32 vs in-memory", oracle, native)
            assert_equal(dimension, "native32 vs legacy64", legacy, native)

    print("PASS: native F32 keys preserve exact scalar persistence")


if __name__ == "__main__":
    main()
