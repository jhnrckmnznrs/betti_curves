#!/usr/bin/env python3
"""Verify the interior neighbor kernel against generic streaming and in-memory persistence."""

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
        "--f32-key-mode",
        args.f32_key_mode,
    ]

    with tempfile.TemporaryDirectory(prefix="betti_neighbor_kernel_equivalence_") as temp:
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
            generic = run(
                args.binary,
                args.input,
                root / f"{dimension}_generic",
                args.slab_depth,
                args.foreground_connectivity,
                f"{dimension}-scalar-stream",
                [*fixed, "--neighbor-kernel", "generic"],
            )
            fast = run(
                args.binary,
                args.input,
                root / f"{dimension}_interior_fast",
                args.slab_depth,
                args.foreground_connectivity,
                f"{dimension}-scalar-stream",
                [*fixed, "--neighbor-kernel", "interior-fast"],
            )
            assert_equal(dimension, "generic vs in-memory", oracle, generic)
            assert_equal(dimension, "interior-fast vs in-memory", oracle, fast)
            assert_equal(dimension, "interior-fast vs generic", generic, fast)

    print("PASS: interior neighbor kernel preserves exact scalar persistence")


if __name__ == "__main__":
    main()
