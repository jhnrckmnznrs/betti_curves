#!/usr/bin/env python3
"""Verify root-carrying local unions against conventional streaming and in-memory persistence."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

SWEEP_RE = re.compile(
    r"PROFILE_SWEEP\s+\S+\s+.*"
    r"union_attempts=(?P<union_attempts>\d+)\s+"
    r"successful_unions=(?P<successful_unions>\d+)\s+"
    r"same_root_unions=(?P<same_root_unions>\d+)\s+"
    r"find_calls=(?P<find_calls>\d+)\s+"
    r"find_parent_steps=(?P<find_parent_steps>\d+)\s+"
    r"active_rechecks=(?P<active_rechecks>\d+)\s+"
    r"active_recheck_failures=(?P<active_recheck_failures>\d+)\s+"
    r"root_carry_attempts=(?P<root_carry_attempts>\d+)\s+"
    r"avoided_find_calls=(?P<avoided_find_calls>\d+)"
)


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
        str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(output)
    ]
    if extra:
        command.extend(extra)
    completed = subprocess.run(command, cwd=work, text=True, capture_output=True)
    log = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{log}")
    return read_intervals(output), log


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


def parse_sweep(dimension: str, label: str, log: str) -> dict[str, int]:
    match = SWEEP_RE.search(log)
    if match is None:
        raise SystemExit(f"FAIL {dimension} {label}: PROFILE_SWEEP diagnostics line not found")
    return {name: int(value) for name, value in match.groupdict().items()}


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
        "--representative-active-check", "recheck",
        "--sweep-diagnostics",
    ]

    with tempfile.TemporaryDirectory(prefix="betti_union_kernel_equivalence_") as temp:
        root = Path(temp)
        for dimension in args.modes:
            oracle, _ = run(
                args.binary, args.input, root / f"{dimension}_oracle",
                args.slab_depth, args.foreground_connectivity, f"{dimension}-scalar",
            )
            conventional, conventional_log = run(
                args.binary, args.input, root / f"{dimension}_conventional",
                args.slab_depth, args.foreground_connectivity, f"{dimension}-scalar-stream",
                [*fixed, "--union-kernel", "conventional"],
            )
            root_carrying, root_log = run(
                args.binary, args.input, root / f"{dimension}_root_carrying",
                args.slab_depth, args.foreground_connectivity, f"{dimension}-scalar-stream",
                [*fixed, "--union-kernel", "root-carrying"],
            )

            assert_equal(dimension, "conventional vs in-memory", oracle, conventional)
            assert_equal(dimension, "root-carrying vs in-memory", oracle, root_carrying)
            assert_equal(dimension, "root-carrying vs conventional", conventional, root_carrying)

            conventional_stats = parse_sweep(dimension, "conventional", conventional_log)
            root_stats = parse_sweep(dimension, "root-carrying", root_log)

            if conventional_stats["active_recheck_failures"] != 0 or root_stats["active_recheck_failures"] != 0:
                raise SystemExit(f"FAIL {dimension}: inactive representative observed")
            if conventional_stats["root_carry_attempts"] != 0 or conventional_stats["avoided_find_calls"] != 0:
                raise SystemExit(f"FAIL {dimension}: conventional kernel reported root-carry counters")
            if root_stats["root_carry_attempts"] != root_stats["union_attempts"]:
                raise SystemExit(
                    f"FAIL {dimension}: root-carry attempts {root_stats['root_carry_attempts']} "
                    f"!= union attempts {root_stats['union_attempts']}"
                )
            if root_stats["avoided_find_calls"] != root_stats["root_carry_attempts"]:
                raise SystemExit(
                    f"FAIL {dimension}: avoided finds {root_stats['avoided_find_calls']} "
                    f"!= root-carry attempts {root_stats['root_carry_attempts']}"
                )
            if root_stats["find_calls"] >= conventional_stats["find_calls"]:
                raise SystemExit(
                    f"FAIL {dimension}: root-carrying did not reduce find calls "
                    f"({root_stats['find_calls']} >= {conventional_stats['find_calls']})"
                )

            removed = conventional_stats["find_calls"] - root_stats["find_calls"]
            print(
                f"PASS {dimension} diagnostics: find calls {conventional_stats['find_calls']} -> "
                f"{root_stats['find_calls']} ({removed} fewer); "
                f"root-carry attempts={root_stats['root_carry_attempts']}"
            )

    print("PASS: root-carrying unions preserve exact scalar persistence and reduce find calls")


if __name__ == "__main__":
    main()
