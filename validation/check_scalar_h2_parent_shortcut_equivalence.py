#!/usr/bin/env python3
"""Verify the H2 parent shortcut against root-carrying find and in-memory persistence."""

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
    r"PROFILE_SWEEP\s+scalar_h2_stream\s+.*"
    r"union_attempts=(?P<union_attempts>\d+)\s+"
    r"successful_unions=(?P<successful_unions>\d+)\s+"
    r"same_root_unions=(?P<same_root_unions>\d+)\s+"
    r"find_calls=(?P<find_calls>\d+)\s+"
    r"find_parent_steps=(?P<find_parent_steps>\d+)\s+"
    r"active_rechecks=(?P<active_rechecks>\d+)\s+"
    r"active_recheck_failures=(?P<active_recheck_failures>\d+)\s+"
    r"root_carry_attempts=(?P<root_carry_attempts>\d+)\s+"
    r"avoided_find_calls=(?P<avoided_find_calls>\d+)\s+"
    r"direct_parent_checks=(?P<direct_parent_checks>\d+)\s+"
    r"direct_parent_hits=(?P<direct_parent_hits>\d+)\s+"
    r"avoided_neighbor_find_calls=(?P<avoided_neighbor_find_calls>\d+)"
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
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    return parser.parse_args()


def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected header in {path}: {header!r}")
        return Counter(tuple(row) for row in reader if row)


def run(binary: Path, input_path: Path, work: Path, slab: int, fg: int, mode: str,
        extra: list[str] | None = None) -> tuple[Counter[tuple[str, str]], str]:
    work.mkdir(parents=True, exist_ok=True)
    output = work / "intervals.csv"
    command = [str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(output)]
    if extra:
        command.extend(extra)
    completed = subprocess.run(command, cwd=work, text=True, capture_output=True)
    log = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{log}")
    return read_intervals(output), log


def assert_equal(label: str, expected: Counter[tuple[str, str]], observed: Counter[tuple[str, str]]) -> None:
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise SystemExit(
            f"FAIL h2 {label}: persistence multisets differ\n"
            f"missing: {missing.most_common(10)}\nextra: {extra.most_common(10)}"
        )
    print(f"PASS h2 {label}: {sum(expected.values())} intervals agree exactly")


def parse_sweep(label: str, log: str) -> dict[str, int]:
    match = SWEEP_RE.search(log)
    if match is None:
        raise SystemExit(f"FAIL h2 {label}: PROFILE_SWEEP diagnostics line not found")
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
        "--union-kernel", "root-carrying",
        "--sweep-diagnostics",
    ]

    with tempfile.TemporaryDirectory(prefix="betti_h2_parent_shortcut_equivalence_") as temp:
        root = Path(temp)
        oracle, _ = run(
            args.binary, args.input, root / "oracle", args.slab_depth,
            args.foreground_connectivity, "h2-scalar",
        )
        find_intervals, find_log = run(
            args.binary, args.input, root / "find", args.slab_depth,
            args.foreground_connectivity, "h2-scalar-stream",
            [*fixed, "--neighbor-root-check", "find"],
        )
        shortcut_intervals, shortcut_log = run(
            args.binary, args.input, root / "parent_shortcut", args.slab_depth,
            args.foreground_connectivity, "h2-scalar-stream",
            [*fixed, "--neighbor-root-check", "parent-shortcut"],
        )

        assert_equal("find vs in-memory", oracle, find_intervals)
        assert_equal("parent-shortcut vs in-memory", oracle, shortcut_intervals)
        assert_equal("parent-shortcut vs find", find_intervals, shortcut_intervals)

        find_stats = parse_sweep("find", find_log)
        shortcut_stats = parse_sweep("parent-shortcut", shortcut_log)

        if find_stats["active_recheck_failures"] or shortcut_stats["active_recheck_failures"]:
            raise SystemExit("FAIL h2: inactive representative observed")
        if find_stats["direct_parent_checks"] != 0 or find_stats["direct_parent_hits"] != 0:
            raise SystemExit("FAIL h2: find reference unexpectedly reported direct-parent activity")
        if shortcut_stats["direct_parent_checks"] != shortcut_stats["root_carry_attempts"]:
            raise SystemExit(
                "FAIL h2: parent shortcut did not check every root-carrying union attempt "
                f"({shortcut_stats['direct_parent_checks']} != {shortcut_stats['root_carry_attempts']})"
            )
        if shortcut_stats["direct_parent_hits"] != shortcut_stats["avoided_neighbor_find_calls"]:
            raise SystemExit("FAIL h2: direct-parent hits and avoided neighbor finds disagree")
        if shortcut_stats["find_calls"] > find_stats["find_calls"]:
            raise SystemExit(
                f"FAIL h2: parent shortcut increased find calls "
                f"({shortcut_stats['find_calls']} > {find_stats['find_calls']})"
            )
        expected_removed = shortcut_stats["avoided_neighbor_find_calls"]
        actual_removed = find_stats["find_calls"] - shortcut_stats["find_calls"]
        if actual_removed != expected_removed:
            raise SystemExit(
                f"FAIL h2: find-call reduction {actual_removed} != avoided neighbor finds {expected_removed}"
            )

        hit_rate = (
            shortcut_stats["direct_parent_hits"] / shortcut_stats["direct_parent_checks"]
            if shortcut_stats["direct_parent_checks"] else 0.0
        )
        print(
            "PASS h2 diagnostics: "
            f"direct-parent hits={shortcut_stats['direct_parent_hits']}/"
            f"{shortcut_stats['direct_parent_checks']} ({100.0 * hit_rate:.2f}%); "
            f"find calls {find_stats['find_calls']} -> {shortcut_stats['find_calls']}"
        )

    print("PASS: H2 parent shortcut preserves exact scalar persistence and avoids neighbor finds")


if __name__ == "__main__":
    main()
