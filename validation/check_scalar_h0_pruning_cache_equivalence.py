#!/usr/bin/env python3
"""Verify all H0 pruning-cache modes against exact in-memory scalar persistence."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

CACHE_MODES = ("off", "4k", "16k", "64k", "256k")
CACHE_RE = re.compile(
    r"PROFILE_CACHE\s+scalar_h0_stream\s+mode=(?P<mode>\S+)\s+"
    r"entries=(?P<entries>\d+)\s+storage_bytes=(?P<storage_bytes>\d+)"
)
SWEEP_RE = re.compile(
    r"PROFILE_SWEEP\s+scalar_h0_stream\s+"
    r"pruning_mask_calls=(?P<pruning_mask_calls>\d+)\s+"
    r"(?:active_state_checks=\d+\s+)?"
    r"active_neighbor_hits=(?P<active_neighbor_hits>\d+)\s+"
    r"representative_visits=(?P<representative_visits>\d+)\s+"
    r"pruning_cache_hits=(?P<pruning_cache_hits>\d+)\s+"
    r"pruning_cache_misses=(?P<pruning_cache_misses>\d+)\s+"
    r"component_mask_computations=(?P<component_mask_computations>\d+)\s+"
    r"union_attempts=(?P<union_attempts>\d+)\s+"
    r"successful_unions=(?P<successful_unions>\d+)\s+"
    r"same_root_unions=(?P<same_root_unions>\d+)\s+"
    r"find_calls=(?P<find_calls>\d+)\s+"
    r"find_parent_steps=(?P<find_parent_steps>\d+)\s+"
    r"active_rechecks=(?P<active_rechecks>\d+)\s+"
    r"active_recheck_failures=(?P<active_recheck_failures>\d+)\s+"
    r"root_carry_attempts=(?P<root_carry_attempts>\d+)\s+"
    r"avoided_find_calls=(?P<avoided_find_calls>\d+)\s+"
    r"pruning_cache_lookups=(?P<pruning_cache_lookups>\d+)"
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
            f"FAIL h0 {label}: persistence multisets differ\n"
            f"missing: {missing.most_common(10)}\nextra: {extra.most_common(10)}"
        )
    print(f"PASS h0 {label}: {sum(expected.values())} intervals agree exactly")


def parse_diagnostics(label: str, log: str) -> tuple[dict[str, int | str], dict[str, int]]:
    cache = CACHE_RE.search(log)
    sweep = SWEEP_RE.search(log)
    if cache is None or sweep is None:
        raise SystemExit(f"FAIL h0 {label}: PROFILE_CACHE/PROFILE_SWEEP diagnostics missing")
    cache_stats: dict[str, int | str] = {
        "mode": cache.group("mode"),
        "entries": int(cache.group("entries")),
        "storage_bytes": int(cache.group("storage_bytes")),
    }
    sweep_stats = {name: int(value) for name, value in sweep.groupdict().items()}
    return cache_stats, sweep_stats


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

    with tempfile.TemporaryDirectory(prefix="betti_h0_cache_equivalence_") as temp:
        root = Path(temp)
        oracle, _ = run(
            args.binary, args.input, root / "oracle", args.slab_depth,
            args.foreground_connectivity, "h0-scalar",
        )
        reference: Counter[tuple[str, str]] | None = None
        for cache_mode in CACHE_MODES:
            intervals, log = run(
                args.binary, args.input, root / cache_mode, args.slab_depth,
                args.foreground_connectivity, "h0-scalar-stream",
                [*fixed, "--h0-pruning-cache", cache_mode],
            )
            assert_equal(f"cache={cache_mode} vs in-memory", oracle, intervals)
            if reference is None:
                reference = intervals
            else:
                assert_equal(f"cache={cache_mode} vs cache=off", reference, intervals)

            cache_stats, sweep = parse_diagnostics(cache_mode, log)
            if cache_stats["mode"] != cache_mode:
                raise SystemExit(f"FAIL h0 {cache_mode}: reported cache mode {cache_stats['mode']!r}")
            if sweep["active_recheck_failures"] != 0:
                raise SystemExit(f"FAIL h0 {cache_mode}: inactive representative observed")
            if cache_mode == "off":
                if cache_stats["entries"] != 0 or cache_stats["storage_bytes"] != 0:
                    raise SystemExit("FAIL h0 cache=off: cache storage was allocated")
                if sweep["pruning_cache_lookups"] or sweep["pruning_cache_hits"] or sweep["pruning_cache_misses"]:
                    raise SystemExit("FAIL h0 cache=off: cache activity was reported")
            else:
                if sweep["pruning_cache_lookups"] != sweep["pruning_cache_hits"] + sweep["pruning_cache_misses"]:
                    raise SystemExit(
                        f"FAIL h0 {cache_mode}: lookups != hits + misses "
                        f"({sweep['pruning_cache_lookups']} != "
                        f"{sweep['pruning_cache_hits']} + {sweep['pruning_cache_misses']})"
                    )
                if sweep["component_mask_computations"] != sweep["pruning_cache_misses"]:
                    raise SystemExit(
                        f"FAIL h0 {cache_mode}: recomputations != misses "
                        f"({sweep['component_mask_computations']} != {sweep['pruning_cache_misses']})"
                    )
            hit_rate = (
                sweep["pruning_cache_hits"] / sweep["pruning_cache_lookups"]
                if sweep["pruning_cache_lookups"] else 0.0
            )
            print(
                f"PASS h0 cache diagnostics {cache_mode}: entries={cache_stats['entries']} "
                f"lookups={sweep['pruning_cache_lookups']} hits={sweep['pruning_cache_hits']} "
                f"hit_rate={100.0 * hit_rate:.2f}% computations={sweep['component_mask_computations']}"
            )

    print("PASS: all H0 pruning-cache modes preserve exact scalar persistence")


if __name__ == "__main__":
    main()
