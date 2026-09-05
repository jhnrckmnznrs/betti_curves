#!/usr/bin/env python3
"""Measure H0 pruning-cache hit/recomputation statistics for all cache sizes."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path

CACHE_MODES = ("off", "4k", "16k", "64k", "256k")
CACHE_RE = re.compile(
    r"PROFILE_CACHE\s+scalar_h0_stream\s+mode=(?P<cache_mode>\S+)\s+"
    r"entries=(?P<cache_entries>\d+)\s+storage_bytes=(?P<cache_storage_bytes>\d+)"
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
PREP_RE = re.compile(
    r"PROFILE_PREP\s+scalar_h0_stream\s+.*local_sweep_seconds=(?P<local_sweep>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    parser.add_argument("--output", type=Path, default=Path("h0_pruning_cache_diagnostics.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="betti_h0_cache_diag_") as temp:
        root = Path(temp)
        for cache_mode in CACHE_MODES:
            output = root / f"{cache_mode}.csv"
            command = [
                str(args.binary.resolve()), str(args.input.resolve()), str(args.slab_depth),
                str(args.foreground_connectivity), "h0-scalar-stream", str(output),
                "--merge-strategy", "scan", "--interface-order", "radix", "--event-order", "verify",
                "--f32-key-mode", args.f32_key_mode, "--neighbor-kernel", "interior-fast",
                "--representative-active-check", "recheck", "--union-kernel", "root-carrying",
                "--h0-pruning-cache", cache_mode, "--sweep-diagnostics",
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            log = completed.stdout + completed.stderr
            if completed.returncode != 0:
                raise RuntimeError(f"cache={cache_mode} failed\n{log}")
            cache_match = CACHE_RE.search(log)
            sweep_match = SWEEP_RE.search(log)
            prep_match = PREP_RE.search(log)
            if cache_match is None or sweep_match is None:
                raise RuntimeError(f"diagnostics missing for cache={cache_mode}")
            raw = {key: int(value) for key, value in sweep_match.groupdict().items()}
            cache = cache_match.groupdict()
            row: dict[str, object] = {
                "h0_pruning_cache": cache["cache_mode"],
                "cache_entries": int(cache["cache_entries"]),
                "cache_storage_bytes": int(cache["cache_storage_bytes"]),
                "local_sweep_seconds": float(prep_match.group("local_sweep")) if prep_match else "",
                **raw,
            }
            lookups = raw["pruning_cache_lookups"]
            row["cache_hit_rate"] = raw["pruning_cache_hits"] / lookups if lookups else 0.0
            row["computation_fraction_vs_no_cache"] = 0.0
            rows.append(row)

    no_cache_computations = int(next(row for row in rows if row["h0_pruning_cache"] == "off")["component_mask_computations"])
    for row in rows:
        computations = int(row["component_mask_computations"])
        row["computation_fraction_vs_no_cache"] = (
            computations / no_cache_computations if no_cache_computations else 0.0
        )
        if int(row["active_recheck_failures"]) != 0:
            raise RuntimeError(f"cache={row['h0_pruning_cache']}: inactive representative observed")
        if row["h0_pruning_cache"] == "off":
            if int(row["pruning_cache_lookups"]) != 0:
                raise RuntimeError("cache=off unexpectedly performed lookups")
        elif int(row["pruning_cache_lookups"]) != int(row["pruning_cache_hits"]) + int(row["pruning_cache_misses"]):
            raise RuntimeError(f"cache={row['h0_pruning_cache']}: lookup accounting mismatch")

    fields = list(rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    print("cache\tentries\tMiB\thit_rate\tcomputations\tvs_off\tsweep_s")
    for row in rows:
        sweep = row["local_sweep_seconds"]
        sweep_text = f"{float(sweep):.6f}" if sweep != "" else "n/a"
        print(
            f"{row['h0_pruning_cache']}\t{row['cache_entries']}\t"
            f"{int(row['cache_storage_bytes']) / (1024**2):.3f}\t"
            f"{float(row['cache_hit_rate']):.2%}\t{row['component_mask_computations']}\t"
            f"{float(row['computation_fraction_vs_no_cache']):.2%}\t{sweep_text}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
