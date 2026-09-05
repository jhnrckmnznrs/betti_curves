#!/usr/bin/env python3
"""Run one diagnostic scalar sweep and summarize pruning/union-find operation counters."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path

SWEEP_RE = re.compile(
    r"PROFILE_SWEEP\s+(?P<name>\S+)\s+"
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
    r"active_recheck_failures=(?P<active_recheck_failures>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    parser.add_argument("--output", type=Path, default=Path("scalar_sweep_diagnostics.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="betti_sweep_diagnostics_") as temp:
        root = Path(temp)
        for mode in ("h0-scalar-stream", "h2-scalar-stream"):
            output = root / f"{mode}.csv"
            command = [
                str(args.binary.resolve()), str(args.input.resolve()), str(args.slab_depth),
                str(args.foreground_connectivity), mode, str(output),
                "--merge-strategy", "scan", "--interface-order", "radix", "--event-order", "verify",
                "--f32-key-mode", args.f32_key_mode, "--neighbor-kernel", "interior-fast",
                "--representative-active-check", "recheck", "--sweep-diagnostics",
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            log = completed.stdout + completed.stderr
            if completed.returncode != 0:
                raise RuntimeError(f"{mode} failed\n{log}")
            match = SWEEP_RE.search(log)
            if match is None:
                raise RuntimeError(f"PROFILE_SWEEP missing for {mode}")
            raw = {key: int(value) for key, value in match.groupdict().items() if key != "name"}
            cache_total = raw["pruning_cache_hits"] + raw["pruning_cache_misses"]
            row: dict[str, object] = {"mode": mode, **raw}
            row["cache_hit_rate"] = raw["pruning_cache_hits"] / cache_total if cache_total else 0.0
            row["active_hits_per_voxel"] = raw["active_neighbor_hits"] / raw["pruning_mask_calls"] if raw["pruning_mask_calls"] else 0.0
            row["representatives_per_voxel"] = raw["representative_visits"] / raw["pruning_mask_calls"] if raw["pruning_mask_calls"] else 0.0
            row["representative_retention"] = raw["representative_visits"] / raw["active_neighbor_hits"] if raw["active_neighbor_hits"] else 0.0
            row["same_root_fraction"] = raw["same_root_unions"] / raw["union_attempts"] if raw["union_attempts"] else 0.0
            row["parent_steps_per_find"] = raw["find_parent_steps"] / raw["find_calls"] if raw["find_calls"] else 0.0
            rows.append(row)

    fields = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(
            f"{row['mode']}: masks={row['pruning_mask_calls']} active_hits/voxel={row['active_hits_per_voxel']:.3f} "
            f"reps/voxel={row['representatives_per_voxel']:.3f} cache_hit={row['cache_hit_rate']:.2%} "
            f"same_root={row['same_root_fraction']:.2%} parent_steps/find={row['parent_steps_per_find']:.3f} "
            f"recheck_failures={row['active_recheck_failures']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
