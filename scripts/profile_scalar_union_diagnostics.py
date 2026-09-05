#!/usr/bin/env python3
"""Compare local sweep operation counters for conventional and root-carrying unions."""

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
    r"active_recheck_failures=(?P<active_recheck_failures>\d+)\s+"
    r"root_carry_attempts=(?P<root_carry_attempts>\d+)\s+"
    r"avoided_find_calls=(?P<avoided_find_calls>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    parser.add_argument("--output", type=Path, default=Path("scalar_union_diagnostics.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="betti_union_diagnostics_") as temp:
        root = Path(temp)
        for mode in ("h0-scalar-stream", "h2-scalar-stream"):
            for kernel in ("conventional", "root-carrying"):
                output = root / f"{mode}_{kernel}.csv"
                command = [
                    str(args.binary.resolve()), str(args.input.resolve()), str(args.slab_depth),
                    str(args.foreground_connectivity), mode, str(output),
                    "--merge-strategy", "scan", "--interface-order", "radix", "--event-order", "verify",
                    "--f32-key-mode", args.f32_key_mode, "--neighbor-kernel", "interior-fast",
                    "--representative-active-check", "recheck", "--union-kernel", kernel,
                    "--sweep-diagnostics",
                ]
                completed = subprocess.run(command, text=True, capture_output=True)
                log = completed.stdout + completed.stderr
                if completed.returncode != 0:
                    raise RuntimeError(f"{mode} {kernel} failed\n{log}")
                match = SWEEP_RE.search(log)
                if match is None:
                    raise RuntimeError(f"PROFILE_SWEEP missing for {mode} {kernel}")
                raw = {key: int(value) for key, value in match.groupdict().items() if key != "name"}
                row: dict[str, object] = {"mode": mode, "union_kernel": kernel, **raw}
                row["same_root_fraction"] = raw["same_root_unions"] / raw["union_attempts"] if raw["union_attempts"] else 0.0
                row["parent_steps_per_find"] = raw["find_parent_steps"] / raw["find_calls"] if raw["find_calls"] else 0.0
                row["finds_per_union_attempt"] = raw["find_calls"] / raw["union_attempts"] if raw["union_attempts"] else 0.0
                row["avoided_find_fraction"] = raw["avoided_find_calls"] / raw["union_attempts"] if raw["union_attempts"] else 0.0
                rows.append(row)

    conventional = {str(row["mode"]): row for row in rows if row["union_kernel"] == "conventional"}
    for row in rows:
        base = conventional[str(row["mode"])]
        row["find_calls_vs_conventional"] = int(row["find_calls"]) - int(base["find_calls"])
        row["find_call_reduction_fraction"] = (
            (int(base["find_calls"]) - int(row["find_calls"])) / int(base["find_calls"])
            if int(base["find_calls"]) else 0.0
        )

    fields = list(rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for mode in ("h0-scalar-stream", "h2-scalar-stream"):
        c = next(row for row in rows if row["mode"] == mode and row["union_kernel"] == "conventional")
        r = next(row for row in rows if row["mode"] == mode and row["union_kernel"] == "root-carrying")
        if int(r["active_recheck_failures"]) != 0:
            raise RuntimeError(f"{mode}: inactive representative observed")
        print(
            f"{mode}: find_calls {c['find_calls']} -> {r['find_calls']} "
            f"({float(r['find_call_reduction_fraction']):.2%} reduction), "
            f"parent_steps {c['find_parent_steps']} -> {r['find_parent_steps']}, "
            f"same_root={float(r['same_root_fraction']):.2%}, "
            f"root_carry_attempts={r['root_carry_attempts']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
