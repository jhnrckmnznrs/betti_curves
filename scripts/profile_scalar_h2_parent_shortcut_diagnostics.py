#!/usr/bin/env python3
"""Collect one diagnostic H2 run for find and parent-shortcut root resolution."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path

SWEEP_RE = re.compile(
    r"PROFILE_SWEEP\s+scalar_h2_stream\s+.*"
    r"union_attempts=(?P<union_attempts>\d+)\s+successful_unions=(?P<successful_unions>\d+)\s+"
    r"same_root_unions=(?P<same_root_unions>\d+)\s+find_calls=(?P<find_calls>\d+)\s+"
    r"find_parent_steps=(?P<find_parent_steps>\d+)\s+active_rechecks=(?P<active_rechecks>\d+)\s+"
    r"active_recheck_failures=(?P<active_recheck_failures>\d+)\s+"
    r"root_carry_attempts=(?P<root_carry_attempts>\d+)\s+avoided_find_calls=(?P<avoided_find_calls>\d+)\s+"
    r"direct_parent_checks=(?P<direct_parent_checks>\d+)\s+direct_parent_hits=(?P<direct_parent_hits>\d+)\s+"
    r"avoided_neighbor_find_calls=(?P<avoided_neighbor_find_calls>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    parser.add_argument("--output", type=Path, default=Path("h2_parent_shortcut_diagnostics.csv"))
    return parser.parse_args()


def run(args: argparse.Namespace, root_check: str, work: Path) -> dict[str, object]:
    output = work / f"{root_check}.csv"
    cmd = [
        str(args.binary.resolve()), str(args.input.resolve()), str(args.slab_depth),
        str(args.foreground_connectivity), "h2-scalar-stream", str(output),
        "--merge-strategy", "scan", "--interface-order", "radix", "--event-order", "verify",
        "--f32-key-mode", args.f32_key_mode, "--neighbor-kernel", "interior-fast",
        "--representative-active-check", "recheck", "--union-kernel", "root-carrying",
        "--neighbor-root-check", root_check, "--sweep-diagnostics",
    ]
    completed = subprocess.run(cmd, cwd=work, text=True, capture_output=True)
    log = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"{root_check} failed:\n{log}")
    match = SWEEP_RE.search(log)
    if match is None:
        raise RuntimeError(f"PROFILE_SWEEP not found for {root_check}\n{log}")
    row: dict[str, object] = {"neighbor_root_check": root_check}
    row.update({name: int(value) for name, value in match.groupdict().items()})
    checks = int(row["direct_parent_checks"])
    hits = int(row["direct_parent_hits"])
    finds = int(row["find_calls"])
    steps = int(row["find_parent_steps"])
    row["direct_parent_hit_rate"] = hits / checks if checks else 0.0
    row["parent_steps_per_find"] = steps / finds if finds else 0.0
    return row


def main() -> None:
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    with tempfile.TemporaryDirectory(prefix="betti_h2_parent_shortcut_diag_") as temp:
        root = Path(temp)
        rows = [run(args, "find", root), run(args, "parent-shortcut", root)]
    find_row, shortcut_row = rows
    if int(find_row["direct_parent_checks"]) != 0:
        raise RuntimeError("find reference unexpectedly performed direct-parent checks")
    if int(shortcut_row["direct_parent_hits"]) != int(shortcut_row["avoided_neighbor_find_calls"]):
        raise RuntimeError("shortcut hit count does not equal avoided neighbor finds")
    if int(shortcut_row["active_recheck_failures"]) != 0:
        raise RuntimeError("inactive representative observed")

    fields = list(rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("root_check\tfind_calls\tparent_steps\tdirect_hits\tdirect_checks\thit_rate")
    for row in rows:
        print(
            f"{row['neighbor_root_check']}\t{row['find_calls']}\t{row['find_parent_steps']}\t"
            f"{row['direct_parent_hits']}\t{row['direct_parent_checks']}\t"
            f"{100.0 * float(row['direct_parent_hit_rate']):.2f}%"
        )
    removed = int(find_row["find_calls"]) - int(shortcut_row["find_calls"])
    print(f"avoided H2 neighbor find calls: {removed}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
