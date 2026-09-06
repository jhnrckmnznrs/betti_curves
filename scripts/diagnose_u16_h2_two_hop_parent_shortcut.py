#!/usr/bin/env python3
"""Instrumented structural diagnostic for the U16 H2 two-hop parent shortcut.

This is deliberately separate from timing: --sweep-diagnostics perturbs the hot
loop and must not be used to estimate the production speedup.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import tempfile
from pathlib import Path

from profile_u16_h2_two_hop_parent_shortcut import (
    COUNTERS,
    INVARIANTS,
    command,
    mode_for_backend,
    run_once,
)


def median(rows: list[dict[str, object]], field: str) -> float:
    return float(statistics.median(float(r[field]) for r in rows))


def main() -> int:
    ap = argparse.ArgumentParser(description="Instrumented U16 H2 two-hop structural diagnostic.")
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/v23_2_1_u16_h2_two_hop_parent_shortcut_diagnostics.csv"),
    )
    args = ap.parse_args()

    if args.repeats < 1:
        ap.error("--repeats must be >= 1")
    if any(depth <= 0 for depth in args.slab_depths):
        ap.error("all --slab-depths must be positive")
    if len(set(args.slab_depths)) != len(args.slab_depths):
        ap.error("--slab-depths must not contain duplicates")

    stack = args.stack.resolve()
    binary = args.binary.resolve()
    if not stack.exists():
        ap.error(f"stack does not exist: {stack}")
    if not binary.exists():
        ap.error(f"binary does not exist: {binary}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="u16_h2_two_hop_diag_") as td_raw:
        td = Path(td_raw)
        seq = 0
        for depth in args.slab_depths:
            for repeat in range(1, args.repeats + 1):
                current: dict[str, dict[str, object]] = {}
                for backend in ("reference", "candidate"):
                    seq += 1
                    mode = mode_for_backend(backend)
                    result = run_once(
                        command(
                            binary,
                            stack,
                            depth,
                            args.foreground_connectivity,
                            td / f"diag_{seq}.csv",
                            mode,
                            diagnostics=True,
                        ),
                        td / f"diag_{seq}.time",
                        mode,
                    )
                    row = {
                        "backend": backend,
                        "slab_depth": depth,
                        "repeat": repeat,
                        **result,
                    }
                    rows.append(row)
                    current[backend] = row
                    print(
                        f"{backend} d={depth} repeat={repeat}: "
                        f"direct={result['direct_parent_hits']}/{result['direct_parent_checks']} "
                        f"two_hop={result['two_hop_parent_hits']}/{result['two_hop_parent_checks']} "
                        f"neighbor_finds={result['neighbor_find_calls']} "
                        f"same_root_depth2={result['same_root_neighbor_find_two_hop']} "
                        f"same_root_gt2={result['same_root_neighbor_find_gt_two_hop']}"
                    )

                ref = current["reference"]
                cand = current["candidate"]
                for metric in INVARIANTS:
                    if int(ref[metric]) != int(cand[metric]):
                        raise RuntimeError(
                            f"invariant failed d={depth} repeat={repeat}: {metric} "
                            f"reference={ref[metric]} candidate={cand[metric]}"
                        )
                if int(ref["two_hop_parent_checks"]) or int(ref["two_hop_parent_hits"]):
                    raise RuntimeError(f"reference unexpectedly executed two-hop checks at d={depth}")

    fields = [
        "backend",
        "slab_depth",
        "repeat",
        "neighbor_root_check",
        *COUNTERS,
    ]
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary_fields = [
        "slab_depth",
        "repeats",
        "reference_direct_parent_checks",
        "reference_direct_parent_hits",
        "reference_direct_parent_hit_rate",
        "reference_neighbor_find_calls",
        "reference_neighbor_find_parent_steps",
        "reference_same_root_neighbor_find_zero_hop",
        "reference_same_root_neighbor_find_one_hop",
        "reference_same_root_neighbor_find_two_hop",
        "reference_same_root_neighbor_find_gt_two_hop",
        "candidate_direct_parent_checks",
        "candidate_direct_parent_hits",
        "candidate_two_hop_parent_checks",
        "candidate_two_hop_parent_hits",
        "candidate_two_hop_hit_rate",
        "candidate_neighbor_find_calls",
        "candidate_neighbor_find_parent_steps",
        "neighbor_finds_avoided_vs_reference",
        "neighbor_find_parent_steps_delta",
    ]
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for depth in sorted(set(int(r["slab_depth"]) for r in rows)):
            ref_rows = [r for r in rows if int(r["slab_depth"]) == depth and r["backend"] == "reference"]
            cand_rows = [r for r in rows if int(r["slab_depth"]) == depth and r["backend"] == "candidate"]
            ref_checks = median(ref_rows, "direct_parent_checks")
            ref_hits = median(ref_rows, "direct_parent_hits")
            cand_checks = median(cand_rows, "two_hop_parent_checks")
            cand_hits = median(cand_rows, "two_hop_parent_hits")
            ref_finds = median(ref_rows, "neighbor_find_calls")
            cand_finds = median(cand_rows, "neighbor_find_calls")
            writer.writerow({
                "slab_depth": depth,
                "repeats": min(len(ref_rows), len(cand_rows)),
                "reference_direct_parent_checks": ref_checks,
                "reference_direct_parent_hits": ref_hits,
                "reference_direct_parent_hit_rate": ref_hits / ref_checks if ref_checks else 0.0,
                "reference_neighbor_find_calls": ref_finds,
                "reference_neighbor_find_parent_steps": median(ref_rows, "neighbor_find_parent_steps"),
                "reference_same_root_neighbor_find_zero_hop": median(ref_rows, "same_root_neighbor_find_zero_hop"),
                "reference_same_root_neighbor_find_one_hop": median(ref_rows, "same_root_neighbor_find_one_hop"),
                "reference_same_root_neighbor_find_two_hop": median(ref_rows, "same_root_neighbor_find_two_hop"),
                "reference_same_root_neighbor_find_gt_two_hop": median(ref_rows, "same_root_neighbor_find_gt_two_hop"),
                "candidate_direct_parent_checks": median(cand_rows, "direct_parent_checks"),
                "candidate_direct_parent_hits": median(cand_rows, "direct_parent_hits"),
                "candidate_two_hop_parent_checks": cand_checks,
                "candidate_two_hop_parent_hits": cand_hits,
                "candidate_two_hop_hit_rate": cand_hits / cand_checks if cand_checks else 0.0,
                "candidate_neighbor_find_calls": cand_finds,
                "candidate_neighbor_find_parent_steps": median(cand_rows, "neighbor_find_parent_steps"),
                "neighbor_finds_avoided_vs_reference": ref_finds - cand_finds,
                "neighbor_find_parent_steps_delta": median(cand_rows, "neighbor_find_parent_steps") - median(ref_rows, "neighbor_find_parent_steps"),
            })

    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
