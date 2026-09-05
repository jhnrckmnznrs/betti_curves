#!/usr/bin/env python3
"""External profiler for stream_betti_curves hierarchical branch trees.

No Rust source instrumentation is required. Runs the existing release binary
under /usr/bin/time -v, sweeps slab depths, parses the program's existing
PROFILE_BRANCH_* lines, and writes raw + median-summary CSV files.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


TIME_KEYS = {
    "User time (seconds)": "user_seconds",
    "System time (seconds)": "system_seconds",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "Minor (reclaiming a frame) page faults": "minor_faults",
    "Major (requiring I/O) page faults": "major_faults",
    "Voluntary context switches": "voluntary_context_switches",
    "Involuntary context switches": "involuntary_context_switches",
    "File system inputs": "fs_inputs",
    "File system outputs": "fs_outputs",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("stack", type=Path)
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--dimensions", nargs="+", choices=("h0", "h2"), default=("h0", "h2"))
    p.add_argument("--slab-depths", nargs="+", type=int, default=(4, 8, 16, 32))
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("profiles/branch_tree_external_profile.csv"))
    return p.parse_args()


def parse_time_file(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        for label, key in TIME_KEYS.items():
            prefix = label + ":"
            if line.startswith(prefix):
                try:
                    out[key] = float(line[len(prefix):].strip())
                except ValueError:
                    pass
                break
    return out


def parse_kv(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def run_one(args: argparse.Namespace, dim: str, depth: int, repeat: int) -> dict[str, object]:
    binary = args.binary.resolve()
    stack = args.stack.resolve()
    mode = f"branch-tree-{dim}-hierarchical"

    with tempfile.TemporaryDirectory(prefix=f"betti_profile_{dim}_d{depth}_") as td:
        td = Path(td)
        time_file = td / "time.txt"
        cmd = [
            "/usr/bin/time", "-v", "-o", str(time_file),
            str(binary), str(stack), str(depth),
            str(args.foreground_connectivity), mode,
        ]

        t0 = time.perf_counter()
        cp = subprocess.run(
            cmd,
            cwd=td,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        wall = time.perf_counter() - t0

        if cp.returncode != 0:
            raise RuntimeError(
                f"{mode} depth={depth} repeat={repeat} failed\n{cp.stdout[-12000:]}"
            )

        upper = dim.upper()
        hierarchy_prefix = f"PROFILE_BRANCH_{upper}_HIERARCHY "
        combine_prefix = f"PROFILE_BRANCH_{upper}_HIER_COMBINE "
        compute_re = re.compile(
            rf"Hierarchical {upper} branch-tree computation took ([0-9.]+) seconds"
        )

        hierarchy = {}
        combines = []
        compute_seconds = None

        for line in cp.stdout.splitlines():
            if line.startswith(hierarchy_prefix):
                hierarchy = parse_kv(line[len(hierarchy_prefix):])
            elif line.startswith(combine_prefix):
                combines.append(parse_kv(line[len(combine_prefix):]))
            else:
                m = compute_re.search(line)
                if m:
                    compute_seconds = float(m.group(1))

        if not hierarchy:
            raise RuntimeError(
                f"Missing {hierarchy_prefix.strip()} output. "
                "Use the already-correct hierarchical binary."
            )

        row: dict[str, object] = {
            "dimension": dim,
            "slab_depth": depth,
            "repeat": repeat,
            "wall_seconds": wall,
            "compute_seconds": compute_seconds if compute_seconds is not None else wall,
            "combine_lines": len(combines),
        }
        row.update(parse_time_file(time_file))

        for k, v in hierarchy.items():
            try:
                row[k] = int(v)
            except ValueError:
                row[k] = v

        integer_combine_fields = (
            "pair_nodes",
            "parent_interface_nodes",
            "parent_attach",
            "parent_interface",
            "cross_retained",
            "cross_candidates",
        )
        for field in integer_combine_fields:
            vals = []
            for c in combines:
                try:
                    vals.append(int(c.get(field, "0")))
                except ValueError:
                    vals.append(0)
            row[f"combine_total_{field}"] = sum(vals)
            row[f"combine_max_{field}"] = max(vals, default=0)

        timing_combine_fields = (
            "setup_seconds",
            "cross_seconds",
            "sort_seconds",
            "merge_seconds",
            "reduce_seconds",
            "finish_seconds",
            "total_seconds",
        )
        for field in timing_combine_fields:
            vals = []
            for c in combines:
                try:
                    vals.append(float(c.get(field, "0")))
                except ValueError:
                    vals.append(0.0)
            row[f"combine_total_{field}"] = sum(vals)
            row[f"combine_max_{field}"] = max(vals, default=0.0)

        candidates = int(row.get("combine_total_cross_candidates", 0))
        retained = int(row.get("combine_total_cross_retained", 0))
        row["cross_retention_fraction"] = (retained / candidates) if candidates else 0.0

        return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def med(group: list[dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in group if key in r and r[key] not in ("", None)]
    return statistics.median(vals) if vals else 0.0


def main() -> int:
    args = parse_args()
    if not Path("/usr/bin/time").exists():
        raise SystemExit("/usr/bin/time is required (Ubuntu package: time)")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    rows = []
    total = len(args.dimensions) * len(args.slab_depths) * args.repeats
    done = 0

    for dim in args.dimensions:
        for depth in args.slab_depths:
            for rep in range(1, args.repeats + 1):
                done += 1
                print(f"[{done}/{total}] {dim.upper()} slab_depth={depth} repeat={rep}", flush=True)
                row = run_one(args, dim, depth, rep)
                rows.append(row)
                print(
                    f"  compute={float(row['compute_seconds']):.3f}s "
                    f"wall={float(row['wall_seconds']):.3f}s "
                    f"rss={float(row.get('max_rss_kb', 0))/1024:.1f} MiB "
                    f"max_pair_nodes={row.get('max_pair_nodes', '?')}"
                )

    write_csv(args.output, rows)

    summary_rows: list[dict[str, object]] = []
    for dim in args.dimensions:
        for depth in args.slab_depths:
            group = [
                r for r in rows
                if r["dimension"] == dim and int(r["slab_depth"]) == depth
            ]
            summary_rows.append({
                "dimension": dim,
                "slab_depth": depth,
                "repeats": len(group),
                "median_compute_seconds": med(group, "compute_seconds"),
                "median_wall_seconds": med(group, "wall_seconds"),
                "median_user_seconds": med(group, "user_seconds"),
                "median_system_seconds": med(group, "system_seconds"),
                "median_max_rss_mb": med(group, "max_rss_kb") / 1024.0,
                "median_minor_faults": med(group, "minor_faults"),
                "median_major_faults": med(group, "major_faults"),
                "median_max_pair_nodes": med(group, "max_pair_nodes"),
                "median_finalized_input_events": med(group, "finalized_input_events"),
                "median_finalized_events": med(group, "finalized_events"),
                "median_contracted_zero_events": med(group, "contracted_zero_events"),
                "median_repaired_history_parent_refs": med(group, "repaired_history_parent_refs"),
                "median_compact_event_bytes": med(group, "compact_event_bytes"),
                "median_full_event_bytes": med(group, "full_event_bytes"),
                "median_finalized_storage_mb": med(group, "finalized_storage_bytes") / (1024.0 * 1024.0),
                "median_leaf_batch_seconds": med(group, "leaf_batch_seconds"),
                "median_leaf_read_thread_seconds": med(group, "leaf_read_thread_seconds"),
                "median_leaf_compute_thread_seconds": med(group, "leaf_compute_thread_seconds"),
                "median_bucket_seconds": med(group, "bucket_seconds"),
                "median_fan_in_seconds": med(group, "fan_in_seconds"),
                "median_root_combine_seconds": med(group, "root_combine_seconds"),
                "median_root_reduce_seconds": med(group, "root_reduce_seconds"),
                "median_canonical_seconds": med(group, "canonical_seconds"),
                "median_combine_setup_seconds": med(group, "combine_total_setup_seconds"),
                "median_combine_cross_seconds": med(group, "combine_total_cross_seconds"),
                "median_combine_sort_seconds": med(group, "combine_total_sort_seconds"),
                "median_combine_merge_seconds": med(group, "combine_total_merge_seconds"),
                "median_combine_reduce_seconds": med(group, "combine_total_reduce_seconds"),
                "median_combine_finish_seconds": med(group, "combine_total_finish_seconds"),
                "median_combine_total_seconds": med(group, "combine_total_total_seconds"),
                "median_combine_total_cross_candidates": med(group, "combine_total_cross_candidates"),
                "median_combine_total_cross_retained": med(group, "combine_total_cross_retained"),
                "median_cross_retention_fraction": med(group, "cross_retention_fraction"),
            })

    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    write_csv(summary_path, summary_rows)

    print("\nMedian summary")
    for r in summary_rows:
        print(
            f"{r['dimension'].upper()} d={r['slab_depth']:>3}: "
            f"{r['median_compute_seconds']:.3f}s, "
            f"{r['median_max_rss_mb']:.1f} MiB RSS, "
            f"events={r['median_finalized_events']:.0f}/"
            f"{r['median_finalized_input_events']:.0f}, "
            f"zero_contract={r['median_contracted_zero_events']:.0f}, "
            f"stored={r['median_finalized_storage_mb']:.1f} MiB, "
            f"leaf={r['median_leaf_batch_seconds']:.3f}s, "
            f"fan-in={r['median_fan_in_seconds'] + r['median_root_combine_seconds']:.3f}s, "
            f"root={r['median_root_reduce_seconds']:.3f}s, "
            f"canon={r['median_canonical_seconds']:.3f}s, "
            f"pair={r['median_max_pair_nodes']:.0f}"
        )

    print(f"\nRaw CSV:     {args.output}")
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
