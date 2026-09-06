#!/usr/bin/env python3
"""Structural profiler for the production U16 H2 d16 leaf sweep.

This is a diagnostic run, not a timing benchmark. Exact counters are collected
with --sweep-diagnostics. Three contiguous per-voxel regions are timed on a
deterministic hashed ~1/1024 sample so Instant::now() is not called for every
voxel:

  1. activation / boundary / interface bookkeeping
  2. active-neighborhood construction + pruning + representative selection
  3. union-find + persistence decisions + local event emission

The production reference is fixed to parent-shortcut with root dedup disabled.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
USER_RE = re.compile(r"User time \(seconds\):\s*([0-9.]+)")
SYS_RE = re.compile(r"System time \(seconds\):\s*([0-9.]+)")
LEAF_PREFIX = "PROFILE_U16_PERSIST_LEAF dimension=h2 "
STRUCT_PREFIX = "PROFILE_U16_H2_SWEEP_STRUCTURE "
SAMPLE_PREFIX = "PROFILE_H2_SWEEP_SAMPLE "


def parse_kv_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def one_line(stdout: str, prefix: str) -> dict[str, str]:
    lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one {prefix.strip()} line, found {len(lines)}")
    return parse_kv_line(lines[0])


def sample_lines(stdout: str) -> list[dict[str, str]]:
    lines = [parse_kv_line(line) for line in stdout.splitlines() if line.startswith(SAMPLE_PREFIX)]
    if not lines:
        raise RuntimeError("missing PROFILE_H2_SWEEP_SAMPLE lines; was --sweep-diagnostics enabled?")
    return lines


def req(pattern: re.Pattern[str], text: str, label: str) -> str:
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"missing {label} in /usr/bin/time output")
    return match.group(1)


def command(binary: Path, stack: Path, depth: int, connectivity: int, output: Path) -> list[str]:
    return [
        str(binary),
        str(stack),
        str(depth),
        str(connectivity),
        "h2-scalar-hierarchical-stream",
        str(output),
        "--local-h2-birth-state", "compact",
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", "packed",
        "--h2-hier-cross-storage", "direct",
        "--h2-hier-outside-structural-pruning", "off",
        "--neighbor-root-check", "parent-shortcut",
        "--sweep-diagnostics",
    ]


def i(name: str, row: dict[str, str]) -> int:
    if name not in row:
        raise RuntimeError(f"missing counter {name!r}")
    return int(row[name])


def f(name: str, row: dict[str, str]) -> float:
    if name not in row:
        raise RuntimeError(f"missing metric {name!r}")
    return float(row[name])


def safe_rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def run_once(binary: Path, stack: Path, depth: int, connectivity: int, temp: Path, repeat: int) -> dict[str, object]:
    out_path = temp / f"intervals_{repeat}.csv"
    time_path = temp / f"time_{repeat}.txt"
    cmd = command(binary, stack, depth, connectivity, out_path)
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"] = "1"
    env["BETTI_PERSIST_H2_ROOT_DEDUP"] = "0"

    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(time_path), *cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    wall = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    leaf = one_line(proc.stdout, LEAF_PREFIX)
    structure = one_line(proc.stdout, STRUCT_PREFIX)
    samples = sample_lines(proc.stdout)
    if leaf.get("neighbor_root_check") != "parent-shortcut":
        raise RuntimeError(f"expected parent-shortcut, got {leaf.get('neighbor_root_check')!r}")
    if leaf.get("root_dedup") != "off":
        raise RuntimeError("structural profile requires root_dedup=off")

    strides = {i("sample_stride", row) for row in samples}
    if len(strides) != 1:
        raise RuntimeError(f"inconsistent sample strides across slabs: {sorted(strides)}")
    sample_stride = strides.pop()
    sampled_voxels = sum(i("sampled_voxels", row) for row in samples)
    sampled_interior = sum(i("sampled_interior_voxels", row) for row in samples)
    sampled_boundary = sum(i("sampled_boundary_voxels", row) for row in samples)
    sampled_representatives = sum(i("sampled_representatives", row) for row in samples)
    activation_ns = sum(i("activation_boundary_ns", row) for row in samples)
    neighborhood_ns = sum(i("neighborhood_pruning_ns", row) for row in samples)
    union_ns = sum(i("union_persistence_emit_ns", row) for row in samples)
    sampled_total_ns = activation_ns + neighborhood_ns + union_ns

    if sampled_voxels == 0:
        raise RuntimeError("hashed sample selected zero voxels")
    if sampled_interior + sampled_boundary != sampled_voxels:
        raise RuntimeError("sampled interior/boundary counts do not sum to sampled voxels")
    total_voxels = i("total_voxels", structure)
    interior_voxels = i("interior_fast_voxels", structure)
    boundary_voxels = i("boundary_voxels", structure)
    if interior_voxels + boundary_voxels != total_voxels:
        raise RuntimeError("exact interior/boundary counts do not sum to total voxels")

    timing = time_path.read_text()
    union_attempts = i("union_attempts", leaf)
    reps = i("representative_visits", structure)
    cache_hits = i("pruning_cache_hits", structure)
    cache_misses = i("pruning_cache_misses", structure)
    active_checks = i("active_state_checks", structure)
    active_hits = i("active_neighbor_hits", structure)

    result: dict[str, object] = {
        "repeat": repeat,
        "slab_depth": depth,
        "foreground_connectivity": connectivity,
        "wall_seconds": wall,
        "user_seconds": float(req(USER_RE, timing, "user time")),
        "system_seconds": float(req(SYS_RE, timing, "system time")),
        "max_rss_mib": float(req(RSS_RE, timing, "maximum RSS")) / 1024.0,
        "scalar_order_seconds": f("scalar_order_seconds", leaf),
        "local_sweep_seconds": f("local_sweep_seconds", leaf),
        "sample_stride": sample_stride,
        "sampled_voxels": sampled_voxels,
        "sampled_interior_voxels": sampled_interior,
        "sampled_boundary_voxels": sampled_boundary,
        "sampled_representatives": sampled_representatives,
        "activation_boundary_ns": activation_ns,
        "neighborhood_pruning_ns": neighborhood_ns,
        "union_persistence_emit_ns": union_ns,
        "sampled_total_ns": sampled_total_ns,
        "activation_boundary_share": safe_rate(activation_ns, sampled_total_ns),
        "neighborhood_pruning_share": safe_rate(neighborhood_ns, sampled_total_ns),
        "union_persistence_emit_share": safe_rate(union_ns, sampled_total_ns),
        "activation_boundary_ns_per_sampled_voxel": safe_rate(activation_ns, sampled_voxels),
        "neighborhood_pruning_ns_per_sampled_voxel": safe_rate(neighborhood_ns, sampled_voxels),
        "union_persistence_emit_ns_per_sampled_voxel": safe_rate(union_ns, sampled_voxels),
        "sampled_representatives_per_voxel": safe_rate(sampled_representatives, sampled_voxels),
        "union_persistence_emit_ns_per_sampled_representative": safe_rate(union_ns, sampled_representatives),
        "total_voxels": total_voxels,
        "interior_fast_voxels": interior_voxels,
        "boundary_voxels": boundary_voxels,
        "interior_fast_fraction": safe_rate(interior_voxels, total_voxels),
        "pruning_mask_calls": i("pruning_mask_calls", structure),
        "active_state_checks": active_checks,
        "active_neighbor_hits": active_hits,
        "active_neighbor_hit_rate": safe_rate(active_hits, active_checks),
        "representative_visits": reps,
        "representatives_per_voxel": safe_rate(reps, total_voxels),
        "pruning_cache_hits": cache_hits,
        "pruning_cache_misses": cache_misses,
        "pruning_cache_hit_rate": safe_rate(cache_hits, cache_hits + cache_misses),
        "component_mask_computations": i("component_mask_computations", structure),
        "union_attempts": union_attempts,
        "successful_unions": i("successful_unions", leaf),
        "same_root_unions": i("same_root_unions", leaf),
        "union_attempts_per_voxel": safe_rate(union_attempts, total_voxels),
        "representative_to_union_ratio": safe_rate(reps, union_attempts),
        "zero_persistence_pairs_elided": i("zero_persistence_pairs_elided", leaf),
        "direct_parent_checks": i("direct_parent_checks", leaf),
        "direct_parent_hits": i("direct_parent_hits", leaf),
        "direct_parent_hit_rate": safe_rate(i("direct_parent_hits", leaf), i("direct_parent_checks", leaf)),
        "neighbor_find_calls": i("neighbor_find_calls", leaf),
        "neighbor_find_parent_steps": i("neighbor_find_parent_steps", leaf),
        "interface_rep_queries": i("interface_rep_queries", structure),
        "interface_rep_writes": i("interface_rep_writes", structure),
        "interface_forced_root_unions": i("interface_forced_root_unions", structure),
        "interface_interface_unions": i("interface_interface_unions", structure),
        "final_pairs": i("final_pairs", structure),
        "attach_events": i("attach_events", structure),
        "outside_events": i("outside_events", structure),
        "interface_events": i("interface_events", structure),
    }
    return result


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Structural U16 H2 local-sweep profiler")
    parser.add_argument("stack", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("profiles/v23_4_u16_h2_sweep_structure.csv"))
    args = parser.parse_args()
    if args.slab_depth <= 0:
        parser.error("--slab-depth must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    stack = args.stack.resolve()
    binary = args.binary.resolve()
    if not stack.exists():
        parser.error(f"stack does not exist: {stack}")
    if not binary.exists():
        parser.error(f"binary does not exist: {binary}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="u16_h2_sweep_structure_") as raw:
        temp = Path(raw)
        for repeat in range(1, args.repeats + 1):
            row = run_once(binary, stack, args.slab_depth, args.foreground_connectivity, temp, repeat)
            rows.append(row)
            print(
                f"repeat={repeat} d={args.slab_depth}: sweep={row['local_sweep_seconds']:.3f}s "
                f"sample shares activation={100*float(row['activation_boundary_share']):.1f}% "
                f"neighborhood={100*float(row['neighborhood_pruning_share']):.1f}% "
                f"union/persistence/events={100*float(row['union_persistence_emit_share']):.1f}%"
            )

    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    numeric_fields = [key for key, value in rows[0].items() if key != "repeat" and isinstance(value, (int, float))]
    summary: dict[str, object] = {
        "repeats": len(rows),
        "slab_depth": args.slab_depth,
        "foreground_connectivity": args.foreground_connectivity,
    }
    for field in numeric_fields:
        if field in ("slab_depth", "foreground_connectivity"):
            continue
        summary[f"median_{field}"] = median([float(row[field]) for row in rows])

    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    shares = {
        "activation/boundary": float(summary["median_activation_boundary_share"]),
        "neighborhood/pruning": float(summary["median_neighborhood_pruning_share"]),
        "union/persistence/events": float(summary["median_union_persistence_emit_share"]),
    }
    dominant = max(shares, key=shares.get)
    print(f"dominant sampled region: {dominant} ({100*shares[dominant]:.1f}%)")
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
