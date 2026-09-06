#!/usr/bin/env python3
"""Profile native-U16 hierarchical H0 persistence across leaf worker counts."""
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
PAR_RE = re.compile(
    r"PROFILE_U16_PERSIST_H0_PARALLEL leaf_workers=(\d+) "
    r"leaf_batch_seconds=([0-9.]+) leaf_profile_work_seconds=([0-9.]+) "
    r"buffered_pairs_peak=(\d+)"
)
LEAF_RE = re.compile(
    r"PROFILE_U16_PERSIST_LEAF dimension=h0 key_bytes=(\d+) "
    r"zero_persistence_pairs_elided=(\d+) union_attempts=(\d+) "
    r"successful_unions=(\d+) scalar_order_seconds=([0-9.]+) "
    r"local_sweep_seconds=([0-9.]+)"
)


def command(binary: Path, stack: Path, depth: int, conn: int, out: Path) -> list[str]:
    return [
        str(binary), str(stack), str(depth), str(conn),
        "h0-scalar-hierarchical-stream", str(out),
        "--h0-birth-buffer", "reuse-input",
        "--h0-event-storage", "direct",
        "--global-h0-uf-layout", "packed",
        "--h0-hier-attach-pruning", "elder-dominated",
    ]


def run_once(cmd: list[str], timing_path: Path, workers: int) -> dict[str, float | int]:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H0_LEAF_WORKERS"] = str(workers)
    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(timing_path), *cmd],
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
    timing = timing_path.read_text()
    rss = RSS_RE.search(timing)
    user = USER_RE.search(timing)
    system = SYS_RE.search(timing)
    par = PAR_RE.search(proc.stdout)
    leaf = LEAF_RE.search(proc.stdout)
    if not all((rss, user, system, par, leaf)):
        raise RuntimeError(f"missing profiling fields\nstdout:\n{proc.stdout}\ntime:\n{timing}")
    return {
        "wall_seconds": wall,
        "max_rss_mib": int(rss.group(1)) / 1024.0,
        "user_seconds": float(user.group(1)),
        "system_seconds": float(system.group(1)),
        "reported_leaf_workers": int(par.group(1)),
        "leaf_batch_seconds": float(par.group(2)),
        "leaf_profile_work_seconds": float(par.group(3)),
        "buffered_pairs_peak": int(par.group(4)),
        "zero_persistence_pairs_elided": int(leaf.group(2)),
        "union_attempts": int(leaf.group(3)),
        "successful_unions": int(leaf.group(4)),
        "scalar_order_seconds": float(leaf.group(5)),
        "local_sweep_seconds": float(leaf.group(6)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--foreground-connectivity", type=int, default=26)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", type=Path, default=Path("profiles/v21_u16_h0_parallel_leaves.csv"))
    args = ap.parse_args()
    if any(w < 1 for w in args.workers):
        raise SystemExit("all --workers values must be positive")

    stack = args.stack.resolve()
    binary = args.binary.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="u16_h0_parallel_prof_") as td_raw:
        td = Path(td_raw)
        seq = 0
        for depth in args.slab_depths:
            for workers in args.workers:
                for _ in range(args.warmup):
                    seq += 1
                    run_once(
                        command(binary, stack, depth, args.foreground_connectivity, td / f"{seq}.csv"),
                        td / f"{seq}.time", workers,
                    )
                for rep in range(1, args.repeats + 1):
                    seq += 1
                    result = run_once(
                        command(binary, stack, depth, args.foreground_connectivity, td / f"{seq}.csv"),
                        td / f"{seq}.time", workers,
                    )
                    rows.append({
                        "slab_depth": depth,
                        "workers": workers,
                        "repeat": rep,
                        **result,
                    })
                    print(
                        f"d={depth} workers={workers} r={rep}: "
                        f"{result['wall_seconds']:.3f}s {result['max_rss_mib']:.1f} MiB "
                        f"leaf_batch={result['leaf_batch_seconds']:.3f}s"
                    )

    fields = list(rows[0])
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = args.output.with_name(args.output.stem + "_summary.csv")
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((int(row["slab_depth"]), int(row["workers"])), []).append(row)
    summary_fields = [
        "slab_depth", "workers", "repeats", "median_wall_seconds", "median_max_rss_mib",
        "median_user_seconds", "median_system_seconds", "median_leaf_batch_seconds",
        "median_leaf_profile_work_seconds", "median_buffered_pairs_peak",
        "median_zero_persistence_pairs_elided", "median_union_attempts",
        "median_successful_unions", "median_scalar_order_seconds", "median_local_sweep_seconds",
    ]
    with summary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for (depth, workers), rs in sorted(groups.items()):
            def med(name: str) -> float:
                return float(statistics.median(float(r[name]) for r in rs))
            writer.writerow({
                "slab_depth": depth,
                "workers": workers,
                "repeats": len(rs),
                "median_wall_seconds": med("wall_seconds"),
                "median_max_rss_mib": med("max_rss_mib"),
                "median_user_seconds": med("user_seconds"),
                "median_system_seconds": med("system_seconds"),
                "median_leaf_batch_seconds": med("leaf_batch_seconds"),
                "median_leaf_profile_work_seconds": med("leaf_profile_work_seconds"),
                "median_buffered_pairs_peak": med("buffered_pairs_peak"),
                "median_zero_persistence_pairs_elided": med("zero_persistence_pairs_elided"),
                "median_union_attempts": med("union_attempts"),
                "median_successful_unions": med("successful_unions"),
                "median_scalar_order_seconds": med("scalar_order_seconds"),
                "median_local_sweep_seconds": med("local_sweep_seconds"),
            })
    print(f"wrote {args.output} and {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
