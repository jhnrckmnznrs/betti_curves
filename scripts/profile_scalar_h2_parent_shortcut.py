#!/usr/bin/env python3
"""Balanced profiler for H2 root-carrying find versus direct-parent shortcut."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

PROFILE_RE = re.compile(
    r"PROFILE\s+scalar_h2_stream\s+prepare_seconds=(?P<prepare>[0-9.]+)\s+"
    r"reduce_seconds=(?P<reduce>[0-9.]+)\s+cleanup_seconds=(?P<cleanup>[0-9.]+)\s+"
    r"total_seconds=(?P<total>[0-9.]+)"
)
PREP_RE = re.compile(
    r"PROFILE_PREP\s+scalar_h2_stream\s+"
    r"decode_seconds=(?P<decode>[0-9.]+)\s+key_conversion_seconds=(?P<key_conversion>[0-9.]+)\s+"
    r"slab_copy_seconds=(?P<slab_copy>[0-9.]+)\s+scalar_order_seconds=(?P<scalar_order>[0-9.]+)\s+"
    r"local_sweep_seconds=(?P<local_sweep>[0-9.]+)\s+local_run_write_seconds=(?P<local_run_write>[0-9.]+)\s+"
    r"cross_interface_seconds=(?P<cross_interface>[0-9.]+)\s+unaccounted_seconds=(?P<unaccounted>[0-9.]+)"
)
TIME_FIELDS = {
    "User time (seconds)": "user_seconds",
    "System time (seconds)": "system_seconds",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "File system inputs": "fs_inputs",
    "File system outputs": "fs_outputs",
    "Major (requiring I/O) page faults": "major_faults",
    "Minor (reclaiming a frame) page faults": "minor_faults",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    parser.add_argument("--output", type=Path, default=Path("h2_parent_shortcut_profile.csv"))
    parser.add_argument("--keep-logs", action="store_true")
    return parser.parse_args()


def canonical_hash(path: Path) -> tuple[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence header in {path}: {header!r}")
        rows = sorted(tuple(row) for row in reader if row)
    digest = hashlib.sha256()
    for birth, death in rows:
        digest.update(f"{birth},{death}\n".encode())
    return digest.hexdigest(), len(rows)


def parse_time_file(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        field = TIME_FIELDS.get(key)
        if field is None:
            continue
        try:
            result[field] = float(value.strip())
        except ValueError:
            pass
    return result


def parse_internal(stdout: str) -> dict[str, float]:
    result: dict[str, float] = {}
    match = PROFILE_RE.search(stdout)
    if match:
        result.update(
            prepare_seconds=float(match.group("prepare")),
            reduce_seconds=float(match.group("reduce")),
            cleanup_seconds=float(match.group("cleanup")),
            internal_total_seconds=float(match.group("total")),
        )
    prep = PREP_RE.search(stdout)
    if prep:
        for name in ("decode", "key_conversion", "slab_copy", "scalar_order", "local_sweep",
                     "local_run_write", "cross_interface", "unaccounted"):
            result[f"{name}_seconds"] = float(prep.group(name))
    return result


def run_once(args: argparse.Namespace, root_check: str, repeat: int, sequence: int,
             root: Path, record: bool) -> dict[str, object] | None:
    tag = "warmup" if not record else f"r{repeat:03d}"
    run_dir = root / f"{sequence:04d}_{root_check}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    intervals = run_dir / "intervals.csv"
    time_path = run_dir / "time.txt"
    log_path = run_dir / "run.log"
    command = [
        "/usr/bin/time", "-v", "-o", str(time_path), str(args.binary.resolve()),
        str(args.input.resolve()), str(args.slab_depth), str(args.foreground_connectivity),
        "h2-scalar-stream", str(intervals),
        "--merge-strategy", "scan", "--interface-order", "radix", "--event-order", "verify",
        "--f32-key-mode", args.f32_key_mode, "--neighbor-kernel", "interior-fast",
        "--representative-active-check", "recheck", "--union-kernel", "root-carrying",
        "--neighbor-root-check", root_check,
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=run_dir, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    wall = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"h2 {root_check} failed; see {log_path}")
    if not intervals.is_file():
        raise RuntimeError(f"h2 {root_check} produced no interval CSV")
    if not record:
        if not args.keep_logs:
            shutil.rmtree(run_dir, ignore_errors=True)
        return None
    digest, count = canonical_hash(intervals)
    row: dict[str, object] = {
        "mode": "h2-scalar-stream", "neighbor_root_check": root_check,
        "repeat": repeat, "run_sequence": sequence, "slab_depth": args.slab_depth,
        "foreground_connectivity": args.foreground_connectivity, "wall_seconds": wall,
        "interval_rows": count, "canonical_sha256": digest,
        **parse_time_file(time_path), **parse_internal(completed.stdout),
    }
    if not args.keep_logs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return row


def median(rows: list[dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return statistics.median(values) if values else float("nan")


def main() -> None:
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    rows: list[dict[str, object]] = []
    sequence = 0
    with tempfile.TemporaryDirectory(prefix="betti_h2_parent_shortcut_profile_") as temp:
        root = Path(temp)
        for _ in range(args.warmup):
            for root_check in ("find", "parent-shortcut"):
                run_once(args, root_check, -1, sequence, root, False)
                sequence += 1
        for repeat in range(args.repeats):
            order = ("find", "parent-shortcut") if repeat % 2 == 0 else ("parent-shortcut", "find")
            for root_check in order:
                row = run_once(args, root_check, repeat, sequence, root, True)
                sequence += 1
                assert row is not None
                rows.append(row)

    outputs = {(str(row["canonical_sha256"]), int(row["interval_rows"])) for row in rows}
    if len(outputs) != 1:
        raise RuntimeError(f"H2 parent-shortcut outputs disagree: {outputs}")

    fields = [
        "mode", "neighbor_root_check", "repeat", "run_sequence", "slab_depth",
        "foreground_connectivity", "wall_seconds", "user_seconds", "system_seconds", "max_rss_kb",
        "prepare_seconds", "local_sweep_seconds", "scalar_order_seconds", "local_run_write_seconds",
        "cross_interface_seconds", "reduce_seconds", "interval_rows", "canonical_sha256",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["neighbor_root_check"])].append(row)
    print("\nMedian H2 neighbor-root-check ablation")
    print("root_check\twall\tprepare\tsweep\treduce\tRSS_MiB")
    for root_check in ("find", "parent-shortcut"):
        group = groups[root_check]
        print(
            f"{root_check}\t{median(group, 'wall_seconds'):.6f}\t"
            f"{median(group, 'prepare_seconds'):.6f}\t{median(group, 'local_sweep_seconds'):.6f}\t"
            f"{median(group, 'reduce_seconds'):.6f}\t{median(group, 'max_rss_kb') / 1024.0:.2f}"
        )
    find_rows = groups["find"]
    shortcut_rows = groups["parent-shortcut"]
    print(
        f"speedup H2 wall: {median(find_rows, 'wall_seconds') / median(shortcut_rows, 'wall_seconds'):.4f}x; "
        f"local sweep: {median(find_rows, 'local_sweep_seconds') / median(shortcut_rows, 'local_sweep_seconds'):.4f}x"
    )
    print(f"PASS: exact H2 persistence agreement; wrote {args.output}")


if __name__ == "__main__":
    main()
