#!/usr/bin/env python3
"""Balanced same-binary profiler for conventional versus root-carrying local unions."""

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
    r"PROFILE\s+(?P<name>\S+)\s+"
    r"prepare_seconds=(?P<prepare>[0-9.]+)\s+"
    r"reduce_seconds=(?P<reduce>[0-9.]+)\s+"
    r"cleanup_seconds=(?P<cleanup>[0-9.]+)\s+"
    r"total_seconds=(?P<total>[0-9.]+)"
)
PREP_RE = re.compile(
    r"PROFILE_PREP\s+(?P<name>\S+)\s+"
    r"decode_seconds=(?P<decode>[0-9.]+)\s+"
    r"key_conversion_seconds=(?P<key_conversion>[0-9.]+)\s+"
    r"slab_copy_seconds=(?P<slab_copy>[0-9.]+)\s+"
    r"scalar_order_seconds=(?P<scalar_order>[0-9.]+)\s+"
    r"local_sweep_seconds=(?P<local_sweep>[0-9.]+)\s+"
    r"local_run_write_seconds=(?P<local_run_write>[0-9.]+)\s+"
    r"cross_interface_seconds=(?P<cross_interface>[0-9.]+)\s+"
    r"unaccounted_seconds=(?P<unaccounted>[0-9.]+)\s+"
    r"total_voxels=(?P<total_voxels>[0-9]+)\s+"
    r"interior_fast_voxels=(?P<interior_fast_voxels>[0-9]+)"
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
    parser.add_argument(
        "--modes", nargs="+", choices=("h0-scalar-stream", "h2-scalar-stream"),
        default=("h0-scalar-stream", "h2-scalar-stream"),
    )
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    parser.add_argument("--output", type=Path, default=Path("scalar_union_kernel_profile.csv"))
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
        for name in (
            "decode", "key_conversion", "slab_copy", "scalar_order", "local_sweep",
            "local_run_write", "cross_interface", "unaccounted",
        ):
            result[f"{name}_seconds"] = float(prep.group(name))
        result["total_voxels"] = float(prep.group("total_voxels"))
        result["interior_fast_voxels"] = float(prep.group("interior_fast_voxels"))
    return result


def run_once(args: argparse.Namespace, kernel: str, mode: str, repeat: int, sequence: int,
             root: Path, record: bool) -> dict[str, object] | None:
    tag = "warmup" if not record else f"r{repeat:03d}"
    run_dir = root / f"{sequence:04d}_{mode}_{kernel}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    intervals = run_dir / "intervals.csv"
    time_path = run_dir / "time.txt"
    log_path = run_dir / "run.log"
    command = [
        "/usr/bin/time", "-v", "-o", str(time_path), str(args.binary.resolve()),
        str(args.input.resolve()), str(args.slab_depth), str(args.foreground_connectivity),
        mode, str(intervals),
        "--merge-strategy", "scan", "--interface-order", "radix", "--event-order", "verify",
        "--f32-key-mode", args.f32_key_mode, "--neighbor-kernel", "interior-fast",
        "--representative-active-check", "recheck", "--union-kernel", kernel,
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=run_dir, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    wall = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{mode} {kernel} failed; see {log_path}")
    if not intervals.is_file():
        raise RuntimeError(f"{mode} {kernel} produced no interval CSV")
    if not record:
        if not args.keep_logs:
            shutil.rmtree(run_dir, ignore_errors=True)
        return None
    digest, count = canonical_hash(intervals)
    row: dict[str, object] = {
        "mode": mode, "union_kernel": kernel, "repeat": repeat,
        "run_sequence": sequence, "slab_depth": args.slab_depth,
        "foreground_connectivity": args.foreground_connectivity,
        "wall_seconds": wall, "interval_rows": count, "canonical_sha256": digest,
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
    with tempfile.TemporaryDirectory(prefix="betti_union_kernel_profile_") as temp:
        root = Path(temp)
        for mode in args.modes:
            for _ in range(args.warmup):
                for kernel in ("conventional", "root-carrying"):
                    run_once(args, kernel, mode, -1, sequence, root, False)
                    sequence += 1
            for repeat in range(args.repeats):
                order = ("conventional", "root-carrying") if repeat % 2 == 0 else ("root-carrying", "conventional")
                for kernel in order:
                    row = run_once(args, kernel, mode, repeat, sequence, root, True)
                    sequence += 1
                    assert row is not None
                    rows.append(row)

    by_mode: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        by_mode[str(row["mode"])].add((str(row["canonical_sha256"]), int(row["interval_rows"])))
    bad = {mode: values for mode, values in by_mode.items() if len(values) != 1}
    if bad:
        raise RuntimeError(f"union-kernel outputs disagree: {bad}")

    fields = [
        "mode", "union_kernel", "repeat", "run_sequence", "slab_depth",
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

    print("\nMedian union-kernel ablation")
    print("mode\tkernel\twall\tprepare\tsweep\treduce\tRSS_MiB")
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mode"]), str(row["union_kernel"]))].append(row)
    for key in sorted(groups):
        group = groups[key]
        print(
            f"{key[0]}\t{key[1]}\t{median(group, 'wall_seconds'):.6f}\t"
            f"{median(group, 'prepare_seconds'):.6f}\t{median(group, 'local_sweep_seconds'):.6f}\t"
            f"{median(group, 'reduce_seconds'):.6f}\t{median(group, 'max_rss_kb') / 1024.0:.2f}"
        )
    for mode in args.modes:
        conventional = groups[(mode, "conventional")]
        root_carrying = groups[(mode, "root-carrying")]
        cw = median(conventional, "wall_seconds")
        rw = median(root_carrying, "wall_seconds")
        cs = median(conventional, "local_sweep_seconds")
        rs = median(root_carrying, "local_sweep_seconds")
        print(f"speedup {mode} wall: {cw / rw:.4f}x; local sweep: {cs / rs:.4f}x")
    print(f"PASS: exact persistence agreement across union kernels; wrote {args.output}")


if __name__ == "__main__":
    main()
