#!/usr/bin/env python3
"""Balanced same-binary profiler for generic versus interior-fast scalar neighbor kernels."""

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
    parser.add_argument("input", type=Path, help="scalar TIFF-stack directory")
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("h0-scalar-stream", "h2-scalar-stream"),
        default=("h0-scalar-stream", "h2-scalar-stream"),
    )
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    parser.add_argument("--output", type=Path, default=Path("scalar_neighbor_kernel_profile.csv"))
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


def parse_internal_profile(stdout: str) -> dict[str, float]:
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
            "decode",
            "key_conversion",
            "slab_copy",
            "scalar_order",
            "local_sweep",
            "local_run_write",
            "cross_interface",
            "unaccounted",
        ):
            result[f"{name}_seconds"] = float(prep.group(name))
        result["total_voxels"] = float(prep.group("total_voxels"))
        result["interior_fast_voxels"] = float(prep.group("interior_fast_voxels"))
    return result


def run_once(
    args: argparse.Namespace,
    kernel: str,
    mode: str,
    repeat: int,
    sequence: int,
    root: Path,
    record: bool,
) -> dict[str, object] | None:
    tag = "warmup" if not record else f"r{repeat:03d}"
    run_dir = root / f"{sequence:04d}_{mode}_{kernel}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    intervals = run_dir / "intervals.csv"
    time_path = run_dir / "time.txt"
    log_path = run_dir / "run.log"
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(time_path),
        str(args.binary.resolve()),
        str(args.input.resolve()),
        str(args.slab_depth),
        str(args.foreground_connectivity),
        mode,
        str(intervals),
        "--merge-strategy",
        "scan",
        "--interface-order",
        "radix",
        "--event-order",
        "verify",
        "--f32-key-mode",
        args.f32_key_mode,
        "--neighbor-kernel",
        kernel,
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=run_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
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

    digest, interval_rows = canonical_hash(intervals)
    row: dict[str, object] = {
        "mode": mode,
        "neighbor_kernel": kernel,
        "f32_key_mode": args.f32_key_mode,
        "repeat": repeat,
        "run_sequence": sequence,
        "slab_depth": args.slab_depth,
        "foreground_connectivity": args.foreground_connectivity,
        "wall_seconds": wall,
        "interval_rows": interval_rows,
        "canonical_sha256": digest,
        **parse_time_file(time_path),
        **parse_internal_profile(completed.stdout),
    }
    if not args.keep_logs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return row


def verify(rows: list[dict[str, object]]) -> None:
    by_mode: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        by_mode[str(row["mode"])].add(
            (str(row["canonical_sha256"]), int(row["interval_rows"]))
        )
    bad = {mode: values for mode, values in by_mode.items() if len(values) != 1}
    if bad:
        raise RuntimeError(f"neighbor-kernel persistence outputs disagree: {bad}")


def write_results(rows: list[dict[str, object]], output: Path) -> None:
    fields = [
        "mode",
        "neighbor_kernel",
        "f32_key_mode",
        "repeat",
        "run_sequence",
        "slab_depth",
        "foreground_connectivity",
        "wall_seconds",
        "user_seconds",
        "system_seconds",
        "max_rss_kb",
        "fs_inputs",
        "fs_outputs",
        "major_faults",
        "minor_faults",
        "prepare_seconds",
        "decode_seconds",
        "key_conversion_seconds",
        "slab_copy_seconds",
        "scalar_order_seconds",
        "local_sweep_seconds",
        "local_run_write_seconds",
        "cross_interface_seconds",
        "unaccounted_seconds",
        "total_voxels",
        "interior_fast_voxels",
        "reduce_seconds",
        "cleanup_seconds",
        "internal_total_seconds",
        "interval_rows",
        "canonical_sha256",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def median(group: list[dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in group if field in row]
    return statistics.median(values) if values else float("nan")


def print_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mode"]), str(row["neighbor_kernel"]))].append(row)

    print("\nMedian detailed preparation profile")
    print(
        "mode\tkernel\twall\tprepare\tdecode\tkey\tcopy\torder\tsweep\tlocal_write\tcross\treduce\tRSS_MiB"
    )
    walls: dict[tuple[str, str], float] = {}
    sweeps: dict[tuple[str, str], float] = {}
    for key in sorted(groups):
        group = groups[key]
        wall = median(group, "wall_seconds")
        sweep = median(group, "local_sweep_seconds")
        walls[key] = wall
        sweeps[key] = sweep
        print(
            f"{key[0]}\t{key[1]}\t{wall:.3f}\t{median(group, 'prepare_seconds'):.3f}"
            f"\t{median(group, 'decode_seconds'):.3f}\t{median(group, 'key_conversion_seconds'):.3f}"
            f"\t{median(group, 'slab_copy_seconds'):.3f}\t{median(group, 'scalar_order_seconds'):.3f}"
            f"\t{sweep:.3f}\t{median(group, 'local_run_write_seconds'):.3f}"
            f"\t{median(group, 'cross_interface_seconds'):.3f}\t{median(group, 'reduce_seconds'):.3f}"
            f"\t{median(group, 'max_rss_kb') / 1024.0:.1f}"
        )
        if key[1] == "interior-fast":
            total = median(group, "total_voxels")
            fast_voxels = median(group, "interior_fast_voxels")
            if total > 0:
                print(f"  fast-path voxel coverage: {100.0 * fast_voxels / total:.2f}%")

    for mode in sorted({mode for mode, _ in groups}):
        generic = walls.get((mode, "generic"))
        fast = walls.get((mode, "interior-fast"))
        generic_sweep = sweeps.get((mode, "generic"))
        fast_sweep = sweeps.get((mode, "interior-fast"))
        if generic is not None and fast is not None:
            print(f"wall speedup {mode} interior-fast vs generic: {generic / fast:.3f}x")
        if generic_sweep is not None and fast_sweep is not None:
            print(
                f"local-sweep speedup {mode} interior-fast vs generic: "
                f"{generic_sweep / fast_sweep:.3f}x"
            )


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats must be >= 1 and --warmup must be >= 0")
    if not args.input.is_dir():
        raise SystemExit(f"input stack not found: {args.input}")
    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    if not Path("/usr/bin/time").is_file():
        raise SystemExit("/usr/bin/time is required")

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="betti_neighbor_profile_") as temp:
        root = Path(temp)
        if args.keep_logs:
            root = args.output.with_suffix("").parent / f"{args.output.stem}_logs"
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)

        sequence = 0
        for mode in args.modes:
            for warmup in range(args.warmup):
                for kernel in ("generic", "interior-fast"):
                    sequence += 1
                    run_once(args, kernel, mode, -warmup - 1, sequence, root, record=False)

            for repeat in range(1, args.repeats + 1):
                order = (
                    ("generic", "interior-fast")
                    if repeat % 2
                    else ("interior-fast", "generic")
                )
                for kernel in order:
                    sequence += 1
                    print(f"run {sequence}: {mode} {kernel} repeat {repeat}", flush=True)
                    row = run_once(args, kernel, mode, repeat, sequence, root, record=True)
                    assert row is not None
                    rows.append(row)

    verify(rows)
    write_results(rows, args.output)
    print_summary(rows)
    print(f"\nPASS: all generic/interior-fast outputs agree; wrote {args.output}")


if __name__ == "__main__":
    main()
