#!/usr/bin/env python3
"""Balanced same-binary profiler for legacy64 versus native32 F32 stream keys."""

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

H0_STORAGE_RE = re.compile(
    r"PROFILE_H0_STORAGE\s+scalar_h0_stream\s+"
    r"pipeline=(?P<pipeline>\S+)\s+"
    r"disk_key_bytes=(?P<disk_key_bytes>\d+)\s+"
    r"interface_nodes=(?P<interface_nodes>\d+)\s+"
    r"interface_birth_bytes=(?P<interface_birth_bytes>\d+)\s+"
    r"local_pair_bytes=(?P<local_pair_bytes>\d+)\s+"
    r"attach_bytes=(?P<attach_bytes>\d+)\s+"
    r"interface_bytes=(?P<interface_bytes>\d+)\s+"
    r"cross_bytes=(?P<cross_bytes>\d+)\s+"
    r"total_run_bytes=(?P<total_run_bytes>\d+)\s+"
    r"estimated_global_parent_bytes=(?P<global_parent_bytes>\d+)\s+"
    r"estimated_global_rank_bytes=(?P<global_rank_bytes>\d+)\s+"
    r"estimated_global_birth_bytes=(?P<global_birth_bytes>\d+)\s+"
    r"estimated_global_total_bytes=(?P<global_total_bytes>\d+)"
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
    parser.add_argument("input", type=Path, help="F32 TIFF-stack directory")
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--slice-limit", type=int, default=0, help="stage only the first N TIFF slices; 0 uses the full stack")
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("h0-scalar-stream", "h2-scalar-stream"),
        default=("h0-scalar-stream", "h2-scalar-stream"),
    )
    parser.add_argument("--output", type=Path, default=Path("scalar_f32_key_profile.csv"))
    parser.add_argument("--keep-logs", action="store_true")
    return parser.parse_args()



def stage_input_subset(input_path: Path, root: Path, limit: int) -> Path:
    if limit == 0:
        return input_path
    if limit < 0:
        raise SystemExit("--slice-limit must be nonnegative")
    slices = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if len(slices) < limit:
        raise SystemExit(f"requested {limit} slices but found only {len(slices)} TIFF files")
    staged = root / "input_subset"
    staged.mkdir(parents=True, exist_ok=True)
    for path in slices[:limit]:
        (staged / path.name).symlink_to(path.resolve())
    return staged

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


def run_once(
    args: argparse.Namespace,
    key_mode: str,
    mode: str,
    repeat: int,
    sequence: int,
    root: Path,
    record: bool,
) -> dict[str, object] | None:
    tag = "warmup" if not record else f"r{repeat:03d}"
    run_dir = root / f"{sequence:04d}_{mode}_{key_mode}_{tag}"
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
        key_mode,
    ]
    if mode == "h0-scalar-stream":
        command.extend(["--global-h0-uf-layout", "parent-rank"])

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
        raise RuntimeError(f"{mode} {key_mode} failed; see {log_path}")
    if not intervals.is_file():
        raise RuntimeError(f"{mode} {key_mode} produced no interval CSV")

    if not record:
        if not args.keep_logs:
            shutil.rmtree(run_dir, ignore_errors=True)
        return None

    digest, interval_rows = canonical_hash(intervals)
    match = PROFILE_RE.search(completed.stdout)
    profile: dict[str, float] = {}
    if match:
        profile = {
            "prepare_seconds": float(match.group("prepare")),
            "reduce_seconds": float(match.group("reduce")),
            "cleanup_seconds": float(match.group("cleanup")),
            "internal_total_seconds": float(match.group("total")),
        }
    storage: dict[str, object] = {}
    storage_match = H0_STORAGE_RE.search(completed.stdout)
    if storage_match:
        storage = {
            "h0_storage_pipeline": storage_match.group("pipeline"),
            **{name: int(value) for name, value in storage_match.groupdict().items() if name != "pipeline"},
        }
    row: dict[str, object] = {
        "mode": mode,
        "f32_key_mode": key_mode,
        "repeat": repeat,
        "run_sequence": sequence,
        "slab_depth": args.slab_depth,
        "foreground_connectivity": args.foreground_connectivity,
        "wall_seconds": wall,
        "interval_rows": interval_rows,
        "canonical_sha256": digest,
        **parse_time_file(time_path),
        **profile,
        **storage,
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
        raise RuntimeError(f"native/legacy persistence outputs disagree: {bad}")


def write_results(rows: list[dict[str, object]], output: Path) -> None:
    fields = [
        "mode",
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
        "reduce_seconds",
        "cleanup_seconds",
        "internal_total_seconds",
        "h0_storage_pipeline",
        "disk_key_bytes",
        "interface_nodes",
        "interface_birth_bytes",
        "local_pair_bytes",
        "attach_bytes",
        "interface_bytes",
        "cross_bytes",
        "total_run_bytes",
        "global_parent_bytes",
        "global_rank_bytes",
        "global_birth_bytes",
        "global_total_bytes",
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
        groups[(str(row["mode"]), str(row["f32_key_mode"]))].append(row)

    print("\nMedian F32-key profile summary")
    print("mode\tkey_mode\twall_s\tprepare_s\treduce_s\tRSS_MiB\ttemp_GiB\tglobal_GiB")
    walls: dict[tuple[str, str], float] = {}
    for key in sorted(groups):
        group = groups[key]
        wall = median(group, "wall_seconds")
        walls[key] = wall
        prep = median(group, "prepare_seconds")
        reduce = median(group, "reduce_seconds")
        rss = median(group, "max_rss_kb") / 1024.0
        temp_gib = median(group, "total_run_bytes") / 2**30 if any("total_run_bytes" in row for row in group) else float("nan")
        global_gib = median(group, "global_total_bytes") / 2**30 if any("global_total_bytes" in row for row in group) else float("nan")
        print(f"{key[0]}\t{key[1]}\t{wall:.3f}\t{prep:.3f}\t{reduce:.3f}\t{rss:.1f}\t{temp_gib:.3f}\t{global_gib:.3f}")

    for mode in sorted({mode for mode, _ in groups}):
        legacy = walls.get((mode, "legacy64"))
        native = walls.get((mode, "native32"))
        if legacy is not None and native is not None:
            print(f"speedup {mode} native32 vs legacy64: {legacy / native:.3f}x")


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
    with tempfile.TemporaryDirectory(prefix="betti_f32_profile_") as temp:
        temp_root = Path(temp)
        args.input = stage_input_subset(args.input, temp_root, args.slice_limit)
        root = temp_root
        if args.keep_logs:
            root = args.output.with_suffix("").parent / f"{args.output.stem}_logs"
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)

        sequence = 0
        for mode in args.modes:
            for warmup in range(args.warmup):
                for key_mode in ("legacy64", "native32"):
                    sequence += 1
                    run_once(args, key_mode, mode, -warmup - 1, sequence, root, record=False)

            for repeat in range(1, args.repeats + 1):
                order = ("legacy64", "native32") if repeat % 2 else ("native32", "legacy64")
                for key_mode in order:
                    sequence += 1
                    print(f"run {sequence}: {mode} {key_mode} repeat {repeat}", flush=True)
                    row = run_once(args, key_mode, mode, repeat, sequence, root, record=True)
                    assert row is not None
                    rows.append(row)

    verify(rows)
    write_results(rows, args.output)
    print_summary(rows)
    print(f"\nPASS: all native32/legacy64 outputs agree; wrote {args.output}")


if __name__ == "__main__":
    main()
