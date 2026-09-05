#!/usr/bin/env python3
"""Balanced A/B profiler for disk-backed exact scalar persistence.

Runs baseline and candidate binaries on the same TIFF stack with alternating
execution order, captures wall/resource/internal phase timings, and requires
canonical persistence multisets to agree before reporting speedups.
"""

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
    parser.add_argument("input", type=Path, help="directory containing one TIFF stack")
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("target/release/betti_curves"),
        help="optimized executable",
    )
    parser.add_argument("--baseline", type=Path, help="pre-optimization executable")
    parser.add_argument("--slab-depth", type=int, default=32)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="unreported warm-up rounds per mode (default: 1)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("h0-scalar-stream", "h2-scalar-stream"),
        default=("h0-scalar-stream", "h2-scalar-stream"),
    )
    parser.add_argument("--output", type=Path, default=Path("scalar_stream_profile.csv"))
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="copy stdout/stderr logs beside the result CSV",
    )
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
        digest.update(birth.encode())
        digest.update(b",")
        digest.update(death.encode())
        digest.update(b"\n")
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
    label: str,
    binary: Path,
    args: argparse.Namespace,
    mode: str,
    repeat: int,
    sequence: int,
    root: Path,
    record: bool,
) -> dict[str, object] | None:
    tag = "warmup" if not record else f"r{repeat:03d}"
    run_dir = root / f"{sequence:04d}_{label}_{mode}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    persistence_path = run_dir / "intervals.csv"
    time_path = run_dir / "time.txt"
    log_path = run_dir / "run.log"
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(time_path),
        str(binary.resolve()),
        str(args.input.resolve()),
        str(args.slab_depth),
        str(args.foreground_connectivity),
        mode,
        str(persistence_path),
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
    wall_seconds = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} {mode} repeat {repeat} failed with code {completed.returncode}; "
            f"see {log_path}"
        )
    if not persistence_path.is_file():
        raise RuntimeError(f"{label} {mode} produced no persistence CSV")

    if not record:
        if not args.keep_logs:
            shutil.rmtree(run_dir, ignore_errors=True)
        return None

    digest, interval_rows = canonical_hash(persistence_path)
    resource_stats = parse_time_file(time_path)
    profile_match = PROFILE_RE.search(completed.stdout)
    profile = {}
    if profile_match:
        profile = {
            "prepare_seconds": float(profile_match.group("prepare")),
            "reduce_seconds": float(profile_match.group("reduce")),
            "cleanup_seconds": float(profile_match.group("cleanup")),
            "internal_total_seconds": float(profile_match.group("total")),
        }

    row: dict[str, object] = {
        "label": label,
        "binary": str(binary.resolve()),
        "mode": mode,
        "repeat": repeat,
        "run_sequence": sequence,
        "slab_depth": args.slab_depth,
        "foreground_connectivity": args.foreground_connectivity,
        "wall_seconds": wall_seconds,
        "interval_rows": interval_rows,
        "canonical_sha256": digest,
        **resource_stats,
        **profile,
    }
    if not args.keep_logs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return row


def verify_hashes(rows: list[dict[str, object]]) -> None:
    by_mode: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_mode[str(row["mode"])].add(str(row["canonical_sha256"]))
    failures = {mode: hashes for mode, hashes in by_mode.items() if len(hashes) != 1}
    if failures:
        details = "; ".join(f"{mode}: {sorted(hashes)}" for mode, hashes in failures.items())
        raise RuntimeError(f"persistence outputs disagree: {details}")


def write_results(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "binary",
        "mode",
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
        "interval_rows",
        "canonical_sha256",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def print_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["label"]), str(row["mode"]))].append(row)

    print("\nMedian profile summary")
    print("label\tmode\twall_s\tmax_rss_GiB\tprepare_s\treduce_s")
    medians: dict[tuple[str, str], float] = {}
    for key in sorted(groups):
        group = groups[key]
        wall = statistics.median(float(row["wall_seconds"]) for row in group)
        medians[key] = wall
        rss_values = [float(row["max_rss_kb"]) for row in group if "max_rss_kb" in row]
        prep_values = [float(row["prepare_seconds"]) for row in group if "prepare_seconds" in row]
        reduce_values = [float(row["reduce_seconds"]) for row in group if "reduce_seconds" in row]
        rss = statistics.median(rss_values) / (1024.0 * 1024.0) if rss_values else float("nan")
        prep = statistics.median(prep_values) if prep_values else float("nan")
        reduce = statistics.median(reduce_values) if reduce_values else float("nan")
        print(f"{key[0]}\t{key[1]}\t{wall:.3f}\t{rss:.3f}\t{prep:.3f}\t{reduce:.3f}")

    for mode in sorted({mode for _, mode in medians}):
        baseline = medians.get(("baseline", mode))
        candidate = medians.get(("candidate", mode))
        if baseline is not None and candidate is not None:
            print(f"speedup {mode}: {baseline / candidate:.3f}x")


def balanced_order(binaries: list[tuple[str, Path]], repeat: int) -> list[tuple[str, Path]]:
    if len(binaries) < 2 or repeat % 2 == 1:
        return binaries
    return list(reversed(binaries))


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats must be >= 1 and --warmup must be >= 0")
    if not args.input.is_dir():
        raise SystemExit(f"input TIFF stack directory does not exist: {args.input}")
    if not Path("/usr/bin/time").is_file():
        raise SystemExit("/usr/bin/time is required for peak-RSS profiling")

    binaries = [("candidate", args.candidate)]
    if args.baseline is not None:
        binaries.insert(0, ("baseline", args.baseline))
    for label, binary in binaries:
        if not binary.is_file():
            raise SystemExit(f"{label} binary does not exist: {binary}")

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="betti_profile_") as temp:
        root = Path(temp)
        if args.keep_logs:
            root = args.output.with_suffix("").parent / f"{args.output.stem}_logs"
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)
        root = root.resolve()
        sequence = 0

        for mode in args.modes:
            for warmup in range(1, args.warmup + 1):
                for label, binary in balanced_order(binaries, warmup):
                    sequence += 1
                    print(f"warmup {label} {mode} {warmup}/{args.warmup}", flush=True)
                    run_once(label, binary, args, mode, -warmup, sequence, root, record=False)

            for repeat in range(1, args.repeats + 1):
                for label, binary in balanced_order(binaries, repeat):
                    sequence += 1
                    print(f"profile {label} {mode} {repeat}/{args.repeats}", flush=True)
                    row = run_once(label, binary, args, mode, repeat, sequence, root, record=True)
                    assert row is not None
                    rows.append(row)

    verify_hashes(rows)
    write_results(rows, args.output)
    print_summary(rows)
    print("\nPASS: all repeated baseline/candidate persistence multisets agree")
    print(f"Wrote detailed profile to {args.output}")


if __name__ == "__main__":
    main()
