#!/usr/bin/env python3
"""Full-factorial profiler for scalar persistence stream optimizations.

The same release binary is run under all requested combinations of:
  merge strategy:    scan | heap
  interface ordering: comparison | radix
  event ordering:    resort | verify

F32 key storage is held fixed per invocation (`native32` by default) so the
original 2x2x2 ablation remains directly interpretable.

Execution order is rotated/reversed between repeats to reduce cache/order bias.
Every measured output is canonicalized as an interval multiset; the script
fails if any configuration changes the persistence result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Config:
    merge: str
    interface: str
    event: str

    @property
    def label(self) -> str:
        if (self.merge, self.interface, self.event) == ("scan", "comparison", "resort"):
            return "legacy"
        if (self.merge, self.interface, self.event) == ("scan", "radix", "verify"):
            return "production-base"
        if (self.merge, self.interface, self.event) == ("heap", "radix", "verify"):
            return "v1-heap"
        return f"m-{self.merge}_i-{self.interface}_e-{self.event}"

    @property
    def cli(self) -> list[str]:
        return [
            "--merge-strategy",
            self.merge,
            "--interface-order",
            self.interface,
            "--event-order",
            self.event,
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("h0-scalar-stream", "h2-scalar-stream"),
        default=("h0-scalar-stream", "h2-scalar-stream"),
    )
    parser.add_argument("--merge-strategies", nargs="+", choices=("scan", "heap"), default=("scan", "heap"))
    parser.add_argument(
        "--interface-orders",
        nargs="+",
        choices=("comparison", "radix"),
        default=("comparison", "radix"),
    )
    parser.add_argument("--event-orders", nargs="+", choices=("resort", "verify"), default=("resort", "verify"))
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    parser.add_argument("--output", type=Path, default=Path("scalar_stream_ablation.csv"))
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
    args: argparse.Namespace,
    config: Config,
    mode: str,
    repeat: int,
    sequence: int,
    root: Path,
    record: bool,
) -> dict[str, object] | None:
    tag = "warmup" if not record else f"r{repeat:03d}"
    run_dir = root / f"{sequence:04d}_{mode}_{config.label}_{tag}"
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
        *config.cli,
        "--f32-key-mode",
        args.f32_key_mode,
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
        raise RuntimeError(
            f"{mode} {config.label} failed with code {completed.returncode}; see {log_path}"
        )
    if not intervals.is_file():
        raise RuntimeError(f"{mode} {config.label} produced no interval CSV")

    if not record:
        if not args.keep_logs:
            shutil.rmtree(run_dir, ignore_errors=True)
        return None

    digest, interval_rows = canonical_hash(intervals)
    profile_match = PROFILE_RE.search(completed.stdout)
    profile: dict[str, float] = {}
    if profile_match:
        profile = {
            "prepare_seconds": float(profile_match.group("prepare")),
            "reduce_seconds": float(profile_match.group("reduce")),
            "cleanup_seconds": float(profile_match.group("cleanup")),
            "internal_total_seconds": float(profile_match.group("total")),
        }

    row: dict[str, object] = {
        "mode": mode,
        "config": config.label,
        "merge_strategy": config.merge,
        "interface_order": config.interface,
        "event_order": config.event,
        "f32_key_mode": args.f32_key_mode,
        "repeat": repeat,
        "run_sequence": sequence,
        "slab_depth": args.slab_depth,
        "foreground_connectivity": args.foreground_connectivity,
        "wall_seconds": wall,
        "interval_rows": interval_rows,
        "canonical_sha256": digest,
        **parse_time_file(time_path),
        **profile,
    }
    if not args.keep_logs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return row


def balanced_configs(configs: list[Config], repeat: int) -> list[Config]:
    if not configs:
        return []
    shift = (repeat - 1) % len(configs)
    order = configs[shift:] + configs[:shift]
    if repeat % 2 == 0:
        order.reverse()
    return order


def verify_hashes(rows: list[dict[str, object]]) -> None:
    by_mode: dict[str, set[str]] = defaultdict(set)
    by_mode_counts: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        mode = str(row["mode"])
        by_mode[mode].add(str(row["canonical_sha256"]))
        by_mode_counts[mode].add(int(row["interval_rows"]))
    failures = []
    for mode in sorted(by_mode):
        if len(by_mode[mode]) != 1 or len(by_mode_counts[mode]) != 1:
            failures.append(
                f"{mode}: hashes={sorted(by_mode[mode])}, counts={sorted(by_mode_counts[mode])}"
            )
    if failures:
        raise RuntimeError("ablation configurations disagree: " + "; ".join(failures))


def write_results(rows: list[dict[str, object]], output: Path) -> None:
    fields = [
        "mode",
        "config",
        "merge_strategy",
        "interface_order",
        "event_order",
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
        "interval_rows",
        "canonical_sha256",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def med(group: list[dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in group if field in row]
    return statistics.median(values) if values else float("nan")


def print_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    config_meta: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        groups[(str(row["mode"]), str(row["config"]))].append(row)
        config_meta[str(row["config"])] = (
            str(row["merge_strategy"]),
            str(row["interface_order"]),
            str(row["event_order"]),
        )

    for mode in sorted({mode for mode, _ in groups}):
        print(f"\n{mode} median ablation summary")
        print("config\tmerge\tinterface\tevent\twall_s\tprepare_s\treduce_s\tRSS_MiB")
        ranked = []
        for (group_mode, label), group in groups.items():
            if group_mode != mode:
                continue
            wall = med(group, "wall_seconds")
            prep = med(group, "prepare_seconds")
            reduce = med(group, "reduce_seconds")
            rss = med(group, "max_rss_kb") / 1024.0
            merge, interface, event = config_meta[label]
            ranked.append((wall, label, merge, interface, event, prep, reduce, rss))
        ranked.sort()
        for wall, label, merge, interface, event, prep, reduce, rss in ranked:
            print(
                f"{label}\t{merge}\t{interface}\t{event}\t"
                f"{wall:.3f}\t{prep:.3f}\t{reduce:.3f}\t{rss:.1f}"
            )
        if ranked:
            print(f"best {mode}: {ranked[0][1]} ({ranked[0][0]:.3f} s)")

        print("factor pooled medians (diagnostic, not a substitute for cell comparisons)")
        for factor, levels in (
            ("merge_strategy", ("scan", "heap")),
            ("interface_order", ("comparison", "radix")),
            ("event_order", ("resort", "verify")),
        ):
            values = {}
            for level in levels:
                selected = [
                    row
                    for row in rows
                    if row["mode"] == mode and row[factor] == level
                ]
                if selected:
                    values[level] = statistics.median(float(row["wall_seconds"]) for row in selected)
            if len(values) == 2:
                a, b = levels
                print(f"  {factor}: {a}={values[a]:.3f}s, {b}={values[b]:.3f}s")


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats must be >= 1 and --warmup must be >= 0")
    if not args.input.is_dir():
        raise SystemExit(f"input TIFF stack directory does not exist: {args.input}")
    if not args.binary.is_file():
        raise SystemExit(f"binary does not exist: {args.binary}")
    if not Path("/usr/bin/time").is_file():
        raise SystemExit("/usr/bin/time is required")

    configs = [
        Config(merge, interface, event)
        for merge, interface, event in itertools.product(
            args.merge_strategies, args.interface_orders, args.event_orders
        )
    ]
    default_config = Config("scan", "radix", "verify")

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="betti_ablation_") as temp:
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
                sequence += 1
                print(f"warmup {mode} {warmup}/{args.warmup} ({default_config.label})", flush=True)
                run_once(args, default_config, mode, -warmup, sequence, root, record=False)

            for repeat in range(1, args.repeats + 1):
                for config in balanced_configs(configs, repeat):
                    sequence += 1
                    print(
                        f"profile {mode} repeat {repeat}/{args.repeats}: {config.label}",
                        flush=True,
                    )
                    row = run_once(args, config, mode, repeat, sequence, root, record=True)
                    assert row is not None
                    rows.append(row)

    verify_hashes(rows)
    write_results(rows, args.output)
    print_summary(rows)
    print("\nPASS: every ablation configuration produced the same persistence multiset per mode")
    print(f"Wrote detailed ablation profile to {args.output}")


if __name__ == "__main__":
    main()
