#!/usr/bin/env python3
"""Profile consolidated production scalar-stream defaults.

Runs H0/H2 production defaults over one or more TIFF stacks and slab depths.
No tuning flags are passed. The profiler records the binary SHA-256, effective
PROFILE_CONFIG, internal phase timings, /usr/bin/time resource statistics, and
canonical persistence hashes. For a given specimen/mode, every repeat and slab
depth must produce the exact same persistence multiset hash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import platform
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

PROFILE_RE = re.compile(
    r"PROFILE\s+scalar_(?P<mode>h[02])_stream\s+prepare_seconds=(?P<prepare>[0-9.]+)\s+"
    r"reduce_seconds=(?P<reduce>[0-9.]+)\s+cleanup_seconds=(?P<cleanup>[0-9.]+)\s+"
    r"total_seconds=(?P<total>[0-9.]+)"
)
PREP_RE = re.compile(
    r"PROFILE_PREP\s+scalar_(?P<mode>h[02])_stream\s+decode_seconds=(?P<decode>[0-9.]+)\s+"
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
SOURCE_TYPE_RE = re.compile(r"source pixel type:\s*(?P<pixel_type>\S+)")
TRIM_RE = re.compile(
    r"PROFILE_TRIM\s+scalar_h2_stream\s+strategy=(?P<strategy>\S+)\s+"
    r"attempted=(?P<attempted>true|false)\s+released=(?P<released>true|false)\s+"
    r"seconds=(?P<seconds>[0-9.]+)"
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--slab-depths", type=int, nargs="+", default=(8, 16, 32))
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--modes", nargs="+", choices=("h0", "h2"), default=("h0", "h2"))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("production_scalar_profile.csv"))
    p.add_argument("--summary-output", type=Path)
    p.add_argument("--keep-logs", action="store_true")
    return p.parse_args()


def file_sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def canonical_hash(path: Path):
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        header = next(r, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence header {header!r}")
        rows = sorted(tuple(row) for row in r if row)
    d = hashlib.sha256()
    for birth, death in rows:
        d.update(f"{birth},{death}\n".encode())
    return d.hexdigest(), len(rows)


def parse_time(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        field = TIME_FIELDS.get(key)
        if field:
            try:
                out[field] = float(value.strip())
            except ValueError:
                pass
    return out


def parse_config(log: str, mode: str):
    prefix = f"PROFILE_CONFIG scalar_{mode}_stream "
    line = next((line for line in log.splitlines() if line.startswith(prefix)), None)
    if line is None:
        raise RuntimeError(f"missing {prefix.strip()}")
    out = {}
    for token in line[len(prefix):].split():
        if "=" in token:
            key, value = token.split("=", 1)
            out[f"config_{key}"] = value
    return out


def glibc_linux() -> bool:
    libc_name, _ = platform.libc_ver()
    return platform.system() == "Linux" and libc_name.lower() == "glibc"


def expected_production_config(mode: str):
    common = {
        "config_merge_strategy": "scan",
        "config_interface_order": "radix",
        "config_event_order": "verify",
        "config_f32_key_mode": "native32",
        "config_neighbor_kernel": "interior-fast",
        "config_representative_active_check": "recheck",
        "config_union_kernel": "root-carrying",
        "config_interface_state": "root-invariant",
        "config_uf_layout": "packed",
    }
    if mode == "h0":
        common.update(
            config_h0_pruning_cache="64k",
            config_active_state="separate",
            config_global_h0_uf_layout="packed",
        )
    else:
        common.update(
            config_neighbor_root_check="parent-shortcut",
            config_active_state="parent-sentinel",
            config_phase_trim="before-reduce" if glibc_linux() else "off",
            config_local_h2_birth_state="compact",
            config_global_h2_birth_state="compact",
            config_global_h2_uf_layout="parent-rank",
        )
    return common


def validate_production_config(row, mode: str, binary_hash: str):
    expected = expected_production_config(mode)
    mismatches = {
        key: (value, row.get(key))
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "effective configuration is not the consolidated production default; "
            f"binary_sha256={binary_hash} mode={mode} mismatches={mismatches}. "
            "The release binary is probably stale. Run `cargo build --release --locked` "
            "and `python3 validation/check_production_scalar_equivalence.py ...` before profiling."
        )


def parse_internal(log: str, mode: str):
    out = {}
    source = SOURCE_TYPE_RE.search(log)
    if source is None:
        raise RuntimeError("missing scalar source pixel type")
    out["source_pixel_type"] = source.group("pixel_type")
    match = next((m for m in PROFILE_RE.finditer(log) if m.group("mode") == mode), None)
    if match is None:
        raise RuntimeError(f"missing PROFILE scalar_{mode}_stream line")
    out.update(
        prepare_seconds=float(match.group("prepare")),
        reduce_seconds=float(match.group("reduce")),
        cleanup_seconds=float(match.group("cleanup")),
        internal_total_seconds=float(match.group("total")),
    )
    match = next((m for m in PREP_RE.finditer(log) if m.group("mode") == mode), None)
    if match:
        for name in (
            "decode", "key_conversion", "slab_copy", "scalar_order", "local_sweep",
            "local_run_write", "cross_interface", "unaccounted",
        ):
            out[f"{name}_seconds"] = float(match.group(name))
        out["total_voxels"] = int(match.group("total_voxels"))
        out["interior_fast_voxels"] = int(match.group("interior_fast_voxels"))
    if mode == "h2":
        m = TRIM_RE.search(log)
        if m:
            out.update(
                trim_strategy=m.group("strategy"),
                trim_attempted=m.group("attempted"),
                trim_released=m.group("released"),
                trim_seconds=float(m.group("seconds")),
            )
    out.update(parse_config(log, mode))
    return out


def run_once(a, binary_hash, specimen_index, input_path, slab, mode, repeat, sequence, root, record):
    tag = "warmup" if not record else f"r{repeat:03d}"
    rd = root / f"{sequence:05d}_{input_path.name}_d{slab}_{mode}_{tag}"
    rd.mkdir(parents=True)
    intervals = rd / "intervals.csv"
    timing = rd / "time.txt"
    log_path = rd / "run.log"
    command = [
        "/usr/bin/time", "-v", "-o", str(timing),
        str(a.binary.resolve()), str(input_path.resolve()), str(slab),
        str(a.foreground_connectivity), f"{mode}-scalar-stream", str(intervals),
    ]
    started = time.perf_counter()
    cp = subprocess.run(command, cwd=rd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    log_path.write_text(cp.stdout, encoding="utf-8")
    if cp.returncode:
        raise RuntimeError(f"production run failed; see {log_path}")
    if not record:
        if not a.keep_logs:
            shutil.rmtree(rd, ignore_errors=True)
        return None
    digest, count = canonical_hash(intervals)
    row = {
        "specimen_index": specimen_index,
        "specimen": input_path.name,
        "input_path": str(input_path.resolve()),
        "mode": mode,
        "slab_depth": slab,
        "repeat": repeat,
        "run_sequence": sequence,
        "binary_sha256": binary_hash,
        "wall_seconds": wall,
        "interval_rows": count,
        "canonical_sha256": digest,
        **parse_time(timing),
        **parse_internal(cp.stdout, mode),
    }
    validate_production_config(row, mode, binary_hash)
    if not a.keep_logs:
        shutil.rmtree(rd, ignore_errors=True)
    return row


def median(rows, field):
    vals = [float(r[field]) for r in rows if r.get(field, "") not in ("", None)]
    return statistics.median(vals) if vals else float("nan")


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    if shutil.which("/usr/bin/time") is None and not Path("/usr/bin/time").is_file():
        raise SystemExit("/usr/bin/time is required")
    for path in a.inputs:
        if not path.is_dir():
            raise SystemExit(f"input stack not found: {path}")
    if any(d <= 0 for d in a.slab_depths):
        raise SystemExit("all slab depths must be positive")

    binary_hash = file_sha256(a.binary)
    configs = [
        (i, path, depth, mode)
        for i, path in enumerate(a.inputs)
        for depth in a.slab_depths
        for mode in a.modes
    ]
    rows = []
    sequence = 0
    with tempfile.TemporaryDirectory(prefix="betti_production_profile_") as td:
        root = Path(td)
        for _ in range(a.warmup):
            for spec_i, path, depth, mode in configs:
                run_once(a, binary_hash, spec_i, path, depth, mode, -1, sequence, root, False)
                sequence += 1
        for repeat in range(a.repeats):
            shift = repeat % len(configs)
            order = configs[shift:] + configs[:shift]
            if repeat % 2:
                order = list(reversed(order))
            for spec_i, path, depth, mode in order:
                rows.append(
                    run_once(a, binary_hash, spec_i, path, depth, mode, repeat, sequence, root, True)
                )
                sequence += 1

    # Exact slab-depth and repeat invariance for each specimen/mode.
    signatures = defaultdict(set)
    configs_seen = defaultdict(set)
    for row in rows:
        key = (row["specimen_index"], row["mode"])
        signatures[key].add((row["canonical_sha256"], row["interval_rows"]))
        configs_seen[key].add(tuple(sorted((k, v) for k, v in row.items() if k.startswith("config_"))))
    bad = {key: sig for key, sig in signatures.items() if len(sig) != 1}
    if bad:
        raise RuntimeError(f"persistence output changed across depth/repeat: {bad}")
    bad_config = {key: cfg for key, cfg in configs_seen.items() if len(cfg) != 1}
    if bad_config:
        raise RuntimeError(f"effective production configuration changed within a mode: {bad_config}")
    source_types = defaultdict(set)
    for row in rows:
        source_types[row["specimen_index"]].add(row.get("source_pixel_type", ""))
    bad_source = {key: values for key, values in source_types.items() if len(values) != 1 or "" in values}
    if bad_source:
        raise RuntimeError(f"source pixel type changed or was not reported: {bad_source}")

    all_fields = []
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=all_fields)
        w.writeheader()
        w.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["specimen_index"], row["specimen"], row["mode"], row["slab_depth"])].append(row)
    summary_rows = []
    for (spec_i, specimen, mode, slab), group in sorted(grouped.items()):
        summary_rows.append({
            "specimen_index": spec_i,
            "specimen": specimen,
            "mode": mode,
            "slab_depth": slab,
            "median_wall_seconds": median(group, "wall_seconds"),
            "median_prepare_seconds": median(group, "prepare_seconds"),
            "median_local_sweep_seconds": median(group, "local_sweep_seconds"),
            "median_reduce_seconds": median(group, "reduce_seconds"),
            "median_peak_rss_mib": median(group, "max_rss_kb") / 1024.0,
            "interval_rows": group[0]["interval_rows"],
            "canonical_sha256": group[0]["canonical_sha256"],
            "binary_sha256": binary_hash,
            "source_pixel_type": group[0].get("source_pixel_type", ""),
        })
    summary_path = a.summary_output or a.output.with_name(a.output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(summary_rows[0]) if summary_rows else [])
        if summary_rows:
            w.writeheader()
            w.writerows(summary_rows)

    print("\nProduction scalar-stream profile")
    print("specimen\tmode\tdepth\twall_s\tprepare_s\tsweep_s\treduce_s\tRSS_MiB")
    for row in summary_rows:
        print(
            f"{row['specimen']}\t{row['mode']}\t{row['slab_depth']}\t"
            f"{row['median_wall_seconds']:.3f}\t{row['median_prepare_seconds']:.3f}\t"
            f"{row['median_local_sweep_seconds']:.3f}\t{row['median_reduce_seconds']:.3f}\t"
            f"{row['median_peak_rss_mib']:.1f}"
        )
    print("\nBest slab depth by specimen/mode (median wall time)")
    by_mode = defaultdict(list)
    for row in summary_rows:
        by_mode[(row["specimen_index"], row["specimen"], row["mode"])].append(row)
    for (_, specimen, mode), group in sorted(by_mode.items()):
        best = min(group, key=lambda r: r["median_wall_seconds"])
        print(
            f"{specimen} {mode}: depth={best['slab_depth']} "
            f"wall={best['median_wall_seconds']:.3f}s RSS={best['median_peak_rss_mib']:.1f}MiB"
        )
    print(f"\nPASS: exact persistence invariant across slab depths/repeats; wrote {a.output}")
    print(f"Summary: {summary_path}")
    print(f"Binary SHA-256: {binary_hash}")


if __name__ == "__main__":
    main()
