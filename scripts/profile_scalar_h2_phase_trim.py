#!/usr/bin/env python3
"""Balanced H2 wall/RSS benchmark for one targeted pre-reduction malloc_trim(0)."""
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
    r"PROFILE_PREP\s+scalar_h2_stream\s+decode_seconds=(?P<decode>[0-9.]+)\s+"
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
CONFIGS = (
    ("parent-rank", "off"),
    ("parent-rank", "before-reduce"),
    ("packed", "off"),
    ("packed", "before-reduce"),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--slab-depth", type=int, default=16)
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    p.add_argument("--output", type=Path, default=Path("scalar_h2_phase_trim_profile.csv"))
    p.add_argument("--keep-logs", action="store_true")
    return p.parse_args()


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


def parse_internal(log: str):
    out = {}
    m = PROFILE_RE.search(log)
    if m:
        out.update(
            prepare_seconds=float(m.group("prepare")),
            reduce_seconds=float(m.group("reduce")),
            cleanup_seconds=float(m.group("cleanup")),
            internal_total_seconds=float(m.group("total")),
        )
    m = PREP_RE.search(log)
    if m:
        for name in (
            "decode", "key_conversion", "slab_copy", "scalar_order", "local_sweep",
            "local_run_write", "cross_interface", "unaccounted",
        ):
            out[f"{name}_seconds"] = float(m.group(name))
    m = TRIM_RE.search(log)
    if not m:
        raise RuntimeError("missing PROFILE_TRIM line")
    out.update(
        trim_strategy=m.group("strategy"),
        trim_attempted=m.group("attempted"),
        trim_released=m.group("released"),
        trim_seconds=float(m.group("seconds")),
    )
    return out


def run_once(a, layout, trim, repeat, sequence, root, record):
    tag = "warmup" if not record else f"r{repeat:03d}"
    rd = root / f"{sequence:04d}_{layout}_{trim}_{tag}"
    rd.mkdir(parents=True)
    intervals = rd / "intervals.csv"
    timing = rd / "time.txt"
    log = rd / "run.log"
    cmd = [
        "/usr/bin/time", "-v", "-o", str(timing),
        str(a.binary.resolve()), str(a.input.resolve()), str(a.slab_depth),
        str(a.foreground_connectivity), "h2-scalar-stream", str(intervals),
        "--merge-strategy", "scan",
        "--interface-order", "radix",
        "--event-order", "verify",
        "--f32-key-mode", a.f32_key_mode,
        "--neighbor-kernel", "interior-fast",
        "--representative-active-check", "recheck",
        "--union-kernel", "root-carrying",
        "--h0-pruning-cache", "64k",
        "--neighbor-root-check", "parent-shortcut",
        "--active-state", "parent-sentinel",
        "--interface-state", "root-invariant",
        "--uf-layout", layout,
        "--phase-trim", trim,
    ]
    started = time.perf_counter()
    cp = subprocess.run(cmd, cwd=rd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    log.write_text(cp.stdout, encoding="utf-8")
    if cp.returncode:
        raise RuntimeError(f"{layout}/{trim} failed; see {log}")
    parsed = parse_internal(cp.stdout)
    if parsed["trim_strategy"] != trim:
        raise RuntimeError(f"trim strategy mismatch: requested {trim}, logged {parsed['trim_strategy']}")
    expected_attempted = "true" if trim == "before-reduce" else "false"
    if parsed["trim_attempted"] != expected_attempted:
        raise RuntimeError(
            f"trim attempted mismatch for {layout}/{trim}: {parsed['trim_attempted']}"
        )
    if not record:
        if not a.keep_logs:
            shutil.rmtree(rd, ignore_errors=True)
        return None
    digest, count = canonical_hash(intervals)
    row = {
        "uf_layout": layout,
        "phase_trim": trim,
        "repeat": repeat,
        "run_sequence": sequence,
        "wall_seconds": wall,
        "interval_rows": count,
        "canonical_sha256": digest,
        **parse_time(timing),
        **parsed,
    }
    if not a.keep_logs:
        shutil.rmtree(rd, ignore_errors=True)
    return row


def median(rows, field):
    values = [float(r[field]) for r in rows if r.get(field, "") != ""]
    return statistics.median(values) if values else float("nan")


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    rows = []
    sequence = 0
    with tempfile.TemporaryDirectory(prefix="betti_h2_phase_trim_profile_") as td:
        root = Path(td)
        for _ in range(a.warmup):
            for layout, trim in CONFIGS:
                run_once(a, layout, trim, -1, sequence, root, False)
                sequence += 1
        for repeat in range(a.repeats):
            shift = repeat % len(CONFIGS)
            order = CONFIGS[shift:] + CONFIGS[:shift]
            if repeat % 2:
                order = tuple(reversed(order))
            for layout, trim in order:
                rows.append(run_once(a, layout, trim, repeat, sequence, root, True))
                sequence += 1

    signatures = {(r["canonical_sha256"], r["interval_rows"]) for r in rows}
    if len(signatures) != 1:
        raise RuntimeError(f"phase-trim outputs disagree: {signatures}")

    fields = [
        "uf_layout", "phase_trim", "repeat", "run_sequence", "wall_seconds",
        "user_seconds", "system_seconds", "max_rss_kb", "prepare_seconds",
        "trim_seconds", "trim_attempted", "trim_released", "local_sweep_seconds",
        "reduce_seconds", "cleanup_seconds", "fs_inputs", "fs_outputs",
        "major_faults", "minor_faults", "interval_rows", "canonical_sha256",
    ]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fields})

    groups = defaultdict(list)
    for row in rows:
        groups[(row["uf_layout"], row["phase_trim"])].append(row)

    print("\nMedian H2 phase-trim ablation")
    print("layout\ttrim\twall_s\tprepare_s\ttrim_s\treduce_s\tRSS_MiB\treleased")
    for key in CONFIGS:
        g = groups[key]
        released = sum(r.get("trim_released") == "true" for r in g)
        print(
            f"{key[0]}\t{key[1]}\t{median(g,'wall_seconds'):.6f}\t"
            f"{median(g,'prepare_seconds'):.6f}\t{median(g,'trim_seconds'):.6f}\t"
            f"{median(g,'reduce_seconds'):.6f}\t{median(g,'max_rss_kb')/1024:.2f}\t"
            f"{released}/{len(g)}"
        )

    for layout in ("parent-rank", "packed"):
        off = groups[(layout, "off")]
        trim = groups[(layout, "before-reduce")]
        print(
            f"{layout} trim effect: wall "
            f"{(median(trim,'wall_seconds')/median(off,'wall_seconds')-1)*100:+.2f}%; "
            f"RSS {(median(trim,'max_rss_kb')/median(off,'max_rss_kb')-1)*100:+.2f}%; "
            f"median trim call {median(trim,'trim_seconds')*1000:.3f} ms"
        )

    packed_trim = groups[("packed", "before-reduce")]
    parent_off = groups[("parent-rank", "off")]
    print(
        "production candidate packed+trim vs parent-rank+off: "
        f"wall {(median(packed_trim,'wall_seconds')/median(parent_off,'wall_seconds')-1)*100:+.2f}%; "
        f"RSS {(median(packed_trim,'max_rss_kb')/median(parent_off,'max_rss_kb')-1)*100:+.2f}%"
    )
    print(f"PASS: exact H2 persistence agreement across all configurations; wrote {a.output}")


if __name__ == "__main__":
    main()
