#!/usr/bin/env python3
"""Balanced benchmark for native32 global H0 parent-rank vs packed UF."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

PROFILE_RE = re.compile(
    r"PROFILE\s+scalar_h0_stream\s+prepare_seconds=(?P<prepare>[0-9.]+)\s+"
    r"reduce_seconds=(?P<reduce>[0-9.]+)\s+cleanup_seconds=(?P<cleanup>[0-9.]+)\s+"
    r"total_seconds=(?P<total>[0-9.]+)"
)
STATE_RE = re.compile(
    r"PROFILE_GLOBAL_H0_STATE\s+scalar_h0_stream\s+layout=(?P<layout>\S+)\s+"
    r"nodes=(?P<nodes>\d+)\s+parent_bytes=(?P<parent>\d+)\s+rank_bytes=(?P<rank>\d+)\s+"
    r"birth_bytes=(?P<birth>\d+)\s+total_bytes=(?P<state_total>\d+)"
)
STORAGE_RE = re.compile(
    r"PROFILE_H0_STORAGE\s+scalar_h0_stream\s+pipeline=(?P<pipeline>\S+)\s+"
    r"disk_key_bytes=(?P<key_bytes>\d+)\s+interface_nodes=(?P<interface_nodes>\d+).*?"
    r"total_run_bytes=(?P<run_bytes>\d+).*?global_h0_uf_layout=(?P<storage_layout>\S+)"
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
LAYOUTS = ("parent-rank", "packed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--slab-depth", type=int, default=1)
    p.add_argument("--slice-limit", type=int, default=2)
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("scalar_h0_global_uf_layout_profile.csv"))
    return p.parse_args()


def stage_subset(src: Path, root: Path, limit: int) -> Path:
    if limit == 0:
        return src
    slices = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    if len(slices) < limit:
        raise SystemExit(f"requested {limit} slices but found {len(slices)}")
    dst = root / "input_subset"
    dst.mkdir()
    for path in slices[:limit]:
        (dst / path.name).symlink_to(path.resolve())
    return dst


def canonical_hash(path: Path):
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        if next(r, None) != ["birth", "death"]:
            raise RuntimeError("unexpected persistence CSV header")
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
    p = PROFILE_RE.search(log)
    state = STATE_RE.search(log)
    storage = STORAGE_RE.search(log)
    if not p or not state or not storage:
        raise RuntimeError("missing H0 profile/state/storage diagnostics")
    return {
        "prepare_seconds": float(p.group("prepare")),
        "reduce_seconds": float(p.group("reduce")),
        "cleanup_seconds": float(p.group("cleanup")),
        "internal_total_seconds": float(p.group("total")),
        "reported_layout": state.group("layout"),
        "global_nodes": int(state.group("nodes")),
        "global_parent_bytes": int(state.group("parent")),
        "global_rank_bytes": int(state.group("rank")),
        "global_birth_bytes": int(state.group("birth")),
        "global_state_total_bytes": int(state.group("state_total")),
        "storage_pipeline": storage.group("pipeline"),
        "disk_key_bytes": int(storage.group("key_bytes")),
        "total_run_bytes": int(storage.group("run_bytes")),
        "storage_layout": storage.group("storage_layout"),
    }


def run_once(a, input_path: Path, layout: str, repeat: int, sequence: int, root: Path, record: bool):
    tag = "warmup" if not record else f"r{repeat:03d}"
    rd = root / f"{sequence:04d}_{layout}_{tag}"
    rd.mkdir(parents=True)
    intervals = rd / "intervals.csv"
    timing = rd / "time.txt"
    cmd = [
        "/usr/bin/time", "-v", "-o", str(timing),
        str(a.binary.resolve()), str(input_path.resolve()), str(a.slab_depth),
        str(a.foreground_connectivity), "h0-scalar-stream", str(intervals),
        "--f32-key-mode", "native32",
        "--global-h0-uf-layout", layout,
    ]
    start = time.perf_counter()
    cp = subprocess.run(cmd, cwd=rd, text=True, capture_output=True, env=os.environ.copy())
    wall = time.perf_counter() - start
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}{cp.stderr}")
    digest, count = canonical_hash(intervals)
    row = {"layout": layout, "repeat": repeat, "sequence": sequence, "wall_seconds": wall}
    row.update(parse_time(timing))
    row.update(parse_internal(cp.stdout + cp.stderr))
    row.update(interval_rows=count, canonical_sha256=digest)
    if row["reported_layout"] != layout or row["storage_layout"] != layout:
        raise RuntimeError(f"expected layout {layout}, diagnostics disagree")
    if row["storage_pipeline"] != "native32-end-to-end" or row["disk_key_bytes"] != 4:
        raise RuntimeError("profile did not exercise native32 end-to-end storage")
    return row


def median(rows, field):
    return statistics.median(float(r[field]) for r in rows)


def main() -> None:
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    help_cp = subprocess.run([str(a.binary.resolve()), "--help"], text=True, capture_output=True)
    if "--global-h0-uf-layout" not in help_cp.stdout + help_cp.stderr:
        raise SystemExit("stale binary: rebuild v1.19 candidate")

    rows = []
    sequence = 0
    with tempfile.TemporaryDirectory(prefix="betti_h0_global_uf_profile_") as td:
        root = Path(td)
        test_input = stage_subset(a.input, root, a.slice_limit)
        for warmup in range(a.warmup):
            order = LAYOUTS if warmup % 2 == 0 else tuple(reversed(LAYOUTS))
            for layout in order:
                run_once(a, test_input, layout, -1, sequence, root, False)
                sequence += 1
        for repeat in range(a.repeats):
            order = LAYOUTS if repeat % 2 == 0 else tuple(reversed(LAYOUTS))
            for layout in order:
                rows.append(run_once(a, test_input, layout, repeat, sequence, root, True))
                sequence += 1

    signatures = {(r["canonical_sha256"], r["interval_rows"]) for r in rows}
    if len(signatures) != 1:
        raise RuntimeError(f"persistence differs across layouts/repeats: {signatures}")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["layout"]].append(row)
    print("layout\twall_s\treduce_s\tpeak_RSS_MiB\tglobal_state_MiB\trank_MiB")
    for layout in LAYOUTS:
        g = groups[layout]
        print(
            f"{layout}\t{median(g,'wall_seconds'):.6f}\t{median(g,'reduce_seconds'):.6f}\t"
            f"{median(g,'max_rss_kb')/1024:.2f}\t{median(g,'global_state_total_bytes')/2**20:.2f}\t"
            f"{median(g,'global_rank_bytes')/2**20:.2f}"
        )
    ref, packed = groups["parent-rank"], groups["packed"]
    print(
        f"\npacked vs parent-rank: RSS {(median(packed,'max_rss_kb')/median(ref,'max_rss_kb')-1)*100:+.2f}%, "
        f"wall {(median(packed,'wall_seconds')/median(ref,'wall_seconds')-1)*100:+.2f}%, "
        f"reduce {(median(packed,'reduce_seconds')/median(ref,'reduce_seconds')-1)*100:+.2f}%, "
        f"explicit global state {(median(packed,'global_state_total_bytes')/median(ref,'global_state_total_bytes')-1)*100:+.2f}%"
    )
    print(f"PASS: exact persistence agreement; wrote {a.output}")


if __name__ == "__main__":
    main()
