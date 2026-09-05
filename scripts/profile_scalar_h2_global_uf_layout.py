#!/usr/bin/env python3
"""Balanced wall/RSS benchmark for compact global H2 parent-rank vs packed UF."""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
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
STATE_RE = re.compile(
    r"PROFILE_GLOBAL_H2_STATE\s+strategy=(?P<strategy>\S+)\s+uf_layout=(?P<layout>\S+)\s+"
    r"nodes=(?P<nodes>\d+)\s+parent_bytes=(?P<parent>\d+)\s+rank_bytes=(?P<rank>\d+)\s+"
    r"birth_bytes=(?P<birth>\d+)\s+total_bytes=(?P<state_total>\d+)"
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--slab-depth", type=int, default=16)
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("scalar_h2_global_uf_layout_profile.csv"))
    return p.parse_args()


def require_v115_binary(binary: Path):
    cp = subprocess.run([str(binary.resolve()), "--help"], text=True, capture_output=True)
    help_text = cp.stdout + cp.stderr
    required = ("--global-h2-birth-state", "--global-h2-uf-layout")
    missing = [option for option in required if option not in help_text]
    if cp.returncode != 0 or missing:
        missing_text = ", ".join(missing) if missing else "v1.15 options"
        raise SystemExit(
            f"stale/non-v1.15 release binary: {binary.resolve()} does not advertise {missing_text}.\n"
            "Rebuild from this source tree with:\n"
            "  cargo build --release --locked\n"
            "Then verify:\n"
            "  target/release/betti_curves --help | grep global-h2-uf-layout"
        )


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
    out = {}
    m = PROFILE_RE.search(log)
    if not m:
        raise RuntimeError("missing PROFILE scalar_h2_stream line")
    out.update(
        prepare_seconds=float(m.group("prepare")),
        reduce_seconds=float(m.group("reduce")),
        cleanup_seconds=float(m.group("cleanup")),
        internal_total_seconds=float(m.group("total")),
    )
    m = STATE_RE.search(log)
    if not m:
        raise RuntimeError("missing PROFILE_GLOBAL_H2_STATE line")
    out.update(
        reported_strategy=m.group("strategy"),
        reported_layout=m.group("layout"),
        global_nodes=int(m.group("nodes")),
        global_parent_bytes=int(m.group("parent")),
        global_rank_bytes=int(m.group("rank")),
        global_birth_bytes=int(m.group("birth")),
        global_state_total_bytes=int(m.group("state_total")),
    )
    return out


def run_once(a, layout: str, repeat: int, sequence: int, root: Path, record: bool):
    tag = "warmup" if not record else f"r{repeat:03d}"
    rd = root / f"{sequence:04d}_{layout}_{tag}"
    rd.mkdir(parents=True)
    intervals = rd / "intervals.csv"
    timing = rd / "time.txt"
    cmd = [
        "/usr/bin/time", "-v", "-o", str(timing),
        str(a.binary.resolve()), str(a.input.resolve()), str(a.slab_depth),
        str(a.foreground_connectivity), "h2-scalar-stream", str(intervals),
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", layout,
    ]
    start = time.perf_counter()
    cp = subprocess.run(cmd, cwd=rd, text=True, capture_output=True)
    wall = time.perf_counter() - start
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}{cp.stderr}")
    digest, count = canonical_hash(intervals)
    row = dict(layout=layout, repeat=repeat, sequence=sequence, wall_seconds=wall)
    row.update(parse_time(timing))
    row.update(parse_internal(cp.stdout + cp.stderr))
    row.update(interval_rows=count, canonical_sha256=digest)
    if row["reported_strategy"] != "compact" or row["reported_layout"] != layout:
        raise RuntimeError(
            f"expected compact/{layout}, got {row['reported_strategy']}/{row['reported_layout']}"
        )
    return row


def median(rows, field):
    return statistics.median(float(r[field]) for r in rows)


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    require_v115_binary(a.binary)
    rows = []
    sequence = 0
    with tempfile.TemporaryDirectory(prefix="betti_h2_global_uf_profile_") as td:
        root = Path(td)
        for warmup in range(a.warmup):
            order = LAYOUTS if warmup % 2 == 0 else tuple(reversed(LAYOUTS))
            for layout in order:
                run_once(a, layout, -1, sequence, root, False)
                sequence += 1
        for repeat in range(a.repeats):
            order = LAYOUTS if repeat % 2 == 0 else tuple(reversed(LAYOUTS))
            for layout in order:
                rows.append(run_once(a, layout, repeat, sequence, root, True))
                sequence += 1

    signatures = {(r["canonical_sha256"], r["interval_rows"]) for r in rows}
    if len(signatures) != 1:
        raise RuntimeError(f"persistence outputs differ across layouts/repeats: {signatures}")

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
            f"{median(g,'max_rss_kb')/1024:.2f}\t"
            f"{median(g,'global_state_total_bytes')/2**20:.2f}\t"
            f"{median(g,'global_rank_bytes')/2**20:.2f}"
        )

    ref = groups["parent-rank"]
    packed = groups["packed"]
    rss_r, rss_p = median(ref, "max_rss_kb"), median(packed, "max_rss_kb")
    wall_r, wall_p = median(ref, "wall_seconds"), median(packed, "wall_seconds")
    state_r, state_p = median(ref, "global_state_total_bytes"), median(packed, "global_state_total_bytes")
    print(
        f"\npacked vs parent-rank: RSS {(rss_p/rss_r-1)*100:+.2f}%, "
        f"wall {(wall_p/wall_r-1)*100:+.2f}%, "
        f"explicit global state {(state_p/state_r-1)*100:+.2f}%"
    )
    print(f"PASS: exact persistence agreement; wrote {a.output}")


if __name__ == "__main__":
    main()
