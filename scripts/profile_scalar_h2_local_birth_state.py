#!/usr/bin/env python3
"""Balanced wall/RSS benchmark for tagged vs compact local H2 birth storage."""
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
PREP_RE = re.compile(
    r"PROFILE_PREP\s+scalar_h2_stream\s+.*?local_sweep_seconds=(?P<sweep>[0-9.]+)"
)
STATE_RE = re.compile(
    r"PROFILE_LOCAL_H2_STATE\s+strategy=(?P<strategy>\S+)\s+nodes=(?P<nodes>\d+)\s+"
    r"key_bytes=(?P<key_bytes>\d+)\s+parent_bytes=(?P<parent>\d+)\s+rank_bytes=(?P<rank>\d+)\s+"
    r"birth_bytes=(?P<birth>\d+)\s+total_bytes=(?P<state_total>\d+)"
)
SOURCE_TYPE_RE = re.compile(r"source pixel type:\s*(?P<pixel_type>\S+)")
CONFIG_RE = re.compile(r"PROFILE_CONFIG\s+scalar_h2_stream\s+.*?f32_key_mode=(?P<f32_key_mode>\S+)")
TIME_FIELDS = {
    "User time (seconds)": "user_seconds",
    "System time (seconds)": "system_seconds",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "File system inputs": "fs_inputs",
    "File system outputs": "fs_outputs",
    "Major (requiring I/O) page faults": "major_faults",
    "Minor (reclaiming a frame) page faults": "minor_faults",
}
STRATEGIES = ("tagged", "compact")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--slab-depth", type=int, default=16)
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("scalar_h2_local_birth_state_profile.csv"))
    return p.parse_args()


def require_v116_binary(binary: Path):
    cp = subprocess.run([str(binary.resolve()), "--help"], text=True, capture_output=True)
    help_text = cp.stdout + cp.stderr
    required = ("--local-h2-birth-state", "--global-h2-birth-state", "--global-h2-uf-layout", "--f32-key-mode")
    missing = [option for option in required if option not in help_text]
    if cp.returncode != 0 or missing:
        missing_text = ", ".join(missing) if missing else "v1.16 options"
        raise SystemExit(
            f"stale/non-v1.16 release binary: {binary.resolve()} does not advertise {missing_text}.\n"
            "Rebuild with:\n  cargo build --release --locked\n"
            "Then verify:\n  target/release/betti_curves --help | grep local-h2-birth-state"
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
    m = PREP_RE.search(log)
    if not m:
        raise RuntimeError("missing PROFILE_PREP scalar_h2_stream line")
    out["local_sweep_seconds"] = float(m.group("sweep"))
    m = SOURCE_TYPE_RE.search(log)
    if not m:
        raise RuntimeError("missing scalar source pixel type")
    out["source_pixel_type"] = m.group("pixel_type")
    m = CONFIG_RE.search(log)
    if not m:
        raise RuntimeError("missing PROFILE_CONFIG scalar_h2_stream line")
    out["reported_f32_key_mode"] = m.group("f32_key_mode")
    m = STATE_RE.search(log)
    if not m:
        raise RuntimeError("missing PROFILE_LOCAL_H2_STATE line")
    out.update(
        reported_strategy=m.group("strategy"),
        local_nodes=int(m.group("nodes")),
        local_key_bytes=int(m.group("key_bytes")),
        local_parent_bytes=int(m.group("parent")),
        local_rank_bytes=int(m.group("rank")),
        local_birth_bytes=int(m.group("birth")),
        local_state_total_bytes=int(m.group("state_total")),
    )
    return out


def run_once(a, strategy: str, repeat: int, sequence: int, root: Path, record: bool):
    tag = "warmup" if not record else f"r{repeat:03d}"
    rd = root / f"{sequence:04d}_{strategy}_{tag}"
    rd.mkdir(parents=True)
    intervals = rd / "intervals.csv"
    timing = rd / "time.txt"
    cmd = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(timing),
        str(a.binary.resolve()),
        str(a.input.resolve()),
        str(a.slab_depth),
        str(a.foreground_connectivity),
        "h2-scalar-stream",
        str(intervals),
        "--f32-key-mode",
        "native32",
        "--local-h2-birth-state",
        strategy,
        "--global-h2-birth-state",
        "compact",
        "--global-h2-uf-layout",
        "parent-rank",
    ]
    start = time.perf_counter()
    cp = subprocess.run(cmd, cwd=rd, text=True, capture_output=True)
    wall = time.perf_counter() - start
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}{cp.stderr}")
    digest, count = canonical_hash(intervals)
    row = dict(strategy=strategy, repeat=repeat, sequence=sequence, wall_seconds=wall)
    row.update(parse_time(timing))
    row.update(parse_internal(cp.stdout + cp.stderr))
    row.update(interval_rows=count, canonical_sha256=digest)
    if row["reported_strategy"] != strategy:
        raise RuntimeError(f"expected local H2 strategy {strategy}, got {row['reported_strategy']}")
    if row["reported_f32_key_mode"] != "native32":
        raise RuntimeError(
            f"expected native32 F32 key mode, got {row['reported_f32_key_mode']}"
        )
    if row["source_pixel_type"] == "F32" and row["local_key_bytes"] != 4:
        raise RuntimeError(
            "F32 input did not reach the native-width local H2 path: "
            f"PROFILE_LOCAL_H2_STATE reported key_bytes={row['local_key_bytes']}"
        )
    return row


def median(rows, field):
    return statistics.median(float(r[field]) for r in rows)


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    require_v116_binary(a.binary)
    rows = []
    sequence = 0
    with tempfile.TemporaryDirectory(prefix="betti_h2_local_birth_profile_") as td:
        root = Path(td)
        for warmup in range(a.warmup):
            order = STRATEGIES if warmup % 2 == 0 else tuple(reversed(STRATEGIES))
            for strategy in order:
                run_once(a, strategy, -1, sequence, root, False)
                sequence += 1
        for repeat in range(a.repeats):
            order = STRATEGIES if repeat % 2 == 0 else tuple(reversed(STRATEGIES))
            for strategy in order:
                rows.append(run_once(a, strategy, repeat, sequence, root, True))
                sequence += 1

    signatures = {(r["canonical_sha256"], r["interval_rows"]) for r in rows}
    if len(signatures) != 1:
        raise RuntimeError(f"persistence outputs differ across strategies/repeats: {signatures}")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["strategy"]].append(row)

    print("strategy\twall_s\tprepare_s\tsweep_s\tpeak_RSS_MiB\tlocal_state_MiB\tbirth_MiB")
    for strategy in STRATEGIES:
        g = groups[strategy]
        print(
            f"{strategy}\t{median(g,'wall_seconds'):.6f}\t{median(g,'prepare_seconds'):.6f}\t"
            f"{median(g,'local_sweep_seconds'):.6f}\t{median(g,'max_rss_kb')/1024:.2f}\t"
            f"{median(g,'local_state_total_bytes')/2**20:.2f}\t"
            f"{median(g,'local_birth_bytes')/2**20:.2f}"
        )

    tagged = groups["tagged"]
    compact = groups["compact"]
    rss_t, rss_c = median(tagged, "max_rss_kb"), median(compact, "max_rss_kb")
    wall_t, wall_c = median(tagged, "wall_seconds"), median(compact, "wall_seconds")
    state_t = median(tagged, "local_state_total_bytes")
    state_c = median(compact, "local_state_total_bytes")
    birth_t = median(tagged, "local_birth_bytes")
    birth_c = median(compact, "local_birth_bytes")
    print(
        f"\ncompact vs tagged: RSS {(rss_c/rss_t-1)*100:+.2f}%, "
        f"wall {(wall_c/wall_t-1)*100:+.2f}%, "
        f"local state {(state_c/state_t-1)*100:+.2f}%, "
        f"birth allocation {(birth_c/birth_t-1)*100:+.2f}%"
    )
    print(f"PASS: exact persistence agreement; wrote {a.output}")


if __name__ == "__main__":
    main()
