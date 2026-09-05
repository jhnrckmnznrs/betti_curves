#!/usr/bin/env python3
"""Stage-by-stage H2 memory audit for parent-rank versus packed local UF layout."""
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
from pathlib import Path

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
TIME_FIELDS = {
    "Maximum resident set size (kbytes)": "external_max_rss_kb",
    "User time (seconds)": "external_user_seconds",
    "System time (seconds)": "external_system_seconds",
    "Minor (reclaiming a frame) page faults": "external_minor_faults",
    "Major (requiring I/O) page faults": "external_major_faults",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--slab-depth", type=int, default=16)
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("h2_memory_audit.csv"))
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
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        field = TIME_FIELDS.get(key)
        if not field:
            continue
        try:
            result[field] = float(value.strip())
        except ValueError:
            pass
    return result


def parse_profile_records(log: str):
    rows = []
    for line_no, line in enumerate(log.splitlines(), 1):
        line = line.strip()
        if line.startswith("PROFILE_MEM "):
            parts = line.split(maxsplit=2)
            scope = parts[1]
            kv = dict(KV_RE.findall(parts[2] if len(parts) > 2 else ""))
            rows.append({"record_type": "mem", "scope": scope, "line_no": line_no, **kv})
        elif line.startswith("PROFILE_MEMVEC "):
            parts = line.split(maxsplit=2)
            scope = parts[1]
            kv = dict(KV_RE.findall(parts[2] if len(parts) > 2 else ""))
            rows.append({"record_type": "vec", "scope": scope, "line_no": line_no, **kv})
    return rows


def run_once(args, layout: str, repeat: int, root: Path):
    run_dir = root / f"r{repeat:02d}_{layout}"
    run_dir.mkdir(parents=True)
    intervals = run_dir / "intervals.csv"
    timing = run_dir / "time.txt"
    log_path = run_dir / "run.log"
    cmd = [
        "/usr/bin/time", "-v", "-o", str(timing),
        str(args.binary.resolve()), str(args.input.resolve()), str(args.slab_depth),
        str(args.foreground_connectivity), "h2-scalar-stream", str(intervals),
        "--merge-strategy", "scan",
        "--interface-order", "radix",
        "--event-order", "verify",
        "--f32-key-mode", args.f32_key_mode,
        "--neighbor-kernel", "interior-fast",
        "--representative-active-check", "recheck",
        "--union-kernel", "root-carrying",
        "--neighbor-root-check", "parent-shortcut",
        "--active-state", "parent-sentinel",
        "--interface-state", "root-invariant",
        "--uf-layout", layout,
        "--h2-memory-audit",
    ]
    start = time.perf_counter()
    cp = subprocess.run(cmd, cwd=run_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - start
    log_path.write_text(cp.stdout, encoding="utf-8")
    if cp.returncode:
        raise RuntimeError(f"{layout} failed; see {log_path}\n{cp.stdout[-4000:]}")
    digest, count = canonical_hash(intervals)
    external = parse_time(timing)
    records = parse_profile_records(cp.stdout)
    if not records:
        raise RuntimeError(f"{layout} produced no PROFILE_MEM records; is this v1.11?")
    base = {
        "uf_layout": layout,
        "repeat": repeat,
        "wall_seconds": wall,
        "interval_rows": count,
        "canonical_sha256": digest,
        **external,
    }
    for row in records:
        row.update(base)
    if not args.keep_logs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return records, (digest, count), base



def main():
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    all_rows = []
    outputs = {"parent-rank": set(), "packed": set()}
    run_bases = []
    with tempfile.TemporaryDirectory(prefix="betti_h2_mem_audit_") as td:
        root = Path(td)
        for repeat in range(args.repeats):
            order = ("parent-rank", "packed") if repeat % 2 == 0 else ("packed", "parent-rank")
            for layout in order:
                rows, signature, base = run_once(args, layout, repeat, root)
                all_rows.extend(rows)
                outputs[layout].add(signature)
                run_bases.append(base)
    signatures = set().union(*outputs.values())
    if len(signatures) != 1:
        raise RuntimeError(f"persistence outputs disagree across layouts/repeats: {outputs}")

    preferred = [
        "uf_layout", "repeat", "record_type", "scope", "stage", "slab", "line_no",
        "rss_kb", "hwm_kb", "rss_anon_kb", "rss_file_kb", "rss_shmem_kb",
        "pss_kb", "pss_anon_kb", "pss_file_kb", "private_clean_kb",
        "private_dirty_kb", "anonymous_kb", "external_max_rss_kb", "wall_seconds",
        "interval_rows", "canonical_sha256",
    ]
    all_fields = set().union(*(row.keys() for row in all_rows))
    fields = preferred + sorted(all_fields.difference(preferred))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    print("\nH2 memory-audit summary")
    print("layout\tmedian external RSS MiB\tmax live RSS MiB\tmax HWM MiB\tstage of max live RSS")
    for layout in ("parent-rank", "packed"):
        bases = [b for b in run_bases if b["uf_layout"] == layout]
        external = statistics.median(float(b["external_max_rss_kb"]) for b in bases) / 1024
        mem_rows = [r for r in all_rows if r["uf_layout"] == layout and r["record_type"] == "mem"]
        max_live = max((float(r.get("rss_kb", 0)) for r in mem_rows), default=0)
        max_hwm = max((float(r.get("hwm_kb", 0)) for r in mem_rows), default=0)
        peak = max(mem_rows, key=lambda r: float(r.get("rss_kb", 0)), default={})
        peak_stage = f"{peak.get('scope','')}:{peak.get('stage','')}[slab={peak.get('slab','')}]"
        print(f"{layout}\t{external:.2f}\t{max_live/1024:.2f}\t{max_hwm/1024:.2f}\t{peak_stage}")

    # Report where packed first exceeds parent-rank in median HWM for matching stage/slab.
    grouped = {}
    for row in all_rows:
        if row["record_type"] != "mem":
            continue
        key = (row.get("scope"), row.get("stage"), row.get("slab"))
        grouped.setdefault((row["uf_layout"], key), []).append(float(row.get("hwm_kb", 0)))
    deltas = []
    keys = {key for layout, key in grouped if layout == "parent-rank"} & {key for layout, key in grouped if layout == "packed"}
    for key in keys:
        ref = statistics.median(grouped[("parent-rank", key)])
        packed = statistics.median(grouped[("packed", key)])
        deltas.append((packed - ref, key, ref, packed))
    if deltas:
        delta, key, ref, packed = max(deltas)
        print(f"largest matched-stage HWM delta: {delta/1024:.2f} MiB at {key}; parent-rank={ref/1024:.2f}, packed={packed/1024:.2f}")
    print(f"PASS: exact H2 persistence agreement; wrote {args.output}")


if __name__ == "__main__":
    main()
