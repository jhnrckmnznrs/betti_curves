#!/usr/bin/env python3
"""Paired CX09T1 profile for the optimized disk-backed hierarchical H0 candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "validation"))
from f32_fixture import stage_f32_subset  # noqa: E402

TIME_FIELDS = {
    "User time (seconds)": "user_seconds",
    "System time (seconds)": "system_seconds",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "File system inputs": "fs_inputs",
    "File system outputs": "fs_outputs",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--slice-limit", type=int, default=0, help="0 stages all CX09T1 slices")
    p.add_argument("--slab-depths", type=int, nargs="+", default=(8, 16, 32))
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("cx09t1_v1_21_hierarchical_stream.csv"))
    p.add_argument("--summary-output", type=Path)
    p.add_argument("--keep-logs", action="store_true")
    return p.parse_args()


def canonical_hash(path: Path):
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        if next(r, None) != ["birth", "death"]:
            raise RuntimeError(f"bad persistence header in {path}")
        rows = sorted(tuple(row) for row in r if row)
    dig = hashlib.sha256()
    for b, d in rows:
        dig.update(f"{b},{d}\n".encode())
    return dig.hexdigest(), len(rows)


def parse_time(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        k, v = line.strip().split(":", 1)
        if k in TIME_FIELDS:
            try:
                out[TIME_FIELDS[k]] = float(v.strip())
            except ValueError:
                pass
    return out


def kv_line(log: str, prefix: str):
    line = next((x.strip() for x in log.splitlines() if x.startswith(prefix)), None)
    if line is None:
        return {}
    out = {}
    for token in line[len(prefix):].strip().split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def run_once(a, root, seq, variant, slab, fixture, repeat):
    rd = root / f"{seq:04d}_{variant}_d{slab}_r{repeat}"
    rd.mkdir(parents=True)
    out = rd / "intervals.csv"
    timing = rd / "time.txt"
    extras = []
    if variant == "flat-direct":
        mode = "h0-scalar-stream"
        extras = [
            "--f32-key-mode", "native32", "--h0-birth-buffer", "reuse-input",
            "--h0-event-storage", "direct", "--event-order", "verify",
            "--global-h0-uf-layout", "packed", "--merge-strategy", "scan",
        ]
    elif variant == "hier-reference":
        mode = "h0-scalar-hierarchical"
    elif variant == "hier-stream":
        mode = "h0-scalar-hierarchical-stream"
        extras = [
            "--f32-key-mode", "native32", "--h0-birth-buffer", "reuse-input",
            "--h0-event-storage", "direct", "--event-order", "verify",
            "--global-h0-uf-layout", "packed",
        ]
    else:
        raise ValueError(variant)

    cmd = [
        "/usr/bin/time", "-v", "-o", str(timing), str(a.binary.resolve()),
        str(fixture.resolve()), str(slab), str(a.foreground_connectivity), mode,
        str(out), *extras,
    ]
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, cwd=rd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    (rd / "run.log").write_text(cp.stdout, encoding="utf-8")
    if cp.returncode:
        raise RuntimeError(f"failed {variant} d{slab}:\n{cp.stdout[-5000:]}")
    digest, count = canonical_hash(out)
    row = {
        "variant": variant, "slab_depth": slab, "repeat": repeat,
        "wall_seconds": wall, "interval_count": count, "canonical_sha256": digest,
    }
    row.update(parse_time(timing))
    if variant == "hier-reference":
        for k, v in kv_line(cp.stdout, "PROFILE_H0_HIERARCHY ").items():
            row[f"hier_{k}"] = v
    elif variant == "hier-stream":
        for k, v in kv_line(cp.stdout, "PROFILE_H0_HIER_STREAM ").items():
            row[f"hier_stream_{k}"] = v
        for k, v in kv_line(cp.stdout, "PROFILE_GLOBAL_H0_STATE scalar_h0_hierarchical_stream ").items():
            row[f"root_{k}"] = v
    else:
        for k, v in kv_line(cp.stdout, "PROFILE_H0_STORAGE scalar_h0_stream ").items():
            row[f"flat_{k}"] = v
    if not a.keep_logs:
        shutil.rmtree(rd, ignore_errors=True)
    return row


def med(vals):
    return statistics.median(float(v) for v in vals)


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    with tempfile.TemporaryDirectory(prefix="betti_v121_profile_") as td:
        root = Path(td)
        fixture = stage_f32_subset(a.input, root / "f32_input", a.slice_limit)
        rows = []
        seq = 0
        # Rotate variant order by repeat to reduce cache/order bias.
        base = ["flat-direct", "hier-reference", "hier-stream"]
        for slab in a.slab_depths:
            for repeat in range(1, a.repeats + 1):
                order = base[(repeat - 1) % len(base):] + base[:(repeat - 1) % len(base)]
                for variant in order:
                    rows.append(run_once(a, root, seq, variant, slab, fixture, repeat))
                    seq += 1

    # Every variant/repeat/depth must have exactly the same barcode.
    hashes = defaultdict(set)
    counts = defaultdict(set)
    for r in rows:
        hashes[r["slab_depth"]].add(r["canonical_sha256"])
        counts[r["slab_depth"]].add(r["interval_count"])
    bad = {d: (hashes[d], counts[d]) for d in hashes if len(hashes[d]) != 1 or len(counts[d]) != 1}
    if bad:
        raise SystemExit(f"persistence mismatch: {bad}")

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["variant"], r["slab_depth"])].append(r)
    summary = []
    for (variant, slab), rs in sorted(grouped.items()):
        s = {
            "variant": variant,
            "slab_depth": slab,
            "repeats": len(rs),
            "interval_count": rs[0]["interval_count"],
            "canonical_sha256": rs[0]["canonical_sha256"],
            "median_wall_seconds": med([r["wall_seconds"] for r in rs]),
            "median_max_rss_mib": med([r["max_rss_kb"] for r in rs]) / 1024.0,
        }
        for key in (
            "hier_max_pair_nodes", "hier_final_interface_nodes",
            "hier_stream_max_pair_nodes", "hier_stream_max_pair_state_bytes",
            "hier_stream_final_interface_nodes", "hier_stream_finalized_pair_bytes",
            "hier_stream_root_attach_bytes", "hier_stream_root_interface_bytes",
            "root_total_bytes", "flat_total_run_bytes", "flat_estimated_global_total_bytes",
        ):
            vals = {r.get(key, "") for r in rs}
            if len(vals) == 1:
                s[key] = rs[0].get(key, "")
        summary.append(s)

    summary_path = a.summary_output or a.output.with_name(a.output.stem + "_summary.csv")
    sf = []
    for r in summary:
        for k in r:
            if k not in sf:
                sf.append(k)
    with summary_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=sf); w.writeheader(); w.writerows(summary)

    print("PASS: all v1.21 hierarchy profile variants are persistence-identical")
    for s in summary:
        print(
            f"{s['variant']:14s} d{s['slab_depth']:<3d} "
            f"wall={s['median_wall_seconds']:.3f}s rss={s['median_max_rss_mib']:.1f} MiB"
        )
    print(f"wrote {a.output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
