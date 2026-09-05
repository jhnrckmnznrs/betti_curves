#!/usr/bin/env python3
"""Paired CX09T1 profile for v1.23 elder-dominated hierarchical attach pruning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from collections import Counter

from sys import path as sys_path
sys_path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from f32_fixture import stage_f32_subset  # noqa: E402

TIME_FIELDS = {
    "User time (seconds)": "user_seconds",
    "System time (seconds)": "system_seconds",
    "Maximum resident set size (kbytes)": "max_rss_kb",
}
COMBINE_RE = re.compile(
    r"PROFILE_H0_HIER_STREAM_COMBINE .*?parent_attach_bytes=(?P<attach>\d+) "
    r"parent_interface_bytes=(?P<interface>\d+) .*?attach_finalized_early=(?P<finalized>\d+) "
    r"attach_propagated=(?P<propagated>\d+) attach_pruning=(?P<strategy>\S+)"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--slice-limit", type=int, default=0, help="0 = all CX09T1 slices")
    p.add_argument("--slab-depths", type=int, nargs="+", default=(8, 16, 32))
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("cx09t1_v1_23_attach_pruning.csv"))
    return p.parse_args()


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


def persistence_digests(path: Path):
    """Return ordered and canonical multiset digests plus the exact row Counter.

    The hierarchical pruning candidate may finalize an interval at an earlier
    fan-in level than the reference path.  That legitimately changes CSV row
    emission order, so the ordered file digest is diagnostic only.  Exactness
    is decided from the Counter and the canonical digest below.
    """
    ordered = hashlib.sha256()
    counts: Counter[tuple[str, str]] = Counter()
    rows = 0
    with path.open("r", newline="", encoding="utf-8") as h:
        reader = csv.reader(h)
        if next(reader, None) != ["birth", "death"]:
            raise RuntimeError("bad persistence header")
        for row in reader:
            if not row:
                continue
            if len(row) != 2:
                raise RuntimeError(f"unexpected persistence row: {row!r}")
            birth, death = row
            raw = f"{birth},{death}\n".encode("utf-8")
            ordered.update(raw)
            counts[(birth, death)] += 1
            rows += 1

    canonical = hashlib.sha256()
    for (birth, death), count in sorted(counts.items()):
        canonical.update(f"{birth},{death},{count}\n".encode("utf-8"))
    return ordered.hexdigest(), canonical.hexdigest(), rows, counts


def kv_line(log: str, prefix: str):
    line = next((x for x in log.splitlines() if x.startswith(prefix)), None)
    if not line:
        return {}
    out = {}
    for token in line[len(prefix):].split():
        if "=" in token:
            k, v = token.split("=", 1); out[k] = v
    return out


def run_one(a, fixture: Path, root: Path, slab: int, strategy: str, repeat: int):
    tag = f"{strategy.replace('-', '_')}_d{slab}_r{repeat}"
    work = root / tag; work.mkdir(parents=True)
    out = work / "intervals.csv"; timing = work / "time.txt"; logp = work / "run.log"
    cmd = [
        "/usr/bin/time", "-v", "-o", str(timing), str(a.binary.resolve()),
        str(fixture.resolve()), str(slab), str(a.foreground_connectivity),
        "h0-scalar-hierarchical-stream", str(out),
        "--f32-key-mode", "native32",
        "--h0-birth-buffer", "reuse-input",
        "--h0-event-storage", "direct",
        "--event-order", "verify",
        "--global-h0-uf-layout", "packed",
        "--h0-hier-attach-pruning", strategy,
    ]
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    logp.write_text(cp.stdout, encoding="utf-8")
    if cp.returncode:
        raise RuntimeError(f"{tag} failed:\n{cp.stdout[-6000:]}")
    ordered_digest, canonical_digest, rows, counts = persistence_digests(out)
    profile = kv_line(cp.stdout, "PROFILE_H0_HIER_STREAM ")
    config = kv_line(cp.stdout, "PROFILE_CONFIG scalar_h0_hierarchical_stream ")
    combines = list(COMBINE_RE.finditer(cp.stdout))
    row = {
        "strategy": strategy, "slab_depth": slab, "repeat": repeat,
        "wall_seconds": wall, "interval_count": rows,
        "output_sha256_ordered": ordered_digest,
        "output_sha256_canonical": canonical_digest,
        "aggregate_parent_attach_bytes": sum(int(m.group("attach")) for m in combines),
        "aggregate_parent_interface_bytes": sum(int(m.group("interface")) for m in combines),
        "combine_attach_finalized_early": sum(int(m.group("finalized")) for m in combines),
        "combine_attach_propagated": sum(int(m.group("propagated")) for m in combines),
    }
    row.update(parse_time(timing))
    for k, v in profile.items(): row[f"hier_{k}"] = v
    for k, v in config.items(): row[f"config_{k}"] = v
    if config.get("h0_hier_attach_pruning") != strategy:
        raise RuntimeError(f"{tag}: pruning configuration mismatch")
    if row.get("max_rss_kb"):
        row["max_rss_mib"] = row["max_rss_kb"] / 1024.0
    return row, counts


def write(path: Path, rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    a=parse_args()
    if not a.binary.is_file(): raise SystemExit(f"binary not found: {a.binary}")
    rows=[]
    with tempfile.TemporaryDirectory(prefix="betti_v123_profile_") as td:
        root=Path(td)
        fixture=stage_f32_subset(a.input, root/"f32", a.slice_limit)
        for slab in a.slab_depths:
            hashes={}
            counters={}
            for repeat in range(1, a.repeats+1):
                # Rotate order by repeat to reduce cache-order bias.
                order=("off","elder-dominated") if repeat % 2 else ("elder-dominated","off")
                for strategy in order:
                    print(f"RUN {strategy} d={slab} repeat={repeat}", flush=True)
                    row, counts = run_one(a,fixture,root,slab,strategy,repeat); rows.append(row); write(a.output,rows)
                    hashes.setdefault(strategy,set()).add((row["interval_count"],row["output_sha256_canonical"]))
                    counters.setdefault(strategy, []).append(counts)
            if len(hashes.get("off",())) != 1 or len(hashes.get("elder-dominated",())) != 1 or hashes["off"] != hashes["elder-dominated"]:
                off_counter = counters.get("off", [Counter()])[0]
                pruned_counter = counters.get("elder-dominated", [Counter()])[0]
                raise SystemExit(
                    f"FAIL d{slab}: canonical persistence differs\n"
                    f"missing={(off_counter-pruned_counter).most_common(10)}\n"
                    f"extra={(pruned_counter-off_counter).most_common(10)}\n"
                    f"canonical_hashes={hashes}"
                )
    print(f"PASS: v1.23 attach-pruning profile preserved exact persistence; wrote {a.output}")


if __name__ == "__main__": main()
