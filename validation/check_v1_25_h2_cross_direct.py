#!/usr/bin/env python3
"""Exactness gate for v1.25 direct cross-interface consumption in hierarchical H2."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from f32_fixture import stage_f32_subset

PROFILE_RE = re.compile(
    r"PROFILE_H2_HIER_STREAM\s+"
    r"leaf_slabs=(?P<leaf>\d+)\s+combines=(?P<combines>\d+)\s+"
    r"max_live_summaries=(?P<live>\d+)\s+max_pair_nodes=(?P<pair>\d+)\s+"
    r"max_pair_state_bytes=(?P<state>\d+)\s+final_interface_nodes=(?P<final>\d+)\s+"
    r"finalized_pair_bytes=(?P<pair_bytes>\d+)\s+root_attach_bytes=(?P<root_attach>\d+)\s+"
    r"root_outside_bytes=(?P<root_outside>\d+)\s+root_interface_bytes=(?P<root_interface>\d+)\s+"
    r"root_materialized=(?P<root>\S+)\s+disk_key_bytes=(?P<key>\d+).*?"
    r"global_h2_birth_state=(?P<birth>\S+)\s+global_h2_uf_layout=(?P<layout>\S+).*?"
    r"attach_finalized_early=(?P<finalized>\d+)\s+attach_propagated=(?P<propagated>\d+)\s+"
    r"outside_propagated=(?P<outside>\d+).*?h2_hier_cross_storage=(?P<cross>\S+)"
)
CROSS_LINE_RE = re.compile(
    r"^PROFILE_H2_HIER_STREAM_(?:COMBINE|FINAL)\b.*$", re.MULTILINE
)
CROSS_RETAINED_RE = re.compile(r"\bcross_retained=(?P<retained>\d+)\b")
CROSS_STORAGE_RE = re.compile(r"\bcross_storage=(?P<storage>\S+)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path,
                   default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slice-limit", type=int, default=64)
    p.add_argument("--slab-depths", type=int, nargs="+", default=(4, 8, 16))
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    return p.parse_args()


def read_intervals(path: Path):
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        if next(r, None) != ["birth", "death"]:
            raise RuntimeError(f"bad persistence header in {path}")
        return Counter(tuple(row) for row in r if row)


def run(binary: Path, inp: Path, work: Path, slab: int, fg: int, mode: str, extra=()):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(inp.resolve()), str(slab), str(fg), mode, str(out), *extra]
    cp = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
    return read_intervals(out), cp.stdout


def equal(label, expected, actual):
    if expected != actual:
        raise SystemExit(
            f"FAIL {label}: persistence multisets differ\n"
            f"missing={(expected-actual).most_common(12)}\n"
            f"extra={(actual-expected).most_common(12)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def parse_profile(log: str, width: int, height: int, slab: int, expected_cross: str):
    m = PROFILE_RE.search(log)
    if not m:
        raise SystemExit(f"FAIL d{slab} {expected_cross}: missing PROFILE_H2_HIER_STREAM")
    if m.group("cross") != expected_cross:
        raise SystemExit(
            f"FAIL d{slab}: requested cross storage {expected_cross}, got {m.group('cross')}"
        )
    vals = {k: int(m.group(k)) for k in (
        "leaf", "combines", "live", "pair", "state", "final", "pair_bytes",
        "root_attach", "root_outside", "root_interface", "key", "finalized",
        "propagated", "outside")}
    if m.group("root") != "false":
        raise SystemExit(f"FAIL d{slab}: root was materialized")
    if any(vals[k] != 0 for k in ("final", "root_attach", "root_outside", "root_interface")):
        raise SystemExit(f"FAIL d{slab}: terminal-free root accounting is not zero")
    if vals["key"] != 4 or m.group("birth") != "compact" or m.group("layout") != "packed":
        raise SystemExit(f"FAIL d{slab}: expected native32 compact/packed H2 storage")
    face = width * height
    if vals["pair"] > 4 * face:
        raise SystemExit(f"FAIL d{slab}: pair frontier {vals['pair']} exceeds 4A={4*face}")
    if vals["state"] > 8 * vals["pair"] + 4:
        raise SystemExit(f"FAIL d{slab}: pair state exceeds packed/native32 bound")
    cross = []
    for line in CROSS_LINE_RE.findall(log):
        retained_match = CROSS_RETAINED_RE.search(line)
        if not retained_match:
            continue
        storage_match = CROSS_STORAGE_RE.search(line)
        # v1.24/v1.25 disk records predate the explicit cross_storage tag.
        # Direct records are always tagged, so an untagged record is safely
        # interpreted as the legacy disk path.
        storage = storage_match.group("storage") if storage_match else "disk"
        cross.append((int(retained_match.group("retained")), storage))
    if not cross:
        raise SystemExit(f"FAIL d{slab}: no hierarchical cross profile records")
    if any(storage != expected_cross for _, storage in cross):
        raise SystemExit(f"FAIL d{slab}: mixed cross storage records: {cross[:5]}")
    retained = sum(value for value, _ in cross)
    print(
        f"PASS d{slab} {expected_cross}: max_pair_nodes={vals['pair']} "
        f"attach_finalized={vals['finalized']:,} attach_propagated={vals['propagated']:,} "
        f"outside_propagated={vals['outside']:,} cross_retained={retained:,}"
    )


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    cp = subprocess.run([str(a.binary.resolve()), "--help"], text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if "--h2-hier-cross-storage" not in (cp.stdout or ""):
        raise SystemExit(
            "FAIL: selected binary does not expose --h2-hier-cross-storage; rebuild v1.25"
        )
    with tempfile.TemporaryDirectory(prefix="betti_v125_h2_cross_") as td:
        root = Path(td)
        f32 = stage_f32_subset(a.input, root / "f32_input", a.slice_limit)
        import tifffile
        first = tifffile.imread(sorted(f32.glob("*.tif*"))[0])
        height, width = first.shape
        oracle, _ = run(a.binary, f32, root / "oracle", max(a.slab_depths),
                        a.foreground_connectivity, "h2-scalar")
        common = ("--f32-key-mode", "native32", "--local-h2-birth-state", "compact",
                  "--global-h2-birth-state", "compact", "--global-h2-uf-layout", "packed")
        for slab in a.slab_depths:
            disk, log_disk = run(
                a.binary, f32, root / f"d{slab}_disk", slab, a.foreground_connectivity,
                "h2-scalar-hierarchical-stream", (*common, "--h2-hier-cross-storage", "disk"),
            )
            direct, log_direct = run(
                a.binary, f32, root / f"d{slab}_direct", slab, a.foreground_connectivity,
                "h2-scalar-hierarchical-stream", (*common, "--h2-hier-cross-storage", "direct"),
            )
            equal(f"d{slab} disk vs in-memory", oracle, disk)
            equal(f"d{slab} direct vs in-memory", oracle, direct)
            equal(f"d{slab} direct vs disk", disk, direct)
            parse_profile(log_disk, width, height, slab, "disk")
            parse_profile(log_direct, width, height, slab, "direct")
    print("PASS: v1.25 direct hierarchical H2 cross consumption preserves exact persistence")


if __name__ == "__main__":
    main()
