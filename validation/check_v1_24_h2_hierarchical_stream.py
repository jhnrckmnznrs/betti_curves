#!/usr/bin/env python3
"""Exactness gate for v1.24 outside-aware hierarchical native-F32 H2."""
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
    r"attach_finalized_early=(?P<finalized>\d+)\s+attach_propagated=(?P<propagated>\d+)"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path,
                   default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slice-limit", type=int, default=64)
    p.add_argument("--slab-depths", type=int, nargs="+", default=(4, 8, 16))
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    return p.parse_args()


def ensure_binary_supports_candidate(binary: Path):
    cp = subprocess.run(
        [str(binary.resolve()), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    help_text = cp.stdout or ""
    required = (
        "h0-scalar-hierarchical-stream",
        "h2-scalar-hierarchical-stream",
    )
    missing = [mode for mode in required if mode not in help_text]
    if missing:
        raise SystemExit(
            "FAIL: the selected betti_curves binary is stale or was built from a different source tree.\n"
            f"binary: {binary.resolve()}\n"
            f"missing candidate mode(s) in --help: {', '.join(missing)}\n"
            "Rebuild target/release/betti_curves from the v1.24 candidate directory, then rerun this gate."
        )


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


def parse_profile(log: str, width: int, height: int, slab: int):
    m = PROFILE_RE.search(log)
    if not m:
        raise SystemExit(f"FAIL d{slab}: missing PROFILE_H2_HIER_STREAM")
    v = {k: int(m.group(k)) for k in (
        "leaf", "combines", "live", "pair", "state", "final", "pair_bytes",
        "root_attach", "root_outside", "root_interface", "key", "finalized", "propagated")}
    if m.group("root") != "false":
        raise SystemExit(f"FAIL d{slab}: root was materialized")
    if any(v[k] != 0 for k in ("final", "root_attach", "root_outside", "root_interface")):
        raise SystemExit(f"FAIL d{slab}: terminal-free root accounting is not zero")
    if v["key"] != 4 or m.group("birth") != "compact" or m.group("layout") != "packed":
        raise SystemExit(f"FAIL d{slab}: expected native32 compact/packed H2 storage")
    face = width * height
    if v["pair"] > 4 * face:
        raise SystemExit(f"FAIL d{slab}: pair frontier {v['pair']} exceeds 4A={4*face}")
    expected_state_bound = 8 * v["pair"] + 4  # distinguished outside adds one parent word at final fan-in
    if v["state"] > expected_state_bound:
        raise SystemExit(
            f"FAIL d{slab}: pair state {v['state']} exceeds packed/native32 bound {expected_state_bound}"
        )
    print(
        f"PASS d{slab} hierarchy profile: leaves={v['leaf']} max_pair_nodes={v['pair']} "
        f"state={v['state']/2**20:.2f} MiB attach_finalized={v['finalized']:,} "
        f"attach_propagated={v['propagated']:,}"
    )
    return v


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    ensure_binary_supports_candidate(a.binary)
    with tempfile.TemporaryDirectory(prefix="betti_v124_h2_hier_") as td:
        root = Path(td)
        f32 = stage_f32_subset(a.input, root / "f32_input", a.slice_limit)
        # Infer staged dimensions without requiring numpy/tifffile metadata helpers here.
        import tifffile
        first = tifffile.imread(sorted(f32.glob("*.tif*"))[0])
        height, width = first.shape

        # Independent in-memory oracle uses the same exact F32-valued fixture.
        oracle, _ = run(a.binary, f32, root / "oracle", max(a.slab_depths),
                        a.foreground_connectivity, "h2-scalar")
        flat, _ = run(
            a.binary, f32, root / "flat", max(a.slab_depths),
            a.foreground_connectivity, "h2-scalar-stream",
            ("--f32-key-mode", "native32", "--local-h2-birth-state", "compact",
             "--global-h2-birth-state", "compact", "--global-h2-uf-layout", "packed"),
        )
        equal("flat native32/packed H2 vs in-memory", oracle, flat)

        reference = None
        for slab in a.slab_depths:
            got, log = run(
                a.binary, f32, root / f"hier_d{slab}", slab,
                a.foreground_connectivity, "h2-scalar-hierarchical-stream",
                ("--f32-key-mode", "native32", "--local-h2-birth-state", "compact",
                 "--global-h2-birth-state", "compact", "--global-h2-uf-layout", "packed"),
            )
            equal(f"hierarchical H2 d{slab} vs in-memory", oracle, got)
            equal(f"hierarchical H2 d{slab} vs flat native32/packed", flat, got)
            if reference is not None:
                equal(f"hierarchical H2 d{slab} vs previous hierarchy depth", reference, got)
            reference = got
            parse_profile(log, width, height, slab)

    print("PASS: v1.24 outside-aware hierarchical H2 preserves exact persistence and the <=4A frontier")


if __name__ == "__main__":
    main()
