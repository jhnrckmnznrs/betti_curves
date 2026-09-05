#!/usr/bin/env python3
"""Exact H2 persistence check for compact global parent-rank vs packed UF layout."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

STATE_RE = re.compile(
    r"PROFILE_GLOBAL_H2_STATE\s+strategy=(?P<strategy>\S+)\s+uf_layout=(?P<layout>\S+)\s+"
    r"nodes=(?P<nodes>\d+)\s+parent_bytes=(?P<parent>\d+)\s+rank_bytes=(?P<rank>\d+)\s+"
    r"birth_bytes=(?P<birth>\d+)\s+total_bytes=(?P<total>\d+)"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path,
                   default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slab-depth", type=int, default=16)
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
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


def read_intervals(path: Path):
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        if next(r, None) != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence CSV header in {path}")
        return Counter(tuple(row) for row in r if row)


def run(binary: Path, input_path: Path, work: Path, slab: int, fg: int, mode: str, extra=()):
    work.mkdir(parents=True, exist_ok=True)
    output = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(output), *extra]
    cp = subprocess.run(cmd, cwd=work, text=True, capture_output=True)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")
    return read_intervals(output), cp.stdout + cp.stderr


def check(label, expected, actual):
    if expected != actual:
        raise SystemExit(
            f"FAIL {label}: persistence multisets differ\n"
            f"missing={(expected-actual).most_common(10)}\nextra={(actual-expected).most_common(10)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def parse_state(label: str, log: str, layout: str):
    m = STATE_RE.search(log)
    if not m:
        raise SystemExit(f"FAIL {label}: missing PROFILE_GLOBAL_H2_STATE")
    if m.group("strategy") != "compact" or m.group("layout") != layout:
        raise SystemExit(
            f"FAIL {label}: expected compact/{layout}, got {m.group('strategy')}/{m.group('layout')}"
        )
    state = {k: int(m.group(k)) for k in ("nodes", "parent", "rank", "birth", "total")}
    print(
        f"PASS {label}: nodes={state['nodes']} parent={state['parent']/2**20:.2f} MiB "
        f"rank={state['rank']/2**20:.2f} MiB birth={state['birth']/2**20:.2f} MiB "
        f"total={state['total']/2**20:.2f} MiB"
    )
    return state


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    require_v115_binary(a.binary)

    with tempfile.TemporaryDirectory(prefix="betti_h2_global_uf_equiv_") as td:
        root = Path(td)
        oracle, _ = run(a.binary, a.input, root / "oracle", a.slab_depth,
                        a.foreground_connectivity, "h2-scalar")
        states = {}
        reference = None
        for layout in ("parent-rank", "packed"):
            intervals, log = run(
                a.binary, a.input, root / layout, a.slab_depth, a.foreground_connectivity,
                "h2-scalar-stream",
                ("--global-h2-birth-state", "compact", "--global-h2-uf-layout", layout),
            )
            check(f"H2 compact/{layout} vs in-memory", oracle, intervals)
            if reference is None:
                reference = intervals
            else:
                check(f"H2 compact/{layout} vs compact/parent-rank", reference, intervals)
            states[layout] = parse_state(f"H2 compact/{layout}", log, layout)

    reference = states["parent-rank"]
    packed = states["packed"]
    if packed["rank"] != 0:
        raise SystemExit(f"FAIL: packed global H2 layout still allocates rank bytes: {packed['rank']}")
    if packed["parent"] != reference["parent"] or packed["birth"] != reference["birth"]:
        raise SystemExit("FAIL: packed layout unexpectedly changed parent or birth capacities")
    saved = reference["total"] - packed["total"]
    if saved <= 0:
        raise SystemExit("FAIL: packed layout did not reduce explicit global state")
    print(
        f"PASS: packed global H2 UF saves {saved/2**20:.2f} MiB "
        f"({100.0*saved/reference['total']:.2f}%) on this run"
    )


if __name__ == "__main__":
    main()
