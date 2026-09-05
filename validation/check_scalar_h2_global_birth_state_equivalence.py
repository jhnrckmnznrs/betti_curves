#!/usr/bin/env python3
"""Exact H2 persistence check for tagged vs compact global birth state."""
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
    r"PROFILE_GLOBAL_H2_STATE\s+strategy=(?P<strategy>\S+)\s+uf_layout=(?P<uf_layout>\S+)\s+nodes=(?P<nodes>\d+)\s+"
    r"parent_bytes=(?P<parent>\d+)\s+rank_bytes=(?P<rank>\d+)\s+"
    r"birth_bytes=(?P<birth>\d+)\s+total_bytes=(?P<total>\d+)"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument(
        "--binary",
        type=Path,
        default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")),
    )
    p.add_argument("--slab-depth", type=int, default=16)
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    return p.parse_args()


def read_intervals(path: Path):
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        header = next(r, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected header {header!r} in {path}")
        return Counter(tuple(row) for row in r if row)


def run(binary: Path, input_path: Path, work: Path, slab: int, fg: int, mode: str, extra=None):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(out)]
    if extra:
        cmd.extend(extra)
    cp = subprocess.run(cmd, cwd=work, text=True, capture_output=True)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}{cp.stderr}")
    return read_intervals(out), cp.stdout + cp.stderr


def check(label: str, expected, actual):
    if expected != actual:
        raise SystemExit(
            f"FAIL {label}: persistence multisets differ\n"
            f"missing={(expected-actual).most_common(10)}\n"
            f"extra={(actual-expected).most_common(10)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def parse_state(label: str, log: str, expected_strategy: str):
    m = STATE_RE.search(log)
    if not m:
        raise SystemExit(f"FAIL {label}: missing PROFILE_GLOBAL_H2_STATE line")
    if m.group("strategy") != expected_strategy:
        raise SystemExit(
            f"FAIL {label}: expected strategy {expected_strategy}, got {m.group('strategy')}"
        )
    result = {k: int(m.group(k)) for k in ("nodes", "parent", "rank", "birth", "total")}
    print(
        f"PASS {label}: strategy={expected_strategy} nodes={result['nodes']} "
        f"birth={result['birth'] / 2**20:.2f} MiB total={result['total'] / 2**20:.2f} MiB"
    )
    return result


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")

    with tempfile.TemporaryDirectory(prefix="betti_h2_global_birth_equiv_") as td:
        root = Path(td)
        oracle, _ = run(
            a.binary,
            a.input,
            root / "oracle",
            a.slab_depth,
            a.foreground_connectivity,
            "h2-scalar",
        )

        states = {}
        reference = None
        for strategy in ("tagged", "compact"):
            intervals, log = run(
                a.binary,
                a.input,
                root / strategy,
                a.slab_depth,
                a.foreground_connectivity,
                "h2-scalar-stream",
                ["--global-h2-birth-state", strategy],
            )
            check(f"H2 {strategy} vs in-memory", oracle, intervals)
            if reference is None:
                reference = intervals
            else:
                check(f"H2 {strategy} vs tagged stream", reference, intervals)
            states[strategy] = parse_state(f"H2 {strategy}", log, strategy)

    tagged = states["tagged"]
    compact = states["compact"]
    if compact["birth"] >= tagged["birth"]:
        raise SystemExit(
            f"FAIL: compact birth state did not shrink: tagged={tagged['birth']} compact={compact['birth']}"
        )
    saved = tagged["total"] - compact["total"]
    print(
        f"PASS: compact global H2 state saves {saved / 2**20:.2f} MiB "
        f"({100.0 * saved / tagged['total']:.2f}%) on this run"
    )


if __name__ == "__main__":
    main()
