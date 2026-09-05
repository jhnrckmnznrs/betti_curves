#!/usr/bin/env python3
"""Exact H2 persistence check for tagged vs compact local H2 birth storage."""
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
    r"PROFILE_LOCAL_H2_STATE\s+strategy=(?P<strategy>\S+)\s+nodes=(?P<nodes>\d+)\s+"
    r"key_bytes=(?P<key_bytes>\d+)\s+parent_bytes=(?P<parent>\d+)\s+rank_bytes=(?P<rank>\d+)\s+"
    r"birth_bytes=(?P<birth>\d+)\s+total_bytes=(?P<total>\d+)"
)
SOURCE_TYPE_RE = re.compile(r"source pixel type:\s*(?P<pixel_type>\S+)")
CONFIG_RE = re.compile(r"PROFILE_CONFIG\s+scalar_h2_stream\s+.*?f32_key_mode=(?P<f32_key_mode>\S+)")


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


def require_v116_binary(binary: Path):
    cp = subprocess.run([str(binary.resolve()), "--help"], text=True, capture_output=True)
    help_text = cp.stdout + cp.stderr
    required = ("--local-h2-birth-state", "--global-h2-birth-state", "--global-h2-uf-layout", "--f32-key-mode")
    missing = [option for option in required if option not in help_text]
    if cp.returncode != 0 or missing:
        missing_text = ", ".join(missing) if missing else "v1.16 options"
        raise SystemExit(
            f"stale/non-v1.16 release binary: {binary.resolve()} does not advertise {missing_text}.\n"
            "Rebuild from this source tree with:\n"
            "  cargo build --release --locked\n"
            "Then verify:\n"
            "  target/release/betti_curves --help | grep local-h2-birth-state"
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
    cmd = [
        str(binary.resolve()),
        str(input_path.resolve()),
        str(slab),
        str(fg),
        mode,
        str(output),
        *extra,
    ]
    cp = subprocess.run(cmd, cwd=work, text=True, capture_output=True)
    if cp.returncode:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
        )
    return read_intervals(output), cp.stdout + cp.stderr


def check(label, expected, actual):
    if expected != actual:
        raise SystemExit(
            f"FAIL {label}: persistence multisets differ\n"
            f"missing={(expected-actual).most_common(10)}\n"
            f"extra={(actual-expected).most_common(10)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def parse_state(label: str, log: str, strategy: str):
    source = SOURCE_TYPE_RE.search(log)
    if not source:
        raise SystemExit(f"FAIL {label}: missing scalar source pixel type")
    config = CONFIG_RE.search(log)
    if not config:
        raise SystemExit(f"FAIL {label}: missing PROFILE_CONFIG scalar_h2_stream")
    if config.group("f32_key_mode") != "native32":
        raise SystemExit(
            f"FAIL {label}: expected f32_key_mode=native32, got {config.group('f32_key_mode')}"
        )
    m = STATE_RE.search(log)
    if not m:
        raise SystemExit(f"FAIL {label}: missing PROFILE_LOCAL_H2_STATE")
    if m.group("strategy") != strategy:
        raise SystemExit(
            f"FAIL {label}: expected local birth strategy {strategy}, got {m.group('strategy')}"
        )
    state = {
        key: int(m.group(key))
        for key in ("nodes", "key_bytes", "parent", "rank", "birth", "total")
    }
    state["source_pixel_type"] = source.group("pixel_type")
    if state["source_pixel_type"] == "F32" and state["key_bytes"] != 4:
        raise SystemExit(
            f"FAIL {label}: F32 input reported key_bytes={state['key_bytes']}; native-width path was not used"
        )
    print(
        f"PASS {label}: nodes={state['nodes']} key={state['key_bytes']} B "
        f"parent={state['parent']/2**20:.2f} MiB rank={state['rank']/2**20:.2f} MiB "
        f"birth={state['birth']/2**20:.2f} MiB total={state['total']/2**20:.2f} MiB"
    )
    return state


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    require_v116_binary(a.binary)

    with tempfile.TemporaryDirectory(prefix="betti_h2_local_birth_equiv_") as td:
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
                (
                    "--f32-key-mode",
                    "native32",
                    "--local-h2-birth-state",
                    strategy,
                    "--global-h2-birth-state",
                    "compact",
                    "--global-h2-uf-layout",
                    "parent-rank",
                ),
            )
            check(f"H2 local {strategy} vs in-memory", oracle, intervals)
            if reference is None:
                reference = intervals
            else:
                check(f"H2 local {strategy} vs tagged", reference, intervals)
            states[strategy] = parse_state(f"H2 local {strategy}", log, strategy)

    tagged = states["tagged"]
    compact = states["compact"]
    if compact["parent"] != tagged["parent"] or compact["rank"] != tagged["rank"]:
        raise SystemExit("FAIL: local birth-state ablation unexpectedly changed UF parent/rank capacities")
    if compact["birth"] >= tagged["birth"]:
        raise SystemExit("FAIL: compact local H2 birth storage did not reduce birth allocation")
    saved = tagged["total"] - compact["total"]
    print(
        f"PASS: compact local H2 births save {saved/2**20:.2f} MiB "
        f"({100.0*saved/tagged['total']:.2f}% of explicit local UF state)"
    )


if __name__ == "__main__":
    main()
