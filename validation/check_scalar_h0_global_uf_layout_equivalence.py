#!/usr/bin/env python3
"""Validate packed global H0 union-find against parent-rank and in-memory H0."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

PIXEL_RE = re.compile(r"source pixel type:\s*(?P<pixel>\S+)")
STATE_RE = re.compile(
    r"PROFILE_GLOBAL_H0_STATE\s+scalar_h0_stream\s+"
    r"layout=(?P<layout>\S+)\s+nodes=(?P<nodes>\d+)\s+"
    r"parent_bytes=(?P<parent>\d+)\s+rank_bytes=(?P<rank>\d+)\s+"
    r"birth_bytes=(?P<birth>\d+)\s+total_bytes=(?P<total>\d+)"
)
STORAGE_RE = re.compile(
    r"PROFILE_H0_STORAGE\s+scalar_h0_stream\s+"
    r"pipeline=(?P<pipeline>\S+)\s+disk_key_bytes=(?P<key_bytes>\d+)\s+"
    r"interface_nodes=(?P<nodes>\d+).*?"
    r"estimated_global_total_bytes=(?P<planned_total>\d+)\s+"
    r"global_h0_uf_layout=(?P<layout>\S+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="F32 TIFF-stack directory")
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slab-depth", type=int, default=1)
    p.add_argument("--slice-limit", type=int, default=2)
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    return p.parse_args()


def stage_subset(src: Path, root: Path, limit: int) -> Path:
    if limit == 0:
        return src
    slices = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    if len(slices) < limit:
        raise SystemExit(f"requested {limit} slices but found {len(slices)}")
    dst = root / "input_subset"
    dst.mkdir()
    for path in slices[:limit]:
        (dst / path.name).symlink_to(path.resolve())
    return dst


def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        if next(r, None) != ["birth", "death"]:
            raise RuntimeError(f"unexpected header in {path}")
        return Counter(tuple(row) for row in r if row)


def run(binary: Path, input_path: Path, work: Path, slab: int, fg: int, mode: str, extra: list[str] | None = None):
    work.mkdir(parents=True)
    output = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(output)]
    if extra:
        cmd.extend(extra)
    cp = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
    return read_intervals(output), cp.stdout


def check_equal(label: str, expected: Counter[tuple[str, str]], observed: Counter[tuple[str, str]]) -> None:
    if expected != observed:
        raise SystemExit(
            f"FAIL {label}: persistence differs\n"
            f"missing={(expected-observed).most_common(10)}\n"
            f"extra={(observed-expected).most_common(10)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def parse_state(label: str, log: str, layout: str) -> dict[str, int]:
    pixel = PIXEL_RE.search(log)
    if not pixel or pixel.group("pixel") != "F32":
        raise SystemExit(f"FAIL {label}: expected source pixel type F32")
    storage = STORAGE_RE.search(log)
    if not storage:
        raise SystemExit(f"FAIL {label}: missing PROFILE_H0_STORAGE")
    if storage.group("pipeline") != "native32-end-to-end" or int(storage.group("key_bytes")) != 4:
        raise SystemExit(f"FAIL {label}: expected native32-end-to-end with 4-byte disk keys")
    if storage.group("layout") != layout:
        raise SystemExit(f"FAIL {label}: storage reports layout={storage.group('layout')}, expected {layout}")
    state = STATE_RE.search(log)
    if not state:
        raise SystemExit(f"FAIL {label}: missing PROFILE_GLOBAL_H0_STATE")
    if state.group("layout") != layout:
        raise SystemExit(f"FAIL {label}: state reports layout={state.group('layout')}, expected {layout}")
    values = {k: int(state.group(k)) for k in ("nodes", "parent", "rank", "birth", "total")}
    if values["total"] != values["parent"] + values["rank"] + values["birth"]:
        raise SystemExit(f"FAIL {label}: state byte accounting is inconsistent")
    print(
        f"PASS {label}: nodes={values['nodes']} parent={values['parent']/2**20:.2f} MiB "
        f"rank={values['rank']/2**20:.2f} MiB birth={values['birth']/2**20:.2f} MiB "
        f"total={values['total']/2**20:.2f} MiB"
    )
    return values


def main() -> None:
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    help_cp = subprocess.run([str(a.binary.resolve()), "--help"], text=True, capture_output=True)
    if "--global-h0-uf-layout" not in help_cp.stdout + help_cp.stderr:
        raise SystemExit("stale binary: rebuild v1.19 candidate; --global-h0-uf-layout is missing")

    fixed = [
        "--f32-key-mode", "native32",
        "--merge-strategy", "scan",
        "--interface-order", "radix",
        "--event-order", "verify",
        "--neighbor-kernel", "interior-fast",
        "--representative-active-check", "recheck",
        "--union-kernel", "root-carrying",
        "--h0-pruning-cache", "64k",
        "--active-state", "separate",
        "--interface-state", "root-invariant",
        "--uf-layout", "packed",
    ]

    with tempfile.TemporaryDirectory(prefix="betti_h0_global_uf_equiv_") as td:
        root = Path(td)
        test_input = stage_subset(a.input, root, a.slice_limit)
        oracle, _ = run(a.binary, test_input, root / "oracle", a.slab_depth, a.foreground_connectivity, "h0-scalar")
        results = {}
        states = {}
        for layout in ("parent-rank", "packed"):
            intervals, log = run(
                a.binary,
                test_input,
                root / layout,
                a.slab_depth,
                a.foreground_connectivity,
                "h0-scalar-stream",
                [*fixed, "--global-h0-uf-layout", layout],
            )
            check_equal(f"H0 native32/{layout} vs in-memory", oracle, intervals)
            results[layout] = intervals
            states[layout] = parse_state(f"H0 native32/{layout}", log, layout)

        check_equal("H0 packed vs parent-rank", results["parent-rank"], results["packed"])
        ref = states["parent-rank"]
        packed = states["packed"]
        if packed["nodes"] != ref["nodes"] or packed["parent"] != ref["parent"] or packed["birth"] != ref["birth"]:
            raise SystemExit("FAIL: packed layout changed node, parent, or birth allocation")
        if ref["rank"] <= 0 or packed["rank"] != 0:
            raise SystemExit("FAIL: packed layout did not eliminate the separate rank vector")
        saved = ref["total"] - packed["total"]
        if saved <= 0:
            raise SystemExit("FAIL: packed layout did not reduce global H0 state")
        print(f"PASS: packed global H0 saves {saved/2**20:.2f} MiB ({100*saved/ref['total']:.2f}%)")

    print("PASS: packed global H0 UF preserves exact persistence")


if __name__ == "__main__":
    main()
