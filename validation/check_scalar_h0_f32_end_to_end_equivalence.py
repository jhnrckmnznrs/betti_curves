#!/usr/bin/env python3
"""Validate end-to-end native32 H0 streaming against legacy64 and in-memory H0."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

STORAGE_RE = re.compile(
    r"PROFILE_H0_STORAGE\s+scalar_h0_stream\s+"
    r"pipeline=(?P<pipeline>\S+)\s+"
    r"disk_key_bytes=(?P<disk_key_bytes>\d+)\s+"
    r"interface_nodes=(?P<interface_nodes>\d+)\s+"
    r"interface_birth_bytes=(?P<interface_birth_bytes>\d+)\s+"
    r"local_pair_bytes=(?P<local_pair_bytes>\d+)\s+"
    r"attach_bytes=(?P<attach_bytes>\d+)\s+"
    r"interface_bytes=(?P<interface_bytes>\d+)\s+"
    r"cross_bytes=(?P<cross_bytes>\d+)\s+"
    r"total_run_bytes=(?P<total_run_bytes>\d+)\s+"
    r"estimated_global_parent_bytes=(?P<global_parent_bytes>\d+)\s+"
    r"estimated_global_rank_bytes=(?P<global_rank_bytes>\d+)\s+"
    r"estimated_global_birth_bytes=(?P<global_birth_bytes>\d+)\s+"
    r"estimated_global_total_bytes=(?P<global_total_bytes>\d+)"
)
PIXEL_RE = re.compile(r"source pixel type:\s*(?P<pixel>\S+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="F32 TIFF-stack directory")
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slab-depth", type=int, default=1)
    p.add_argument("--slice-limit", type=int, default=2, help="stage only the first N TIFF slices; 0 uses the full stack")
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    return p.parse_args()



def stage_input_subset(input_path: Path, root: Path, limit: int) -> Path:
    if limit == 0:
        return input_path
    if limit < 0:
        raise SystemExit("--slice-limit must be nonnegative")
    slices = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if len(slices) < limit:
        raise SystemExit(f"requested {limit} slices but found only {len(slices)} TIFF files")
    staged = root / "input_subset"
    staged.mkdir(parents=True, exist_ok=True)
    for path in slices[:limit]:
        (staged / path.name).symlink_to(path.resolve())
    return staged

def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected header in {path}: {header!r}")
        return Counter(tuple(row) for row in reader if row)


def run(binary: Path, input_path: Path, work: Path, slab: int, fg: int, mode: str, extra: list[str] | None = None):
    work.mkdir(parents=True, exist_ok=True)
    output = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(output)]
    if extra:
        cmd.extend(extra)
    completed = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{completed.stdout}")
    return read_intervals(output), completed.stdout


def assert_equal(label: str, expected: Counter[tuple[str, str]], observed: Counter[tuple[str, str]]) -> None:
    if expected != observed:
        missing = expected - observed
        extra = observed - expected
        raise SystemExit(
            f"FAIL {label}: persistence multisets differ\n"
            f"missing={missing.most_common(10)}\nextra={extra.most_common(10)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def parse_storage(label: str, log: str, expected_key_bytes: int, expected_pipeline: str) -> dict[str, int]:
    pixel = PIXEL_RE.search(log)
    if not pixel or pixel.group("pixel") != "F32":
        raise SystemExit(f"FAIL {label}: expected source pixel type F32")
    match = STORAGE_RE.search(log)
    if not match:
        raise SystemExit(f"FAIL {label}: missing PROFILE_H0_STORAGE")
    if match.group("pipeline") != expected_pipeline:
        raise SystemExit(
            f"FAIL {label}: expected pipeline {expected_pipeline}, got {match.group('pipeline')}"
        )
    key_bytes = int(match.group("disk_key_bytes"))
    if key_bytes != expected_key_bytes:
        raise SystemExit(f"FAIL {label}: expected {expected_key_bytes}-byte disk keys, got {key_bytes}")
    values = {name: int(value) for name, value in match.groupdict().items() if name != "pipeline"}
    nodes = values["interface_nodes"]
    if values["interface_birth_bytes"] != nodes * key_bytes:
        raise SystemExit(f"FAIL {label}: interface birth file size does not equal nodes * key bytes")
    expected_global = nodes * (4 + 1 + key_bytes)
    if values["global_total_bytes"] != expected_global:
        raise SystemExit(
            f"FAIL {label}: expected global state {expected_global}, got {values['global_total_bytes']}"
        )
    print(
        f"PASS {label}: disk_key_bytes={key_bytes} "
        f"interface_birth={values['interface_birth_bytes']/2**20:.2f} MiB "
        f"global_state={values['global_total_bytes']/2**20:.2f} MiB"
    )
    return values


def main() -> None:
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    if not a.input.is_dir():
        raise SystemExit(f"input stack not found: {a.input}")

    fixed = [
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
        "--global-h0-uf-layout", "parent-rank",
    ]

    with tempfile.TemporaryDirectory(prefix="betti_h0_f32_e2e_equiv_") as td:
        root = Path(td)
        test_input = stage_input_subset(a.input, root, a.slice_limit)
        oracle, _ = run(a.binary, test_input, root / "oracle", a.slab_depth, a.foreground_connectivity, "h0-scalar")
        legacy, legacy_log = run(
            a.binary, test_input, root / "legacy64", a.slab_depth, a.foreground_connectivity,
            "h0-scalar-stream", [*fixed, "--f32-key-mode", "legacy64"],
        )
        native, native_log = run(
            a.binary, test_input, root / "native32", a.slab_depth, a.foreground_connectivity,
            "h0-scalar-stream", [*fixed, "--f32-key-mode", "native32"],
        )

        assert_equal("H0 legacy64 vs in-memory", oracle, legacy)
        assert_equal("H0 native32 end-to-end vs in-memory", oracle, native)
        assert_equal("H0 native32 end-to-end vs legacy64", legacy, native)
        legacy_state = parse_storage("legacy64", legacy_log, 8, "wide64")
        native_state = parse_storage("native32", native_log, 4, "native32-end-to-end")

        if legacy_state["interface_nodes"] != native_state["interface_nodes"]:
            raise SystemExit("FAIL: native32 changed the interface-node count")
        if native_state["global_total_bytes"] >= legacy_state["global_total_bytes"]:
            raise SystemExit("FAIL: native32 did not reduce estimated global H0 state")
        saved = legacy_state["global_total_bytes"] - native_state["global_total_bytes"]
        print(
            f"PASS: native32 removes {saved/2**20:.2f} MiB "
            f"({100.0*saved/legacy_state['global_total_bytes']:.2f}%) from explicit global H0 state"
        )

    print("PASS: end-to-end native F32 H0 storage preserves exact persistence")


if __name__ == "__main__":
    main()
