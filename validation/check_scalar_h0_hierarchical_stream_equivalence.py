#!/usr/bin/env python3
"""Validate optimized disk-backed hierarchical H0 against flat and in-memory H0."""
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

HIER_RE = re.compile(
    r"PROFILE_H0_HIER_STREAM "
    r"leaf_slabs=(?P<leaves>\d+) combines=(?P<combines>\d+) "
    r"max_live_summaries=(?P<live>\d+) max_pair_nodes=(?P<pair>\d+) "
    r"max_pair_state_bytes=(?P<state>\d+) final_interface_nodes=(?P<final>\d+) "
    r"finalized_pair_bytes=(?P<pairs>\d+) root_attach_bytes=(?P<root_attach>\d+) "
    r"root_interface_bytes=(?P<root_interface>\d+) .*?root_materialized=(?P<materialized>true|false) "
    r"disk_key_bytes=(?P<key>\d+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument(
        "--binary",
        type=Path,
        default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")),
    )
    p.add_argument("--slice-limit", type=int, default=64)
    p.add_argument("--slab-depths", type=int, nargs="+", default=(4, 8, 16))
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    return p.parse_args()


def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        header = next(rows, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence header in {path}: {header}")
        return Counter(tuple(row) for row in rows if row)


def run(binary: Path, inp: Path, work: Path, slab: int, fg: int, mode: str, extra=()):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "intervals.csv"
    cmd = [str(binary.resolve()), str(inp.resolve()), str(slab), str(fg), mode, str(out), *extra]
    cp = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
    return read_intervals(out), cp.stdout


def equal(label: str, expected, actual) -> None:
    if expected != actual:
        raise SystemExit(
            f"FAIL {label}\nmissing={(expected-actual).most_common(10)}\n"
            f"extra={(actual-expected).most_common(10)}"
        )
    print(f"PASS {label}: {sum(expected.values())} intervals agree exactly")


def main() -> None:
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")

    with tempfile.TemporaryDirectory(prefix="betti_h0_hier_stream_") as td:
        root = Path(td)
        f32 = stage_f32_subset(a.input, root / "f32_input", a.slice_limit)
        common = (
            "--f32-key-mode", "native32",
            "--h0-birth-buffer", "reuse-input",
            "--h0-event-storage", "direct",
            "--event-order", "verify",
            "--global-h0-uf-layout", "packed",
        )
        oracle, _ = run(
            a.binary, f32, root / "oracle", max(a.slab_depths),
            a.foreground_connectivity, "h0-scalar"
        )

        # Read face area from one staged TIFF without relying on filename conventions.
        try:
            import tifffile
        except ImportError as exc:
            raise SystemExit("validation requires tifffile") from exc
        first = sorted(f32.glob("*.tif"))[0]
        arr = tifffile.imread(first)
        face_area = int(arr.shape[0]) * int(arr.shape[1])

        for slab in a.slab_depths:
            flat, _ = run(
                a.binary, f32, root / f"flat_d{slab}", slab,
                a.foreground_connectivity, "h0-scalar-stream", common
            )
            hier, log = run(
                a.binary, f32, root / f"hier_d{slab}", slab,
                a.foreground_connectivity, "h0-scalar-hierarchical-stream", common
            )
            equal(f"flat d{slab} vs in-memory", oracle, flat)
            equal(f"hierarchical-stream d{slab} vs in-memory", oracle, hier)
            equal(f"hierarchical-stream d{slab} vs flat", flat, hier)

            match = HIER_RE.search(log)
            if not match:
                raise SystemExit(f"FAIL d{slab}: missing PROFILE_H0_HIER_STREAM")
            pair_nodes = int(match.group("pair"))
            final_nodes = int(match.group("final"))
            key_bytes = int(match.group("key"))
            if pair_nodes > 4 * face_area:
                raise SystemExit(
                    f"FAIL d{slab}: pair frontier {pair_nodes} exceeds 4A={4*face_area}"
                )
            root_attach = int(match.group("root_attach"))
            root_interface = int(match.group("root_interface"))
            materialized = match.group("materialized") == "true"
            staged_depth = len(list(f32.glob("*.tif")))
            leaf_count = (staged_depth + slab - 1) // slab
            if leaf_count > 1:
                if final_nodes != 0:
                    raise SystemExit(
                        f"FAIL d{slab}: terminal-free final fan-in retained {final_nodes} interface nodes"
                    )
                if root_attach != 0 or root_interface != 0:
                    raise SystemExit(
                        f"FAIL d{slab}: terminal-free final fan-in materialized root bytes "
                        f"attach={root_attach}, interface={root_interface}"
                    )
                if materialized:
                    raise SystemExit(f"FAIL d{slab}: root_materialized=true for multi-slab hierarchy")
            if key_bytes != 4:
                raise SystemExit(f"FAIL d{slab}: expected 4-byte F32 disk keys, got {key_bytes}")
            print(
                f"PASS d{slab}: hierarchical frontier={pair_nodes} <= 4A={4*face_area}, "
                f"root_materialized={materialized}, root_bytes={root_attach + root_interface}, "
                f"key_bytes={key_bytes}"
            )

    print("PASS: optimized disk-backed hierarchical H0 preserves exact persistence")


if __name__ == "__main__":
    main()
