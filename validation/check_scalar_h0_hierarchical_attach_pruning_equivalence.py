#!/usr/bin/env python3
"""Validate elder-dominated hierarchical H0 attach pruning exactly on a staged F32 CX09T1 subset."""
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
    r"PROFILE_H0_HIER_STREAM .*?attach_finalized_early=(?P<finalized>\d+) "
    r"attach_propagated=(?P<propagated>\d+) h0_hier_attach_pruning=(?P<strategy>\S+)"
)
COMBINE_RE = re.compile(
    r"PROFILE_H0_HIER_STREAM_COMBINE .*?parent_attach_bytes=(?P<attach>\d+) "
    r"parent_interface_bytes=(?P<interface>\d+) .*?attach_finalized_early=(?P<finalized>\d+) "
    r"attach_propagated=(?P<propagated>\d+) attach_pruning=(?P<strategy>\S+)"
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slice-limit", type=int, default=64)
    p.add_argument("--slab-depths", type=int, nargs="+", default=(4, 8, 16))
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    return p.parse_args()


def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        if next(r, None) != ["birth", "death"]:
            raise RuntimeError(f"unexpected header in {path}")
        return Counter(tuple(row) for row in r if row)


def run(binary: Path, inp: Path, work: Path, slab: int, fg: int, strategy: str):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "intervals.csv"
    cmd = [
        str(binary.resolve()), str(inp.resolve()), str(slab), str(fg),
        "h0-scalar-hierarchical-stream", str(out),
        "--f32-key-mode", "native32",
        "--h0-birth-buffer", "reuse-input",
        "--h0-event-storage", "direct",
        "--event-order", "verify",
        "--global-h0-uf-layout", "packed",
        "--h0-hier-attach-pruning", strategy,
    ]
    cp = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if cp.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{cp.stdout}")
    return read_intervals(out), cp.stdout


def equal(label: str, a, b) -> None:
    if a != b:
        raise SystemExit(
            f"FAIL {label}\nmissing={(a-b).most_common(10)}\nextra={(b-a).most_common(10)}"
        )
    print(f"PASS {label}: {sum(a.values())} intervals agree exactly")


def stats(log: str, expected: str):
    m = PROFILE_RE.search(log)
    if not m:
        raise SystemExit("FAIL: missing attach-pruning PROFILE_H0_HIER_STREAM fields")
    if m.group("strategy") != expected:
        raise SystemExit(f"FAIL: expected strategy {expected}, got {m.group('strategy')}")
    combine = list(COMBINE_RE.finditer(log))
    return {
        "finalized": int(m.group("finalized")),
        "propagated": int(m.group("propagated")),
        "parent_attach_bytes": sum(int(x.group("attach")) for x in combine),
        "parent_interface_bytes": sum(int(x.group("interface")) for x in combine),
        "combines": len(combine),
    }


def main() -> None:
    a = args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    with tempfile.TemporaryDirectory(prefix="betti_h0_attach_prune_") as td:
        root = Path(td)
        f32 = stage_f32_subset(a.input, root / "f32", a.slice_limit)
        for slab in a.slab_depths:
            off, off_log = run(a.binary, f32, root / f"off_d{slab}", slab, a.foreground_connectivity, "off")
            pruned, pruned_log = run(a.binary, f32, root / f"pruned_d{slab}", slab, a.foreground_connectivity, "elder-dominated")
            equal(f"elder-dominated d{slab} vs off", off, pruned)
            so = stats(off_log, "off")
            sp = stats(pruned_log, "elder-dominated")
            if sp["finalized"] <= 0:
                raise SystemExit(f"FAIL d{slab}: elder-dominated pruning did not finalize any attach events")
            if sp["propagated"] >= so["propagated"]:
                raise SystemExit(
                    f"FAIL d{slab}: propagated attaches did not decrease: off={so['propagated']} pruned={sp['propagated']}"
                )
            print(
                f"PASS d{slab}: early_finalized={sp['finalized']:,}, "
                f"propagated {so['propagated']:,}->{sp['propagated']:,}, "
                f"aggregate parent attach bytes {so['parent_attach_bytes']/2**20:.2f}->{sp['parent_attach_bytes']/2**20:.2f} MiB"
            )
    print("PASS: elder-dominated hierarchical H0 attach pruning preserves exact persistence")


if __name__ == "__main__":
    main()
