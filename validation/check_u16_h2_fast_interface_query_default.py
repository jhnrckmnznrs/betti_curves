#!/usr/bin/env python3
"""Regression gate for the v23.6 default-on U16 H2 fast interface query."""
from __future__ import annotations

import argparse
import collections
import csv
import os
import subprocess
import tempfile
from pathlib import Path

Interval = tuple[str, str]


def rows(path: Path) -> list[Interval]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence header in {path}: {reader.fieldnames}")
        return [(row["birth"], row["death"]) for row in reader]


def run(cmd: list[str], setting: str) -> str:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"] = "1"
    env["BETTI_PERSIST_H2_ROOT_DEDUP"] = "0"

    if setting == "default":
        env.pop("BETTI_PERSIST_H2_FAST_INTERFACE_QUERY", None)
        expected = "on"
    elif setting == "on":
        env["BETTI_PERSIST_H2_FAST_INTERFACE_QUERY"] = "1"
        expected = "on"
    elif setting == "off":
        env["BETTI_PERSIST_H2_FAST_INTERFACE_QUERY"] = "0"
        expected = "off"
    else:
        raise ValueError(setting)

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}):\n{' '.join(cmd)}\n{proc.stdout}")
    if f"h2_fast_interface_query={expected}" not in proc.stdout:
        raise RuntimeError(
            f"{setting} run did not report h2_fast_interface_query={expected}\n{proc.stdout}"
        )
    if "h2_root_dedup=off" not in proc.stdout:
        raise RuntimeError("production-default gate requires root dedup off")
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depth", type=int, default=16)
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    args = ap.parse_args()

    stack = args.stack.resolve()
    binary = args.binary.resolve()
    if not stack.exists():
        ap.error(f"stack does not exist: {stack}")
    if not binary.exists():
        ap.error(f"binary does not exist: {binary}")
    if args.slab_depth <= 0:
        ap.error("slab depth must be positive")

    extra = [
        "--local-h2-birth-state", "compact",
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", "packed",
        "--h2-hier-cross-storage", "direct",
        "--h2-hier-outside-structural-pruning", "off",
        "--neighbor-root-check", "parent-shortcut",
        "--interface-state", "root-invariant",
    ]

    with tempfile.TemporaryDirectory(prefix="u16_h2_fast_iface_default_") as td_raw:
        td = Path(td_raw)
        outputs: dict[str, Path] = {}
        for setting in ("default", "on", "off"):
            out = td / f"{setting}.csv"
            cmd = [
                str(binary),
                str(stack),
                str(args.slab_depth),
                str(args.foreground_connectivity),
                "h2-scalar-hierarchical-stream",
                str(out),
                *extra,
            ]
            run(cmd, setting)
            outputs[setting] = out

        default_rows = rows(outputs["default"])
        on_rows = rows(outputs["on"])
        off_rows = rows(outputs["off"])
        if default_rows != on_rows:
            raise RuntimeError("FAIL: unset/default fast-interface output differs from explicit =1")
        if collections.Counter(default_rows) != collections.Counter(off_rows):
            raise RuntimeError("FAIL: production default interval multiset differs from explicit reference =0")
        if default_rows != off_rows:
            raise RuntimeError("FAIL: production default ordered persistence rows differ from explicit reference =0")

    print(
        f"PASS v23.6 default gate: unset=>on, =1=>on, =0=>off; "
        f"{len(default_rows)} ordered intervals identical at d={args.slab_depth}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
