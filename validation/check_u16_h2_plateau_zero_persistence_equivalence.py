#!/usr/bin/env python3
"""Exact same-binary equivalence for U16 H2 plateau-zero pair elision."""
from __future__ import annotations

import argparse
import collections
import csv
import os
import subprocess
import tempfile
from pathlib import Path


def rows(path: Path) -> collections.Counter[tuple[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence header in {path}: {reader.fieldnames}")
        return collections.Counter((row["birth"], row["death"]) for row in reader)


def run(cmd: list[str], enabled: bool) -> str:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"] = "1" if enabled else "0"
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if proc.returncode:
        raise RuntimeError(
            f"command failed with code {proc.returncode}:\n{' '.join(cmd)}\n{proc.stdout}"
        )
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
    extra = [
        "--local-h2-birth-state", "compact",
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", "packed",
        "--h2-hier-cross-storage", "direct",
        "--h2-hier-outside-structural-pruning", "off",
    ]
    with tempfile.TemporaryDirectory(prefix="u16_h2_plateau_zero_equiv_") as td_raw:
        td = Path(td_raw)
        off = td / "h2_off.csv"
        on = td / "h2_on.csv"
        base = [
            str(binary), str(stack), str(args.slab_depth),
            str(args.foreground_connectivity), "h2-scalar-hierarchical-stream",
        ]
        off_log = run(base + [str(off)] + extra, enabled=False)
        on_log = run(base + [str(on)] + extra, enabled=True)
        a = rows(off)
        b = rows(on)
        if a != b:
            missing = list((a - b).items())[:10]
            extra_rows = list((b - a).items())[:10]
            raise RuntimeError(
                "FAIL H2: plateau-zero on/off interval multisets differ\n"
                f"missing from enabled: {missing}\nextra in enabled: {extra_rows}"
            )
        if "h2_plateau_zero_elision=off" not in off_log:
            raise RuntimeError("FAIL H2: disabled run did not report plateau-zero elision off")
        if "h2_plateau_zero_elision=on" not in on_log:
            raise RuntimeError("FAIL H2: enabled run did not report plateau-zero elision on")
        print(
            f"PASS H2: {sum(a.values())} exact intervals; "
            "plateau-zero elision matches materialize-and-filter reference"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
