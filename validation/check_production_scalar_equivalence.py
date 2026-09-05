#!/usr/bin/env python3
"""Validate production scalar-stream defaults against the in-memory oracle.

The production stream is invoked with no tuning/ablation flags. For each
requested homology degree, one in-memory scalar persistence multiset is used as
an independent oracle and the production stream is checked at every requested
slab depth. The checker also validates the effective production configuration
reported by PROFILE_CONFIG.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument(
        "--binary",
        type=Path,
        default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")),
    )
    p.add_argument("--slab-depths", type=int, nargs="+", default=(8, 16, 32))
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--modes", nargs="+", choices=("h0", "h2"), default=("h0", "h2"))
    return p.parse_args()


def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        header = next(r, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected persistence header in {path}: {header!r}")
        return Counter(tuple(row) for row in r if row)


def parse_profile_config(stdout: str, mode: str) -> dict[str, str]:
    prefix = f"PROFILE_CONFIG scalar_{mode}_stream "
    line = next((line for line in stdout.splitlines() if line.startswith(prefix)), None)
    if line is None:
        raise RuntimeError(f"missing {prefix.strip()} line")
    result: dict[str, str] = {}
    for token in line[len(prefix):].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


def glibc_linux() -> bool:
    libc_name, _ = platform.libc_ver()
    return platform.system() == "Linux" and libc_name.lower() == "glibc"


def expected_config(mode: str) -> dict[str, str]:
    common = {
        "merge_strategy": "scan",
        "interface_order": "radix",
        "event_order": "verify",
        "f32_key_mode": "native32",
        "neighbor_kernel": "interior-fast",
        "representative_active_check": "recheck",
        "union_kernel": "root-carrying",
        "interface_state": "root-invariant",
        "uf_layout": "packed",
    }
    if mode == "h0":
        common.update(
            h0_pruning_cache="64k",
            active_state="separate",
            global_h0_uf_layout="packed",
        )
    else:
        common.update(
            neighbor_root_check="parent-shortcut",
            active_state="parent-sentinel",
            phase_trim="before-reduce" if glibc_linux() else "off",
            local_h2_birth_state="compact",
            global_h2_birth_state="compact",
            global_h2_uf_layout="parent-rank",
        )
    return common


def run(binary: Path, input_path: Path, work: Path, slab: int, fg: int, mode: str):
    work.mkdir(parents=True, exist_ok=True)
    output = work / "intervals.csv"
    command = [
        str(binary.resolve()),
        str(input_path.resolve()),
        str(slab),
        str(fg),
        mode,
        str(output),
    ]
    cp = subprocess.run(command, cwd=work, text=True, capture_output=True)
    if cp.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
        )
    return read_intervals(output), cp.stdout


def main() -> None:
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    if not a.input.is_dir():
        raise SystemExit(f"input stack not found: {a.input}")
    if any(depth <= 0 for depth in a.slab_depths):
        raise SystemExit("all slab depths must be positive")

    with tempfile.TemporaryDirectory(prefix="betti_production_equivalence_") as td:
        root = Path(td)
        oracle_depth = a.slab_depths[0]
        for dim in a.modes:
            oracle, _ = run(
                a.binary,
                a.input,
                root / f"{dim}_oracle",
                oracle_depth,
                a.foreground_connectivity,
                f"{dim}-scalar",
            )
            expected = expected_config(dim)
            for slab in a.slab_depths:
                observed, stdout = run(
                    a.binary,
                    a.input,
                    root / f"{dim}_stream_d{slab}",
                    slab,
                    a.foreground_connectivity,
                    f"{dim}-scalar-stream",
                )
                if observed != oracle:
                    missing = oracle - observed
                    extra = observed - oracle
                    raise SystemExit(
                        f"FAIL {dim} depth={slab}: production stream differs from in-memory oracle\n"
                        f"missing: {missing.most_common(10)}\nextra: {extra.most_common(10)}"
                    )
                config = parse_profile_config(stdout, dim)
                mismatches = {
                    key: (value, config.get(key))
                    for key, value in expected.items()
                    if config.get(key) != value
                }
                if mismatches:
                    raise SystemExit(
                        f"FAIL {dim} depth={slab}: production config mismatch: {mismatches}\n"
                        f"reported: {config}"
                    )
                print(
                    f"PASS {dim} depth={slab}: {sum(observed.values())} intervals; "
                    f"production defaults verified"
                )

    print("PASS: production scalar-stream defaults agree exactly with the in-memory oracle")


if __name__ == "__main__":
    main()
