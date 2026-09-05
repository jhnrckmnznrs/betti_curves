#!/usr/bin/env python3
"""Check every scalar-stream ablation cell against the in-memory exact oracle."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")),
    )
    parser.add_argument("--slab-depth", type=int, default=16)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument("--modes", nargs="+", choices=("h0", "h2"), default=("h0", "h2"))
    parser.add_argument("--f32-key-mode", choices=("legacy64", "native32"), default="native32")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="check only legacy and measured production-base cells instead of all eight",
    )
    return parser.parse_args()


def read_intervals(path: Path) -> Counter[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["birth", "death"]:
            raise RuntimeError(f"unexpected header in {path}: {header!r}")
        return Counter(tuple(row) for row in reader if row)


def run(
    binary: Path,
    input_path: Path,
    work: Path,
    slab: int,
    fg: int,
    mode: str,
    extra_args: list[str] | None = None,
) -> Counter[tuple[str, str]]:
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
    if extra_args:
        command.extend(extra_args)
    completed = subprocess.run(command, cwd=work, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return read_intervals(output)


def configs(quick: bool) -> list[tuple[str, str, str]]:
    if quick:
        return [
            ("scan", "comparison", "resort"),
            ("scan", "radix", "verify"),
        ]
    return list(
        itertools.product(
            ("scan", "heap"),
            ("comparison", "radix"),
            ("resort", "verify"),
        )
    )


def label(config: tuple[str, str, str]) -> str:
    if config == ("scan", "comparison", "resort"):
        return "legacy"
    if config == ("scan", "radix", "verify"):
        return "production-base"
    if config == ("heap", "radix", "verify"):
        return "v1-heap"
    return "/".join(config)


def main() -> None:
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    if not args.input.is_dir():
        raise SystemExit(f"input stack not found: {args.input}")

    with tempfile.TemporaryDirectory(prefix="betti_ablation_equivalence_") as temp:
        root = Path(temp)
        for dimension in args.modes:
            expected = run(
                args.binary,
                args.input,
                root / f"{dimension}_in_memory",
                args.slab_depth,
                args.foreground_connectivity,
                f"{dimension}-scalar",
            )
            count = sum(expected.values())
            for config in configs(args.quick):
                merge, interface, event = config
                observed = run(
                    args.binary,
                    args.input,
                    root / f"{dimension}_{merge}_{interface}_{event}",
                    args.slab_depth,
                    args.foreground_connectivity,
                    f"{dimension}-scalar-stream",
                    [
                        "--merge-strategy",
                        merge,
                        "--interface-order",
                        interface,
                        "--event-order",
                        event,
                        "--f32-key-mode",
                        args.f32_key_mode,
                    ],
                )
                if observed != expected:
                    missing = expected - observed
                    extra = observed - expected
                    raise SystemExit(
                        f"FAIL {dimension} {label(config)}: stream/in-memory multisets differ\n"
                        f"missing from stream: {missing.most_common(10)}\n"
                        f"extra in stream: {extra.most_common(10)}"
                    )
                print(f"PASS {dimension} {label(config)}: {count} intervals agree exactly")

    print("PASS: every requested scalar-stream ablation cell matches the in-memory oracle")


if __name__ == "__main__":
    main()
