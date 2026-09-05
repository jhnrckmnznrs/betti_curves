#!/usr/bin/env python3
"""Compare scalar streaming persistence against the in-memory scalar path.

This is intended for a representative stack that is small enough to run both
algorithms. Equality is checked as an exact multiset of CSV intervals, so row
ordering is irrelevant.
"""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--slab-depth", type=int, default=2)
    parser.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("h0", "h2"),
        default=("h0", "h2"),
    )
    parser.add_argument("--merge-strategy", choices=("scan", "heap"), default="scan")
    parser.add_argument(
        "--interface-order", choices=("comparison", "radix"), default="radix"
    )
    parser.add_argument("--event-order", choices=("resort", "verify"), default="verify")
    parser.add_argument(
        "--f32-key-mode", choices=("legacy64", "native32"), default="native32"
    )
    parser.add_argument(
        "--neighbor-kernel", choices=("generic", "interior-fast"), default="generic"
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
    command = [str(binary.resolve()), str(input_path.resolve()), str(slab), str(fg), mode, str(output)]
    if extra_args:
        command.extend(extra_args)
    completed = subprocess.run(command, cwd=work, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return read_intervals(output)


def main() -> None:
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    if not args.input.is_dir():
        raise SystemExit(f"input stack not found: {args.input}")

    with tempfile.TemporaryDirectory(prefix="betti_equivalence_") as temp:
        root = Path(temp)
        for dimension in args.modes:
            in_memory_mode = f"{dimension}-scalar"
            stream_mode = f"{dimension}-scalar-stream"
            expected = run(
                args.binary,
                args.input,
                root / f"{dimension}_in_memory",
                args.slab_depth,
                args.foreground_connectivity,
                in_memory_mode,
            )
            observed = run(
                args.binary,
                args.input,
                root / f"{dimension}_stream",
                args.slab_depth,
                args.foreground_connectivity,
                stream_mode,
                [
                    "--merge-strategy",
                    args.merge_strategy,
                    "--interface-order",
                    args.interface_order,
                    "--event-order",
                    args.event_order,
                    "--f32-key-mode",
                    args.f32_key_mode,
                    "--neighbor-kernel",
                    args.neighbor_kernel,
                ],
            )
            if observed != expected:
                missing = expected - observed
                extra = observed - expected
                raise SystemExit(
                    f"FAIL {dimension}: stream/in-memory interval multisets differ\n"
                    f"missing from stream: {missing.most_common(10)}\n"
                    f"extra in stream: {extra.most_common(10)}"
                )
            print(f"PASS {dimension}: {sum(observed.values())} intervals agree exactly")


if __name__ == "__main__":
    main()
