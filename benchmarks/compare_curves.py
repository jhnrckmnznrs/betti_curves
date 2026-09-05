#!/usr/bin/env python3
"""Compare sparse Betti-curve CSVs numerically."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[tuple[float, int]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "threshold" not in reader.fieldnames:
            raise SystemExit(f"{path}: expected threshold column")
        value_cols = [c for c in reader.fieldnames if c != "threshold"]
        if len(value_cols) != 1:
            raise SystemExit(f"{path}: expected exactly one Betti-value column")
        value_col = value_cols[0]
        return [(float(row["threshold"]), int(row[value_col])) for row in reader]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--atol", type=float, default=0.0)
    a = p.parse_args()
    left, right = read(a.a), read(a.b)
    if len(left) != len(right):
        raise SystemExit(f"row-count mismatch: {len(left)} != {len(right)}")
    for i, ((ta, va), (tb, vb)) in enumerate(zip(left, right)):
        if abs(ta - tb) > a.atol or va != vb:
            raise SystemExit(f"mismatch at row {i}: {(ta, va)} != {(tb, vb)}")
    print(f"PASS: {len(left)} sparse curve rows agree")


if __name__ == "__main__":
    main()
