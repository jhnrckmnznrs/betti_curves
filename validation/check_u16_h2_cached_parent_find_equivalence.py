#!/usr/bin/env python3
"""Exact U16 H2 persistence equivalence: one-hop reference vs cached-parent find."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
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


def multiset_digest(intervals: list[Interval]) -> str:
    counts = collections.Counter(intervals)
    digest = hashlib.sha256()
    for interval, count in sorted(counts.items()):
        digest.update(interval[0].encode()); digest.update(b"\0")
        digest.update(interval[1].encode()); digest.update(b"\0")
        digest.update(str(count).encode()); digest.update(b"\n")
    return digest.hexdigest()


def run(cmd: list[str]) -> str:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"] = "1"
    env["BETTI_PERSIST_H2_ROOT_DEDUP"] = "0"
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if proc.returncode:
        raise RuntimeError(f"command failed with code {proc.returncode}:\n{' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def compare(depth: int, reference: list[Interval], candidate: list[Interval], require_order: bool) -> None:
    a = collections.Counter(reference); b = collections.Counter(candidate)
    if a != b:
        missing = list((a - b).items())[:10]
        extra = list((b - a).items())[:10]
        raise RuntimeError(
            f"FAIL H2 d={depth}: reference/cached-parent interval multisets differ\n"
            f"missing from cached-parent: {missing}\nextra in cached-parent: {extra}"
        )
    if require_order and reference != candidate:
        first = next(i for i, (left, right) in enumerate(zip(reference, candidate)) if left != right)
        raise RuntimeError(
            f"FAIL H2 d={depth}: multisets match but row order differs at index {first}: "
            f"one-hop={reference[first]} cached-parent={candidate[first]}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    ap.add_argument("--allow-order-difference", action="store_true")
    args = ap.parse_args()

    if any(depth <= 0 for depth in args.slab_depths):
        ap.error("all --slab-depths must be positive")
    if len(set(args.slab_depths)) != len(args.slab_depths):
        ap.error("--slab-depths must not contain duplicates")
    stack = args.stack.resolve(); binary = args.binary.resolve()
    if not stack.exists(): ap.error(f"stack does not exist: {stack}")
    if not binary.exists(): ap.error(f"binary does not exist: {binary}")

    extra = [
        "--local-h2-birth-state", "compact",
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", "packed",
        "--h2-hier-cross-storage", "direct",
        "--h2-hier-outside-structural-pruning", "off",
    ]
    baseline_multiset: collections.Counter[Interval] | None = None

    with tempfile.TemporaryDirectory(prefix="u16_h2_cached_parent_equiv_") as td_raw:
        td = Path(td_raw)
        for depth in args.slab_depths:
            ref_path = td / f"h2_d{depth}_one_hop.csv"
            cand_path = td / f"h2_d{depth}_cached_parent.csv"
            base = [str(binary), str(stack), str(depth), str(args.foreground_connectivity), "h2-scalar-hierarchical-stream"]
            ref_log = run(base + [str(ref_path)] + extra + ["--neighbor-root-check", "parent-shortcut"])
            cand_log = run(base + [str(cand_path)] + extra + ["--neighbor-root-check", "parent-cached-find"])
            reference = rows(ref_path); candidate = rows(cand_path)
            compare(depth, reference, candidate, require_order=not args.allow_order_difference)

            if "neighbor_root_check=parent-shortcut" not in ref_log:
                raise RuntimeError(f"FAIL H2 d={depth}: one-hop run did not report parent-shortcut")
            if "neighbor_root_check=parent-cached-find" not in cand_log:
                raise RuntimeError(f"FAIL H2 d={depth}: cached-parent run did not report parent-cached-find")
            if "h2_root_dedup=off" not in ref_log or "h2_root_dedup=off" not in cand_log:
                raise RuntimeError(f"FAIL H2 d={depth}: equivalence gate requires root dedup off")

            current = collections.Counter(reference)
            if baseline_multiset is None:
                baseline_multiset = current
            elif current != baseline_multiset:
                missing = list((baseline_multiset - current).items())[:10]
                extra_rows = list((current - baseline_multiset).items())[:10]
                raise RuntimeError(
                    f"FAIL H2 d={depth}: slab-depth invariance failed against d={args.slab_depths[0]}\n"
                    f"missing: {missing}\nextra: {extra_rows}"
                )

            digest = multiset_digest(reference)
            order_note = "ordered rows" if not args.allow_order_difference else "interval multiset"
            print(
                f"PASS H2 d={depth}: {len(reference)} exact intervals; cached-parent matches one-hop "
                f"reference ({order_note}); sha256={digest}"
            )

    print(
        f"PASS H2 cached-parent find gate across slab depths {','.join(map(str, args.slab_depths))}: "
        "same-binary A/B equivalence and slab-depth invariance"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
