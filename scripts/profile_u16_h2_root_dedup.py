#!/usr/bin/env python3
"""Paired A/B profile for native-U16 H2 persistence UF-root deduplication.

The benchmark deliberately alternates run order within each pair so that slow
thermal/frequency/cache drift cannot systematically favour either backend.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable

RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
USER_RE = re.compile(r"User time \(seconds\):\s*([0-9.]+)")
SYS_RE = re.compile(r"System time \(seconds\):\s*([0-9.]+)")
LEAF_RE = re.compile(
    r"PROFILE_U16_PERSIST_LEAF dimension=h2 key_bytes=4 plateau_zero_elision=(on|off) "
    r"zero_persistence_pairs_elided=(\d+) union_attempts=(\d+) successful_unions=(\d+) "
    r"scalar_order_seconds=([0-9.]+) local_sweep_seconds=([0-9.]+) root_dedup=(on|off) "
    r"same_root_unions=(\d+) root_dedup_inputs=(\d+) root_dedup_unique=(\d+) "
    r"root_dedup_skipped=(\d+)"
)

METRICS = [
    "wall_seconds",
    "max_rss_mib",
    "user_seconds",
    "system_seconds",
    "zero_persistence_pairs_elided",
    "union_attempts",
    "successful_unions",
    "same_root_unions",
    "root_dedup_inputs",
    "root_dedup_unique",
    "root_dedup_skipped",
    "scalar_order_seconds",
    "local_sweep_seconds",
]

# These counters describe state transitions rather than redundant work. They
# must agree between the two members of every A/B pair.
PAIR_INVARIANTS = (
    "zero_persistence_pairs_elided",
    "successful_unions",
)


def command(binary: Path, stack: Path, depth: int, conn: int, out: Path) -> list[str]:
    return [
        str(binary),
        str(stack),
        str(depth),
        str(conn),
        "h2-scalar-hierarchical-stream",
        str(out),
        "--local-h2-birth-state",
        "compact",
        "--global-h2-birth-state",
        "compact",
        "--global-h2-uf-layout",
        "packed",
        "--h2-hier-cross-storage",
        "direct",
        "--h2-hier-outside-structural-pruning",
        "off",
    ]


def required_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"missing {label} in /usr/bin/time output")
    return match


def run_once(cmd: list[str], timing_path: Path, enabled: bool) -> dict[str, float | int | str]:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"] = "1"
    env["BETTI_PERSIST_H2_ROOT_DEDUP"] = "1" if enabled else "0"
    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(timing_path), *cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    wall = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    timing = timing_path.read_text()
    match = LEAF_RE.search(proc.stdout)
    if not match:
        raise RuntimeError("missing v23 PROFILE_U16_PERSIST_LEAF H2 line")
    expected = "on" if enabled else "off"
    if match.group(7) != expected:
        raise RuntimeError(
            f"requested root_dedup={expected}, profile reported root_dedup={match.group(7)}"
        )
    return {
        "wall_seconds": wall,
        "max_rss_mib": float(required_match(RSS_RE, timing, "maximum RSS").group(1)) / 1024.0,
        "user_seconds": float(required_match(USER_RE, timing, "user time").group(1)),
        "system_seconds": float(required_match(SYS_RE, timing, "system time").group(1)),
        "plateau_zero_elision": match.group(1),
        "zero_persistence_pairs_elided": int(match.group(2)),
        "union_attempts": int(match.group(3)),
        "successful_unions": int(match.group(4)),
        "scalar_order_seconds": float(match.group(5)),
        "local_sweep_seconds": float(match.group(6)),
        "root_dedup": match.group(7),
        "same_root_unions": int(match.group(8)),
        "root_dedup_inputs": int(match.group(9)),
        "root_dedup_unique": int(match.group(10)),
        "root_dedup_skipped": int(match.group(11)),
    }


def median(values: Iterable[float]) -> float:
    return float(statistics.median(values))


def mad(values: Iterable[float]) -> float:
    xs = [float(x) for x in values]
    centre = median(xs)
    return median(abs(x - centre) for x in xs)


def iqr(values: Iterable[float]) -> float:
    xs = [float(x) for x in values]
    if len(xs) < 2:
        return 0.0
    q1, _, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return float(q3 - q1)


def cv(values: Iterable[float]) -> float:
    xs = [float(x) for x in values]
    if len(xs) < 2:
        return 0.0
    mean = statistics.fmean(xs)
    return float(statistics.stdev(xs) / mean) if mean else 0.0


def pct_delta(off: float, on: float) -> float:
    return 100.0 * (on - off) / off if off else 0.0


def two_sided_sign_test_p(deltas: Iterable[float]) -> tuple[int, int, float]:
    """Return (negative_count, non_tied_count, exact two-sided sign-test p)."""
    xs = [float(x) for x in deltas if float(x) != 0.0]
    n = len(xs)
    if n == 0:
        return 0, 0, 1.0
    negatives = sum(x < 0.0 for x in xs)
    tail_k = min(negatives, n - negatives)
    tail = sum(math.comb(n, k) for k in range(tail_k + 1)) / (2**n)
    return negatives, n, min(1.0, 2.0 * tail)


def pair_order(pair_id: int, start_with: str) -> tuple[str, str]:
    first = start_with if pair_id % 2 == 1 else ("on" if start_with == "off" else "off")
    second = "on" if first == "off" else "off"
    return first, second


def assert_pair_invariants(depth: int, pair_id: int, results: dict[str, dict[str, object]]) -> None:
    off = results["off"]
    on = results["on"]
    for metric in PAIR_INVARIANTS:
        if int(off[metric]) != int(on[metric]):
            raise RuntimeError(
                f"pair invariant failed at d={depth}, pair={pair_id}: "
                f"{metric} off={off[metric]} on={on[metric]}"
            )
    if int(on["same_root_unions"]) != 0:
        raise RuntimeError(
            f"dedup-enabled run still reported same-root unions at d={depth}, pair={pair_id}: "
            f"{on['same_root_unions']}"
        )
    if int(on["root_dedup_skipped"]) != int(off["same_root_unions"]):
        raise RuntimeError(
            f"dedup accounting mismatch at d={depth}, pair={pair_id}: "
            f"on skipped={on['root_dedup_skipped']} vs off same_root={off['same_root_unions']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Interleaved paired A/B profile of U16 H2 root deduplication."
    )
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="unrecorded paired warmups per depth (default: 1)",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=11,
        help="recorded A/B pairs per depth (default: 11)",
    )
    ap.add_argument(
        "--start-with",
        choices=("off", "on"),
        default="off",
        help="backend that runs first in odd-numbered pairs; even pairs reverse it",
    )
    ap.add_argument("--output", type=Path, default=Path("profiles/v23_u16_h2_root_dedup_paired.csv"))
    args = ap.parse_args()

    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if args.repeats < 2:
        ap.error("--repeats must be >= 2 for robust spread statistics")
    if any(depth <= 0 for depth in args.slab_depths):
        ap.error("all --slab-depths must be positive")
    if len(set(args.slab_depths)) != len(args.slab_depths):
        ap.error("--slab-depths must not contain duplicates")

    stack = args.stack.resolve()
    binary = args.binary.resolve()
    if not stack.exists():
        ap.error(f"stack does not exist: {stack}")
    if not binary.exists():
        ap.error(f"binary does not exist: {binary}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="u16_h2_root_dedup_paired_") as td_raw:
        td = Path(td_raw)
        seq = 0
        for depth in args.slab_depths:
            # Warmups are paired and alternate too, so both paths see equivalent
            # preconditioning before recorded measurements begin.
            for warm in range(1, args.warmup + 1):
                for backend in pair_order(warm, args.start_with):
                    enabled = backend == "on"
                    seq += 1
                    print(f"warmup={warm} root_dedup={backend} h2 d={depth}")
                    run_once(
                        command(binary, stack, depth, args.foreground_connectivity, td / f"{seq}.csv"),
                        td / f"{seq}.time",
                        enabled,
                    )

            for pair_id in range(1, args.repeats + 1):
                order = pair_order(pair_id, args.start_with)
                current: dict[str, dict[str, object]] = {}
                for run_position, backend in enumerate(order, start=1):
                    enabled = backend == "on"
                    seq += 1
                    result = run_once(
                        command(binary, stack, depth, args.foreground_connectivity, td / f"{seq}.csv"),
                        td / f"{seq}.time",
                        enabled,
                    )
                    row: dict[str, object] = {
                        "backend": backend,
                        "slab_depth": depth,
                        "pair_id": pair_id,
                        "run_position": run_position,
                        "pair_order": "-".join(order),
                        "sequence": seq,
                        **result,
                    }
                    rows.append(row)
                    current[backend] = row
                    print(
                        f"pair={pair_id:02d} order={'/'.join(order)} pos={run_position} "
                        f"root_dedup={backend} d={depth}: "
                        f"{result['wall_seconds']:.3f}s {result['max_rss_mib']:.1f} MiB "
                        f"attempts={result['union_attempts']} same_root={result['same_root_unions']} "
                        f"skipped={result['root_dedup_skipped']}"
                    )

                assert_pair_invariants(depth, pair_id, current)
                off = current["off"]
                on = current["on"]
                pair_row: dict[str, object] = {
                    "slab_depth": depth,
                    "pair_id": pair_id,
                    "pair_order": "-".join(order),
                }
                for metric in ("wall_seconds", "local_sweep_seconds", "user_seconds", "system_seconds", "max_rss_mib"):
                    off_value = float(off[metric])
                    on_value = float(on[metric])
                    pair_row[f"off_{metric}"] = off_value
                    pair_row[f"on_{metric}"] = on_value
                    pair_row[f"delta_{metric}"] = on_value - off_value
                    pair_row[f"pct_delta_{metric}"] = pct_delta(off_value, on_value)
                pair_row["off_union_attempts"] = int(off["union_attempts"])
                pair_row["on_union_attempts"] = int(on["union_attempts"])
                pair_row["union_attempts_eliminated"] = int(off["union_attempts"]) - int(on["union_attempts"])
                pair_row["pct_union_attempts_eliminated"] = (
                    100.0 * int(pair_row["union_attempts_eliminated"]) / int(off["union_attempts"])
                    if int(off["union_attempts"])
                    else 0.0
                )
                pair_rows.append(pair_row)

    fields = list(rows[0])
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = args.output.with_name(args.output.stem + "_summary.csv")
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["backend"]), int(row["slab_depth"])), []).append(row)
    fields2 = ["backend", "slab_depth", "repeats"]
    for metric in METRICS:
        fields2.extend(
            [
                f"median_{metric}",
                f"mad_{metric}",
                f"iqr_{metric}",
            ]
        )
    fields2.append("cv_wall_seconds")
    with summary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields2)
        writer.writeheader()
        for (backend, depth), rs in sorted(groups.items()):
            out: dict[str, object] = {"backend": backend, "slab_depth": depth, "repeats": len(rs)}
            for metric in METRICS:
                xs = [float(r[metric]) for r in rs]
                out[f"median_{metric}"] = median(xs)
                out[f"mad_{metric}"] = mad(xs)
                out[f"iqr_{metric}"] = iqr(xs)
            out["cv_wall_seconds"] = cv(float(r["wall_seconds"]) for r in rs)
            writer.writerow(out)

    pairs = args.output.with_name(args.output.stem + "_pairs.csv")
    pair_fields = list(pair_rows[0])
    with pairs.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pair_rows)

    paired_summary = args.output.with_name(args.output.stem + "_paired_summary.csv")
    paired_groups: dict[int, list[dict[str, object]]] = {}
    for row in pair_rows:
        paired_groups.setdefault(int(row["slab_depth"]), []).append(row)
    paired_metrics = (
        "pct_delta_wall_seconds",
        "pct_delta_local_sweep_seconds",
        "pct_delta_user_seconds",
        "pct_delta_system_seconds",
        "pct_delta_max_rss_mib",
        "pct_union_attempts_eliminated",
    )
    paired_fields = ["slab_depth", "pairs"]
    for metric in paired_metrics:
        paired_fields.extend([f"median_{metric}", f"mad_{metric}", f"iqr_{metric}"])
    paired_fields.extend(
        [
            "median_off_wall_seconds",
            "median_on_wall_seconds",
            "median_off_local_sweep_seconds",
            "median_on_local_sweep_seconds",
            "on_faster_wall_pairs",
            "non_tied_wall_pairs",
            "on_faster_wall_fraction",
            "two_sided_sign_test_p_wall",
            "on_faster_local_sweep_pairs",
            "non_tied_local_sweep_pairs",
            "on_faster_local_sweep_fraction",
            "two_sided_sign_test_p_local_sweep",
            "median_pct_delta_wall_off_first",
            "median_pct_delta_wall_on_first",
            "wall_order_effect_pp",
        ]
    )
    with paired_summary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=paired_fields)
        writer.writeheader()
        for depth, rs in sorted(paired_groups.items()):
            out: dict[str, object] = {"slab_depth": depth, "pairs": len(rs)}
            for metric in paired_metrics:
                xs = [float(r[metric]) for r in rs]
                out[f"median_{metric}"] = median(xs)
                out[f"mad_{metric}"] = mad(xs)
                out[f"iqr_{metric}"] = iqr(xs)
            out["median_off_wall_seconds"] = median(float(r["off_wall_seconds"]) for r in rs)
            out["median_on_wall_seconds"] = median(float(r["on_wall_seconds"]) for r in rs)
            out["median_off_local_sweep_seconds"] = median(
                float(r["off_local_sweep_seconds"]) for r in rs
            )
            out["median_on_local_sweep_seconds"] = median(
                float(r["on_local_sweep_seconds"]) for r in rs
            )

            wall_deltas = [float(r["delta_wall_seconds"]) for r in rs]
            wall_wins, wall_non_tied, wall_p = two_sided_sign_test_p(wall_deltas)
            out["on_faster_wall_pairs"] = wall_wins
            out["non_tied_wall_pairs"] = wall_non_tied
            out["on_faster_wall_fraction"] = wall_wins / wall_non_tied if wall_non_tied else 0.0
            out["two_sided_sign_test_p_wall"] = wall_p

            sweep_deltas = [float(r["delta_local_sweep_seconds"]) for r in rs]
            sweep_wins, sweep_non_tied, sweep_p = two_sided_sign_test_p(sweep_deltas)
            out["on_faster_local_sweep_pairs"] = sweep_wins
            out["non_tied_local_sweep_pairs"] = sweep_non_tied
            out["on_faster_local_sweep_fraction"] = (
                sweep_wins / sweep_non_tied if sweep_non_tied else 0.0
            )
            out["two_sided_sign_test_p_local_sweep"] = sweep_p

            off_first = [
                float(r["pct_delta_wall_seconds"]) for r in rs if r["pair_order"] == "off-on"
            ]
            on_first = [
                float(r["pct_delta_wall_seconds"]) for r in rs if r["pair_order"] == "on-off"
            ]
            off_first_median = median(off_first) if off_first else 0.0
            on_first_median = median(on_first) if on_first else 0.0
            out["median_pct_delta_wall_off_first"] = off_first_median
            out["median_pct_delta_wall_on_first"] = on_first_median
            out["wall_order_effect_pp"] = on_first_median - off_first_median
            writer.writerow(out)

    print(f"wrote {args.output}")
    print(f"wrote {summary}")
    print(f"wrote {pairs}")
    print(f"wrote {paired_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
