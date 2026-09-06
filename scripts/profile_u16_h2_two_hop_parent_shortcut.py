#!/usr/bin/env python3
"""Paired A/B profile for the U16 H2 two-hop parent shortcut candidate.

Reference: --neighbor-root-check parent-shortcut
Candidate: --neighbor-root-check parent-two-hop
Root dedup is forced off because v23.1 showed it is a runtime regression.
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
LEAF_PREFIX = "PROFILE_U16_PERSIST_LEAF dimension=h2 "

COUNTERS = (
    "zero_persistence_pairs_elided",
    "union_attempts",
    "successful_unions",
    "same_root_unions",
    "direct_parent_checks",
    "direct_parent_hits",
    "two_hop_parent_checks",
    "two_hop_parent_hits",
    "avoided_neighbor_find_calls",
    "neighbor_find_calls",
    "neighbor_find_parent_steps",
    "neighbor_find_zero_hop",
    "neighbor_find_one_hop",
    "neighbor_find_two_hop",
    "neighbor_find_gt_two_hop",
    "same_root_neighbor_find_zero_hop",
    "same_root_neighbor_find_one_hop",
    "same_root_neighbor_find_two_hop",
    "same_root_neighbor_find_gt_two_hop",
)

INVARIANTS = (
    "zero_persistence_pairs_elided",
    "union_attempts",
    "successful_unions",
    "same_root_unions",
)

SUMMARY_METRICS = (
    "wall_seconds",
    "max_rss_mib",
    "user_seconds",
    "system_seconds",
    "scalar_order_seconds",
    "local_sweep_seconds",
    *COUNTERS,
)


def command(binary: Path, stack: Path, depth: int, conn: int, out: Path, mode: str, *, diagnostics: bool = False) -> list[str]:
    cmd = [
        str(binary), str(stack), str(depth), str(conn),
        "h2-scalar-hierarchical-stream", str(out),
        "--local-h2-birth-state", "compact",
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", "packed",
        "--h2-hier-cross-storage", "direct",
        "--h2-hier-outside-structural-pruning", "off",
        "--neighbor-root-check", mode,
    ]
    if diagnostics:
        cmd.append("--sweep-diagnostics")
    return cmd


def parse_leaf(stdout: str) -> dict[str, str]:
    line = next((line for line in stdout.splitlines() if line.startswith(LEAF_PREFIX)), None)
    if line is None:
        raise RuntimeError("missing PROFILE_U16_PERSIST_LEAF H2 line")
    result: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def required_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"missing {label} in /usr/bin/time output")
    return match


def run_once(cmd: list[str], timing_path: Path, expected_mode: str) -> dict[str, object]:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"] = "1"
    env["BETTI_PERSIST_H2_ROOT_DEDUP"] = "0"
    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(timing_path), *cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    wall = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    timing = timing_path.read_text()
    leaf = parse_leaf(proc.stdout)
    if leaf.get("neighbor_root_check") != expected_mode:
        raise RuntimeError(
            f"requested neighbor_root_check={expected_mode}, profile reported "
            f"{leaf.get('neighbor_root_check')!r}"
        )
    if leaf.get("root_dedup") != "off":
        raise RuntimeError("two-hop benchmark must run with root_dedup=off")
    out: dict[str, object] = {
        "wall_seconds": wall,
        "max_rss_mib": float(required_match(RSS_RE, timing, "maximum RSS").group(1)) / 1024.0,
        "user_seconds": float(required_match(USER_RE, timing, "user time").group(1)),
        "system_seconds": float(required_match(SYS_RE, timing, "system time").group(1)),
        "plateau_zero_elision": leaf["plateau_zero_elision"],
        "root_dedup": leaf["root_dedup"],
        "neighbor_root_check": leaf["neighbor_root_check"],
        "scalar_order_seconds": float(leaf["scalar_order_seconds"]),
        "local_sweep_seconds": float(leaf["local_sweep_seconds"]),
    }
    for field in COUNTERS:
        out[field] = int(leaf[field])
    return out


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


def pct_delta(reference: float, candidate: float) -> float:
    return 100.0 * (candidate - reference) / reference if reference else 0.0


def two_sided_sign_test_p(deltas: Iterable[float]) -> tuple[int, int, float]:
    xs = [float(x) for x in deltas if float(x) != 0.0]
    n = len(xs)
    if n == 0:
        return 0, 0, 1.0
    wins = sum(x < 0.0 for x in xs)
    tail_k = min(wins, n - wins)
    tail = sum(math.comb(n, k) for k in range(tail_k + 1)) / (2**n)
    return wins, n, min(1.0, 2.0 * tail)


def pair_order(pair_id: int, start_with: str) -> tuple[str, str]:
    first = start_with if pair_id % 2 == 1 else ("candidate" if start_with == "reference" else "reference")
    second = "candidate" if first == "reference" else "reference"
    return first, second


def mode_for_backend(backend: str) -> str:
    return "parent-shortcut" if backend == "reference" else "parent-two-hop"


def assert_pair_invariants(depth: int, pair_id: int, results: dict[str, dict[str, object]]) -> None:
    ref = results["reference"]
    cand = results["candidate"]
    for metric in INVARIANTS:
        if int(ref[metric]) != int(cand[metric]):
            raise RuntimeError(
                f"pair invariant failed at d={depth}, pair={pair_id}: "
                f"{metric} reference={ref[metric]} candidate={cand[metric]}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired U16 H2 one-hop vs two-hop parent-shortcut benchmark.")
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--start-with", choices=("reference", "candidate"), default="reference")
    ap.add_argument("--output", type=Path, default=Path("profiles/v23_2_1_u16_h2_two_hop_parent_shortcut_paired.csv"))
    ap.add_argument("--diagnostic-repeats", type=int, default=1,
                    help="separate instrumented runs per mode/depth after timing (default: 1)")
    args = ap.parse_args()

    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if args.repeats < 2:
        ap.error("--repeats must be >= 2")
    if args.diagnostic_repeats < 1:
        ap.error("--diagnostic-repeats must be >= 1")
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

    with tempfile.TemporaryDirectory(prefix="u16_h2_two_hop_paired_") as td_raw:
        td = Path(td_raw)
        seq = 0
        for depth in args.slab_depths:
            for warm in range(1, args.warmup + 1):
                for backend in pair_order(warm, args.start_with):
                    seq += 1
                    mode = mode_for_backend(backend)
                    print(f"warmup={warm} backend={backend} mode={mode} h2 d={depth}")
                    run_once(
                        command(binary, stack, depth, args.foreground_connectivity, td / f"{seq}.csv", mode),
                        td / f"{seq}.time", mode,
                    )

            for pair_id in range(1, args.repeats + 1):
                order = pair_order(pair_id, args.start_with)
                current: dict[str, dict[str, object]] = {}
                for run_position, backend in enumerate(order, start=1):
                    seq += 1
                    mode = mode_for_backend(backend)
                    result = run_once(
                        command(binary, stack, depth, args.foreground_connectivity, td / f"{seq}.csv", mode),
                        td / f"{seq}.time", mode,
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
                        f"{backend} d={depth}: {result['wall_seconds']:.3f}s "
                        f"sweep={result['local_sweep_seconds']:.3f}s "
                        f"two_hop_hits={result['two_hop_parent_hits']} "
                        f"neighbor_finds={result['neighbor_find_calls']}"
                    )

                assert_pair_invariants(depth, pair_id, current)
                ref = current["reference"]
                cand = current["candidate"]
                pair: dict[str, object] = {
                    "slab_depth": depth,
                    "pair_id": pair_id,
                    "pair_order": "-".join(order),
                }
                for metric in ("wall_seconds", "local_sweep_seconds", "user_seconds", "system_seconds", "max_rss_mib"):
                    rv = float(ref[metric]); cvv = float(cand[metric])
                    pair[f"reference_{metric}"] = rv
                    pair[f"candidate_{metric}"] = cvv
                    pair[f"delta_{metric}"] = cvv - rv
                    pair[f"pct_delta_{metric}"] = pct_delta(rv, cvv)
                for metric in (
                    "direct_parent_hits", "two_hop_parent_checks", "two_hop_parent_hits",
                    "avoided_neighbor_find_calls", "neighbor_find_calls", "neighbor_find_parent_steps",
                    "same_root_neighbor_find_two_hop", "same_root_neighbor_find_gt_two_hop",
                ):
                    pair[f"reference_{metric}"] = int(ref[metric])
                    pair[f"candidate_{metric}"] = int(cand[metric])
                    pair[f"delta_{metric}"] = int(cand[metric]) - int(ref[metric])
                pair["candidate_two_hop_hit_rate"] = (
                    int(cand["two_hop_parent_hits"]) / int(cand["two_hop_parent_checks"])
                    if int(cand["two_hop_parent_checks"]) else 0.0
                )
                pair["neighbor_finds_avoided_vs_reference"] = int(ref["neighbor_find_calls"]) - int(cand["neighbor_find_calls"])
                pair_rows.append(pair)

    # Timing runs intentionally keep sweep diagnostics OFF so instrumentation does not
    # perturb the A/B measurement.  Collect structural counters separately.
    diagnostic_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="u16_h2_two_hop_diagnostics_") as td_raw:
        td = Path(td_raw)
        seq = 0
        for depth in args.slab_depths:
            for diagnostic_run in range(1, args.diagnostic_repeats + 1):
                current: dict[str, dict[str, object]] = {}
                for backend in ("reference", "candidate"):
                    seq += 1
                    mode = mode_for_backend(backend)
                    result = run_once(
                        command(
                            binary, stack, depth, args.foreground_connectivity,
                            td / f"diag_{seq}.csv", mode, diagnostics=True,
                        ),
                        td / f"diag_{seq}.time", mode,
                    )
                    row = {
                        "backend": backend,
                        "slab_depth": depth,
                        "diagnostic_run": diagnostic_run,
                        **result,
                    }
                    diagnostic_rows.append(row)
                    current[backend] = row
                    print(
                        f"diagnostic={diagnostic_run} {backend} d={depth}: "
                        f"direct_hits={result['direct_parent_hits']} "
                        f"two_hop_checks={result['two_hop_parent_checks']} "
                        f"two_hop_hits={result['two_hop_parent_hits']} "
                        f"neighbor_finds={result['neighbor_find_calls']} "
                        f"same_root_depth2={result['same_root_neighbor_find_two_hop']} "
                        f"same_root_gt2={result['same_root_neighbor_find_gt_two_hop']}"
                    )

                ref = current["reference"]
                cand = current["candidate"]
                for metric in INVARIANTS:
                    if int(ref[metric]) != int(cand[metric]):
                        raise RuntimeError(
                            f"diagnostic invariant failed at d={depth}, run={diagnostic_run}: "
                            f"{metric} reference={ref[metric]} candidate={cand[metric]}"
                        )
                if int(ref["two_hop_parent_checks"]) != 0 or int(ref["two_hop_parent_hits"]) != 0:
                    raise RuntimeError(
                        f"reference unexpectedly executed two-hop checks at d={depth}, "
                        f"run={diagnostic_run}"
                    )

    diagnostics_path = args.output.with_name(args.output.stem + "_diagnostics.csv")
    with diagnostics_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(diagnostic_rows[0]))
        writer.writeheader(); writer.writerows(diagnostic_rows)

    diagnostics_summary_path = args.output.with_name(args.output.stem + "_diagnostics_summary.csv")
    diag_fields = [
        "slab_depth", "diagnostic_repeats",
        "reference_direct_parent_checks", "reference_direct_parent_hits",
        "reference_neighbor_find_calls", "reference_neighbor_find_parent_steps",
        "reference_same_root_neighbor_find_zero_hop",
        "reference_same_root_neighbor_find_one_hop",
        "reference_same_root_neighbor_find_two_hop",
        "reference_same_root_neighbor_find_gt_two_hop",
        "candidate_direct_parent_checks", "candidate_direct_parent_hits",
        "candidate_two_hop_parent_checks", "candidate_two_hop_parent_hits",
        "candidate_two_hop_hit_rate", "candidate_neighbor_find_calls",
        "candidate_neighbor_find_parent_steps",
        "neighbor_finds_avoided_vs_reference",
        "neighbor_find_parent_steps_delta",
    ]
    with diagnostics_summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=diag_fields); writer.writeheader()
        for depth in sorted(set(int(r["slab_depth"]) for r in diagnostic_rows)):
            ref_rows = [r for r in diagnostic_rows if int(r["slab_depth"]) == depth and r["backend"] == "reference"]
            cand_rows = [r for r in diagnostic_rows if int(r["slab_depth"]) == depth and r["backend"] == "candidate"]
            def med(rows, field):
                return median(float(r[field]) for r in rows)
            checks = med(cand_rows, "two_hop_parent_checks")
            hits = med(cand_rows, "two_hop_parent_hits")
            out = {
                "slab_depth": depth,
                "diagnostic_repeats": min(len(ref_rows), len(cand_rows)),
                "reference_direct_parent_checks": med(ref_rows, "direct_parent_checks"),
                "reference_direct_parent_hits": med(ref_rows, "direct_parent_hits"),
                "reference_neighbor_find_calls": med(ref_rows, "neighbor_find_calls"),
                "reference_neighbor_find_parent_steps": med(ref_rows, "neighbor_find_parent_steps"),
                "reference_same_root_neighbor_find_zero_hop": med(ref_rows, "same_root_neighbor_find_zero_hop"),
                "reference_same_root_neighbor_find_one_hop": med(ref_rows, "same_root_neighbor_find_one_hop"),
                "reference_same_root_neighbor_find_two_hop": med(ref_rows, "same_root_neighbor_find_two_hop"),
                "reference_same_root_neighbor_find_gt_two_hop": med(ref_rows, "same_root_neighbor_find_gt_two_hop"),
                "candidate_direct_parent_checks": med(cand_rows, "direct_parent_checks"),
                "candidate_direct_parent_hits": med(cand_rows, "direct_parent_hits"),
                "candidate_two_hop_parent_checks": checks,
                "candidate_two_hop_parent_hits": hits,
                "candidate_two_hop_hit_rate": hits / checks if checks else 0.0,
                "candidate_neighbor_find_calls": med(cand_rows, "neighbor_find_calls"),
                "candidate_neighbor_find_parent_steps": med(cand_rows, "neighbor_find_parent_steps"),
                "neighbor_finds_avoided_vs_reference": med(ref_rows, "neighbor_find_calls") - med(cand_rows, "neighbor_find_calls"),
                "neighbor_find_parent_steps_delta": med(cand_rows, "neighbor_find_parent_steps") - med(ref_rows, "neighbor_find_parent_steps"),
            }
            writer.writerow(out)

    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = args.output.with_name(args.output.stem + "_summary.csv")
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["backend"]), int(row["slab_depth"])), []).append(row)
    summary_fields = ["backend", "slab_depth", "repeats"]
    for metric in SUMMARY_METRICS:
        summary_fields += [f"median_{metric}", f"mad_{metric}", f"iqr_{metric}"]
    summary_fields += ["cv_wall_seconds", "median_two_hop_hit_rate"]
    with summary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields); writer.writeheader()
        for (backend, depth), rs in sorted(groups.items()):
            out: dict[str, object] = {"backend": backend, "slab_depth": depth, "repeats": len(rs)}
            for metric in SUMMARY_METRICS:
                xs = [float(r[metric]) for r in rs]
                out[f"median_{metric}"] = median(xs); out[f"mad_{metric}"] = mad(xs); out[f"iqr_{metric}"] = iqr(xs)
            out["cv_wall_seconds"] = cv(float(r["wall_seconds"]) for r in rs)
            rates = [
                int(r["two_hop_parent_hits"]) / int(r["two_hop_parent_checks"])
                if int(r["two_hop_parent_checks"]) else 0.0 for r in rs
            ]
            out["median_two_hop_hit_rate"] = median(rates)
            writer.writerow(out)

    pairs_path = args.output.with_name(args.output.stem + "_pairs.csv")
    with pairs_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)

    paired_summary = args.output.with_name(args.output.stem + "_paired_summary.csv")
    paired_groups: dict[int, list[dict[str, object]]] = {}
    for row in pair_rows:
        paired_groups.setdefault(int(row["slab_depth"]), []).append(row)
    fields = [
        "slab_depth", "pairs",
        "median_pct_delta_wall_seconds", "mad_pct_delta_wall_seconds", "iqr_pct_delta_wall_seconds",
        "median_pct_delta_local_sweep_seconds", "mad_pct_delta_local_sweep_seconds", "iqr_pct_delta_local_sweep_seconds",
        "candidate_faster_wall_pairs", "non_tied_wall_pairs", "candidate_faster_wall_fraction", "two_sided_sign_test_p_wall",
        "candidate_faster_local_sweep_pairs", "non_tied_local_sweep_pairs", "candidate_faster_local_sweep_fraction", "two_sided_sign_test_p_local_sweep",
        "median_candidate_two_hop_parent_checks", "median_candidate_two_hop_parent_hits", "median_candidate_two_hop_hit_rate",
        "median_neighbor_finds_avoided_vs_reference", "median_delta_neighbor_find_parent_steps",
        "median_reference_same_root_neighbor_find_two_hop", "median_reference_same_root_neighbor_find_gt_two_hop",
        "median_pct_delta_wall_reference_first", "median_pct_delta_wall_candidate_first", "wall_order_effect_pp",
    ]
    with paired_summary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for depth, rs in sorted(paired_groups.items()):
            wall_pct = [float(r["pct_delta_wall_seconds"]) for r in rs]
            sweep_pct = [float(r["pct_delta_local_sweep_seconds"]) for r in rs]
            wall_wins, wall_n, wall_p = two_sided_sign_test_p(float(r["delta_wall_seconds"]) for r in rs)
            sweep_wins, sweep_n, sweep_p = two_sided_sign_test_p(float(r["delta_local_sweep_seconds"]) for r in rs)
            ref_first = [float(r["pct_delta_wall_seconds"]) for r in rs if r["pair_order"] == "reference-candidate"]
            cand_first = [float(r["pct_delta_wall_seconds"]) for r in rs if r["pair_order"] == "candidate-reference"]
            ref_first_med = median(ref_first) if ref_first else 0.0
            cand_first_med = median(cand_first) if cand_first else 0.0
            out = {
                "slab_depth": depth, "pairs": len(rs),
                "median_pct_delta_wall_seconds": median(wall_pct), "mad_pct_delta_wall_seconds": mad(wall_pct), "iqr_pct_delta_wall_seconds": iqr(wall_pct),
                "median_pct_delta_local_sweep_seconds": median(sweep_pct), "mad_pct_delta_local_sweep_seconds": mad(sweep_pct), "iqr_pct_delta_local_sweep_seconds": iqr(sweep_pct),
                "candidate_faster_wall_pairs": wall_wins, "non_tied_wall_pairs": wall_n, "candidate_faster_wall_fraction": wall_wins / wall_n if wall_n else 0.0, "two_sided_sign_test_p_wall": wall_p,
                "candidate_faster_local_sweep_pairs": sweep_wins, "non_tied_local_sweep_pairs": sweep_n, "candidate_faster_local_sweep_fraction": sweep_wins / sweep_n if sweep_n else 0.0, "two_sided_sign_test_p_local_sweep": sweep_p,
                "median_candidate_two_hop_parent_checks": median(float(r["candidate_two_hop_parent_checks"]) for r in rs),
                "median_candidate_two_hop_parent_hits": median(float(r["candidate_two_hop_parent_hits"]) for r in rs),
                "median_candidate_two_hop_hit_rate": median(float(r["candidate_two_hop_hit_rate"]) for r in rs),
                "median_neighbor_finds_avoided_vs_reference": median(float(r["neighbor_finds_avoided_vs_reference"]) for r in rs),
                "median_delta_neighbor_find_parent_steps": median(float(r["delta_neighbor_find_parent_steps"]) for r in rs),
                "median_reference_same_root_neighbor_find_two_hop": median(float(r["reference_same_root_neighbor_find_two_hop"]) for r in rs),
                "median_reference_same_root_neighbor_find_gt_two_hop": median(float(r["reference_same_root_neighbor_find_gt_two_hop"]) for r in rs),
                "median_pct_delta_wall_reference_first": ref_first_med,
                "median_pct_delta_wall_candidate_first": cand_first_med,
                "wall_order_effect_pp": cand_first_med - ref_first_med,
            }
            writer.writerow(out)

    print(f"wrote {args.output}")
    print(f"wrote {summary}")
    print(f"wrote {pairs_path}")
    print(f"wrote {paired_summary}")
    print(f"wrote {diagnostics_path}")
    print(f"wrote {diagnostics_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
