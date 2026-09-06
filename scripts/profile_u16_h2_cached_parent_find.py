#!/usr/bin/env python3
"""Paired A/B benchmark for the U16 H2 cached-parent find candidate.

Reference: --neighbor-root-check parent-shortcut
Candidate: --neighbor-root-check parent-cached-find

Timing runs keep sweep diagnostics off. A separate instrumented pass records
find-depth counters without contaminating the timing comparison.
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

RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
USER_RE = re.compile(r"User time \(seconds\):\s*([0-9.]+)")
SYS_RE = re.compile(r"System time \(seconds\):\s*([0-9.]+)")
LEAF_PREFIX = "PROFILE_U16_PERSIST_LEAF dimension=h2 "

COUNTERS = (
    "zero_persistence_pairs_elided", "union_attempts", "successful_unions", "same_root_unions",
    "direct_parent_checks", "direct_parent_hits", "avoided_neighbor_find_calls",
    "neighbor_find_calls", "neighbor_find_parent_steps", "neighbor_find_zero_hop",
    "neighbor_find_one_hop", "neighbor_find_two_hop", "neighbor_find_gt_two_hop",
    "same_root_neighbor_find_zero_hop", "same_root_neighbor_find_one_hop",
    "same_root_neighbor_find_two_hop", "same_root_neighbor_find_gt_two_hop",
)
INVARIANTS = ("zero_persistence_pairs_elided", "union_attempts", "successful_unions", "same_root_unions")


def mode_for_backend(backend: str) -> str:
    return "parent-shortcut" if backend == "reference" else "parent-cached-find"


def command(binary: Path, stack: Path, depth: int, conn: int, out: Path, mode: str, diagnostics: bool = False) -> list[str]:
    cmd = [
        str(binary), str(stack), str(depth), str(conn), "h2-scalar-hierarchical-stream", str(out),
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
    line = next((x for x in stdout.splitlines() if x.startswith(LEAF_PREFIX)), None)
    if line is None:
        raise RuntimeError("missing PROFILE_U16_PERSIST_LEAF H2 line")
    out: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def req(pattern: re.Pattern[str], text: str, label: str) -> str:
    m = pattern.search(text)
    if m is None:
        raise RuntimeError(f"missing {label} in /usr/bin/time output")
    return m.group(1)


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
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    leaf = parse_leaf(proc.stdout)
    if leaf.get("neighbor_root_check") != expected_mode:
        raise RuntimeError(f"requested {expected_mode}, profile reported {leaf.get('neighbor_root_check')!r}")
    if leaf.get("root_dedup") != "off":
        raise RuntimeError("cached-parent benchmark requires root_dedup=off")
    timing = timing_path.read_text()
    out: dict[str, object] = {
        "wall_seconds": wall,
        "max_rss_mib": float(req(RSS_RE, timing, "maximum RSS")) / 1024.0,
        "user_seconds": float(req(USER_RE, timing, "user time")),
        "system_seconds": float(req(SYS_RE, timing, "system time")),
        "neighbor_root_check": leaf["neighbor_root_check"],
        "scalar_order_seconds": float(leaf["scalar_order_seconds"]),
        "local_sweep_seconds": float(leaf["local_sweep_seconds"]),
    }
    for field in COUNTERS:
        out[field] = int(leaf[field])
    return out


def median(xs): return float(statistics.median(float(x) for x in xs))
def mad(xs):
    vals = [float(x) for x in xs]; c = median(vals); return median(abs(x-c) for x in vals)
def iqr(xs):
    vals = [float(x) for x in xs]
    if len(vals) < 2: return 0.0
    q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive"); return float(q3-q1)
def pct(a: float, b: float) -> float: return 100.0*(b-a)/a if a else 0.0


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    vals = [x for x in deltas if x != 0.0]; n = len(vals)
    if not n: return 0, 0, 1.0
    wins = sum(x < 0.0 for x in vals); k = min(wins, n-wins)
    tail = sum(math.comb(n, i) for i in range(k+1)) / (2**n)
    return wins, n, min(1.0, 2.0*tail)


def pair_order(pair_id: int, start_with: str) -> tuple[str, str]:
    first = start_with if pair_id % 2 else ("candidate" if start_with == "reference" else "reference")
    return first, ("candidate" if first == "reference" else "reference")


def assert_invariants(depth: int, pair_id: int, ref: dict[str, object], cand: dict[str, object]) -> None:
    for metric in INVARIANTS:
        if int(ref[metric]) != int(cand[metric]):
            raise RuntimeError(
                f"invariant failed d={depth} pair={pair_id}: {metric} "
                f"reference={ref[metric]} candidate={cand[metric]}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired U16 H2 parent-shortcut vs cached-parent-find benchmark")
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--start-with", choices=("reference", "candidate"), default="reference")
    ap.add_argument("--diagnostic-repeats", type=int, default=1)
    ap.add_argument("--output", type=Path, default=Path("profiles/v23_3_u16_h2_cached_parent_find_paired.csv"))
    args = ap.parse_args()
    if args.warmup < 0 or args.repeats < 2 or args.diagnostic_repeats < 1:
        ap.error("require warmup>=0, repeats>=2, diagnostic-repeats>=1")
    if any(d <= 0 for d in args.slab_depths) or len(set(args.slab_depths)) != len(args.slab_depths):
        ap.error("slab depths must be positive and unique")
    stack = args.stack.resolve(); binary = args.binary.resolve()
    if not stack.exists(): ap.error(f"stack does not exist: {stack}")
    if not binary.exists(): ap.error(f"binary does not exist: {binary}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []; pairs: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="u16_h2_cached_parent_paired_") as raw:
        td = Path(raw); seq = 0
        for depth in args.slab_depths:
            for warm in range(1, args.warmup+1):
                for backend in pair_order(warm, args.start_with):
                    seq += 1; mode = mode_for_backend(backend)
                    print(f"warmup={warm} {backend} d={depth} mode={mode}")
                    run_once(command(binary, stack, depth, args.foreground_connectivity, td/f"w{seq}.csv", mode), td/f"w{seq}.time", mode)
            for pair_id in range(1, args.repeats+1):
                order = pair_order(pair_id, args.start_with); current = {}
                for pos, backend in enumerate(order, 1):
                    seq += 1; mode = mode_for_backend(backend)
                    r = run_once(command(binary, stack, depth, args.foreground_connectivity, td/f"r{seq}.csv", mode), td/f"r{seq}.time", mode)
                    row = {"backend": backend, "slab_depth": depth, "pair_id": pair_id, "run_position": pos, "pair_order": "-".join(order), "sequence": seq, **r}
                    rows.append(row); current[backend] = row
                    print(f"pair={pair_id:02d} {'/'.join(order)} pos={pos} {backend} d={depth}: wall={r['wall_seconds']:.3f}s sweep={r['local_sweep_seconds']:.3f}s")
                ref, cand = current["reference"], current["candidate"]
                assert_invariants(depth, pair_id, ref, cand)
                pr: dict[str, object] = {"slab_depth": depth, "pair_id": pair_id, "pair_order": "-".join(order)}
                for metric in ("wall_seconds", "local_sweep_seconds", "user_seconds", "system_seconds", "max_rss_mib"):
                    a, b = float(ref[metric]), float(cand[metric]); pr[f"reference_{metric}"] = a; pr[f"candidate_{metric}"] = b; pr[f"delta_{metric}"] = b-a; pr[f"pct_delta_{metric}"] = pct(a,b)
                pairs.append(pr)

    # Separate diagnostics: these are structural only, never used for timing claims.
    diag_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="u16_h2_cached_parent_diag_") as raw:
        td = Path(raw); seq = 0
        for depth in args.slab_depths:
            for drun in range(1, args.diagnostic_repeats+1):
                current = {}
                for backend in ("reference", "candidate"):
                    seq += 1; mode = mode_for_backend(backend)
                    r = run_once(command(binary, stack, depth, args.foreground_connectivity, td/f"d{seq}.csv", mode, True), td/f"d{seq}.time", mode)
                    row = {"backend": backend, "slab_depth": depth, "diagnostic_run": drun, **r}; diag_rows.append(row); current[backend] = row
                    print(f"diagnostic={drun} {backend} d={depth}: finds={r['neighbor_find_calls']} zero_hop={r['neighbor_find_zero_hop']} parent_steps={r['neighbor_find_parent_steps']}")
                assert_invariants(depth, drun, current["reference"], current["candidate"])

    # Raw timing rows.
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    # Backend summary.
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    fields = ["backend", "slab_depth", "repeats"]
    metrics = ("wall_seconds", "local_sweep_seconds", "user_seconds", "system_seconds", "max_rss_mib")
    for m in metrics: fields += [f"median_{m}", f"mad_{m}", f"iqr_{m}"]
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for backend in ("reference", "candidate"):
            for depth in args.slab_depths:
                rs = [r for r in rows if r["backend"] == backend and int(r["slab_depth"]) == depth]
                out: dict[str, object] = {"backend": backend, "slab_depth": depth, "repeats": len(rs)}
                for m in metrics:
                    vals = [float(r[m]) for r in rs]; out[f"median_{m}"] = median(vals); out[f"mad_{m}"] = mad(vals); out[f"iqr_{m}"] = iqr(vals)
                writer.writerow(out)

    pairs_path = args.output.with_name(args.output.stem + "_pairs.csv")
    with pairs_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)

    paired_path = args.output.with_name(args.output.stem + "_paired_summary.csv")
    pfields = [
        "slab_depth", "pairs", "median_pct_delta_wall_seconds", "mad_pct_delta_wall_seconds", "iqr_pct_delta_wall_seconds",
        "median_pct_delta_local_sweep_seconds", "mad_pct_delta_local_sweep_seconds", "iqr_pct_delta_local_sweep_seconds",
        "candidate_faster_wall_pairs", "non_tied_wall_pairs", "candidate_faster_wall_fraction", "two_sided_sign_test_p_wall",
        "candidate_faster_local_sweep_pairs", "non_tied_local_sweep_pairs", "candidate_faster_local_sweep_fraction", "two_sided_sign_test_p_local_sweep",
        "median_pct_delta_wall_reference_first", "median_pct_delta_wall_candidate_first", "wall_order_effect_pp",
    ]
    with paired_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=pfields); writer.writeheader()
        for depth in args.slab_depths:
            rs = [r for r in pairs if int(r["slab_depth"]) == depth]
            wall = [float(r["pct_delta_wall_seconds"]) for r in rs]; sweep = [float(r["pct_delta_local_sweep_seconds"]) for r in rs]
            ww, wn, wp = sign_test([float(r["delta_wall_seconds"]) for r in rs]); sw, sn, sp = sign_test([float(r["delta_local_sweep_seconds"]) for r in rs])
            rf = [float(r["pct_delta_wall_seconds"]) for r in rs if r["pair_order"] == "reference-candidate"]
            cf = [float(r["pct_delta_wall_seconds"]) for r in rs if r["pair_order"] == "candidate-reference"]
            rfm = median(rf) if rf else 0.0; cfm = median(cf) if cf else 0.0
            writer.writerow({
                "slab_depth": depth, "pairs": len(rs),
                "median_pct_delta_wall_seconds": median(wall), "mad_pct_delta_wall_seconds": mad(wall), "iqr_pct_delta_wall_seconds": iqr(wall),
                "median_pct_delta_local_sweep_seconds": median(sweep), "mad_pct_delta_local_sweep_seconds": mad(sweep), "iqr_pct_delta_local_sweep_seconds": iqr(sweep),
                "candidate_faster_wall_pairs": ww, "non_tied_wall_pairs": wn, "candidate_faster_wall_fraction": ww/wn if wn else 0.0, "two_sided_sign_test_p_wall": wp,
                "candidate_faster_local_sweep_pairs": sw, "non_tied_local_sweep_pairs": sn, "candidate_faster_local_sweep_fraction": sw/sn if sn else 0.0, "two_sided_sign_test_p_local_sweep": sp,
                "median_pct_delta_wall_reference_first": rfm, "median_pct_delta_wall_candidate_first": cfm, "wall_order_effect_pp": cfm-rfm,
            })

    diag_path = args.output.with_name(args.output.stem + "_diagnostics.csv")
    dfields = ["backend", "slab_depth", "diagnostic_run", "neighbor_root_check", *COUNTERS]
    with diag_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=dfields, extrasaction="ignore"); writer.writeheader(); writer.writerows(diag_rows)
    diag_summary_path = args.output.with_name(args.output.stem + "_diagnostics_summary.csv")
    dsfields = [
        "slab_depth", "reference_neighbor_find_calls", "candidate_neighbor_find_calls", "neighbor_finds_avoided",
        "reference_neighbor_find_zero_hop", "candidate_neighbor_find_zero_hop",
        "reference_neighbor_find_parent_steps", "candidate_neighbor_find_parent_steps", "parent_steps_delta",
        "reference_direct_parent_hits", "candidate_direct_parent_hits",
    ]
    with diag_summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=dsfields); writer.writeheader()
        for depth in args.slab_depths:
            rr = [r for r in diag_rows if r["backend"] == "reference" and int(r["slab_depth"]) == depth]
            cc = [r for r in diag_rows if r["backend"] == "candidate" and int(r["slab_depth"]) == depth]
            rf = median(r["neighbor_find_calls"] for r in rr); cf = median(r["neighbor_find_calls"] for r in cc)
            rp = median(r["neighbor_find_parent_steps"] for r in rr); cp = median(r["neighbor_find_parent_steps"] for r in cc)
            writer.writerow({
                "slab_depth": depth, "reference_neighbor_find_calls": rf, "candidate_neighbor_find_calls": cf, "neighbor_finds_avoided": rf-cf,
                "reference_neighbor_find_zero_hop": median(r["neighbor_find_zero_hop"] for r in rr), "candidate_neighbor_find_zero_hop": median(r["neighbor_find_zero_hop"] for r in cc),
                "reference_neighbor_find_parent_steps": rp, "candidate_neighbor_find_parent_steps": cp, "parent_steps_delta": cp-rp,
                "reference_direct_parent_hits": median(r["direct_parent_hits"] for r in rr), "candidate_direct_parent_hits": median(r["direct_parent_hits"] for r in cc),
            })

    for path in (args.output, summary_path, pairs_path, paired_path, diag_path, diag_summary_path): print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
