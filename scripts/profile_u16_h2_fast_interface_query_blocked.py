#!/usr/bin/env python3
"""Blocked paired benchmark for the U16 H2 fast interface query.

Design (pre-registered for v23.5.2):
  * d16 only
  * 5 blocks
  * 7 measured A/B pairs per block
  * 1 unmeasured warmup pair at the start of every block
  * alternating A/B order both within and across blocks
  * optional Linux perf-stat counters (auto-probed)
  * a cooldown between blocks

Reference: BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=0
Candidate: BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=1
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
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
    "zero_persistence_pairs_elided",
    "union_attempts",
    "successful_unions",
    "same_root_unions",
    "neighbor_find_calls",
    "neighbor_find_parent_steps",
)
INVARIANTS = COUNTERS
PERF_EVENTS = ("cycles", "instructions", "branches", "branch-misses")


def command(binary: Path, stack: Path, depth: int, conn: int, out: Path) -> list[str]:
    return [
        str(binary), str(stack), str(depth), str(conn),
        "h2-scalar-hierarchical-stream", str(out),
        "--local-h2-birth-state", "compact",
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", "packed",
        "--h2-hier-cross-storage", "direct",
        "--h2-hier-outside-structural-pruning", "off",
        "--neighbor-root-check", "parent-shortcut",
        "--interface-state", "root-invariant",
    ]


def parse_leaf(stdout: str) -> dict[str, str]:
    line = next((x for x in stdout.splitlines() if x.startswith(LEAF_PREFIX)), None)
    if line is None:
        raise RuntimeError("missing PROFILE_U16_PERSIST_LEAF")
    out: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def req(pat: re.Pattern[str], text: str, label: str) -> str:
    m = pat.search(text)
    if m is None:
        raise RuntimeError(f"missing {label}")
    return m.group(1)


def median(xs) -> float:
    vals = [float(x) for x in xs]
    return float(statistics.median(vals)) if vals else 0.0


def mad(xs) -> float:
    vals = [float(x) for x in xs]
    if not vals:
        return 0.0
    c = median(vals)
    return median(abs(x - c) for x in vals)


def iqr(xs) -> float:
    vals = [float(x) for x in xs]
    if len(vals) < 2:
        return 0.0
    q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    return float(q3 - q1)


def pct(a: float, b: float) -> float:
    return 100.0 * (b - a) / a if a else 0.0


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    vals = [x for x in deltas if x != 0.0]
    n = len(vals)
    if not n:
        return 0, 0, 1.0
    wins = sum(x < 0 for x in vals)
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return wins, n, min(1.0, 2.0 * tail)


def pair_order(global_pair_id: int, start_with: str) -> tuple[str, str]:
    first = start_with if global_pair_id % 2 else (
        "candidate" if start_with == "reference" else "reference"
    )
    return first, ("candidate" if first == "reference" else "reference")


def read_loadavg() -> tuple[float, float, float]:
    try:
        parts = Path("/proc/loadavg").read_text().split()
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return 0.0, 0.0, 0.0


def read_freq_khz() -> tuple[float, float, float]:
    vals: list[float] = []
    for p in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
        try:
            vals.append(float(p.read_text().strip()))
        except Exception:
            pass
    if not vals:
        return 0.0, 0.0, 0.0
    return min(vals), sum(vals) / len(vals), max(vals)


def parse_perf(path: Path) -> dict[str, float]:
    out = {e.replace("-", "_"): 0.0 for e in PERF_EVENTS}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        cols = [x.strip() for x in line.split(",")]
        if len(cols) < 3:
            continue
        event = next((e for e in PERF_EVENTS if e in cols), None)
        if event is None:
            continue
        raw = cols[0].replace(" ", "")
        if raw.startswith("<") or not raw:
            continue
        try:
            out[event.replace("-", "_")] = float(raw)
        except ValueError:
            pass
    return out


def probe_perf(perf_bin: str, td: Path) -> tuple[bool, str]:
    if not perf_bin or shutil.which(perf_bin) is None:
        return False, "perf executable not found"
    out = td / "perf_probe.csv"
    proc = subprocess.run(
        [perf_bin, "stat", "--no-big-num", "-x,", "-o", str(out),
         "-e", ",".join(PERF_EVENTS), "--", "true"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip().replace("\n", " ")
        return False, msg[:300] or f"perf probe failed rc={proc.returncode}"
    parsed = parse_perf(out)
    if parsed["cycles"] <= 0 or parsed["instructions"] <= 0:
        return False, "perf probe returned no usable cycles/instructions"
    return True, "ok"


def run_once(
    cmd: list[str], timing_path: Path, perf_path: Path, fast: bool,
    perf_enabled: bool, perf_bin: str,
) -> dict[str, object]:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"] = "1"
    env["BETTI_PERSIST_H2_ROOT_DEDUP"] = "0"
    env["BETTI_PERSIST_H2_FAST_INTERFACE_QUERY"] = "1" if fast else "0"

    load1, load5, load15 = read_loadavg()
    fmin, favg, fmax = read_freq_khz()

    wrapped = cmd
    if perf_enabled:
        wrapped = [
            perf_bin, "stat", "--no-big-num", "-x,", "-o", str(perf_path),
            "-e", ",".join(PERF_EVENTS), "--", *cmd,
        ]

    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(timing_path), *wrapped],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    wall = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(wrapped)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    expected = "on" if fast else "off"
    if f"h2_fast_interface_query={expected}" not in proc.stdout:
        raise RuntimeError(f"expected h2_fast_interface_query={expected}")
    if "h2_root_dedup=off" not in proc.stdout:
        raise RuntimeError("benchmark requires root dedup off")

    leaf = parse_leaf(proc.stdout)
    t = timing_path.read_text()
    out: dict[str, object] = {
        "wall_seconds": wall,
        "user_seconds": float(req(USER_RE, t, "user time")),
        "system_seconds": float(req(SYS_RE, t, "system time")),
        "max_rss_mib": float(req(RSS_RE, t, "RSS")) / 1024.0,
        "scalar_order_seconds": float(leaf["scalar_order_seconds"]),
        "local_sweep_seconds": float(leaf["local_sweep_seconds"]),
        "loadavg_1": load1,
        "loadavg_5": load5,
        "loadavg_15": load15,
        "cpu_freq_min_khz": fmin,
        "cpu_freq_avg_khz": favg,
        "cpu_freq_max_khz": fmax,
        "perf_available": int(perf_enabled),
    }
    for k in COUNTERS:
        out[k] = int(leaf[k])
    perf = parse_perf(perf_path) if perf_enabled else {
        "cycles": 0.0, "instructions": 0.0, "branches": 0.0, "branch_misses": 0.0
    }
    out.update(perf)
    if perf_enabled and float(out["instructions"]) > 0:
        out["cycles_per_instruction"] = float(out["cycles"]) / float(out["instructions"])
        out["branch_miss_rate"] = (
            float(out["branch_misses"]) / float(out["branches"])
            if float(out["branches"]) > 0 else 0.0
        )
    else:
        out["cycles_per_instruction"] = 0.0
        out["branch_miss_rate"] = 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depth", type=int, default=16)
    ap.add_argument("--foreground-connectivity", type=int, default=26, choices=(6, 18, 26))
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--pairs-per-block", type=int, default=7)
    ap.add_argument("--warmup-pairs-per-block", type=int, default=1)
    ap.add_argument("--cooldown-seconds", type=float, default=10.0)
    ap.add_argument("--start-with", choices=("reference", "candidate"), default="reference")
    ap.add_argument("--perf", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--perf-binary", default="perf")
    ap.add_argument("--output", type=Path,
                    default=Path("profiles/v23_5_2_u16_h2_fast_interface_query_d16_blocked.csv"))
    args = ap.parse_args()

    if args.slab_depth <= 0:
        ap.error("--slab-depth must be positive")
    if args.blocks < 2 or args.pairs_per_block < 2:
        ap.error("require blocks>=2 and pairs-per-block>=2")
    if args.warmup_pairs_per_block < 0 or args.cooldown_seconds < 0:
        ap.error("warmup/cooldown must be nonnegative")

    stack = args.stack.resolve()
    binary = args.binary.resolve()
    if not stack.exists():
        ap.error(f"stack does not exist: {stack}")
    if not binary.exists():
        ap.error(f"binary does not exist: {binary}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    sequence = 0
    global_pair_id = 0

    with tempfile.TemporaryDirectory(prefix="u16_h2_fast_iface_blocked_") as td_raw:
        td = Path(td_raw)
        perf_ok, perf_reason = probe_perf(args.perf_binary, td)
        if args.perf == "off":
            perf_enabled = False
            perf_reason = "disabled by --perf off"
        elif args.perf == "on":
            if not perf_ok:
                raise RuntimeError(f"--perf on requested but perf is unavailable: {perf_reason}")
            perf_enabled = True
        else:
            perf_enabled = perf_ok
        print(f"perf_available={str(perf_enabled).lower()} reason={perf_reason}")

        for block_id in range(1, args.blocks + 1):
            if block_id > 1 and args.cooldown_seconds > 0:
                print(f"block={block_id}: cooldown {args.cooldown_seconds:g}s")
                time.sleep(args.cooldown_seconds)

            print(f"=== block {block_id}/{args.blocks} ===")
            for warm in range(1, args.warmup_pairs_per_block + 1):
                global_pair_id += 1
                order = pair_order(global_pair_id, args.start_with)
                for backend in order:
                    sequence += 1
                    fast = backend == "candidate"
                    run_once(
                        command(binary, stack, args.slab_depth, args.foreground_connectivity,
                                td / f"warm_b{block_id}_{warm}_{backend}.csv"),
                        td / f"warm_b{block_id}_{warm}_{backend}.time",
                        td / f"warm_b{block_id}_{warm}_{backend}.perf",
                        fast, perf_enabled, args.perf_binary,
                    )
                print(f"block={block_id} warmup_pair={warm} order={'-'.join(order)}")

            for pair_in_block in range(1, args.pairs_per_block + 1):
                global_pair_id += 1
                order = pair_order(global_pair_id, args.start_with)
                current: dict[str, dict[str, object]] = {}
                for pos, backend in enumerate(order, 1):
                    sequence += 1
                    fast = backend == "candidate"
                    r = run_once(
                        command(binary, stack, args.slab_depth, args.foreground_connectivity,
                                td / f"b{block_id}_p{pair_in_block}_{backend}.csv"),
                        td / f"b{block_id}_p{pair_in_block}_{backend}.time",
                        td / f"b{block_id}_p{pair_in_block}_{backend}.perf",
                        fast, perf_enabled, args.perf_binary,
                    )
                    row = {
                        "backend": backend,
                        "slab_depth": args.slab_depth,
                        "block_id": block_id,
                        "pair_in_block": pair_in_block,
                        "global_pair_id": global_pair_id,
                        "run_position": pos,
                        "pair_order": "-".join(order),
                        "sequence": sequence,
                        **r,
                    }
                    rows.append(row)
                    current[backend] = row
                    print(
                        f"block={block_id} pair={pair_in_block} {'/'.join(order)} pos={pos} {backend}: "
                        f"wall={float(r['wall_seconds']):.3f}s sweep={float(r['local_sweep_seconds']):.3f}s "
                        f"load1={float(r['loadavg_1']):.2f}"
                    )

                ref, cand = current["reference"], current["candidate"]
                for m in INVARIANTS:
                    if int(ref[m]) != int(cand[m]):
                        raise RuntimeError(
                            f"invariant failed block={block_id} pair={pair_in_block}: "
                            f"{m} {ref[m]} != {cand[m]}"
                        )
                pr: dict[str, object] = {
                    "slab_depth": args.slab_depth,
                    "block_id": block_id,
                    "pair_in_block": pair_in_block,
                    "global_pair_id": global_pair_id,
                    "pair_order": "-".join(order),
                    "perf_available": int(perf_enabled),
                }
                for m in (
                    "wall_seconds", "local_sweep_seconds", "user_seconds", "system_seconds",
                    "max_rss_mib", "cycles", "instructions", "branches", "branch_misses",
                    "cycles_per_instruction", "branch_miss_rate",
                ):
                    a = float(ref[m]); b = float(cand[m])
                    pr[f"reference_{m}"] = a
                    pr[f"candidate_{m}"] = b
                    pr[f"delta_{m}"] = b - a
                    pr[f"pct_delta_{m}"] = pct(a, b)
                pr["reference_loadavg_1"] = ref["loadavg_1"]
                pr["candidate_loadavg_1"] = cand["loadavg_1"]
                pr["reference_cpu_freq_avg_khz"] = ref["cpu_freq_avg_khz"]
                pr["candidate_cpu_freq_avg_khz"] = cand["cpu_freq_avg_khz"]
                pairs.append(pr)

    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    pairs_path = args.output.with_name(args.output.stem + "_pairs.csv")
    with pairs_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]))
        w.writeheader(); w.writerows(pairs)

    blocks_path = args.output.with_name(args.output.stem + "_blocks.csv")
    block_fields = [
        "block_id", "pairs", "median_pct_delta_wall_seconds", "median_pct_delta_local_sweep_seconds",
        "candidate_faster_wall_pairs", "non_tied_wall_pairs", "two_sided_sign_test_p_wall",
        "median_pct_delta_cycles", "median_pct_delta_instructions", "median_pct_delta_branches",
        "median_pct_delta_branch_misses", "perf_available",
    ]
    with blocks_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=block_fields); w.writeheader()
        for block_id in range(1, args.blocks + 1):
            rs = [r for r in pairs if int(r["block_id"]) == block_id]
            wall = [float(r["pct_delta_wall_seconds"]) for r in rs]
            sweep = [float(r["pct_delta_local_sweep_seconds"]) for r in rs]
            ww, wn, wp = sign_test([float(r["delta_wall_seconds"]) for r in rs])
            out = {
                "block_id": block_id,
                "pairs": len(rs),
                "median_pct_delta_wall_seconds": median(wall),
                "median_pct_delta_local_sweep_seconds": median(sweep),
                "candidate_faster_wall_pairs": ww,
                "non_tied_wall_pairs": wn,
                "two_sided_sign_test_p_wall": wp,
                "median_pct_delta_cycles": median(float(r["pct_delta_cycles"]) for r in rs) if perf_enabled else 0.0,
                "median_pct_delta_instructions": median(float(r["pct_delta_instructions"]) for r in rs) if perf_enabled else 0.0,
                "median_pct_delta_branches": median(float(r["pct_delta_branches"]) for r in rs) if perf_enabled else 0.0,
                "median_pct_delta_branch_misses": median(float(r["pct_delta_branch_misses"]) for r in rs) if perf_enabled else 0.0,
                "perf_available": int(perf_enabled),
            }
            w.writerow(out)
            print(
                f"block {block_id}: wall median {out['median_pct_delta_wall_seconds']:+.3f}% "
                f"sweep {out['median_pct_delta_local_sweep_seconds']:+.3f}% wins {ww}/{wn}"
            )

    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    wall = [float(r["pct_delta_wall_seconds"]) for r in pairs]
    sweep = [float(r["pct_delta_local_sweep_seconds"]) for r in pairs]
    ww, wn, wp = sign_test([float(r["delta_wall_seconds"]) for r in pairs])
    sw, sn, sp = sign_test([float(r["delta_local_sweep_seconds"]) for r in pairs])
    rf = [float(r["pct_delta_wall_seconds"]) for r in pairs if r["pair_order"] == "reference-candidate"]
    cf = [float(r["pct_delta_wall_seconds"]) for r in pairs if r["pair_order"] == "candidate-reference"]
    block_rows = list(csv.DictReader(blocks_path.open()))
    negative_blocks = sum(float(r["median_pct_delta_wall_seconds"]) < 0 for r in block_rows)
    summary = {
        "slab_depth": args.slab_depth,
        "blocks": args.blocks,
        "pairs_per_block": args.pairs_per_block,
        "pairs": len(pairs),
        "cooldown_seconds": args.cooldown_seconds,
        "perf_available": int(perf_enabled),
        "perf_probe_reason": perf_reason,
        "median_pct_delta_wall_seconds": median(wall),
        "mad_pct_delta_wall_seconds": mad(wall),
        "iqr_pct_delta_wall_seconds": iqr(wall),
        "median_pct_delta_local_sweep_seconds": median(sweep),
        "mad_pct_delta_local_sweep_seconds": mad(sweep),
        "iqr_pct_delta_local_sweep_seconds": iqr(sweep),
        "candidate_faster_wall_pairs": ww,
        "non_tied_wall_pairs": wn,
        "two_sided_sign_test_p_wall": wp,
        "candidate_faster_local_sweep_pairs": sw,
        "non_tied_local_sweep_pairs": sn,
        "two_sided_sign_test_p_local_sweep": sp,
        "negative_wall_median_blocks": negative_blocks,
        "median_pct_delta_wall_reference_first": median(rf),
        "median_pct_delta_wall_candidate_first": median(cf),
        "wall_order_effect_pp": median(cf) - median(rf),
        "median_pct_delta_cycles": median(float(r["pct_delta_cycles"]) for r in pairs) if perf_enabled else 0.0,
        "median_pct_delta_instructions": median(float(r["pct_delta_instructions"]) for r in pairs) if perf_enabled else 0.0,
        "median_pct_delta_branches": median(float(r["pct_delta_branches"]) for r in pairs) if perf_enabled else 0.0,
        "median_pct_delta_branch_misses": median(float(r["pct_delta_branch_misses"]) for r in pairs) if perf_enabled else 0.0,
    }
    with summary_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary))
        w.writeheader(); w.writerow(summary)

    print(
        f"overall: wall median {summary['median_pct_delta_wall_seconds']:+.3f}% "
        f"sweep {summary['median_pct_delta_local_sweep_seconds']:+.3f}% "
        f"wall wins {ww}/{wn} p={wp:.6f} negative blocks={negative_blocks}/{args.blocks}"
    )
    if perf_enabled:
        print(
            f"perf: cycles {summary['median_pct_delta_cycles']:+.3f}% "
            f"instructions {summary['median_pct_delta_instructions']:+.3f}% "
            f"branches {summary['median_pct_delta_branches']:+.3f}% "
            f"branch-misses {summary['median_pct_delta_branch_misses']:+.3f}%"
        )
    print(f"wrote {args.output}\nwrote {pairs_path}\nwrote {blocks_path}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
