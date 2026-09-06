#!/usr/bin/env python3
"""Evaluate the pre-registered v23.5.2 blocked confirmation gate."""
from __future__ import annotations
import argparse, csv
from pathlib import Path


def truth(v: bool) -> str:
    return "PASS" if v else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.summary.open()))
    if len(rows) != 1:
        raise SystemExit("expected exactly one summary row")
    r = rows[0]
    pairs = int(r["pairs"])
    blocks = int(r["blocks"])
    ppb = int(r["pairs_per_block"])
    wall = float(r["median_pct_delta_wall_seconds"])
    sweep = float(r["median_pct_delta_local_sweep_seconds"])
    wins = int(r["candidate_faster_wall_pairs"])
    p = float(r["two_sided_sign_test_p_wall"])
    neg_blocks = int(r["negative_wall_median_blocks"])
    rf = float(r["median_pct_delta_wall_reference_first"])
    cf = float(r["median_pct_delta_wall_candidate_first"])
    perf = bool(int(r["perf_available"]))
    cyc = float(r["median_pct_delta_cycles"])
    ins = float(r["median_pct_delta_instructions"])

    checks = [
        ("design_is_5x7", blocks == 5 and ppb == 7 and pairs == 35),
        ("median_wall_is_faster", wall < 0.0),
        ("median_local_sweep_is_faster", sweep < 0.0),
        ("wall_wins_at_least_24_of_35", wins >= 24),
        ("wall_sign_test_p_lt_0_05", p < 0.05),
        ("at_least_4_of_5_blocks_have_negative_wall_median", neg_blocks >= 4),
        ("reference_first_median_is_faster", rf < 0.0),
        ("candidate_first_median_is_faster", cf < 0.0),
    ]
    timing_pass = all(v for _, v in checks)
    if perf:
        perf_status = "CORROBORATES" if cyc < 0.0 and ins < 0.0 else "MIXED"
    else:
        perf_status = "UNAVAILABLE"

    verdict = "PROMOTE_CANDIDATE" if timing_pass else "KEEP_EXPERIMENTAL"
    lines = [
        "v23.5.2 blocked fast-interface-query confirmation",
        f"verdict={verdict}",
        f"perf_status={perf_status}",
        "",
        f"pairs={pairs} blocks={blocks} pairs_per_block={ppb}",
        f"median_wall_delta_pct={wall:+.6f}",
        f"median_local_sweep_delta_pct={sweep:+.6f}",
        f"candidate_faster_wall_pairs={wins}/{pairs}",
        f"two_sided_sign_test_p_wall={p:.9f}",
        f"negative_wall_median_blocks={neg_blocks}/{blocks}",
        f"reference_first_median_wall_delta_pct={rf:+.6f}",
        f"candidate_first_median_wall_delta_pct={cf:+.6f}",
    ]
    if perf:
        lines += [
            f"median_cycles_delta_pct={cyc:+.6f}",
            f"median_instructions_delta_pct={ins:+.6f}",
        ]
    lines += ["", "pre_registered_checks:"]
    lines += [f"  {name}={truth(ok)}" for name, ok in checks]
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
