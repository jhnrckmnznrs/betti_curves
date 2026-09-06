#!/usr/bin/env python3
"""Evaluate the pre-registered v23.5.1 d16 confirmation gate.

The input is the paired-summary CSV produced by
scripts/profile_u16_h2_fast_interface_query.py for 31 d16 pairs.

Promotion gate (all required):
  * exactly 31 pairs at slab depth 16;
  * median paired wall delta < 0%;
  * median paired local-sweep delta < 0%;
  * candidate faster in at least 23/31 non-tied wall pairs;
  * exact two-sided wall sign-test p < 0.05;
  * median wall delta is < 0% in both run-order strata.

This evaluator intentionally does not change source defaults. It only records a
PROMOTE_CANDIDATE or KEEP_EXPERIMENTAL recommendation for the follow-up freeze.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"missing/invalid {key!r} in paired summary") from exc


def i(row: dict[str, str], key: str) -> int:
    return int(round(f(row, key)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paired_summary", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    with args.paired_summary.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    d16_rows = [row for row in rows if i(row, "slab_depth") == 16]
    if len(d16_rows) != 1:
        raise SystemExit(f"expected exactly one d16 row, got {len(d16_rows)}")
    r = d16_rows[0]

    depth = i(r, "slab_depth")
    pairs = i(r, "pairs")
    wall = f(r, "median_pct_delta_wall_seconds")
    sweep = f(r, "median_pct_delta_local_sweep_seconds")
    wins = i(r, "candidate_faster_wall_pairs")
    non_tied = i(r, "non_tied_wall_pairs")
    p_wall = f(r, "two_sided_sign_test_p_wall")
    ref_first = f(r, "median_pct_delta_wall_reference_first")
    cand_first = f(r, "median_pct_delta_wall_candidate_first")

    checks = [
        ("slab_depth_is_16", depth == 16, f"d={depth}"),
        ("exactly_31_pairs", pairs == 31, f"pairs={pairs}"),
        ("median_wall_improves", wall < 0.0, f"median_wall={wall:+.4f}%"),
        ("median_sweep_improves", sweep < 0.0, f"median_sweep={sweep:+.4f}%"),
        ("wall_wins_at_least_23", wins >= 23 and non_tied == 31, f"wins={wins}/{non_tied}"),
        ("wall_sign_test_p_lt_0_05", p_wall < 0.05, f"p={p_wall:.8g}"),
        ("reference_first_median_improves", ref_first < 0.0, f"ref-first={ref_first:+.4f}%"),
        ("candidate_first_median_improves", cand_first < 0.0, f"cand-first={cand_first:+.4f}%"),
    ]
    passed = all(ok for _, ok, _ in checks)
    verdict = "PROMOTE_CANDIDATE" if passed else "KEEP_EXPERIMENTAL"

    lines = [
        "v23.5.1 fast-interface-query confirmation",
        f"verdict={verdict}",
        "",
    ]
    for name, ok, detail in checks:
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    lines += [
        "",
        "Interpretation:",
        "PROMOTE_CANDIDATE means the pre-registered performance gate passed;",
        "the source default should only be flipped in a separate freeze change after",
        "reviewing exact-equivalence output and the raw paired CSV for anomalies.",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    out = args.output or args.paired_summary.with_name(
        args.paired_summary.stem.replace("_paired_summary", "") + "_decision.txt"
    )
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
