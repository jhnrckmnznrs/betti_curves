#!/usr/bin/env python3
"""Symbolic equivalence for H2 zero-persistence pair elision.

The optimized path removes a finite pair at the union decision when birth == death.
The reference path materializes that pair and filters it immediately before emission.
Because the union-find state transition is identical in both paths, the emitted
positive finite pairs and all structural actions must match exactly.
"""
from __future__ import annotations

import argparse
import random


def reference(actions: list[tuple[str, int, int]]) -> tuple[list[tuple[str, int, int]], int]:
    out: list[tuple[str, int, int]] = []
    zero = 0
    for kind, birth, death in actions:
        if kind == "pair":
            if birth < death:
                out.append((kind, birth, death))
            else:
                assert birth == death
                zero += 1
        else:
            out.append((kind, birth, death))
    return out, zero


def optimized(actions: list[tuple[str, int, int]]) -> tuple[list[tuple[str, int, int]], int]:
    out: list[tuple[str, int, int]] = []
    zero = 0
    for kind, birth, death in actions:
        if kind == "pair" and birth == death:
            zero += 1
            continue
        if kind == "pair":
            assert birth < death
        out.append((kind, birth, death))
    return out, zero


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=22016)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    structural = ("attach", "outside", "interface")
    for case in range(args.cases):
        actions: list[tuple[str, int, int]] = []
        threshold = rng.randrange(0, 65536)
        for _ in range(rng.randrange(1, 80)):
            if rng.random() < 0.82:
                # H2 finite pairs satisfy birth <= death; ties are zero persistence.
                birth = rng.randrange(0, threshold + 1) if threshold else 0
                if rng.random() < 0.82:
                    death = birth
                else:
                    death = rng.randrange(birth + 1, 65536) if birth < 65535 else birth
                actions.append(("pair", birth, death))
            else:
                kind = rng.choice(structural)
                actions.append((kind, threshold, rng.randrange(0, 65536)))
        ref, ref_zero = reference(actions)
        opt, opt_zero = optimized(actions)
        if ref != opt or ref_zero != opt_zero:
            raise AssertionError(
                f"case {case} mismatch\nref={ref[:20]} zero={ref_zero}\n"
                f"opt={opt[:20]} zero={opt_zero}"
            )
    print(f"PASS H2: {args.cases} plateau-zero persistence-elision cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
