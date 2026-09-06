#!/usr/bin/env python3
"""Symbolic check for hierarchical leaf attach early-finalization.

The optimized leaf rule replaces a non-promoting boundary/internal Attach by a
finalized child->current-boundary-parent record. Higher-level parent changes are
then repaired by the existing deferred-parent resolver. This script checks the
strict filtration-time rule used by that repair against direct replay.
"""

from __future__ import annotations

import argparse
import random


def resolve_h0(parent: int, transitions: list[tuple[int, int, int]], death: int) -> int:
    redirect: dict[int, int] = {}
    for value, child, survivor in sorted(transitions):
        if value < death:
            redirect[child] = survivor
    seen = set()
    while parent in redirect:
        if parent in seen:
            raise AssertionError("cycle")
        seen.add(parent)
        parent = redirect[parent]
    return parent


def direct_h0(parent: int, transitions: list[tuple[int, int, int]], death: int) -> int:
    current = parent
    for value, child, survivor in sorted(transitions):
        if value >= death:
            break
        if current == child:
            current = survivor
    return current


def resolve_h2(parent: int, transitions: list[tuple[int, int, int]], death: int) -> int:
    redirect: dict[int, int] = {}
    for value, child, survivor in sorted(transitions, reverse=True):
        if value > death:
            redirect[child] = survivor
    seen = set()
    while parent in redirect:
        if parent in seen:
            raise AssertionError("cycle")
        seen.add(parent)
        parent = redirect[parent]
    return parent


def direct_h2(parent: int, transitions: list[tuple[int, int, int]], death: int) -> int:
    current = parent
    for value, child, survivor in sorted(transitions, reverse=True):
        if value <= death:
            break
        if current == child:
            current = survivor
    return current


def one_case(rng: random.Random) -> None:
    death = rng.randrange(1, 65535)

    # H0 redirect chains are created in increasing filtration order: once a
    # branch dies, only its survivor can die later.
    h0_parent = 1
    h0_transitions: list[tuple[int, int, int]] = []
    h0_values = sorted(rng.sample(range(0, 65536), rng.randrange(1, 16)))
    child = h0_parent
    for next_id, value in enumerate(h0_values, start=2):
        h0_transitions.append((value, child, next_id))
        child = next_id

    if resolve_h0(h0_parent, h0_transitions, death) != direct_h0(
        h0_parent, h0_transitions, death
    ):
        raise AssertionError(("H0", death, h0_transitions))

    # H2 is the reversed filtration, so valid redirect chains are created in
    # decreasing threshold order.
    h2_parent = 1
    h2_transitions: list[tuple[int, int, int]] = []
    h2_values = sorted(
        rng.sample(range(0, 65536), rng.randrange(1, 16)), reverse=True
    )
    child = h2_parent
    for next_id, value in enumerate(h2_values, start=2):
        h2_transitions.append((value, child, next_id))
        child = next_id

    if resolve_h2(h2_parent, h2_transitions, death) != direct_h2(
        h2_parent, h2_transitions, death
    ):
        raise AssertionError(("H2", death, h2_transitions))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=20260905)
    args = p.parse_args()
    rng = random.Random(args.seed)
    for _ in range(args.cases):
        one_case(rng)
    print(f"PASS: {args.cases} randomized H0/H2 early-finalization repair cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
