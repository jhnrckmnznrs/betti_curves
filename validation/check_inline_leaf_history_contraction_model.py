#!/usr/bin/env python3
"""Symbolic equivalence check for v11 inline leaf-history contraction.

The model creates several independent leaf branch forests. It compares:

1. v10-style behavior: send every raw leaf death to one central online
   finalized-history contractor, then process higher-level survivor deaths.
2. v11-style behavior: contract each leaf's zero-persistence history locally,
   then send only retained positive leaf events to the same central contractor.

Leaf branch IDs are disjoint and higher-level events are generated only after
all leaf summaries, matching the hierarchical no-future-reference invariant.
This complements, but does not replace, the real image-level flat-vs-
hierarchical equivalence validator.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

OUTSIDE = -1


def contract(events, *, h2: bool):
    retained = []
    by_parent = defaultdict(list)
    for event in events:
        child, birth, death, parent, parent_birth = event
        zero = birth == death
        handles = by_parent.pop(child, [])
        keep = []
        moved = []
        for index in handles:
            record = retained[index]
            redirect = zero or (record[2] < death if h2 else record[2] > death)
            if redirect:
                record[3] = parent
                record[4] = parent_birth
                moved.append(index)
            else:
                keep.append(index)
        if keep:
            by_parent[child].extend(keep)
        if parent != OUTSIDE and moved:
            by_parent[parent].extend(moved)
        if zero:
            continue

        index = len(retained)
        retained.append([child, birth, death, parent, parent_birth])
        if parent != OUTSIDE:
            by_parent[parent].append(index)
    return [tuple(record) for record in retained]


def normalized(events):
    # Every finite branch dies at most once in these generated forests.
    return {event[0]: event[1:] for event in events}


def make_leaf(rng, start_id: int, count: int, *, h2: bool):
    survivor = start_id
    births = {survivor: rng.randrange(40, 64) if h2 else rng.randrange(0, 24)}
    parent_deaths = {survivor: -1 if h2 else 64}
    parents = {}
    deaths = {}

    # Parents always have lower IDs. Reversing IDs later is therefore a
    # postorder in which all references to an internal parent exist before
    # that parent's finalization event is observed.
    for offset in range(1, count + 1):
        child = start_id + offset
        candidates = list(range(start_id, child))
        parent = rng.choice(candidates)
        p_birth = births[parent]
        p_death = parent_deaths[parent]
        if h2:
            birth = rng.randint(max(0, p_death), p_birth)
            death = rng.randint(max(0, p_death), birth)
        else:
            birth = rng.randint(p_birth, min(63, p_death))
            death = rng.randint(birth, min(63, p_death))
        births[child] = birth
        deaths[child] = death
        parent_deaths[child] = death
        parents[child] = parent

    events = []
    for child in range(start_id + count, start_id, -1):
        parent = parents[child]
        events.append((child, births[child], deaths[child], parent, births[parent]))
    return events, survivor, births[survivor], start_id + count + 1


def make_case(rng, *, leaves: int, branches_per_leaf: int, h2: bool):
    raw_leaves = []
    survivors = []
    next_id = 0
    for _ in range(leaves):
        events, survivor, survivor_birth, next_id = make_leaf(
            rng, next_id, branches_per_leaf, h2=h2
        )
        raw_leaves.append(events)
        survivors.append((survivor, survivor_birth))

    # Finalize leaf survivors only after every leaf has been emitted, as the
    # hierarchy does during fan-in. Use an immortal Outside sentinel as the
    # final parent. Positive and diagonal survivor deaths are both exercised.
    higher = []
    for survivor, birth in survivors:
        if h2:
            death = rng.randint(0, birth)
        else:
            death = rng.randint(birth, 63)
        higher.append((survivor, birth, death, OUTSIDE, 0))

    raw = [event for leaf in raw_leaves for event in leaf] + higher
    staged_leaves = [event for leaf in raw_leaves for event in contract(leaf, h2=h2)]
    staged = staged_leaves + higher
    return raw, staged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20_000)
    parser.add_argument("--leaves", type=int, default=8)
    parser.add_argument("--branches-per-leaf", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    for h2 in (False, True):
        for case in range(args.cases):
            raw, staged = make_case(
                rng,
                leaves=args.leaves,
                branches_per_leaf=args.branches_per_leaf,
                h2=h2,
            )
            expected = normalized(contract(raw, h2=h2))
            actual = normalized(contract(staged, h2=h2))
            if actual != expected:
                raise SystemExit(
                    f"FAIL {'H2' if h2 else 'H0'} symbolic case {case}: "
                    "inline leaf contraction differs from central-only contraction"
                )
        print(
            f"PASS {'H2' if h2 else 'H0'}: {args.cases} "
            "inline leaf-history contraction cases"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
