#!/usr/bin/env python3
"""Randomized model for v16 plateau-native leaf zero-death elision.

The production leaf reducer processes H0 in nondecreasing filtration order and
H2 in nonincreasing filtration order.  A finite branch born and killed at the
current value t cannot already be the provisional parent of an earlier retained
positive event: before t it does not exist, and at t an older parent of a
positive death must have strictly earlier (H0) / later (H2) birth.

Therefore a local [t,t) death may be suppressed at the union decision itself.
This model generates valid monotone elder-rule merge streams and checks that
v15's generic contraction and v16's pre-history zero elision retain exactly the
same positive events and repair counts, including H2 merges into Outside.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

OUTSIDE = 2**32 - 1


@dataclass
class Event:
    value: int
    child: int
    birth: int
    parent: int


def contract_reference(events: list[Event], births: dict[int, int], h2: bool):
    retained: list[Event] = []
    by_parent: dict[int, list[int]] = {}
    contracted = repaired = 0

    for incoming in events:
        e = Event(incoming.value, incoming.child, incoming.birth, incoming.parent)
        zero = e.birth == e.value
        indices = by_parent.pop(e.child, None)
        if indices:
            keep: list[int] = []
            moved: list[int] = []
            for i in indices:
                prev = retained[i]
                move = zero or (prev.value < e.value if h2 else prev.value > e.value)
                if move:
                    prev.parent = e.parent
                    repaired += 1
                    moved.append(i)
                else:
                    keep.append(i)
            if keep:
                by_parent[e.child] = keep
            if moved and e.parent != OUTSIDE:
                by_parent.setdefault(e.parent, []).extend(moved)

        if zero:
            contracted += 1
            continue

        i = len(retained)
        retained.append(e)
        if e.parent != OUTSIDE:
            by_parent.setdefault(e.parent, []).append(i)

    return [(e.value, e.child, e.birth, e.parent) for e in retained], contracted, repaired


def contract_elided(events: list[Event], births: dict[int, int], h2: bool):
    # Zero events never enter the contractor. Positive events still use the
    # reference machinery, matching the v16 Rust implementation's safety net.
    elided = sum(e.birth == e.value for e in events)
    positive = [e for e in events if e.birth != e.value]
    retained, contracted_inside, repaired = contract_reference(positive, births, h2)
    assert contracted_inside == 0
    return retained, elided, repaired


def elder(a: int, b: int, births: dict[int, int], h2: bool) -> tuple[int, int]:
    if a == OUTSIDE:
        return a, b
    if b == OUTSIDE:
        return b, a
    ba, bb = births[a], births[b]
    if h2:
        a_older = ba > bb or (ba == bb and a < b)
    else:
        a_older = ba < bb or (ba == bb and a < b)
    return (a, b) if a_older else (b, a)


def random_valid_stream(rng: random.Random, h2: bool) -> tuple[list[Event], dict[int, int]]:
    levels = rng.randint(4, 28)
    births: dict[int, int] = {}
    born_at: dict[int, list[int]] = {v: [] for v in range(levels)}
    next_id = 0
    for value in range(levels):
        for _ in range(rng.randint(1, 8)):
            births[next_id] = value
            born_at[value].append(next_id)
            next_id += 1

    active: set[int] = set()
    events: list[Event] = []
    values = range(levels - 1, -1, -1) if h2 else range(levels)

    for value in values:
        active.update(born_at[value])

        # Several same-level and cross-level merges, preserving elder-rule
        # component identities. These are the only events a sequential UF can
        # generate at this threshold.
        attempts = rng.randint(0, min(18, max(0, len(active) - 1)))
        for _ in range(attempts):
            if len(active) < 2:
                break
            a, b = rng.sample(tuple(active), 2)
            parent, child = elder(a, b, births, h2)
            events.append(Event(value, child, births[child], parent))
            active.remove(child)

        # Model H2 contact with the distinguished outside component.
        if h2 and active and rng.random() < 0.35:
            child = rng.choice(tuple(active))
            events.append(Event(value, child, births[child], OUTSIDE))
            active.remove(child)

    # Sanity: production filtration order is monotone.
    death_values = [e.value for e in events]
    assert death_values == sorted(death_values, reverse=h2)
    return events, births


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for h2 in (False, True):
        label = "H2" if h2 else "H0"
        zero_total = 0
        for case in range(args.cases):
            events, births = random_valid_stream(rng, h2)
            ref = contract_reference(events, births, h2)
            got = contract_elided(events, births, h2)
            zero_total += ref[1]
            if ref != got:
                raise AssertionError(
                    f"{label} case {case} mismatch\nreference={ref}\nelided={got}\nevents={events}"
                )
            # For valid monotone leaf streams there is no existing positive
            # parent reference that requires local repair.
            if ref[2] != 0:
                raise AssertionError(f"{label} case {case}: unexpected local repair count {ref[2]}")
        print(
            f"PASS {label}: {args.cases} monotone plateau-elision cases; "
            f"zero deaths elided={zero_total}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
