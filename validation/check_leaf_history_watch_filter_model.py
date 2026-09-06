#!/usr/bin/env python3
"""Randomized equivalence model for the v15 leaf-history watch filter.

The filtered contractor must produce exactly the same retained compact events
and repair counts as the reference contractor that always probes the parent
map. H0 and H2 use opposite filtration inequalities; H2 also includes the
Outside sentinel.
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


def run(events: list[Event], h2: bool, filtered: bool):
    retained: list[Event] = []
    by_parent: dict[int, list[int]] = {}
    watched: set[int] = set()
    contracted = repaired = lookups = skips = 0

    for incoming in events:
        e = Event(incoming.value, incoming.child, incoming.birth, incoming.parent)
        zero = e.birth == e.value

        can_watch = e.child != OUTSIDE
        if filtered and can_watch and e.child not in watched:
            indices = None
            skips += 1
        else:
            lookups += 1
            indices = by_parent.pop(e.child, None) if can_watch else None
            watched.discard(e.child)

        if indices:
            keep, moved = [], []
            for i in indices:
                prev = retained[i]
                should_move = zero or (prev.value < e.value if h2 else prev.value > e.value)
                if should_move:
                    assert prev.parent == e.child
                    prev.parent = e.parent
                    repaired += 1
                    moved.append(i)
                else:
                    keep.append(i)
            if keep:
                by_parent[e.child] = keep
                watched.add(e.child)
            if moved and e.parent != OUTSIDE:
                by_parent.setdefault(e.parent, []).extend(moved)
                watched.add(e.parent)

        if zero:
            contracted += 1
            continue

        i = len(retained)
        retained.append(e)
        if e.parent != OUTSIDE:
            by_parent.setdefault(e.parent, []).append(i)
            watched.add(e.parent)

    packed = [(e.value, e.child, e.birth, e.parent) for e in retained]
    return packed, contracted, repaired, lookups, skips


def random_case(rng: random.Random, h2: bool) -> list[Event]:
    n = rng.randint(1, 180)
    max_id = rng.randint(8, 80)
    events: list[Event] = []
    for _ in range(n):
        child = rng.randrange(max_id)
        parent = rng.randrange(max_id)
        while parent == child:
            parent = rng.randrange(max_id)
        if h2 and rng.random() < 0.08:
            parent = OUTSIDE
        value = rng.randrange(0, 32)
        if rng.random() < 0.82:
            birth = value  # model the overwhelmingly common diagonal death
        else:
            birth = rng.randrange(0, 32)
            if birth == value:
                birth = (birth + 1) % 32
        events.append(Event(value, child, birth, parent))
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20260906)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for h2 in (False, True):
        label = 'H2' if h2 else 'H0'
        for case in range(args.cases):
            events = random_case(rng, h2)
            ref = run(events, h2, False)
            got = run(events, h2, True)
            if ref[:3] != got[:3]:
                raise AssertionError(
                    f'{label} case {case} mismatch\nreference={ref[:3]}\nfiltered={got[:3]}\nevents={events}'
                )
        print(f'PASS {label}: {args.cases} leaf-history watch-filter cases')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
