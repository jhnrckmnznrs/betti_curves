#!/usr/bin/env python3
"""Symbolic check of online finalized-history contraction.

This does not replace image-level equivalence validation. It stress-tests the
filtration-time parent-repair rule on randomized branch-death forests whose
generation order is postorder (all stale references to a branch are generated
before that branch becomes final).
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict


def reference(events, root, h2=False):
    buckets = defaultdict(list)
    for order, event in enumerate(events):
        buckets[event[2]].append((order, event))
    redirect = {}
    nodes = {}
    values = sorted(buckets, reverse=h2)
    for value in values:
        repaired = []
        for order, (child, birth, death, parent, parent_birth) in buckets[value]:
            current = parent
            seen = set()
            while current != root and current in redirect:
                if current in seen:
                    raise AssertionError("redirect cycle")
                seen.add(current)
                current = redirect[current]
            repaired.append((order, (child, birth, death, current, parent_birth)))

        diagonal = {}
        positive = []
        for order, event in repaired:
            child, birth, death, parent, _ = event
            if birth == death:
                diagonal[child] = parent
            else:
                positive.append((order, event))

        def resolve_diagonal(parent):
            seen = set()
            while parent != root and parent in diagonal:
                if parent in seen:
                    raise AssertionError("diagonal cycle")
                seen.add(parent)
                parent = diagonal[parent]
            return parent

        for _, (child, birth, death, parent, _) in positive:
            nodes[child] = (birth, death, resolve_diagonal(parent))
        for _, (child, _birth, _death, parent, _) in repaired:
            redirect[child] = parent
    return nodes


def online(events, root, h2=False):
    buckets = defaultdict(list)
    reverse = defaultdict(list)
    for child, birth, death, parent, parent_birth in events:
        zero = birth == death
        handles = reverse.pop(child, [])
        keep = []
        moved = []
        for event_value, index in handles:
            record = buckets[event_value][index]
            redirect = zero or (event_value < death if h2 else event_value > death)
            if redirect:
                record[3] = parent
                record[4] = parent_birth
                moved.append((event_value, index))
            else:
                keep.append((event_value, index))
        if keep:
            reverse[child] = keep
        if parent != root and moved:
            reverse[parent].extend(moved)
        if zero:
            continue
        index = len(buckets[death])
        buckets[death].append([child, birth, death, parent, parent_birth])
        if parent != root:
            reverse[parent].append((death, index))

    retained = []
    for value in sorted(buckets, reverse=h2):
        retained.extend(tuple(record) for record in buckets[value])
    return reference(retained, root, h2=h2)


def make_case(rng, n, h2=False):
    births = [rng.randrange(0, 32) for _ in range(n)]
    root = n
    parents = []
    for child in range(n):
        if h2:
            eligible = [i for i in range(child) if births[i] >= births[child]]
        else:
            eligible = [i for i in range(child) if births[i] <= births[child]]
        parents.append(root if not eligible or rng.random() < 0.12 else rng.choice(eligible))

    if h2:
        deaths = [rng.randrange(0, birth + 1) for birth in births]
    else:
        deaths = [rng.randrange(birth, 40) for birth in births]

    children = defaultdict(list)
    for child, parent in enumerate(parents):
        children[parent].append(child)
    order = []

    def visit(node):
        for child in children[node]:
            visit(child)
        if node != root:
            order.append(node)

    visit(root)
    events = []
    for child in order:
        parent = parents[child]
        parent_birth = 0 if parent == root else births[parent]
        events.append((child, births[child], deaths[child], parent, parent_birth))
    return events, root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--branches", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    for h2 in (False, True):
        for case in range(args.cases):
            events, root = make_case(rng, args.branches, h2=h2)
            expected = reference(events, root, h2=h2)
            actual = online(events, root, h2=h2)
            if actual != expected:
                raise SystemExit(
                    f"FAIL {'H2' if h2 else 'H0'} symbolic case {case}: "
                    f"online contraction differs from threshold replay"
                )
        print(f"PASS {'H2' if h2 else 'H0'}: {args.cases} symbolic contraction cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
