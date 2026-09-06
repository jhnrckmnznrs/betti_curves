#!/usr/bin/env python3
"""Symbolic equivalence check for recursive hierarchy-history contraction.

For a binary hierarchy, compare:

1. central-only replay of every raw finalized event generated in each subtree;
2. recursively contracting each child history, appending the current node's
   finalized events, and contracting again before returning to the parent.

The children have disjoint branch IDs before their parent combine. Parent-node
finalizations are generated only after both child histories exist, which is the
hierarchical no-future-reference invariant used by the Rust implementation.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass

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
    return {event[0]: event[1:] for event in events}


@dataclass
class Node:
    raw: list[tuple[int, int, int, int, int]]
    staged: list[tuple[int, int, int, int, int]]
    survivor: int
    survivor_birth: int


def make_leaf(rng: random.Random, next_id: int, count: int, *, h2: bool) -> tuple[Node, int]:
    survivor = next_id
    births = {survivor: rng.randrange(40, 64) if h2 else rng.randrange(0, 24)}
    limit = {survivor: 0 if h2 else 63}
    parents = {}
    deaths = {}

    for offset in range(1, count + 1):
        child = next_id + offset
        parent = rng.randrange(next_id, child)
        p_birth = births[parent]
        if h2:
            birth = rng.randint(0, p_birth)
            death = rng.randint(0, birth)
        else:
            birth = rng.randint(p_birth, 63)
            death = rng.randint(birth, 63)
        births[child] = birth
        deaths[child] = death
        parents[child] = parent
        limit[child] = death

    raw = []
    for child in range(next_id + count, next_id, -1):
        parent = parents[child]
        raw.append((child, births[child], deaths[child], parent, births[parent]))
    staged = contract(raw, h2=h2)
    return Node(raw, staged, survivor, births[survivor]), next_id + count + 1


def combine_nodes(rng: random.Random, left: Node, right: Node, *, h2: bool) -> Node:
    # Pick the elder survivor exactly as the union-find would.
    if h2:
        if (left.survivor_birth, -left.survivor) >= (right.survivor_birth, -right.survivor):
            parent, child = left, right
        else:
            parent, child = right, left
        # Superlevel merge death cannot exceed the younger birth. Exercise both
        # diagonal and positive persistence whenever possible.
        if rng.random() < 0.55:
            death = child.survivor_birth
        else:
            death = rng.randint(0, child.survivor_birth)
    else:
        if (left.survivor_birth, left.survivor) <= (right.survivor_birth, right.survivor):
            parent, child = left, right
        else:
            parent, child = right, left
        if rng.random() < 0.55:
            death = child.survivor_birth
        else:
            death = rng.randint(child.survivor_birth, 63)

    final = (
        child.survivor,
        child.survivor_birth,
        death,
        parent.survivor,
        parent.survivor_birth,
    )
    raw = left.raw + right.raw + [final]
    staged = contract(left.staged + right.staged + [final], h2=h2)
    return Node(raw, staged, parent.survivor, parent.survivor_birth)


def make_case(rng: random.Random, *, leaves: int, branches_per_leaf: int, h2: bool) -> Node:
    nodes = []
    next_id = 0
    for _ in range(leaves):
        node, next_id = make_leaf(rng, next_id, branches_per_leaf, h2=h2)
        nodes.append(node)

    # Pairwise binary reduction, matching the semantic requirement rather than
    # one particular slab-count shape.
    while len(nodes) > 1:
        next_nodes = []
        it = iter(nodes)
        for left in it:
            right = next(it, None)
            if right is None:
                next_nodes.append(left)
            else:
                next_nodes.append(combine_nodes(rng, left, right, h2=h2))
        nodes = next_nodes

    root = nodes[0]
    # Final root death into the distinguished outside sentinel so parent repair
    # across every hierarchy level is exercised.
    if h2:
        death = rng.randint(0, root.survivor_birth)
    else:
        death = rng.randint(root.survivor_birth, 63)
    final = (root.survivor, root.survivor_birth, death, OUTSIDE, 0)
    return Node(root.raw + [final], contract(root.staged + [final], h2=h2), OUTSIDE, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20_000)
    parser.add_argument("--leaves", type=int, default=8)
    parser.add_argument("--branches-per-leaf", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    for h2 in (False, True):
        for case in range(args.cases):
            root = make_case(
                rng,
                leaves=args.leaves,
                branches_per_leaf=args.branches_per_leaf,
                h2=h2,
            )
            expected = normalized(contract(root.raw, h2=h2))
            actual = normalized(root.staged)
            if actual != expected:
                raise SystemExit(
                    f"FAIL {'H2' if h2 else 'H0'} symbolic case {case}: "
                    "recursive contraction differs from central-only contraction"
                )
        print(f"PASS {'H2' if h2 else 'H0'}: {args.cases} recursive hierarchy-history cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
