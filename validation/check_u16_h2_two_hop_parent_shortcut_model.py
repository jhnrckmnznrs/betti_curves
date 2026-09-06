#!/usr/bin/env python3
"""Randomized symbolic check for the read-only two-hop H2 parent shortcut."""
from __future__ import annotations

import argparse
import random


def find(parent: list[int], node: int) -> int:
    while parent[node] != node:
        node = parent[node]
    return node


def candidate_same_root(parent: list[int], node: int, current_root: int) -> bool:
    if node == current_root:
        return True
    p = parent[node]
    if p == node:
        return False
    if p == current_root:
        return True
    if parent[p] == p:
        return False
    return parent[p] == current_root


def random_forest(rng: random.Random, n: int) -> list[int]:
    parent = list(range(n))
    # Parent indices never increase, guaranteeing an acyclic rooted forest.
    for node in range(1, n):
        if rng.random() < 0.78:
            parent[node] = rng.randrange(node + 1)
    return parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2302)
    args = ap.parse_args()
    if args.cases <= 0:
        ap.error("--cases must be positive")

    rng = random.Random(args.seed)
    checked = 0
    two_hop_hits = 0
    for _ in range(args.cases):
        n = rng.randint(2, 128)
        parent = random_forest(rng, n)
        roots = [i for i in range(n) if parent[i] == i]
        current_root = rng.choice(roots)
        for node in range(n):
            reference = find(parent, node) == current_root
            shortcut = candidate_same_root(parent, node, current_root)
            if shortcut and not reference:
                raise AssertionError(
                    f"unsound shortcut: node={node} current_root={current_root} parent={parent}"
                )
            p = parent[node]
            if (
                node != current_root
                and p != node
                and p != current_root
                and parent[p] != p
                and parent[p] == current_root
            ):
                two_hop_hits += 1
                if not reference:
                    raise AssertionError("two-hop hit did not resolve to current root")
            checked += 1

    print(
        f"PASS two-hop parent-shortcut model: cases={args.cases} "
        f"node_checks={checked} exact_two_hop_hits={two_hop_hits}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
