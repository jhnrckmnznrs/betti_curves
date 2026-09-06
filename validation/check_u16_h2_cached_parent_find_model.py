#!/usr/bin/env python3
"""Randomized symbolic equivalence for the cached-parent H2 find path.

The candidate reuses the first parent already loaded by the direct-parent probe
and continues the same path-halving algorithm without re-reading that first
parent.  This oracle checks root, traversed-edge count, and the final parent
array against the ordinary path-halving find.
"""
from __future__ import annotations

import argparse
import random


def find_reference(parent: list[int], node: int) -> tuple[int, int]:
    steps = 0
    while parent[node] != node:
        steps += 1
        p = parent[node]
        if parent[p] != p:
            parent[node] = parent[p]
        node = p
    return node, steps


def find_cached(parent: list[int], node: int) -> tuple[int, int]:
    first = parent[node]
    if first == node:
        return node, 0
    steps = 1
    p = first
    while True:
        if parent[p] == p:
            return p, steps
        gp = parent[p]
        parent[node] = gp
        node = p
        p = gp
        steps += 1


def random_forest(rng: random.Random, n: int) -> list[int]:
    parent = list(range(n))
    for node in range(1, n):
        if rng.random() < 0.82:
            parent[node] = rng.randrange(node + 1)
    return parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2330)
    args = ap.parse_args()
    if args.cases <= 0:
        ap.error("--cases must be positive")

    rng = random.Random(args.seed)
    checks = 0
    nonroot_checks = 0
    total_steps = 0
    for _ in range(args.cases):
        n = rng.randint(2, 160)
        forest = random_forest(rng, n)
        for node in range(n):
            ref_parent = forest.copy()
            cand_parent = forest.copy()
            ref_root, ref_steps = find_reference(ref_parent, node)
            cand_root, cand_steps = find_cached(cand_parent, node)
            if (ref_root, ref_steps) != (cand_root, cand_steps):
                raise AssertionError(
                    f"root/step mismatch node={node}: reference={(ref_root, ref_steps)} "
                    f"candidate={(cand_root, cand_steps)} forest={forest}"
                )
            if ref_parent != cand_parent:
                raise AssertionError(
                    f"path-halving state mismatch node={node}:\n"
                    f"reference={ref_parent}\ncandidate={cand_parent}\nforest={forest}"
                )
            checks += 1
            nonroot_checks += int(forest[node] != node)
            total_steps += ref_steps

    print(
        f"PASS cached-parent find model: cases={args.cases} node_checks={checks} "
        f"nonroot_checks={nonroot_checks} traversed_edges={total_steps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
