#!/usr/bin/env python3
"""Randomized equivalence model for v18 H2 shell pruning + UF-root dedup.

The model checks two exact transformations used by the H2 leaf kernel:
1. Among active 6-connected face neighbors of a newly activated voxel, retain
   only one face per connected component of the already-active 3x3x3 shell.
2. Among those retained faces, keep only the first occurrence of each exact
   pre-existing UF root before connecting the new voxel.

Both transformations may remove only edges that are already redundant before
that edge would be inserted, so the connected-component partition after the
activation must be identical to the all-face reference.
"""
from __future__ import annotations

import argparse
import random

OFFSETS_26 = [
    (x, y, z)
    for z in (-1, 0, 1)
    for y in (-1, 0, 1)
    for x in (-1, 0, 1)
    if (x, y, z) != (0, 0, 0)
]
INDEX = {offset: i for i, offset in enumerate(OFFSETS_26)}
FACE_ORDER = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
FACE_BITS = [INDEX[offset] for offset in FACE_ORDER]


def adjacent6(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    return sum(abs(x - y) for x, y in zip(a, b)) == 1


ADJ = [
    [j for j, b in enumerate(OFFSETS_26) if i != j and adjacent6(a, b)]
    for i, a in enumerate(OFFSETS_26)
]


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def shell_representatives(active: list[bool]) -> list[int]:
    remaining = {bit for bit in FACE_BITS if active[bit]}
    reps: list[int] = []
    while remaining:
        seed = next(bit for bit in FACE_BITS if bit in remaining)
        reps.append(seed)
        seen = {seed}
        stack = [seed]
        while stack:
            bit = stack.pop()
            for nxt in ADJ[bit]:
                if active[nxt] and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        remaining.difference_update(seen)
    return reps


def connected_partition(active: list[bool], center_edges: list[int]) -> tuple[int, ...]:
    # 26 shell nodes + center at 26. Only active shell nodes participate.
    center = 26
    dsu = DSU(27)
    for i, is_active in enumerate(active):
        if not is_active:
            continue
        for j in ADJ[i]:
            if j > i and active[j]:
                dsu.union(i, j)
    for bit in center_edges:
        dsu.union(center, bit)

    labels = []
    for i, is_active in enumerate(active):
        labels.append(dsu.find(i) if is_active else -1)
    labels.append(dsu.find(center))
    # Normalize root IDs so two equivalent partitions compare directly.
    mapping: dict[int, int] = {}
    out = []
    next_id = 0
    for root in labels:
        if root < 0:
            out.append(-1)
        else:
            if root not in mapping:
                mapping[root] = next_id
                next_id += 1
            out.append(mapping[root])
    return tuple(out)


def root_dedup_reference(root_labels: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for root in root_labels:
        if root not in seen:
            seen.add(root)
            out.append(root)
    return out


def run(cases: int, seed: int) -> None:
    rng = random.Random(seed)
    shell_pruned = 0
    root_pruned = 0
    for case in range(cases):
        # Bias toward the dense masks observed in descending H2 filtrations.
        p = rng.uniform(0.15, 0.95)
        active = [rng.random() < p for _ in range(26)]
        all_faces = [bit for bit in FACE_BITS if active[bit]]
        reps = shell_representatives(active)
        if len(reps) < len(all_faces):
            shell_pruned += len(all_faces) - len(reps)

        ref_partition = connected_partition(active, all_faces)
        pruned_partition = connected_partition(active, reps)
        if ref_partition != pruned_partition:
            raise AssertionError(
                f"shell pruning changed connectivity at case {case}: "
                f"faces={all_faces} reps={reps}"
            )

        # Model exact UF-root dedup on the shell representatives. Equal labels
        # mean the endpoints were already globally connected before center edges.
        root_labels = [rng.randrange(max(1, len(reps))) for _ in reps]
        unique_labels = root_dedup_reference(root_labels)
        root_pruned += len(root_labels) - len(unique_labels)

        # Connecting center to one endpoint per pre-existing component is the
        # same partition update as connecting it to every endpoint.
        ref_groups = {label for label in root_labels}
        dedup_groups = set(unique_labels)
        if ref_groups != dedup_groups:
            raise AssertionError(f"root dedup changed component set at case {case}")

    print(
        f"PASS H2: {cases} shell/root pruning cases; "
        f"shell_edges_pruned={shell_pruned} root_duplicates_pruned={root_pruned}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0xB2_18)
    args = parser.parse_args()
    run(args.cases, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
