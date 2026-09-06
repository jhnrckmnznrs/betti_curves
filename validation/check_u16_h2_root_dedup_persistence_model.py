#!/usr/bin/env python3
"""Randomized exactness model for U16 H2 persistence root deduplication.

For one newly activated voxel, each active representative neighbor belongs to a
pre-existing union-find component. The v23 transformation resolves those roots
*before* inserting any center edge and retains the first representative of each
distinct root. In the reference sequence, every later representative of a root
already joined through the center is a same-root no-op, so deleting it cannot
change persistence actions or the final component metadata.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Meta:
    outside: bool
    birth: int
    interface: int | None


def older(a: Meta, b: Meta) -> Meta:
    outside = a.outside or b.outside
    birth = max(a.birth, b.birth)
    # The real kernel carries a deterministic interface representative. For the
    # model, retain the current component's representative when present, else
    # the incoming component's. Duplicate-root edges never reach this update.
    interface = a.interface if a.interface is not None else b.interface
    return Meta(outside, birth, interface)


def merge_action(a: Meta, b: Meta, t: int) -> tuple[str, int | None, int | None]:
    """Compact deterministic proxy for the action class/state used by H2."""
    if a.interface is None and b.interface is None:
        if a.outside and b.outside:
            return ("none", None, None)
        younger = b.birth if a.outside else a.birth if b.outside else min(a.birth, b.birth)
        return ("pair", t, younger)
    if a.interface is not None and b.interface is None:
        return ("attach", a.interface, b.birth)
    if a.interface is None and b.interface is not None:
        return ("attach", b.interface, a.birth)
    return ("interface", a.interface, b.interface)


def run_sequence(labels: list[int], components: dict[int, Meta], center: Meta, t: int):
    joined: set[int] = set()
    state = center
    actions: list[tuple[str, int | None, int | None]] = []
    same_root = 0
    for label in labels:
        if label in joined:
            same_root += 1
            continue
        incoming = components[label]
        actions.append(merge_action(state, incoming, t))
        state = older(state, incoming)
        joined.add(label)
    return actions, state, same_root


def first_distinct(labels: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def run(cases: int, seed: int) -> None:
    rng = random.Random(seed)
    total_inputs = total_unique = 0
    for case in range(cases):
        n_components = rng.randint(1, 6)
        components = {
            i: Meta(
                outside=rng.random() < 0.15,
                birth=rng.randint(0, 65535),
                interface=(rng.randrange(1000) if rng.random() < 0.25 else None),
            )
            for i in range(n_components)
        }
        n_neighbors = rng.randint(0, 26)
        labels = [rng.randrange(n_components) for _ in range(n_neighbors)]
        center = Meta(
            outside=rng.random() < 0.10,
            birth=rng.randint(0, 65535),
            interface=(rng.randrange(1000) if rng.random() < 0.20 else None),
        )
        t = rng.randint(0, 65535)
        unique = first_distinct(labels)
        ref_actions, ref_state, ref_noops = run_sequence(labels, components, center, t)
        dedup_actions, dedup_state, dedup_noops = run_sequence(unique, components, center, t)
        if ref_actions != dedup_actions or ref_state != dedup_state or dedup_noops != 0:
            raise AssertionError(
                f"case {case}: root dedup changed persistence proxy\n"
                f"labels={labels}\nunique={unique}\n"
                f"reference={ref_actions, ref_state}\n"
                f"dedup={dedup_actions, dedup_state}"
            )
        if ref_noops != len(labels) - len(unique):
            raise AssertionError("same-root no-op count disagrees with duplicate-root count")
        total_inputs += len(labels)
        total_unique += len(unique)

    print(
        f"PASS H2 persistence root dedup: {cases} cases; "
        f"inputs={total_inputs} unique={total_unique} "
        f"duplicates_elided={total_inputs-total_unique}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0x23112)
    args = ap.parse_args()
    run(args.cases, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
