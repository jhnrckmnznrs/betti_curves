#!/usr/bin/env python3
"""Randomized model check for the compact merge-tree leaf UF state.

The production v14 candidate replaces a full branch object per voxel with a
local elder voxel ID (plus an Outside sentinel for H2). Birth and global branch
ID are reconstructed on demand. This model compares the legacy full-branch
union semantics with that compact representation over randomized union orders,
boundary visibility, ties, and H2 Outside states.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

NO_REP = -1
OUTSIDE = None

@dataclass(frozen=True, order=True)
class H0Branch:
    birth: int
    gid: int


def h0_older(a: H0Branch, b: H0Branch) -> H0Branch:
    return min(a, b)


def h2_older(a, b):
    if a is OUTSIDE or b is OUTSIDE:
        return OUTSIDE
    # branch is (birth, gid); larger birth is older, lower ID breaks ties
    return a if (a[0] > b[0] or (a[0] == b[0] and a[1] <= b[1])) else b


def h2_younger(a, b):
    older = h2_older(a, b)
    y = b if older == a else a
    return y


class LegacyH0:
    def __init__(self, values, reps, base):
        n = len(values)
        self.parent = list(range(n))
        self.rank = [0] * n
        self.branch = [H0Branch(values[i], base + i) for i in range(n)]
        self.rep = reps[:]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b, value, finalize):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        ba, bb = self.branch[ra], self.branch[rb]
        older = h0_older(ba, bb)
        younger = bb if older == ba else ba
        pa, pb = self.rep[ra], self.rep[rb]
        ha, hb = pa != NO_REP, pb != NO_REP
        if ba == bb:
            action = ("interface", value, pa, pb) if ha and hb else None
        elif not ha and not hb:
            action = ("final", value, younger, older)
        elif ha and not hb:
            if finalize and older == ba:
                action = ("final", value, bb, ba)
            else:
                action = ("attach", value, pa, bb)
        elif not ha and hb:
            if finalize and older == bb:
                action = ("final", value, ba, bb)
            else:
                action = ("attach", value, pb, ba)
        else:
            action = ("interface", value, pa, pb)
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.branch[ra] = older
        self.rep[ra] = pa if ha else pb
        return action


class CompactH0:
    def __init__(self, values, reps, base):
        n = len(values)
        self.values = values
        self.base = base
        self.parent = list(range(n))
        self.rank = [0] * n
        self.elder = list(range(n))
        self.rep = reps[:]

    def branch(self, local):
        return H0Branch(self.values[local], self.base + local)

    def older_local(self, a, b):
        return a if (self.values[a], a) <= (self.values[b], b) else b

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b, value, finalize):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        ea, eb = self.elder[ra], self.elder[rb]
        older_l = self.older_local(ea, eb)
        ba, bb = self.branch(ea), self.branch(eb)
        older = self.branch(older_l)
        younger = bb if older_l == ea else ba
        pa, pb = self.rep[ra], self.rep[rb]
        ha, hb = pa != NO_REP, pb != NO_REP
        if ea == eb:
            action = ("interface", value, pa, pb) if ha and hb else None
        elif not ha and not hb:
            action = ("final", value, younger, older)
        elif ha and not hb:
            if finalize and older_l == ea:
                action = ("final", value, bb, ba)
            else:
                action = ("attach", value, pa, bb)
        elif not ha and hb:
            if finalize and older_l == eb:
                action = ("final", value, ba, bb)
            else:
                action = ("attach", value, pb, ba)
        else:
            action = ("interface", value, pa, pb)
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.elder[ra] = older_l
        self.rep[ra] = pa if ha else pb
        return action


class LegacyH2:
    def __init__(self, values, reps, outside, base):
        n = len(values)
        self.parent = list(range(n))
        self.rank = [0] * n
        self.branch = [OUTSIDE if outside[i] else (values[i], base + i) for i in range(n)]
        self.rep = reps[:]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b, value, finalize):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        ba, bb = self.branch[ra], self.branch[rb]
        older = h2_older(ba, bb)
        younger = h2_younger(ba, bb)
        pa, pb = self.rep[ra], self.rep[rb]
        ha, hb = pa != NO_REP, pb != NO_REP
        if not ha and not hb:
            action = None if younger is OUTSIDE else ("final", value, younger, older)
        elif ha and not hb:
            if finalize and older == ba:
                action = None if younger is OUTSIDE else ("final", value, younger, ba)
            else:
                action = ("attach", value, pa, bb)
        elif not ha and hb:
            if finalize and older == bb:
                action = None if younger is OUTSIDE else ("final", value, younger, bb)
            else:
                action = ("attach", value, pb, ba)
        else:
            action = ("interface", value, pa, pb)
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.branch[ra] = older
        self.rep[ra] = pa if ha else pb
        return action


class CompactH2:
    OUT = -1
    def __init__(self, values, reps, outside, base):
        n = len(values)
        self.values = values
        self.base = base
        self.parent = list(range(n))
        self.rank = [0] * n
        self.elder = [self.OUT if outside[i] else i for i in range(n)]
        self.rep = reps[:]

    def branch(self, local):
        return OUTSIDE if local == self.OUT else (self.values[local], self.base + local)

    def older_local(self, a, b):
        if a == self.OUT or b == self.OUT:
            return self.OUT
        va, vb = self.values[a], self.values[b]
        return a if (va > vb or (va == vb and a <= b)) else b

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b, value, finalize):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        ea, eb = self.elder[ra], self.elder[rb]
        older_l = self.older_local(ea, eb)
        ba, bb = self.branch(ea), self.branch(eb)
        older = self.branch(older_l)
        younger = h2_younger(ba, bb)
        pa, pb = self.rep[ra], self.rep[rb]
        ha, hb = pa != NO_REP, pb != NO_REP
        if not ha and not hb:
            action = None if younger is OUTSIDE else ("final", value, younger, older)
        elif ha and not hb:
            if finalize and older_l == ea:
                action = None if younger is OUTSIDE else ("final", value, younger, ba)
            else:
                action = ("attach", value, pa, bb)
        elif not ha and hb:
            if finalize and older_l == eb:
                action = None if younger is OUTSIDE else ("final", value, younger, bb)
            else:
                action = ("attach", value, pb, ba)
        else:
            action = ("interface", value, pa, pb)
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.elder[ra] = older_l
        self.rep[ra] = pa if ha else pb
        return action


def run_case(rng: random.Random, h2: bool):
    n = rng.randint(2, 24)
    values = [rng.randrange(0, 16) for _ in range(n)]  # ties deliberately common
    reps = [i if rng.random() < 0.45 else NO_REP for i in range(n)]
    base = rng.randrange(0, 1_000_000)
    outside = [rng.random() < 0.1 for _ in range(n)]
    legacy = LegacyH2(values, reps, outside, base) if h2 else LegacyH0(values, reps, base)
    compact = CompactH2(values, reps, outside, base) if h2 else CompactH0(values, reps, base)
    edges = [(rng.randrange(n), rng.randrange(n), rng.randrange(0, 16), rng.choice([False, True])) for _ in range(rng.randint(n, 5*n))]
    for a, b, value, finalize in edges:
        got = compact.union(a, b, value, finalize)
        expected = legacy.union(a, b, value, finalize)
        if got != expected:
            raise AssertionError(("H2" if h2 else "H0", values, reps, outside, (a,b,value,finalize), expected, got))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0xB3771)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for h2 in (False, True):
        for _ in range(args.cases):
            run_case(rng, h2)
        print(f"PASS {'H2' if h2 else 'H0'}: {args.cases} compact leaf-state equivalence cases")

if __name__ == "__main__":
    main()
