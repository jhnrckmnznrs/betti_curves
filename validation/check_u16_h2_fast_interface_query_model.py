#!/usr/bin/env python3
"""Independent arithmetic oracle for the H2 fast root-invariant interface lookup."""
from __future__ import annotations
import argparse, random

def reference(index: int, depth: int, face_size: int):
    z, face = divmod(index, face_size)
    if z == 0: return face
    if z + 1 == depth: return face_size + face
    return None

def fast(index: int, depth: int, face_size: int):
    if index < face_size: return index
    if depth <= 1: return None
    upper_start=(depth-1)*face_size
    if index >= upper_start: return face_size + (index-upper_start)
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cases", type=int, default=20000); ap.add_argument("--seed", type=int, default=2305)
    args=ap.parse_args(); rng=random.Random(args.seed)
    checked=0
    fixed=[(d,f) for d in range(1,10) for f in (1,2,7,31,257,4096)]
    for depth,face_size in fixed:
        for index in range(depth*face_size):
            assert reference(index,depth,face_size)==fast(index,depth,face_size)
            checked += 1
    for _ in range(args.cases):
        depth=rng.randint(1,256); face_size=rng.randint(1,1_000_000)
        index=rng.randrange(depth*face_size)
        assert reference(index,depth,face_size)==fast(index,depth,face_size)
        checked += 1
    print(f"PASS fast interface query model: {checked} valid slab indices")
    return 0

if __name__=="__main__": raise SystemExit(main())
