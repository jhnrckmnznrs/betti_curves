#!/usr/bin/env python3
"""Symbolically check streaming fan-in ordering against stable-sort ordering."""
from __future__ import annotations
import argparse
import random


def merge_h0(streams):
    cur=[0]*5; out=[]
    total=sum(map(len, streams))
    priority=[0,0,1,1,2]
    while len(out)<total:
        candidates=[]
        for i, st in enumerate(streams):
            if cur[i] < len(st):
                candidates.append((st[cur[i]][0], priority[i], i))
        _,_,i=min(candidates)
        out.append((i, streams[i][cur[i]]))
        cur[i]+=1
    return out


def merge_h2(streams):
    # streams are outside-L, outside-R, finite-L, finite-R, interface-L,
    # interface-R, cross, already filtered and descending by value.
    cur=[0]*7; out=[]
    total=sum(map(len, streams)); priority=[0,0,1,1,2,2,3]
    while len(out)<total:
        best=None
        for i, st in enumerate(streams):
            if cur[i] < len(st):
                cand=(st[cur[i]][0], priority[i], i)
                if best is None or cand[0] > best[0] or (cand[0]==best[0] and cand[1:] < best[1:]):
                    best=cand
        i=best[2]
        out.append((i, streams[i][cur[i]])); cur[i]+=1
    return out


def run(cases, seed):
    rng=random.Random(seed)
    for _ in range(cases):
        h0=[]
        for _s in range(5):
            vals=sorted(rng.randrange(16) for _ in range(rng.randrange(0,40)))
            h0.append([(v,j) for j,v in enumerate(vals)])
        actual=merge_h0(h0)
        flat=[]
        p=[0,0,1,1,2]
        for si, st in enumerate(h0):
            flat += [(v,p[si],si,j,item) for j,item in enumerate(st) for v in [item[0]]]
        expected=[(si,item) for _,_,si,_,item in sorted(flat, key=lambda x:(x[0],x[1],x[2],x[3]))]
        assert actual==expected

        raw=[]
        for _s in range(2):
            vals=sorted((rng.randrange(16) for _ in range(rng.randrange(0,40))), reverse=True)
            raw.append([(v, rng.randrange(2)==0, j) for j,v in enumerate(vals)]) # outside?
        interfaces=[]
        for _s in range(2):
            vals=sorted((rng.randrange(16) for _ in range(rng.randrange(0,40))), reverse=True)
            interfaces.append([(v,j) for j,v in enumerate(vals)])
        vals=sorted((rng.randrange(16) for _ in range(rng.randrange(0,40))), reverse=True)
        cross=[(v,j) for j,v in enumerate(vals)]
        hs=[
            [x for x in raw[0] if x[1]], [x for x in raw[1] if x[1]],
            [x for x in raw[0] if not x[1]], [x for x in raw[1] if not x[1]],
            interfaces[0], interfaces[1], cross,
        ]
        actual=merge_h2(hs)
        flat=[]; p=[0,0,1,1,2,2,3]
        for si, st in enumerate(hs):
            flat += [(item[0],p[si],si,j,item) for j,item in enumerate(st)]
        expected=[(si,item) for _,_,si,_,item in sorted(flat, key=lambda x:(-x[0],x[1],x[2],x[3]))]
        assert actual==expected
    print(f"PASS: {cases} randomized H0/H2 streaming fan-in ordering cases")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cases',type=int,default=20000); ap.add_argument('--seed',type=int,default=20260905)
    a=ap.parse_args(); run(a.cases,a.seed)
if __name__=='__main__': main()
