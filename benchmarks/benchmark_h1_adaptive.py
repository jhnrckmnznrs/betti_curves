"""Run a bounded comparison of three *exact* Morse stopping policies.

Example: python benchmarks/benchmark_h1_adaptive.py --size 32 --seed 11
Memory/RSS is process-wide; run each alternative in a separate Python process
for independent peak-RSS measurements when benchmarking large volumes.
"""
import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from betti_curves_py.h1_morse import h1_persistence


def run(volume, work, rate, directory, max_cells):
    t = time.perf_counter()
    r = h1_persistence(volume, output_dir=directory, connectivity=26,
        max_cells=max_cells, max_operations=work, indexed_scheduler=True,
        max_heap_entries=100000, adaptive_cohomology_ns_per_cell=rate,
        adaptive_window=4096, adaptive_patience=2,
        store_pair_ids=False, max_seconds=120, cohomology_max_seconds=120)
    # The barcode multiset, rather than only its size, must agree.
    pairs=np.empty(len(r),dtype=[('birth', '<u2'),('death', '<u2')])
    pairs['birth']=r.births.astype('<u2',copy=False)
    pairs['death']=r.deaths.astype('<u2',copy=False)
    pairs.sort(order=['birth','death'])
    signature=hashlib.sha256(pairs.tobytes()).hexdigest()
    return dict(seconds=round(time.perf_counter()-t,5), bars=len(r),barcode_sha256=signature,
                remaining=r.statistics['remaining'],
                cohomology_seconds=r.statistics['cohomology_seconds'],
                adaptive_stopped=r.statistics['adaptive_stopped'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,default=32)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--levels',type=int,default=8)
    p.add_argument('--adaptive-ns-per-cell',type=float,default=800.)
    p.add_argument('--max-operations',type=int,default=80_000_000)
    p.add_argument('--output-json',type=Path)
    a=p.parse_args()
    if a.size<1 or not 2<=a.levels<=256:
        p.error('size must be positive and levels in 2..256')
    volume=np.random.default_rng(a.seed).integers(0,a.levels,(a.size,)*3,dtype=np.uint8)
    max_cells=max(3_000_000,(2*a.size+1)**3)
    with tempfile.TemporaryDirectory(prefix='adaptive_h1_') as d:
        results={}
        for name,operations,rate in [('minimal',1,0.),('adaptive',a.max_operations,a.adaptive_ns_per_cell),
                                     ('extensive',a.max_operations,0.)]:
            results[name]=run(volume,operations,rate,str(Path(d)/name),max_cells)
        if len({x['barcode_sha256'] for x in results.values()})!=1:
            raise AssertionError('Different persistence barcodes across stopping policies')
        report=dict(size=a.size,seed=a.seed,levels=a.levels,results=results)
        print(json.dumps(report,indent=2))
        if a.output_json:a.output_json.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
