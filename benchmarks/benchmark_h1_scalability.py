"""Independent, reproducible one-run H1 scalability experiment.

Requires optional `morse` and, with --validate, `test` dependencies.
This script intentionally does not bundle data or run multiple large jobs.
"""
from __future__ import annotations
import argparse,json,resource,tempfile,time
import numpy as np
from betti_curves_py.h1_morse import h1_persistence


def validate_independent(volume, h1):
    from scipy.ndimage import label,generate_binary_structure
    from skimage.measure import euler_number
    fg26=generate_binary_structure(3,3)
    bg6=generate_binary_structure(3,1)
    thresholds=np.unique(volume)
    independent=[]
    for threshold in thresholds:
        fg=volume<=threshold
        _,beta0=label(fg,structure=fg26)
        labels,num_background=label(~fg,structure=bg6)
        boundary=np.unique(np.concatenate([
            labels[0].ravel(),labels[-1].ravel(),labels[:,0].ravel(),
            labels[:,-1].ravel(),labels[:,:,0].ravel(),labels[:,:,-1].ravel(),
        ]))
        beta2=num_background-len(boundary)+int(0 in boundary)
        independent.append(int(beta0+beta2-euler_number(fg,connectivity=3)))
    observed=h1.at(thresholds)
    if not np.array_equal(observed,independent):
        raise AssertionError(f'Independent H1 disagrees: {observed.tolist()} vs {independent}')
    return dict(thresholds=thresholds.tolist(),betti1=independent)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=32)
    parser.add_argument('--levels',type=int,default=8)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--indexed-scheduler',action='store_true')
    parser.add_argument('--max-heap-entries',type=int,default=100_000)
    parser.add_argument('--max-operations',type=int,default=80_000_000)
    parser.add_argument('--max-annotation-entries',type=int,default=50_000_000)
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--output-json',default=None)
    args=parser.parse_args()
    if args.size<=0 or not 1<=args.levels<=256: parser.error('positive size; levels 1..256 required')
    volume=np.random.default_rng(args.seed).integers(0,args.levels,(args.size,)*3,dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix='h1_scaling_') as catalog:
        start=time.perf_counter()
        result=h1_persistence(volume,output_dir=catalog,indexed_scheduler=args.indexed_scheduler,
            max_heap_entries=args.max_heap_entries,max_operations=args.max_operations,
            max_cells=max(3_000_000,(2*args.size+1)**3),store_pair_ids=False,
            cohomology_max_annotation_entries=args.max_annotation_entries,
            max_seconds=180,cohomology_max_seconds=180)
        elapsed=time.perf_counter()-start
        report=dict(size=args.size,levels=args.levels,seed=args.seed,
            indexed_scheduler=args.indexed_scheduler,seconds=elapsed,bar_count=len(result),
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            statistics=result.statistics)
        if args.validate:report['independent']=validate_independent(volume,result)
        if args.output_json:
            from pathlib import Path
            Path(args.output_json).write_text(json.dumps(report,indent=2))
        print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
