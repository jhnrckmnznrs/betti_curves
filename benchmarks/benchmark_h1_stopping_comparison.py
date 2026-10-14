"""Compare fixed and structural heuristics; verify exact barcode multisets."""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
from betti_curves_py.h1_morse import h1_persistence

def experiment(size,seed,kind='random',face_ns=2200.,cell_ns=800.):
    from benchmark_h1_cost_model import volume_for
    vol=volume_for(size,seed,kind)
    variants={
        'minimal':(1,0.,0.),
        'cell-window':(40_000_000,cell_ns,0.),
        'face-window':(40_000_000,0.,face_ns),
        'extensive':(40_000_000,0.,0.),
    }
    out=[]
    with tempfile.TemporaryDirectory(prefix='h1_cost_compare_') as td:
        for name,(budget,cellrate,facerate) in variants.items():
            t=time.perf_counter()
            result=h1_persistence(vol,output_dir=Path(td)/name,
                indexed_scheduler=True,store_pair_ids=False,
                max_operations=budget,max_heap_entries=100_000,
                adaptive_cohomology_ns_per_cell=cellrate,
                adaptive_cohomology_ns_per_face=facerate,
                adaptive_window=4096,adaptive_patience=2,
                max_cells=max(3_000_000,(2*size+1)**3),max_seconds=120,
                cohomology_max_seconds=120)
            elapsed=time.perf_counter()-t
            pairs=np.stack([result.births,result.deaths],axis=1).astype('uint16')
            pairs=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
            out.append(dict(name=name,seconds=elapsed,bars=len(pairs),
                 barcode_sha256=hashlib.sha256(pairs.tobytes()).hexdigest(),
                 stats=result.statistics))
            print(f'{size} {seed} {kind} {name}: {elapsed:.3f}s '
                  f'coho={result.statistics["cohomology_seconds"]:.3f}s '
                  f'critical={result.statistics["remaining"]} '
                  f'faces={result.statistics["critical_faces"]} '
                  f'face_incidences={result.statistics["critical_face_incidences"]} '
                  f'stopped={result.statistics["adaptive_stopped"]}',flush=True)
    assert len({x['barcode_sha256'] for x in out})==1
    return dict(size=size,seed=seed,kind=kind,results=out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',type=int,nargs='+',default=[48,64])
    p.add_argument('--seeds',type=int,nargs='+',default=[11])
    p.add_argument('--kinds',nargs='+',default=['random'])
    p.add_argument('--face-ns',type=float,default=2200.)
    p.add_argument('--output',default='stopping_comparison.json')
    a=p.parse_args();rows=[]
    for s in a.sizes:
        for kind in a.kinds:
            for seed in a.seeds:
                rows.append(experiment(s,seed,kind,face_ns=a.face_ns))
                Path(a.output).write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
