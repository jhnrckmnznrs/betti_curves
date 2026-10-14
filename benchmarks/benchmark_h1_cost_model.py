"""Reproducible CPU profiling and *held-out-volume* cohomology timing models.

Collects exact Morse and H1 runs with varying reduction work. Each run
records pre-cohomology structural features and *post hoc* sparse-annotation
work, avoiding leakage from actual operations into prospective predictions.

python benchmarks/benchmark_h1_cost_model.py --sizes 24 32 --seeds 2 7 --output /tmp/h1_cost.json
"""
from __future__ import annotations
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np
from betti_curves_py._morse.native_backend import CompactMorse


def volume_for(size, seed, kind):
    rng = np.random.default_rng(seed)
    if kind == 'random':
        return rng.integers(0, 8, (size,)*3, dtype=np.uint8)
    if kind == 'smooth':
        # Nonperiodic smoothing with a fixed rule; SciPy is a benchmark-only extra.
        from scipy.ndimage import gaussian_filter
        z=gaussian_filter(rng.normal(size=(size,)*3), sigma=2.0)
        return np.floor(7.999*np.clip((z-z.min())/(z.max()-z.min()), 0, 1)).astype('uint8')
    if kind == 'tunnel':
        v=np.full((size,)*3,5,dtype=np.uint8)
        a=max(1,size//5); b=size-a
        v[a:b,a:b,a:b]=0
        v[a:b,size//2,size//2]=5
        return v
    raise ValueError(kind)


def measure(size,seed,kind,budget,cache_dir):
    v=volume_for(size,seed,kind)
    with tempfile.TemporaryDirectory(prefix='h1_cost_',dir=cache_dir) as td:
        t=time.perf_counter()
        m=CompactMorse(v,td,max_cells=max(3_000_000,(2*size+1)**3),
            indexed_scheduler=True,max_operations=budget,max_heap_entries=100000,
            max_seconds=120,max_trace_bytes=1000000000)
        morse_seconds=time.perf_counter()-t
        pre=m.critical_structure_profile()
        t=time.perf_counter()
        bars=m.h1_records(max_operations=100_000_000,max_seconds=120,max_annotation_entries=50000000)
        coho_seconds=time.perf_counter()-t
        actual=m.cohomology_profile()
        return dict(size=size,seed=seed,kind=kind,budget=budget,
            group=f'{size}-{seed}-{kind}',morse_seconds=morse_seconds,
            cohomology_seconds=coho_seconds,total_seconds=morse_seconds+coho_seconds,
            bars=int(len(bars)),critical=pre,observed=actual)


def compare_models(rows):
    """OO-group cross-validation, nonnegative linear models, no leakage."""
    from scipy.optimize import nnls
    data=[r for r in rows if r['cohomology_seconds']>0]
    names=['cell-count','edges+faces+incidences','edges+faces+incidences+degree2','post-hoc annotation work']
    columns=[['critical_cells'],['critical_edges','critical_faces','face_incidences'],
        ['critical_edges','critical_faces','face_incidences','face_degree_squared'],
        ['annotation_xor_terms','annotation_updates','face_incidences']]
    def xdata(r, keys):
        return np.asarray([r['critical'].get(k,r['observed'].get(k,0)) for k in keys],dtype=float)
    groups=sorted(set(r['group'] for r in data))
    results=[]
    for name,keys in zip(names,columns):
        ys=[]; ps=[]
        folds=[]
        for g in groups:
            train=[r for r in data if r['group']!=g]
            test=[r for r in data if r['group']==g]
            if not train or not test:continue
            xtr=np.stack([xdata(r,keys) for r in train]); ytr=np.asarray([r['cohomology_seconds'] for r in train]);
            xt=np.stack([xdata(r,keys) for r in test]);yt=np.asarray([r['cohomology_seconds'] for r in test])
            # Unit normalization reduces ill-conditioning; nonnegative slopes
            # and intercept exclude spurious negative cost contributions.
            scale=np.maximum(np.percentile(xtr,80,axis=0),1)
            w,_=nnls(np.column_stack([np.ones(len(train)),xtr/scale]),ytr)
            prediction=np.column_stack([np.ones(len(test)),xt/scale])@w
            ys.extend(yt.tolist());ps.extend(prediction.tolist())
            folds.append(dict(group=g,actual_mean=float(yt.mean()),predicted_mean=float(prediction.mean())))
        if not ys:continue
        y=np.array(ys);p=np.array(ps)
        results.append(dict(model=name,mae_seconds=float(np.mean(abs(y-p))),
            mean_seconds=float(y.mean()),mape_percent=float(100*np.mean(abs(y-p)/np.maximum(y,1e-4))),
            r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.var(y)>0 else None,
            folds=folds))
    return results


def calibration_coefficients(rows):
    """Nonnegative timing coefficients for the *observed training workloads*.

    The coefficients are measured associations, not unbiased estimates of
    future marginal savings. Do not use them as universal default values.
    """
    from scipy.optimize import nnls
    out={}
    for key in ('critical_cells','critical_faces','face_incidences'):
        x=np.asarray([r['critical'][key] for r in rows],dtype=float)
        y=np.asarray([r['cohomology_seconds'] for r in rows],dtype=float)
        scale=max(float(np.percentile(x,80)),1.)
        w,_=nnls(np.column_stack([np.ones(len(x)),x/scale]),y)
        out[key]=dict(intercept_seconds=float(w[0]),ns_per_unit=float(1e9*w[1]/scale))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',type=int,nargs='+',default=[24,32])
    p.add_argument('--seeds',type=int,nargs='+',default=[2,7,13])
    p.add_argument('--kinds',nargs='+',choices=['random','smooth','tunnel'],default=['random','smooth'])
    p.add_argument('--budgets',type=int,nargs='+',default=[1,1_000_000,40_000_000])
    p.add_argument('--output',type=Path,default=Path('h1_cost_study.json'))
    p.add_argument('--cache-dir',type=Path,default=None)
    a=p.parse_args()
    rows=[]
    for size in a.sizes:
        for seed in a.seeds:
            for kind in a.kinds:
                for budget in a.budgets:
                    row=measure(size,seed,kind,budget,a.cache_dir)
                    rows.append(row)
                    print(f"{size:3} {kind:7} seed={seed:3} budget={budget:9} critical={row['critical']['critical_cells']:9} "
                          f"inc={row['critical']['face_incidences']:10} coho={row['cohomology_seconds']:.4f}s "
                          f"Morse={row['morse_seconds']:.4f}s",flush=True)
                    a.output.write_text(json.dumps(dict(rows=rows),indent=2))
    models=compare_models(rows)
    a.output.write_text(json.dumps(dict(rows=rows,models=models,calibration=calibration_coefficients(rows)),indent=2))
    print(json.dumps([{k:v for k,v in item.items() if k!='folds'} for item in models],indent=2))
if __name__=='__main__':main()
