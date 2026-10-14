"""Lightweight provenance-preserving persistence diagram and attribution bridge."""
from __future__ import annotations
import numpy as np


def diagram_rows(engine):
    """Finite H1 points (birth, death) and matching computational pair IDs."""
    catalog=engine.h1_pairs()
    return (np.asarray([[p[1],p[2]] for p in catalog],dtype=np.float64).reshape(-1,2),
            tuple(p[0] for p in catalog))


def selected_loops(engine,scores,*,score_ids,top_k=5,
                   max_operations=100_000_000,max_output_entries=2_000_000):
    """Explain selected top-attributed H1 pairs with certified spatial loops.

    `scores` are signed point attributions from an independent model. This
    function does not train, alter or interpret the model's causal semantics.
    Pair IDs must align *exactly* with the diagram rows.
    """
    _,ids=diagram_rows(engine)
    values=np.asarray(scores,dtype=np.float64)
    if values.shape!=(len(ids),) or not np.isfinite(values).all():
        raise ValueError('Expected one finite score per persistence diagram point')
    if tuple(score_ids)!=ids:raise ValueError('Pair IDs must match all diagram rows exactly')
    if not isinstance(top_k,int) or isinstance(top_k,bool) or top_k<0:
        raise ValueError('top_k must be a nonnegative integer')
    result=[]
    order=sorted(range(len(values)),key=lambda j:(-abs(values[j]),j))
    for i in order[:top_k]:
        cert=engine.replay_h1_native(ids[i],max_operations=max_operations,
                                    max_output_entries=max_output_entries)
        cert['attribution']=float(values[i]);result.append(cert)
    return result
