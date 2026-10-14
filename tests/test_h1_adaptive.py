"""Exactness and diagnostics of optional marginal-cost Morse stopping."""
import numpy as np
import pytest
from betti_curves_py.h1_morse import h1_persistence


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('seed', [2, 5, 11])
def test_adaptive_preserves_exact_bars(seed, dtype, tmp_path):
    data=np.random.default_rng(seed).integers(0, 8, (9, 9, 9), dtype=dtype)
    base=h1_persistence(data, output_dir=tmp_path/'base',indexed_scheduler=True,
                        max_operations=20_000_000)
    stop=h1_persistence(data, output_dir=tmp_path/'adapt',indexed_scheduler=True,
                        max_operations=20_000_000,
                        adaptive_cohomology_ns_per_cell=0.001,
                        adaptive_window=8,adaptive_patience=1)
    assert sorted(base.pairs())==sorted(stop.pairs())
    assert np.array_equal(base.at(np.arange(8)),stop.at(np.arange(8)))
    assert stop.statistics['adaptive_windows']>=0
    if len(stop):
        cert=stop.reconstruct(stop.pair_id_at(0))
        assert cert['verified']


def test_adaptive_stops_early_on_costly_scheduler(tmp_path):
    data=np.random.default_rng(11).integers(0,8,(24,24,24),dtype=np.uint8)
    stop=h1_persistence(data,output_dir=tmp_path/'early',indexed_scheduler=True,
                        max_operations=100_000_000,
                        adaptive_cohomology_ns_per_cell=0.001,
                        adaptive_window=32,adaptive_patience=2)
    assert stop.statistics['adaptive_stopped']==1
    assert stop.statistics['adaptive_windows']>=2
    assert stop.statistics['remaining']<stop.statistics['precollapse_remaining']
    assert stop.statistics['hit_budget']==0
    no_extra=h1_persistence(data,output_dir=tmp_path/'none',indexed_scheduler=True,
                            max_operations=1)
    assert sorted(stop.pairs())==sorted(no_extra.pairs())


def test_adaptive_disabled_by_default(tmp_path):
    data=np.full((8,8,8),5,dtype=np.uint8)
    data[1:7,1:7,1:7]=0;data[1:7,3,3]=5
    old=h1_persistence(data,output_dir=tmp_path/'old',indexed_scheduler=True)
    explicit=h1_persistence(data,output_dir=tmp_path/'explicit',indexed_scheduler=True,
                            adaptive_cohomology_ns_per_cell=0)
    assert sorted(old.pairs())==sorted(explicit.pairs())
    assert explicit.statistics['adaptive_stopped']==0
    assert explicit.statistics['adaptive_windows']==0


@pytest.mark.parametrize('kwargs', [
    dict(adaptive_cohomology_ns_per_cell=-1),
    dict(adaptive_cohomology_ns_per_cell=float('nan')),
    dict(adaptive_cohomology_ns_per_cell=float('inf')),
    dict(adaptive_cohomology_ns_per_cell=1.,indexed_scheduler=False),
    dict(adaptive_window=0),
    dict(adaptive_patience=0),
])
def test_invalid_adaptive_config_rejected(tmp_path,kwargs):
    data=np.zeros((3,3,3),dtype=np.uint8)
    with pytest.raises(ValueError,match='adaptive|Adaptive'):
        h1_persistence(data,output_dir=tmp_path/'invalid',**kwargs)
