"""Exactness and conservation tests for prospective Morse cost features."""
import numpy as np
import pytest
from betti_curves_py.h1_morse import h1_persistence
from betti_curves_py._morse.native_backend import CompactMorse


@pytest.mark.parametrize('seed',[2,7,11])
@pytest.mark.parametrize('budget',[1,5_000_000])
@pytest.mark.parametrize('indexed',[True,False])
def test_incremental_boundary_metrics_match_critical_csr(tmp_path,seed,budget,indexed):
    v=np.random.default_rng(seed).integers(0,8,(9,9,9),dtype=np.uint8)
    c=CompactMorse(v,tmp_path/'morse',indexed_scheduler=indexed,
                   max_operations=budget)
    prof=c.critical_structure_profile()
    assert c.stats['critical_faces']==prof['critical_faces']
    assert c.stats['critical_face_incidences']==prof['face_incidences']
    assert prof['face_degree_squared']>=prof['face_incidences']
    assert prof['critical_edges']<=prof['critical_cells']
    assert prof['critical_faces']<=prof['critical_cells']


@pytest.mark.parametrize('seed',[3,11])
@pytest.mark.parametrize('rate',[(1500.,0.),(0.,400.),(1200.,100.)])
def test_structural_adaptive_preserves_barcodes_and_certificates(tmp_path,seed,rate):
    v=np.random.default_rng(seed).integers(0,8,(10,10,10),dtype=np.uint8)
    base=h1_persistence(v,output_dir=tmp_path/'base',indexed_scheduler=True,
                        max_operations=1,store_pair_ids=False)
    adapt=h1_persistence(v,output_dir=tmp_path/'adapt',indexed_scheduler=True,
                         adaptive_cohomology_ns_per_face=rate[0],
                         adaptive_cohomology_ns_per_incidence=rate[1],
                         adaptive_window=16,adaptive_patience=1,
                         max_operations=20000000,store_pair_ids=False)
    assert sorted(base.pairs())==sorted(adapt.pairs())
    assert adapt.statistics['critical_face_incidences']>=0
    if len(adapt):
        assert adapt.reconstruct(adapt.pair_id_at(0))['verified']


@pytest.mark.parametrize('key,value',[
    ('adaptive_cohomology_ns_per_face',-1.),
    ('adaptive_cohomology_ns_per_incidence',float('nan')),
    ('adaptive_cohomology_ns_per_face',float('inf')),
    ('adaptive_cohomology_ns_per_incidence',-1.),
])
def test_invalid_rate_rejected(tmp_path,key,value):
    v=np.zeros((3,3,3),dtype=np.uint8)
    with pytest.raises(ValueError,match='adaptive'):
        h1_persistence(v,output_dir=tmp_path/'bad',indexed_scheduler=True,**{key:value})


def test_structural_adaptive_requires_indexed(tmp_path):
    v=np.zeros((3,3,3),dtype=np.uint8)
    with pytest.raises(ValueError,match='Adaptive'):
        h1_persistence(v,output_dir=tmp_path/'bad',adaptive_cohomology_ns_per_face=300.)


def test_counters_and_cache_are_observational(tmp_path):
    v=np.random.default_rng(2).integers(0,8,(12,12,12),dtype=np.uint8)
    c=CompactMorse(v,tmp_path/'catalog',indexed_scheduler=True,max_operations=1)
    a=c.critical_structure_profile();r=c.h1_records()
    p=c.cohomology_profile()
    assert p is not None
    assert p['critical_edges']==a['critical_edges']
    assert p['critical_faces']==a['critical_faces']
    assert p['face_incidences']==a['face_incidences']
    assert p['face_degree_squared']==a['face_degree_squared']
    assert p['finite_h1_pairs']==len(r)
    assert p['cohomology_operations']>=0
    assert p['annotation_updates']>0
    assert p['annotation_xor_terms']>=p['annotation_updates']
    reopened=CompactMorse.open(tmp_path/'catalog')
    assert reopened.cohomology_profile()==p


def test_public_statistics_expose_observed_sparsity(tmp_path):
    v=np.random.default_rng(22).integers(0,8,(8,8,8),dtype=np.uint8)
    r=h1_persistence(v,output_dir=tmp_path/'catalog',
                     indexed_scheduler=True,max_operations=1)
    prof=r.statistics['cohomology_profile']
    assert prof['critical_cells']==r.statistics['remaining']
    assert prof['face_incidences']==r.statistics['critical_face_incidences']
    assert prof['finite_h1_pairs']==len(r)
