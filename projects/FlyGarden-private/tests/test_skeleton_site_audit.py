import numpy as np
import pytest
from flygarden.skeleton_site_audit import nearest_segments

def test_long_edge_beats_nearest_vertex_and_nearest_center():
    v=np.array([[-10000,0,0],[10000,0,0],[9000,100,0],[9000,101,0]],float)
    d,edge,t=nearest_segments([[9000,0,0]],v,[[0,1],[2,3]])
    assert d[0]==0 and edge[0]==0 and t[0]==pytest.approx(.95)

def test_exact_search_matches_exhaustive_projection():
    rng=np.random.default_rng(23);v=rng.normal(size=(80,3))*10000
    e=np.column_stack([np.arange(79),np.arange(1,80)]);p=rng.normal(size=(20,3))*10000
    d,_,_=nearest_segments(p,v,e)
    expected=[]
    for q in p:
        u=v[e[:,1]]-v[e[:,0]];t=np.clip(((q-v[e[:,0]])*u).sum(axis=1)/(u*u).sum(axis=1),0,1)
        expected.append(np.linalg.norm(q-(v[e[:,0]]+t[:,None]*u),axis=1).min())
    assert np.allclose(d,expected,atol=1e-7)

def test_degenerate_segment_and_invalid_geometry():
    d,_,_=nearest_segments([[1,0,0]],[[0,0,0],[0,0,0]],[[0,1]])
    assert d[0]==1
    with pytest.raises(ValueError):nearest_segments([[np.nan,0,0]],[[0,0,0]],[[0,0]])
