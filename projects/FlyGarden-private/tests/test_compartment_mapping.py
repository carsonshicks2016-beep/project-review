import numpy as np
import pytest
from flygarden.compartment_mapping import CoordinateDecoder, partner_glomerulus, unique_mesh_assignment


def test_exact_root_ids_and_independent_carry_across_boundaries():
    d=CoordinateDecoder()
    a='720575940623650823';b='720575940629327659';c='720575940632403986'
    assert d.decode([a,b,'1','2','3'])==(a,b,(1,2,3))
    assert d.decode(['',c,'4','5','6'])==(a,c,(4,5,6))
    assert d.decode([b,'','7','8','9'])==(b,c,(7,8,9))
    assert d.decode(['','','10','11','12'])==(b,c,(10,11,12))


def test_missing_and_bad_roots_do_not_change_decoder():
    d=CoordinateDecoder()
    with pytest.raises(ValueError):d.decode(['','','1','2','3'])
    assert d.pre is None and d.post is None
    d.decode(['720575940623650823','720575940629327659','1','2','3'])
    before=(d.pre,d.post)
    for row in [['1.2','2','1','2','3'],['1','2','-1','2','3'],['1','2','NaN','2','3']]:
        with pytest.raises(ValueError):d.decode(row)
        assert (d.pre,d.post)==before


def test_partner_labels_keep_multiglomerular_and_aliases_unavailable():
    assert partner_glomerulus({'cell_type':'ORN_DM1'})==('DM1','ORN_annotation')
    assert partner_glomerulus({'cell_type':'DM1_adPN','cell_class':'ALPN','cell_sub_class':'uniglomerular'})==('DM1','uniglomerular_PN_annotation')
    for c in ['M_smPNm1','VP1d+VP4_l2PN1','CB1321']:
        assert partner_glomerulus({'cell_type':c,'cell_class':'ALPN','cell_sub_class':'multiglomerular'})[0] is None
    assert partner_glomerulus({'cell_type':'ORN_VM6v'})[0]=='VM6v' # No silent atlas alias.


def test_unique_assignment_rejects_overlap_and_outside():
    labels=unique_mesh_assignment([[0,0],[1,0],[0,1],[1,1]],['DM1','DM2'])
    np.testing.assert_array_equal(labels,['unavailable_outside','DM1','DM2','unavailable_overlap'])


def test_native_mesh_engine_known_enclosures():
    import trimesh
    first=trimesh.creation.box(extents=[2,2,2])
    second=first.copy();second.apply_translation([1,0,0])
    points=np.array([[-.5,0,0],[.5,0,0],[1.5,0,0],[3,0,0]])
    memberships=np.column_stack([first.contains(points),second.contains(points)])
    np.testing.assert_array_equal(unique_mesh_assignment(memberships,['A','B']),
                                  ['A','unavailable_overlap','B','unavailable_outside'])
