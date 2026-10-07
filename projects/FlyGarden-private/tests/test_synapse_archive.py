import numpy as np
import pyarrow as pa
import pytest
from flygarden.synapse_archive import select_root_rows,pair_counts,reconcile_counts

ROOT=720575940637666446

def test_both_directions_exact_ids_and_source_rows():
    b=pa.record_batch({'pre_pt_root_id':pa.array([ROOT,2,3],type=pa.int64()),
                       'post_pt_root_id':pa.array([2,ROOT,4],type=pa.int64()),
                       'id':[10,11,12],'pre_pt_position_x':[7,8,9]})
    t=select_root_rows(b,np.array([ROOT],np.int64),100)
    assert t['source_row'].to_pylist()==[100,101]
    assert t['pre_pt_root_id'].to_pylist()[0]==ROOT
    assert t['pre_pt_position_x'].to_pylist()==[7,8]
    assert pair_counts(t)=={(ROOT,2):1,(2,ROOT):1}

def test_unmodeled_partners_never_repair_missing_imported_sites():
    rows=reconcile_counts({(ROOT,2):4},{(ROOT,2):3,(ROOT,999):1})
    assert rows[0]['exact_match'] is False
    assert rows[1]['imported_pair'] is False

def test_reject_float_root_ids():
    b=pa.record_batch({'pre_pt_root_id':[float(ROOT)],'post_pt_root_id':[2]})
    with pytest.raises(ValueError):select_root_rows(b,np.array([ROOT],np.int64))

def test_chunked_file_scan_matches_exact_counts(tmp_path):
    import pyarrow.ipc as ipc
    from collections import Counter
    path=tmp_path/'archive.feather'
    batches=[pa.record_batch({'pre_pt_root_id':pa.array([ROOT,2],type=pa.int64()),
                              'post_pt_root_id':pa.array([2,ROOT],type=pa.int64()),
                              'id':[first,first+1]}) for first in [1,3]]
    with pa.OSFile(str(path),'wb') as f:
        with ipc.new_file(f,batches[0].schema) as writer:
            for b in batches:writer.write_batch(b)
    counts=Counter();rows=[];offset=0
    with pa.OSFile(str(path),'rb') as f:
        reader=ipc.open_file(f)
        for i in range(reader.num_record_batches):
            b=reader.get_batch(i);t=select_root_rows(b,np.array([ROOT],np.int64),offset)
            counts.update(pair_counts(t));rows.extend(t['source_row'].to_pylist());offset+=b.num_rows
    assert counts=={(ROOT,2):2,(2,ROOT):2}
    assert rows==[0,1,2,3]
    assert all(r['exact_match'] for r in reconcile_counts(dict(counts),counts))

def test_release_and_reception_locations_stay_distinct_for_autapse():
    import pandas as pd
    from flygarden.synapse_archive import directional_endpoints
    frame=pd.DataFrame({'pre_pt_root_id':[ROOT],'post_pt_root_id':[ROOT],'id':[123],'source_row':[9],
                        'neuropil':['AL_L'],'pre_pt_position_x':[100],'pre_pt_position_y':[200],'pre_pt_position_z':[300],
                        'post_pt_position_x':[400],'post_pt_position_y':[500],'post_pt_position_z':[600]})
    endpoints=directional_endpoints(frame,[ROOT])
    assert len(endpoints)==2
    assert endpoints.loc[endpoints.direction=='input','x'].tolist()==[400]
    assert endpoints.loc[endpoints.direction=='output','x'].tolist()==[100]
    assert endpoints.root_id.tolist()==[str(ROOT),str(ROOT)]
    assert endpoints.synapse_id.tolist()==['123','123']
