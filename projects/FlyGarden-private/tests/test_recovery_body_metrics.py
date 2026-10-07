import importlib.util
from pathlib import Path
import numpy as np

def test_stationary_body_is_not_useful_forward_motion():
    path=Path(__file__).resolve().parents[1]/'scripts/report_recovery_body.py'
    spec=importlib.util.spec_from_file_location('body_auditor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    initial={'position':[0,0,1],'heading':0}
    rows=[{'body':{**initial,'flipped':False},'motor':[.65,.65],
           'physical_contacts':{'load_weighted_tangential_speed_mm_per_second':None}} for _ in range(2400)]
    result=module.metrics(initial,rows)
    assert result['finite'] and result['stop_speed_mm_s']==0
    assert result['phases'][1]['net_mm']==0 and result['active_stall_seconds']==60
    assert result['contact_coverage']==0
    rows[-1]['body']={**initial,'flipped':False,'position':[np.nan,0,1]}
    assert not module.metrics(initial,rows)['finite']
