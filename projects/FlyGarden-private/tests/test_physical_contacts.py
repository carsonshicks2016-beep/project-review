import numpy as np
from flygarden.synchronized import SynchronizedBody
from flygarden.physical_contacts import physical_contacts

def test_native_foot_contact_probe_is_read_only_and_sees_ground_loads():
    body=SynchronizedBody(seed=9901)
    try:
        body.advance(.1,[.65,.65]);before=body.snapshot()
        values=physical_contacts(body);after=body.snapshot()
        assert np.array_equal(before['physics'],after['physics'])
        assert np.isfinite(values['foot_solver_ground_forces_model_units']).all()
        assert np.linalg.norm(values['foot_solver_ground_forces_model_units'])>0
        assert values['contact_points']
        assert values['load_weighted_tangential_speed_mm_per_second'] is not None
        assert values['load_weighted_tangential_speed_mm_per_second']>=0
        assert np.array_equal(before['cpg']['curr_phases'],after['cpg']['curr_phases'])
    finally:body.close()
