"""Read-only solver contact and tangential velocity diagnostics.

These are physical measurements, not a neural encoder. Force magnitudes retain
the model's native units. Adhesion actuator forces are not included in the
solver contact-force totals. MuJoCo's contact frame is normal then tangents.
Reference: https://mujoco.readthedocs.io/en/stable/computation/
"""
import numpy as np
import mujoco as mj
from flygym.anatomy import BodySegment,LEGS

def physical_contacts(body):
    sim=body.sim;model=sim.mj_model;data=sim.mj_data
    mapping=sim._internal_geomid_by_bodyseg_by_fly[body.fly.name]
    foot_geom={int(mapping[BodySegment(f'{leg}_tarsus5')]):i for i,leg in enumerate(LEGS)}
    ground=set(int(i) for i in sim._internal_ground_geom_ids)
    # Use FlyGym's own public ground filtering for force vectors.
    forces=sim.get_bodysegment_contact_forces(body.fly.name,[BodySegment(f'{leg}_tarsus5') for leg in LEGS],ground_only=True)
    contacts=[]
    for index in range(data.ncon):
        c=data.contact[index];g1,g2=int(c.geom1),int(c.geom2)
        if c.efc_address<0:continue
        if g1 in foot_geom and g2 in ground:foot,other=g1,g2
        elif g2 in foot_geom and g1 in ground:foot,other=g2,g1
        else:continue
        force=np.zeros(6);mj.mj_contactForce(model,data,index,force)
        jac=np.zeros((3,model.nv));rot=np.zeros_like(jac)
        mj.mj_jac(model,data,jac,rot,c.pos,int(model.geom_bodyid[foot]));velocity=user@example.com
        mj.mj_jac(model,data,jac,rot,c.pos,int(model.geom_bodyid[other]));velocity-=user@example.com
        tangent=np.asarray(c.frame).reshape(3,3)[1:]@velocity
        contacts.append({'leg':str(LEGS[foot_geom[foot]]),'normal_force_model_units':float(force[0]),
                         'tangential_speed_mm_per_second':float(np.linalg.norm(tangent)),
                         'position':np.asarray(c.pos).tolist()})
    loads=np.array([max(0,c['normal_force_model_units']) for c in contacts])
    speeds=np.array([c['tangential_speed_mm_per_second'] for c in contacts])
    rms=float(np.sqrt(np.sum(loads*speeds**2)/loads.sum())) if loads.sum()>0 else None
    return {'foot_solver_ground_forces_model_units':np.asarray(forces).tolist(),
            'contact_points':contacts,'load_weighted_tangential_speed_mm_per_second':rms,
            'scope':'Read-only tarsus5 solver contacts; model force units; adhesion actuator loads excluded; no brain input.'}
