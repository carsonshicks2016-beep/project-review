"""Pinned NeuroMechFly 2.1 body and supplied HybridTurningController."""
import copy
import numpy as np
import mujoco as mj
from flygym import Simulation
from flygym.anatomy import BodySegment,ContactBodiesPreset,LEGS
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import HybridTurningController,HybridControllerObservation,LocomotionAction,PreprogrammedSteps,apply_locomotion_action,make_locomotion_fly

class Body:
    dt=.0001
    control_dt=.0005
    def __init__(self,seed=0,blocks=(),spawn=(0.,0.),vision=False,foods=(),predator=None):
        self.arena=FlatGroundWorld()
        for i,b in enumerate(blocks):
            z=b.get('z',2.)
            g=self.arena.mjcf_root.worldbody.add_geom(name=f'obstacle_{i}',type=mj.mjtGeom.mjGEOM_BOX,pos=(b['x'],b['y'],z/2),size=(b['w']/2,b['h']/2,z/2),rgba=(.2,.3,.25,1),contype=0,conaffinity=0)
            self.arena.ground_geoms.append(g)
        for f in foods:
            self.arena.mjcf_root.worldbody.add_geom(name=f"food_{f['id']}",type=mj.mjtGeom.mjGEOM_SPHERE,pos=[f['x'],f['y'],.6],size=[1.,0,0],rgba=[.7,.9,.35,1] if f['odor']==0 else [.65,.45,.9,1],contype=0,conaffinity=0)
        predator_body=self.arena.mjcf_root.worldbody.add_body(name='visual_predator',mocap=True,pos=[0,0,-100])
        predator_body.add_geom(type=mj.mjtGeom.mjGEOM_ELLIPSOID,size=[2.4,1.3,.9],rgba=[.8,.2,.1,1],contype=0,conaffinity=0)
        self.fly=make_locomotion_fly(name='garden_fly',add_adhesion=True,colorize=True)
        self.fly.add_vision()
        for side in ('l','r'):
            self.fly.mjcf_root.body(f'{side}_funiculus').add_site(name=f'odor_{side}',pos=[.02,0,-.1],size=[.02,.02,.02],group=4)
        self.arena.add_fly(self.fly,[*spawn,.8],Rotation3D('quat',[1,0,0,0]),bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,add_ground_contact_sensors=False)
        self.sim=Simulation(self.arena,timestep=self.dt)
        self.preprogrammed=PreprogrammedSteps();dofs=self.fly.get_actuated_jointdofs_order('position')
        self.controller=HybridTurningController(timestep=self.control_dt,retraction_persistence_steps=4,retraction_persistence_initiation_threshold=4.,preprogrammed_steps=self.preprogrammed,output_dof_order=dofs)
        self.sim.reset();self.controller.reset(seed=seed)
        action=LocomotionAction(joint_angles=self.preprogrammed.default_pose_by_dof_order(dofs),adhesion_onoff=np.ones(6,dtype=bool))
        apply_locomotion_action(self.sim,self.fly.name,action);self.sim.warmup()
        self.seed=seed;self.steps=0
        order=self.fly.get_bodysegs_order();self.thorax=order.index(BodySegment('c_thorax'));self.feet=[order.index(BodySegment(f'{leg}_tarsus5')) for leg in LEGS]
        self.thorax_id=int(self.sim._internal_bodyids_by_fly[self.fly.name][self.thorax])
        self.antenna_ids=[i for i in range(self.sim.mj_model.nsite) if self.sim.mj_model.site(i).name.endswith(('odor_l','odor_r'))]
        if len(self.antenna_ids)!=2:raise RuntimeError('Missing physical antennal sampling sites')
        self.update_visual_world(foods,predator or {'enabled':False})
    def advance(self,dt,motor,capture=None):
        n=round(dt/self.dt)
        if abs(n*self.dt-dt)>1e-9:raise ValueError('Body window must match timestep')
        motor=np.clip(motor,0,1.2)
        if n % round(self.control_dt/self.dt):raise ValueError('Window must match control interval')
        stride=round(self.control_dt/self.dt)
        next_sample=(int(round(self.steps*self.dt*30))+1)/30
        for _ in range(n//stride):
            obs=HybridControllerObservation.from_sim(self.sim,self.fly.name)
            action=self.controller.step(motor,obs);apply_locomotion_action(self.sim,self.fly.name,action);mj.mj_step(self.sim.mj_model,self.sim.mj_data,nstep=stride);self.steps+=stride
            if capture is not None and self.steps*self.dt+1e-10>=next_sample:
                capture(self.steps*self.dt,self.recording_pose());next_sample+=1/30
        if not all(np.isfinite(x).all() for x in (self.sim.mj_data.qpos,self.sim.mj_data.qvel,self.sim.mj_data.qacc)):raise RuntimeError('Non-finite body state')
        mj.mj_forward(self.sim.mj_model,self.sim.mj_data)
        return self.observation()
    def recording_pose(self):
        # Use separate derived state: sampling cannot perturb physics or gait observations.
        if not hasattr(self,'_recording_data'):self._recording_data=mj.MjData(self.sim.mj_model)
        d=self._recording_data;m=self.sim.mj_model;source=self.sim.mj_data
        d.qpos[:]=source.qpos;d.mocap_pos[:]=source.mocap_pos;d.mocap_quat[:]=source.mocap_quat
        mj.mj_kinematics(m,d)
        ids=self.sim._internal_bodyids_by_fly[self.fly.name];r=d.xmat[self.thorax_id].reshape(3,3)
        return dict(body=dict(position=d.xpos[self.thorax_id].tolist(),heading=float(np.arctan2(r[1,0],r[0,0])),feet=d.xpos[ids[self.feet]].tolist(),flipped=bool(r[2,2]<0),steps=self.steps,antennae=d.site_xpos[self.antenna_ids].tolist()),visuals=[dict(id=i,p=d.geom_xpos[i].tolist(),r=d.geom_xmat[i].tolist()) for i in range(m.ngeom) if m.geom_type[i]==7])
    def observation(self):
        pos=self.sim.get_body_positions(self.fly.name);r=self.sim.mj_data.xmat[self.thorax_id].reshape(3,3);contacts=HybridControllerObservation.from_sim(self.sim,self.fly.name).stumbling_contact_forces
        return dict(position=pos[self.thorax].tolist(),heading=float(np.arctan2(r[1,0],r[0,0])),feet=pos[self.feet].tolist(),contacts=contacts.reshape(-1,3).tolist(),flipped=bool(r[2,2]<0),steps=self.steps,antennae=self.sim.mj_data.site_xpos[self.antenna_ids].tolist())
    def snapshot(self):
        flags=mj.mjtState.mjSTATE_INTEGRATION;s=np.empty(mj.mj_stateSize(self.sim.mj_model,flags));mj.mj_getState(self.sim.mj_model,self.sim.mj_data,s,flags)
        c=self.controller;cpg=c.cpg_network
        return dict(signature=[self.sim.mj_model.nq,self.sim.mj_model.nv,self.sim.mj_model.nbody,self.sim.mj_model.ngeom,self.sim.mj_model.nmocap],physics=s,steps=self.steps,controller={k:copy.deepcopy(getattr(c,k)) for k in ('retraction_correction','stumbling_correction','retraction_persistence_counter','last_info')},cpg={k:copy.deepcopy(getattr(cpg,k)) for k in ('curr_phases','curr_magnitudes','intrinsic_amps','intrinsic_freqs')},rng=cpg.random_state.get_state())
    def restore(self,s):
        signature=[self.sim.mj_model.nq,self.sim.mj_model.nv,self.sim.mj_model.nbody,self.sim.mj_model.ngeom,self.sim.mj_model.nmocap]
        if s.get('signature')!=signature:raise ValueError('Checkpoint body schema differs; restore a learned-fly checkpoint instead')
        mj.mj_setState(self.sim.mj_model,self.sim.mj_data,s['physics'],mj.mjtState.mjSTATE_INTEGRATION);mj.mj_forward(self.sim.mj_model,self.sim.mj_data)
        # Rebuild derived poses, then restore solver warm-start and integration state exactly.
        mj.mj_setState(self.sim.mj_model,self.sim.mj_data,s['physics'],mj.mjtState.mjSTATE_INTEGRATION);self.steps=s['steps']
        for k,v in s['controller'].items():setattr(self.controller,k,copy.deepcopy(v))
        for k,v in s['cpg'].items():setattr(self.controller.cpg_network,k,copy.deepcopy(v))
        if 'rng' in s:self.controller.cpg_network.random_state.set_state(s['rng'])
    def visuals(self):
        d=self.sim.mj_data;m=self.sim.mj_model
        return [dict(id=i,p=d.geom_xpos[i].tolist(),r=d.geom_xmat[i].tolist()) for i in range(m.ngeom) if m.geom_type[i]==7]
    def geometry(self):
        m=self.sim.mj_model;meshes={};geoms=[]
        for i in range(m.ngeom):
            if m.geom_type[i]!=7:continue
            mid=int(m.geom_dataid[i]);v=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid]);f=int(m.mesh_faceadr[mid]);nf=int(m.mesh_facenum[mid])
            if str(mid) not in meshes:meshes[str(mid)]=dict(vertices=m.mesh_vert[v:v+nv].flatten().tolist(),faces=m.mesh_face[f:f+nf].flatten().tolist())
            material=int(m.geom_matid[i]);color=m.mat_rgba[material] if material>=0 else m.geom_rgba[i]
            geoms.append(dict(id=i,mesh=str(mid),color=color.tolist()))
        return dict(meshes=meshes,geoms=geoms)
    def update_visual_world(self,foods,predator):
        d=self.sim.mj_data;m=self.sim.mj_model
        d.mocap_pos[0]=[predator.get('x',0),predator.get('y',0),1.2] if predator.get('enabled') else [0,0,-100]
        h=predator.get('heading',0);d.mocap_quat[0]=[np.cos(h/2),0,0,np.sin(h/2)]
        for f in foods:
            gid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,f"food_{f['id']}")
            if gid>=0:m.geom_rgba[gid,3]=1. if f['units']>0 else 0.
        mj.mj_forward(m,d)
    def retinal_readouts(self):return self.sim.get_ommatidia_readouts(self.fly.name)
    def eye_frames(self):return self.sim.get_raw_vision(self.fly.name)
    def close(self):
        if self.sim.renderer is not None:self.sim.renderer.close()
        if self.sim.eye_renderer is not None:self.sim.eye_renderer.close()
