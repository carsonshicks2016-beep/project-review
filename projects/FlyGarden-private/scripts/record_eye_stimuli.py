"""Actual FlyGym eye-camera stimuli in an isolated, stationary physical pose."""
import json,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np,mujoco as mj
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.body import Body
from flygym.compose import FlatGroundWorld
from flygarden.vision_encoding import EyeExpansion
from flygarden.recording import atomic_json,space_check
from scripts.audit_brain_reference import sha
OUT=ROOT/'reports/brain-integration/stage6-20261006'
CUES=('blank','static_left','static_right','loom_left','loom_right','moving_pattern','wall')
class VisualBench(FlatGroundWorld):
 def __init__(self):
  super().__init__()
  for name in ('sphere','wall','pattern'):
   body=self.mjcf_root.worldbody.add_body(name='bench_'+name,mocap=True,pos=[0,0,-100])
   if name=='sphere':body.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE,size=[1,0,0],rgba=[.03,.03,.03,1],contype=0,conaffinity=0)
   else:
    body.add_geom(type=mj.mjtGeom.mjGEOM_BOX,size=[.06,4,4],rgba=[.03,.03,.03,1] if name=='wall' else [1,1,1,1],contype=0,conaffinity=0)
    if name=='pattern':
     for y in np.arange(-3.6,4,.8):body.add_geom(type=mj.mjtGeom.mjGEOM_BOX,pos=[-.08,y,0],size=[.04,.2,4],rgba=[.02,.02,.02,1],contype=0,conaffinity=0)
def record():
 started=time.perf_counter();out=OUT/'eye-stimuli';out.mkdir(exist_ok=False)
 with patch('flygarden.body.FlatGroundWorld',VisualBench):body=Body(seed=6101)
 try:
  d=body.sim.mj_data;m=body.sim.mj_model
  mocap={name:int(m.body('bench_'+name).mocapid[0]) for name in ('sphere','wall','pattern')}
  baseline=body.eye_frames();ids=body.sim._intern_eye_camera_ids_by_fly[body.fly.name];eyes=[]
  for i,c in enumerate(ids):
   eyes.append({'side':('left','right')[i],'name':m.camera(c).name,'position':d.cam_xpos[c].tolist(),'forward':(-d.cam_xmat[c].reshape(3,3)[:,2]).tolist()})
  initial=body.snapshot();summaries=[]
  for cue in CUES:
   encoder=EyeExpansion(baseline);frames=[];rows=[];blank_encoder=EyeExpansion(np.full_like(baseline,127))
   for tick in range(90):
    t=tick/30
    for k in mocap.values():d.mocap_pos[k]=[0,0,-100]
    if .5<=t<1.5 and cue!='blank':
     theta=np.pi/3 if cue.endswith('left') or cue=='moving_pattern' else -np.pi/3 if cue.endswith('right') else 0
     r=20-(t-.5)*16 if cue.startswith('loom') or cue=='wall' else 8
     xy=np.array([np.cos(theta),np.sin(theta)])*r
     kind='pattern' if cue=='moving_pattern' else 'wall' if cue=='wall' else 'sphere'
     if kind=='pattern':xy+=np.array([-np.sin(theta),np.cos(theta)])*(t-.5)*3
     d.mocap_pos[mocap[kind]]=[*xy,3];d.mocap_quat[mocap[kind]]=[np.cos(theta/2),0,0,np.sin(theta/2)]
    mj.mj_forward(m,d);f=body.eye_frames();features=encoder.advance(f,1/30);blank_features=blank_encoder.advance(np.full_like(f,127),1/30);assert all(z['lplc2_hz']==0 for z in blank_features)
    frames.append(f);rows.append({'time':t,'features':features,'blanked_features':blank_features})
   space_check(out,128*1024**2)
   with (out/f'{cue}.npz').open('wb') as file:np.savez_compressed(file,frames=np.array(frames),baseline=baseline,times=np.arange(90)/30)
   atomic_json(out/f'{cue}.json',rows)
   selected=rows[30];summaries.append({'cue':cue,'peak_lplc2_hz':[max(r['features'][s]['lplc2_hz'] for r in rows) for s in range(2)],'peak_dark_area':[max(r['features'][s]['dark_area_fraction'] for r in rows) for s in range(2)]})
   Image.fromarray(np.concatenate(frames[30],axis=1)).save(out/f'{cue}.png');print(cue,summaries[-1],flush=True)
  after=body.snapshot();assert np.array_equal(initial['physics'][:m.nq],after['physics'][:m.nq])
  # Verify handedness from measured camera directions and visible static stimuli.
  left=next(s for s in summaries if s['cue']=='static_left')['peak_dark_area'];right=next(s for s in summaries if s['cue']=='static_right')['peak_dark_area']
  assert eyes[0]['forward'][1]>0 and eyes[1]['forward'][1]<0 and left[0]>left[1] and right[1]>right[0]
  atomic_json(out/'manifest.json',{'status':'complete','eyes':eyes,'image_shape':list(baseline.shape),'eye_order':['left','right'],'cues':CUES,'summaries':summaries,'orientation_verified':True,'physics':'Stationary initialized body pose, visual mocap stimuli only; no locomotion or hidden coordinate input to encoder.','sources':{s:sha(ROOT/s) for s in ('flygarden/body.py','flygarden/vision_encoding.py','scripts/record_eye_stimuli.py')},'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'wall_seconds':time.perf_counter()-started})
 finally:body.close()
if __name__=='__main__':record()
