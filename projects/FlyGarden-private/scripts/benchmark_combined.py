import sys,json,time,resource
import mujoco as mj
import imageio.v2 as imageio
from PIL import Image,ImageDraw
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.body import Body
from flygarden.brain import FullBrain,ROOT
from flygarden.world import World
from flygarden.trial_recording import EmbodiedRecording
print('Building complete neural + body loop',flush=True)
brain=FullBrain();world=World();body=Body(blocks=world.arena['blocks'],spawn=world.arena['spawn'],foods=world.arena['foods'],predator=world.arena['predator']);brain.advance(.01);t=time.perf_counter();frames=[]
renderer=mj.Renderer(body.sim.mj_model,width=960,height=540);camera=mj.MjvCamera();camera.distance=12;camera.azimuth=-90;camera.elevation=-30
recording=EmbodiedRecording(body,world,brain,seed=0,controller='Full connectome + supplied gait benchmark',diagnostic=True)
writer=imageio.get_writer(ROOT/'reports/full-brain-walking.mp4',fps=10,codec='libx264',quality=8)
for i in range(100):
 body.update_visual_world(world.arena['foods'],world.arena['predator'])
 sensory=world.sensory(body.observation());brain_start=float(brain.network.t/__import__('brian2').second);motor=brain.advance(.1,odor=sensory['odor'],p9=(.65,.65));obs=body.advance(.1,motor,capture=recording.capture);reward=world.advance(.1,obs);brain.reinforce(reward);recording.step(sensory,motor,.1,brain_start,reinforcement=reward)
 frames.append(dict(time=(i+1)*.1,body=obs,motor=motor.tolist(),rates=brain.last_rates,visuals=body.visuals()))
 camera.lookat[:]=obs['position'];renderer.update_scene(body.sim.mj_data,camera);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,32),fill=(14,23,20));draw.text((12,9),f'Full imported brain + supplied gait | {(i+1)*.1:.1f} simulated seconds | experimental adapter',fill='white');writer.append_data(np.asarray(frame))
 if i%10==9:print(f'{(i+1)/10}/10 seconds; status={world.status}',flush=True)
 if world.status!='running':break
recording.finish(dict(status=world.status,time=world.time,food=world.collected));writer.close();renderer.close();wall=time.perf_counter()-t;seconds=len(frames)*.1
report=dict(status='completed',simulated_seconds=seconds,wall_seconds=wall,throughput=seconds/wall,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,neurons=brain.n,connections=brain.edges,body='NeuroMechFly 2.1',physics_dt=Body.dt,walking_control_dt=Body.control_dt,world_status=world.status,initial_stimulus='Engineered tonic P9 exploration at 65 Hz',learned=False)
(ROOT/'reports/combined-benchmark.json').write_text(json.dumps(report,indent=2));(ROOT/'reports/combined-trace.json').write_text(json.dumps(frames));(ROOT/'reports/body-geometry.json').write_text(json.dumps(body.geometry()));print(json.dumps(report,indent=2),flush=True);body.close()
