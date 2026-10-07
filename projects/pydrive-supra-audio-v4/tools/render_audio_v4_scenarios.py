#!/usr/bin/env python3
"""Render deterministic procedural listening scenarios; no samples are read."""
from __future__ import annotations
import argparse, csv, wave
from dataclasses import replace
from pathlib import Path
import numpy as np
from supra.audio_v4 import AudioEvents,AudioFrame,EngineSynthV4,RATIOS_787B,SpatialRendererV4

SR=48000; BLOCK=1024
def write(path,x):
    x=np.clip(x,-.98,.98); pcm=(x*32767).astype('<i2')
    channels=1 if x.ndim==1 else x.shape[1]
    with wave.open(str(path),'wb') as w: w.setnchannels(channels); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
def render(name,frames):
    synth=EngineSynthV4(SR,78755); chunks=[]
    for frame,seconds in frames:
        for _ in range(int(seconds*SR/BLOCK)):
            stems=synth.synthesize_stems(frame,BLOCK)
            chunks.append(sum(stems[k] for k in ('exhaust','intake','mechanical','accessories','transmission','tyres','brakes','chassis'))*.55)
    return name,np.concatenate(chunks)
def scenarios():
    idle=AudioFrame(rpm=1100,engine_state=2)
    return [
      ('startup_idle',[(replace(idle,engine_state=1),.7),(idle,2.3)]),
      ('loaded_sweep',[(AudioFrame(rpm=r,throttle=.9),.35) for r in range(1500,9001,500)]),
      ('five_gear_acceleration',[(AudioFrame(rpm=7800,throttle=1,gear=g,ratio=RATIOS_787B[g-1],events=AudioEvents(shift=g-1)),.7) for g in range(1,6)]),
      ('matched_downshift',[(AudioFrame(rpm=6500,throttle=.2,gear=3,ratio=RATIOS_787B[2],clutch=.3,shift_phase=.8,shift_mismatch=.05,events=AudioEvents(shift=1)),1.5)]),
      ('mismatched_downshift',[(AudioFrame(rpm=6500,throttle=.2,gear=3,ratio=RATIOS_787B[2],clutch=.3,shift_phase=.8,shift_mismatch=.8,events=AudioEvents(shift=1)),1.5)]),
      ('braking_lockup',[(AudioFrame(rpm=5000,speed=65,brake=.9,wheel_slip=(-.8,-.8,-.3,-.3),brake_temperature=(.9,.9,.7,.7)),2)]),
      ('surfaces',[(AudioFrame(rpm=4500,speed=45,wheel_slip=(.7,)*4,wheel_surface=(s,)*4,kerb=1 if s==4 else 0),.8) for s in range(6)]),
      ('landing_collision',[(AudioFrame(rpm=4000,events=AudioEvents(landing=1),landing_force=4),.8),(AudioFrame(rpm=3000,events=AudioEvents(landing=1,collision=1),collision_impulse=7,damage=.3),1.2)]),
      ('limiter',[(AudioFrame(rpm=9000,throttle=1,limiter=1),2)]),
      ('shutdown',[(replace(idle,engine_state=3),.8),(replace(idle,engine_state=0,rpm=0),1.2)])]
def spatial(name,mode):
    renderer=SpatialRendererV4(SR,1); chunks=[]; total=int(5*SR/BLOCK)
    for i in range(total):
        x=-90+180*i/max(1,total-1); car=AudioFrame(x=x,y=0,yaw=0,vx=72,vy=0,speed=72,rpm=8200,throttle=.9)
        if mode=='onboard': listener=(x-.3,0,72,0,0); perspective=2
        elif mode=='chase': listener=(x-8,1,72,0,0); perspective=1
        else: listener=(0,18,0,0,-np.pi/2); perspective=0
        chunks.append(renderer.render([car],listener,BLOCK,perspective=perspective))
    return name,np.concatenate(chunks)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path('runtime/audio_v4_scenarios')); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    for name,frames in scenarios(): n,x=render(name,frames); write(a.out/f'{n}.wav',x)
    spatial_names=[]
    for name,mode in (('onboard_pass','onboard'),('chase_pass','chase'),('trackside_flyby','trackside')):
        n,x=spatial(name,mode); write(a.out/f'{n}.wav',x); spatial_names.append(name)
    with (a.out/'listening_matrix.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['scenario','r26b_identity','mechanical_plausibility','shift_authenticity','perspective','doppler','fatigue','missing_or_overpowered','notes'])
        for name,_ in scenarios(): w.writerow([name,'','','','','','','',''])
        for name in spatial_names: w.writerow([name,'','','','','','','',''])
    print(a.out)
if __name__=='__main__': main()
