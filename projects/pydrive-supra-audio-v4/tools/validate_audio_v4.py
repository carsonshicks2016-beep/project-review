#!/usr/bin/env python3
"""Audio V5 procedural/mechanical/spatial/runtime acceptance gates."""
from __future__ import annotations
import json,time,tracemalloc
from pathlib import Path
import numpy as np
import supra.audio_v4 as av
from supra.audio_v4 import *

def req(ok,msg):
    if not ok: raise AssertionError(msg)
def rms(x): return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
def peak_hz(x,sr):
    y=np.abs(np.fft.rfft(x*np.hanning(len(x)))); return float(np.fft.rfftfreq(len(x),1/sr)[np.argmax(y)])
def main():
    result={"schema":SCHEMA_VERSION,"identity":"mazda787b-procedural-v1","checks":{}}
    req(SCHEMA_VERSION==5 and len(RATIOS_787B)==5,"contract/drivetrain")
    f=AudioFrame(); req(len(av._encode(f))==CAR_STRIDE and av._decode(av._encode(f)).wheel_surface==(0,0,0,0),"schema roundtrip")
    for sr in (44100,48000,96000):
      s=EngineSynthV4(sr)
      last=0
      for rpm in (1000,3000,6000,7800,9000):
       d=s.synthesize_stems(AudioFrame(rpm=rpm,throttle=.8),4093)
       req(all(np.isfinite(x).all() and len(x)==4093 for x in d.values()),f"finite {sr}")
       hz=peak_hz(d['exhaust'],sr); req(hz>last,f"pitch not monotonic {sr}/{rpm}"); last=hz
    result['checks']['rates_blocks_harmonics']='pass'
    s=EngineSynthV4(48000); base=AudioFrame(rpm=5000,gear=2,ratio=RATIOS_787B[1]); s.reset(base)
    req(rms(s.synthesize_stems(base,2048)['chassis'])<1e-8,'false reset transient')
    matched=AudioFrame(**{**vars(base),'clutch':.3,'shift_phase':.8,'shift_mismatch':.03,'events':AudioEvents(shift=1)})
    poor=AudioFrame(**{**vars(matched),'shift_mismatch':.9,'events':AudioEvents(shift=2)})
    a=s.synthesize_stems(matched,4096); b=s.synthesize_stems(poor,4096)
    req(rms(b['transmission'])>rms(a['transmission']),'mismatch chatter')
    result['checks']['mechanical_shift_no_flare']='pass'
    wheel=[]
    for w in range(4):
      slips=[0.0]*4; slips[w]=.9; q=AudioFrame(rpm=4500,speed=45,wheel_slip=tuple(slips),wheel_surface=(w%6,)*4)
      d=EngineSynthV4(48000,7).synthesize_stems(q,4096); levels=[rms(d[k]) for k in ('tyre_fl','tyre_fr','tyre_rl','tyre_rr')]
      req(int(np.argmax(levels))==w,'wheel isolation'); wheel.append(levels)
    hot=EngineSynthV4(48000).synthesize_stems(AudioFrame(speed=60,brake=.9,brake_temperature=(.9,)*4),4096)
    req(rms(hot['brakes'])>.001,'hot brake')
    result['checks']['per_wheel_surface_brake']='pass'
    lim=EngineSynthV4(48000); below=lim.synthesize_stems(AudioFrame(rpm=8999,throttle=1),4096)['exhaust']; at=lim.synthesize_stems(AudioFrame(rpm=9000,throttle=1,limiter=1),4096)['exhaust']
    req(np.count_nonzero(np.isclose(at,0,atol=1e-7))>np.count_nonzero(np.isclose(below,0,atol=1e-7)),'limiter synchronization')
    result['checks']['limiter_9000']='pass'
    sr=48000; d=FractionalDelay(sr); impulse=np.zeros(sr,np.float32); impulse[0]=1; arrival=int(np.argmax(np.abs(d.process(impulse,100))))
    req(abs(arrival-round(sr*100/SPEED_OF_SOUND))<=2,'100m delay'); result['checks']['propagation_ms']=round(arrival/sr*1000,3)
    tone=np.sin(2*np.pi*400*np.arange(sr)/sr).astype(np.float32)
    ap=FractionalDelay(sr); ap.process(tone,120); aa=ap.process(tone,80)[5000:30000]
    re=FractionalDelay(sr); re.process(tone,80); rr=re.process(tone,120)[5000:30000]
    req(peak_hz(aa,sr)>400>peak_hz(rr,sr),'delay Doppler sign')
    fallback=RadialDoppler(); req(peak_hz(fallback.process(tone,40),sr)<400,'fallback recede')
    result['checks']['doppler_delay_and_fallback']='pass'
    # Physical exhaust directivity after delay history has filled.
    car=AudioFrame(x=0,y=0,yaw=0,rpm=7000,throttle=1)
    front=SpatialRendererV4(sr,1); rear=SpatialRendererV4(sr,1)
    for _ in range(40): of=front.render([car],(15,0,0,0,0),512,perspective=0); ob=rear.render([car],(-15,0,0,0,0),512,perspective=0)
    req(rms(ob)>rms(of),'rear exhaust directivity')
    cut=front.render([car],(0,20,0,0,0),512,perspective=0,cut=1); req(np.max(np.abs(np.diff(cut[:,0])))<.3,'camera cut discontinuity')
    result['checks']['source_directivity_and_cut']='pass'
    perf={}
    for sr in (44100,48000,96000):
      for cars in (1,8,32):
       r=SpatialRendererV4(sr,cars,propagation=True); fs=[AudioFrame(x=5+i*.5,y=2,rpm=7000,throttle=.8,wheel_slip=(.2,)*4) for i in range(cars)]; times=[]
       for _ in range(8):
        t=time.perf_counter(); out=r.render(fs,(0,0,0,0,0),512); times.append(time.perf_counter()-t)
        req(np.isfinite(out).all() and np.max(np.abs(out))<=.981,'output safety')
       p99=float(np.percentile(times[2:],99)); frac=p99/(512/sr); perf[f'{sr}/{cars}']={'p99_ms':round(p99*1000,3),'deadline_fraction':round(frac,3)}
       req(frac<(.25 if cars==1 else .70),f'callback budget {sr}/{cars}: {frac}')
    result['runtime']=perf
    tracemalloc.start(); r=SpatialRendererV4(48000,1); fs=[car]
    for _ in range(10): r.render(fs,(0,0,0,0,0),512)
    before=tracemalloc.take_snapshot()
    for _ in range(30): r.render(fs,(0,0,0,0,0),512)
    after=tracemalloc.take_snapshot(); growth=sum(max(0,x.size_diff) for x in after.compare_to(before,'lineno'))
    tracemalloc.stop(); req(growth<65536,f'steady allocation growth {growth}')
    result['checks']['steady_heap_growth_bytes']=growth
    profile=Path(__file__).parents[1]/'supra/data/mazda787b_procedural_identity_v1.json'; p=json.loads(profile.read_text())
    req(p['profile_version']==result['identity'] and not p['recording_derived'],'identity profile')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
