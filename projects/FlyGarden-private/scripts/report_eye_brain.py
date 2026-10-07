"""Verify actual eye-derived recordings, neural response, motor and body gates."""
import json,sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from flygarden.vision_encoding import EyeExpansion
from scripts.audit_brain_reference import sha
from scripts.diagnose_eye_brain import SEEDS,CONDITIONS,image_rates,populations,SOURCES,OUT

def report():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 protocol=json.loads((OUT/'protocol.json').read_text());assert all(sha(ROOT/s)==v for s,v in protocol['sources'].items())
 em=json.loads((OUT/'eye-stimuli/manifest.json').read_text());assert em['orientation_verified']
 for name,h in em['files'].items():assert sha(OUT/'eye-stimuli'/name)==h,name
 for cue in em['cues']:
  with np.load(OUT/'eye-stimuli'/f'{cue}.npz') as z:
   encoder=EyeExpansion(z['baseline']);rows=json.loads((OUT/'eye-stimuli'/f'{cue}.json').read_text());assert len(rows)==90
   for i,frame in enumerate(z['frames']):
    measured=encoder.advance(frame,1/30)
    for side in range(2):
     for key,value in measured[side].items():assert np.isclose(value,rows[i]['features'][side][key]),(cue,i,side,key)
 rows_by_case={};spikes=0;rss=0;wall=0
 for cue in CONDITIONS:
  for seed in SEEDS:
   folder=OUT/'trials'/f'{cue}-{seed}';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and m['neurons']==138639 and m['connections']==15091983 and not m['learning'];assert m['weights_sha256_before']==m['weights_sha256_after'];assert m['sources']==protocol['sources'] and len(m['chunks'])==30
   rows=json.loads((folder/'bins.json').read_text());rows_by_case[(cue,seed)]=rows
   for tick,c in enumerate(m['chunks']):
    path=folder/c['file'];assert sha(path)==c['sha256'];expected,stamps=image_rates(cue,tick)
    assert rows[tick]['image_times']==stamps and all(t<tick*.1+1e-9 for t in stamps)
    with np.load(path) as z:
     assert np.array_equal(z['requested_hz'],expected);assert np.array_equal(np.bincount(z['spike_i'],minlength=138639),z['counts']);assert len(z['spike_i'])==c['spikes'];assert np.all(z['spike_t']>=tick*.1-1e-9) and np.all(z['spike_t']<(tick+1)*.1+1e-9)
     channels=np.array(m['channels']);assert not z['input_counts'][expected[channels]==0].any();spikes+=len(z['spike_i'])
    assert np.isfinite(rows[tick]['candidate_motor']).all() and np.all(np.array(rows[tick]['candidate_motor'])>=0) and np.all(np.array(rows[tick]['candidate_motor'])<=1.2)
   rss=max(rss,m['peak_rss_bytes']);wall+=m['wall_seconds']
  if cue.startswith('blanked_'):
   for seed in SEEDS:
    blank=OUT/'trials'/f'blank-{seed}';test=OUT/'trials'/f'{cue}-{seed}'
    for tick in range(30):
     with np.load(blank/f'window-{tick:04d}.npz') as a,np.load(test/f'window-{tick:04d}.npz') as b:
      for k in a.files:assert np.array_equal(a[k],b[k]),(cue,seed,k)
    assert all(r['candidate_motor']==[0.,0.] for r in rows_by_case[(cue,seed)])
 repeat=OUT/'repeat'/'loom_left-6199';rm=json.loads((repeat/'manifest.json').read_text());assert rm['status']=='complete'
 for c in rm['chunks']:
  assert sha(repeat/c['file'])==c['sha256']
  with np.load(repeat/c['file']) as a,np.load(OUT/'trials'/'loom_left-6199'/c['file']) as b:
   for key in a.files:assert np.array_equal(a[key],b[key])
 gates=[]
 for seed in SEEDS:
  for side in ('left','right'):
   rows=rows_by_case[('loom_'+side,seed)][6:16];blank=rows_by_case[('blank',seed)][6:16]
   downstream={k:float(np.mean([r['population_hz'][k]-b['population_hz'][k] for r,b in zip(rows,blank)])) for k in ('DNp01_left','DNp01_right','DNp09_left','DNp09_right','DNa02_left','DNa02_right')}
   rms=float(np.sqrt(np.mean((np.array([r['candidate_motor'] for r in rows])-np.array([r['candidate_motor'] for r in blank]))**2)));turn=float(np.mean([r['candidate_turn'] for r in rows]));gates.append({'seed':seed,'side':side,'downstream_mean_hz':downstream,'neural_passed':max(downstream.values())>5,'motor_rms_difference':rms,'motor_passed':rms>.02,'mean_turn':turn})
 mirrored_signs=all(next(g['mean_turn'] for g in gates if g['seed']==seed and g['side']=='left')*next(g['mean_turn'] for g in gates if g['seed']==seed and g['side']=='right')<0 for seed in SEEDS)
 direction=all(abs(g['mean_turn'])>.02 for g in gates) and mirrored_signs
 body=json.loads((OUT/'body-replay/summary.json').read_text());assert body['status']=='complete' and all(r['finite'] and r['frames']==90 for r in body['results'])
 recovery={f'{cue}-{seed}':sum(r['total_spikes'] for r in rows_by_case[(cue,seed)][25:30]) for cue in CONDITIONS for seed in SEEDS}
 results={'final_half_second_spikes':recovery,'status':'complete','trials':27,'chunks':810,'simulated_seconds':81,'raw_spikes':spikes,'eye_orientation_passed':True,'raw_image_feature_reconstruction_passed':True,'future_frame_access':False,'blanking_exact_raw_parity':True,'repeat_exact_raw_parity':True,'weight_hash_parity':True,'neural_gate_passed':all(g['neural_passed'] for g in gates),'motor_gate_passed':all(g['motor_passed'] for g in gates),'mirrored_turn_signs_passed':bool(mirrored_signs),'direction_gate_passed':bool(direction),'gates':gates,'body_replay':body,'peak_rss_bytes':rss,'worker_wall_seconds':wall}
 results['integration_gate_passed']=results['neural_gate_passed'] and results['motor_gate_passed'] and results['direction_gate_passed'];atomic_json(OUT/'results.json',results)
 fig,ax=plt.subplots(4,1,figsize=(11,10),sharex=True)
 for cue,col in [('blank','#777777'),('loom_left','#2675c9'),('loom_right','#d77922')]:
  rows=rows_by_case[(cue,6199)];t=[r['time'] for r in rows]
  ax[0].plot(t,[r['input_hz'][0]-r['input_hz'][1] for r in rows],label=cue,color=col)
  ax[1].plot(t,[r['population_hz']['DNp01_left']+r['population_hz']['DNp01_right'] for r in rows],color=col)
  ax[2].plot(t,[r['population_hz']['DNa02_left']-r['population_hz']['DNa02_right'] for r in rows],color=col)
  ax[3].plot(t,[r['candidate_turn'] for r in rows],color=col)
 for a in ax:a.axvspan(.5,1.5,color='#dddddd',alpha=.4);a.grid(alpha=.2)
 ax[0].set(ylabel='LPLC2 input L−R (Hz)',title='Actual eye-image input: held-out seed 6199');ax[0].legend();ax[1].set(ylabel='DNp01 L+R (Hz)');ax[2].set(ylabel='DNa02 L−R (Hz)');ax[3].set(ylabel='Candidate turn command',xlabel='Simulated seconds');fig.tight_layout();fig.savefig(OUT/'vision-response.png',dpi=150);plt.close(fig)
 # A representative normal-time movie shows actual eye frames and measured activity.
 import imageio.v2 as imageio
 from PIL import Image,ImageDraw
 with np.load(OUT/'eye-stimuli/loom_left.npz') as z:
  temp=OUT/'vision-demo.tmp.mp4';space_check(OUT,32*1024**2)
  with imageio.get_writer(temp,fps=30,codec='libx264',macro_block_size=1,ffmpeg_log_level='error') as video:
   rows=rows_by_case[('loom_left',6199)]
   for tick,frames in enumerate(z['frames']):
    t=tick/30;canvas=Image.new('RGB',(1280,720),'#10161e');draw=ImageDraw.Draw(canvas)
    draw.text((20,15),f'Actual fly eye images - loom left | simulated time {t:.2f}s',fill='white',font_size=23)
    for side in range(2):canvas.paste(Image.fromarray(frames[side]).resize((360,410)),(20+side*370,65));draw.text((20+side*370,485),('Left eye','Right eye')[side],fill='white',font_size=20)
    completed=[r for r in rows if r['time']<=t+1e-9];r=completed[-1] if completed else None
    draw.text((790,65),'Measured neuron population rates',fill='white',font_size=20)
    for i,key in enumerate(('LPLC2_left','LPLC2_right','DNp01_left','DNp01_right','DNa02_left','DNa02_right','DNp09_left','DNp09_right')):
     value=r['population_hz'][key] if r else 0;y=110+i*55;draw.text((790,y),f'{key}: {value:.1f} Hz',fill='white',font_size=18);draw.rectangle((790,y+25,790+min(400,value*2),y+37),fill='#388bd0')
    motor=r['candidate_motor'] if r else [0,0];draw.text((20,540),f'Candidate gait command: {motor[0]:.3f}, {motor[1]:.3f}',fill='white',font_size=22)
    draw.text((20,595),'Experimental image-area expansion encoding; retina/T4/T5 stages bypassed.',fill='#ccd0d7',font_size=19)
    draw.text((20,630),'Activity shown for latest complete 100ms window. Stationary sensory bench; learning frozen.',fill='#ccd0d7',font_size=18)
    video.append_data(np.asarray(canvas))
  temp.replace(OUT/'vision-demo.mp4')
 lines=['# Step 6 — actual-eye vision integration diagnostic','',f"Actual eye acquisition and an image-only neural adapter are implemented. The full integration gate {'passed' if results['integration_gate_passed'] else 'failed'}; this profile remains experimental.",'','## Input and mapping','', 'Verified left/right eye camera orientation using world camera transforms and visible static objects. Inputs are actual fisheye-corrected FlyGym RGB images: two 512×450 eyes. A moving stripe panel, stationary objects, unilateral approaching objects and an approaching wall were rendered in an initialized stationary body pose.','', 'The encoder accepts images and elapsed time only. It measures brightness, absolute image change and positive growth of a dark connected component relative to the initial view. A fixed, bounded engineered gain maps area growth to 0–100 Hz external events. Left/right rates drive all 108 left and 102 right annotated LPLC2 neurons by exact root ID. This bypasses photoreceptor, lamina, medulla and T4/T5 processing; no cell-specific receptive fields or retinotopy are inferred. Brightness and image-change signals are diagnostic readouts, not independent stimulation channels.','', 'The imported table contains 189 direct LPLC2-to-DNp01 connection records representing 1,080 anatomical synapses; exact endpoints and signed counts are retained in lplc2-dnp01-connections.json. These counts do not establish physiological efficacy.','', 'The population choice is supported by [LPLC2 looming research](https://www.nature.com/articles/nature24626). DNp01 is monitored as an annotated downstream population, consistent with its [Giant Fiber classification](https://www.virtualflybrain.org/term/dnp01-vfb_fw036981/); it is not converted into an invented walking escape command. The [published visual-system model](https://www.nature.com/articles/s41586-024-07939-3) required substantial physiological calibration; this adapter does not reproduce that model.','', 'All 138,639 neurons and 15,091,983 connection records remain present. Synaptic weight hashes match before and after every trial. No odor input, tonic walking stimulation, direct descending stimulation, learning, geometric threat proxy or escape reflex is supplied. Stimulated input cells use the reference zero-refractory convention; other cells use 2.2 ms.','', 'Image frames are averaged only after their 100ms acquisition window ends, then delivered in the next neural window. Every source frame precedes its neural window; this introduces an explicit sampling delay without access to future frames.','', '## Results','',f"27 primary full-network trials across seeds 6101, 6102 and held-out 6199: 81 simulated seconds, 810 chunks and {spikes:,} raw spikes. A separate held-out repeat adds three seconds and 30 chunks.",'',f"Downstream neural gate: **{'PASS' if results['neural_gate_passed'] else 'FAIL'}**. Candidate walking-command gate: **{'PASS' if results['motor_gate_passed'] else 'FAIL'}**. Mirrored turning gate: **{'PASS' if direction else 'FAIL'}**.",'', '| Seed | Cue | Highest downstream mean Hz | Motor RMS change | Mean turn |','|---|---|---:|---:|---:|']
 for g in gates:lines.append(f"| {g['seed']} | {g['side']} | {max(g['downstream_mean_hz'].values()):.1f} | {g['motor_rms_difference']:.4f} | {g['mean_turn']:.4f} |")
 lines+=['','![Measured neural response](vision-response.png)','','[Watch the synchronized actual-eye demonstration](vision-demo.mp4)','','## Controls, limitations and body replay','', 'Blanking the eye images removes all encoded stimulation and yields exact raw spike/input/count parity with blank-scene trials on every seed, with zero candidate commands. Every raw spike histogram matches the monitor, all chunk hashes and time intervals pass, and all rates reconstruct from the saved eye features. Features themselves reconstruct from the saved RGB images. Repeating the held-out left loom reproduces every raw spike ID/time, count and input event exactly. Eight focused encoder, timing, mapping and decoder tests passed.','', f"Final-half-second neural recovery: {sum(v==0 for v in recovery.values())}/{len(recovery)} primary trials contain zero spikes in the final 500 ms. Background drive is absent in this isolated vision test. Motor smoothing can outlast neural activity and is reported separately.",'', f"Mirrored turn signs are {'consistent on all three seeds' if mirrored_signs else 'not consistent across seeds'}, but the committed direction gate also requires mean magnitude above 0.02 in the response window. The magnitude gate is not relaxed after seeing the results. Bilateral DNp09 remains silent during the looming response, so this diagnostic supplies no forward walking drive.",'', 'The static-object onset also produces an expansion pulse, and a translating pattern can produce false positives. This is an engineered dark-area growth proxy, not validated LPLC2 selectivity or optical flow. Uniform population stimulation discards retinal location within each eye. Head-fixed sensory recordings isolate the pathway; body replay does not close the sensory loop.','', '| Replay | Heading change (rad) | Travel (mm) | Falls |','|---|---:|---:|---:|']
 for r in body['results']:lines.append(f"| {r['cue']} | {r['heading_change_radians']:.6f} | {r['travel_mm']:.6f} | {r['flipped_samples']} |")
 lines+=['',body['scope'],'', 'All replay poses are finite and each run retains 90 body frames. Small passive movement is not counted as a successful response.','','## Runtime and reproducibility','',f"One neural worker at a time. Primary worker time {wall:.1f}s; maximum worker memory {rss/1024**3:.2f} GiB; throughput {81/wall:.3f} simulated seconds per wall second including startup. Two-GiB free-space reserve checks remain active; no recordings/checkpoints were deleted.",'', 'Protocol, root mapping, stimulus hashes, source snapshots, raw chunks, test logs and numeric results are retained beside this report. Rerender and verify existing results with `.venv-next/bin/python scripts/report_eye_brain.py`. Acquisition and body replay refuse existing output folders. For an independent rerun, use a separate project copy with stage6 output absent, then run record_eye_stimuli.py, diagnose_eye_brain.py, the held-out repeat with --cue loom_left --seed 6199 in repeat/loom_left-6199, replay_eye_motor.py and report_eye_brain.py. The neural batch reuses complete trials only under unchanged source hashes; incomplete trials are preserved.','', 'Production controller and application settings remain unchanged. '+('A separate closed-loop embodied validation is still necessary before any arena promotion.' if results['integration_gate_passed'] else 'Actual visual input now reaches measured neural circuitry, but the failed integration gates block treating it as reliable visually controlled walking. Further work must address the visual model and appropriate motor pathways rather than adding a hidden escape navigator.')]
 (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');snapshot=OUT/'source-snapshot'
 for source in (*SOURCES,'scripts/replay_eye_motor.py','scripts/report_eye_brain.py','tests/test_vision_encoding.py','tests/test_eye_brain.py'):
  dest=snapshot/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,dest)
 atomic_json(OUT/'runtime.json',{'python':sys.version,'numpy':np.__version__,'brian2':__import__('brian2').__version__,'report_sha256':sha(Path(__file__)),'video_sha256':sha(OUT/'vision-demo.mp4')})
 print('Verified',spikes,'spikes; neural/motor/direction gates:',results['neural_gate_passed'],results['motor_gate_passed'],results['direction_gate_passed'])
if __name__=='__main__':report()
