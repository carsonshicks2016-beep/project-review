"""Full-network vision diagnostic driven exclusively by saved actual eye frames."""
import argparse,json,sys,time,subprocess,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from scripts.audit_brain_reference import sha
from scripts.diagnose_sensorimotor_pathway import mapping
SEEDS=(6101,6102,6199)
CONDITIONS=('blank','static_left','static_right','loom_left','loom_right','moving_pattern','wall','blanked_loom_left','blanked_loom_right')
OUT=ROOT/'reports/brain-integration/stage6-20261006'
SOURCES=('scripts/diagnose_eye_brain.py','scripts/record_eye_stimuli.py','flygarden/vision_encoding.py','flygarden/descending.py','flygarden/brain.py','flygarden/body.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet','reports/brain-integration/stage6-20261006/eye-stimuli/manifest.json')
def populations():
 ids,pops,ann=mapping();order={int(v):i for i,v in enumerate(ids)}
 for cell in ('LPLC2','LC4','DNp01'):
  for side in ('left','right'):
   rows=ann[ann.cell_type.eq(cell)&ann.side.eq(side)];assert len(rows)>0,(cell,side)
   pops[cell+'_'+side]=[{'root_id':str(int(r.root_id)),'index':order[int(r.root_id)],'side':side,'cell_type':cell} for r in rows.itertuples()]
 return ids,pops
def image_rates(cue,tick):
 base=cue.removeprefix('blanked_');rows=json.loads((OUT/'eye-stimuli'/f'{base}.json').read_text())
 if tick==0:return np.zeros(2),[]
 selected=rows[(tick-1)*3:tick*3];assert len(selected)==3 and all(r['time']<tick*.1+1e-10 for r in selected)
 key='blanked_features' if cue.startswith('blanked_') else 'features'
 return np.mean([[r[key][side]['lplc2_hz'] for side in range(2)] for r in selected],axis=0),[r['time'] for r in selected]
def worker(folder,cue,seed):
 import brian2 as b
 from flygarden.brain import FullBrain
 from flygarden.descending import DescendingDecoder
 started=time.perf_counter();brain=FullBrain(seed=seed,learning=False,record_spikes=True);_,pops=populations()
 groups=[np.array([n['index'] for n in pops['LPLC2_'+side]],dtype=np.int32) for side in ('left','right')];targets=np.concatenate(groups);channels=np.concatenate([np.full(len(g),s,dtype=np.int32) for s,g in enumerate(groups)])
 gen=b.SpikeGeneratorGroup(len(targets),np.array([],dtype=np.int32),np.array([])*b.second,clock=brain.clock);external=b.Synapses(gen,brain.neurons,on_pre='v_post += 68.75*mV',clock=brain.clock);external.connect(i=np.arange(len(targets)),j=targets);im=b.SpikeMonitor(gen,record=False);brain.network.add(gen,external,im);brain.inputs.active=False;brain.input.active=False
 brain.neurons.rfc=2.2*b.ms;selected=np.any([image_rates(cue,t)[0]>0 for t in range(30)],axis=0);brain.neurons.rfc[targets[selected[channels]]]=0*b.ms
 rng=np.random.default_rng(seed);previous=np.zeros(brain.n,dtype=np.int64);previous_input=np.zeros(len(targets),dtype=np.int64);decoder=DescendingDecoder();rows=[]
 weight_hash=__import__('hashlib').sha256(np.asarray(brain.synapses.w[:]/b.mV).tobytes()).hexdigest()
 folder.mkdir(parents=True,exist_ok=False);manifest={'status':'running','cue':cue,'seed':seed,'neurons':brain.n,'connections':brain.edges,'learning':False,'input_indices':targets.tolist(),'input_root_ids':[str(brain.ids[i]) for i in targets],'channels':channels.tolist(),'chunks':[],'sources':{s:sha(ROOT/s) for s in SOURCES},'weights_sha256_before':weight_hash,'mapping':pops,'clock_seconds':.0001,'bin_seconds':.1,'scope':'Image-only engineered uniform LPLC2 stimulation; no retinotopy, no direct descending stimulation, no geometric threat proxy, no reflex.'};atomic_json(folder/'manifest.json',manifest)
 for tick in range(30):
  t=tick*.1;rates,stamps=image_rates(cue,tick);samples=rng.random((1000,len(targets)))<rates[channels]*.0001;steps,ii=np.nonzero(samples);gen.set_spikes(ii,(t+steps*.0001)*b.second,sorted=True);brain.network.run(.1*b.second,namespace={})
  counts=np.asarray(brain.monitor.count[:],dtype=np.int64);delta=counts-previous;previous=counts.copy();ic=np.asarray(im.count[:],dtype=np.int64);inputs=ic-previous_input;previous_input=ic.copy();si=np.asarray(brain.monitor.i[:],dtype=np.int32).copy();st=np.asarray(brain.monitor.t[:]/b.second).copy();assert len(si)==delta.sum()
  rates_out={k:float(delta[[n['index'] for n in v]].mean()/.1) if v else None for k,v in pops.items()};motor=decoder.advance(.1,rates_out)
  assert np.isfinite(brain.neurons.v[:]/b.mV).all() and np.isfinite(brain.neurons.g[:]/b.mV).all()
  rows.append({'time':(tick+1)*.1,'image_times':stamps,'input_hz':rates.tolist(),'population_hz':rates_out,'candidate_motor':motor.tolist(),'candidate_turn':float(motor[1]-motor[0]),'total_spikes':int(delta.sum())})
  path=folder/f'window-{tick:04d}.npz';space_check(folder,delta.nbytes+si.nbytes+st.nbytes+1024**2)
  with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,counts=delta,input_counts=inputs,spike_i=si,spike_t=st,requested_hz=rates)
  path.with_suffix('.tmp').replace(path);brain.clear_recorded_spikes();manifest['chunks'].append({'file':path.name,'sha256':sha(path),'spikes':len(si)});atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'bins.json',rows)
 after=__import__('hashlib').sha256(np.asarray(brain.synapses.w[:]/b.mV).tobytes()).hexdigest();assert after==weight_hash and brain.plasticity.updates==0
 manifest.update(status='complete',weights_sha256_after=after,wall_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic_json(folder/'manifest.json',manifest)
def run(root):
 root.mkdir(parents=True,exist_ok=True);protocol=root/'protocol.json'
 if not protocol.exists():
  _,pops=populations();atomic_json(protocol,{'schema_version':1,'seeds':SEEDS,'held_out_seed':6199,'conditions':CONDITIONS,'sources':{s:sha(ROOT/s) for s in SOURCES},'mapping':pops,'sources_supporting_mapping':['https://www.nature.com/articles/nature24626','https://www.virtualflybrain.org/term/dnp01-vfb_fw036981/'],'protocol':'3s continuous state, 100us neural clock, preceding completed 100ms of actual eye frames averaged into next neural window; no future frame access. One worker; full network; all weights fixed; no odor, direct walking input, reward or escape override.','gates':{'neural':'For both looming sides on all seeds, downstream DNp01, DNa02 or DNp09 must exceed matched blank by5Hz during0.6-1.6s. Direct LPLC2 stimulation alone is insufficient.','motor':'For both looming sides on all seeds, RMS candidate motor difference from blank during0.6-1.6s must exceed0.02. Eye blanking must remove stimulus-driven neural and motor response.','direction':'Mirrored looming must produce opposite-signed mean turning, absolute >0.02, on all seeds; no decoder tuning.','promotion':'All gates and embodied validation required; partial vision interface remains experimental. Static/translation false positives explicitly reported.'}})
 p=json.loads(protocol.read_text());assert all(sha(ROOT/s)==v for s,v in p['sources'].items());progress={'status':'running','planned':27,'completed':[]};start=time.perf_counter()
 try:
  for cue in CONDITIONS:
   for seed in SEEDS:
    folder=root/'trials'/f'{cue}-{seed}'
    if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text())['status']=='complete':progress['completed'].append(folder.name);continue
    if folder.exists():raise RuntimeError('Preserve incomplete trial before retry '+str(folder))
    space_check(root,64*1024**2);progress['current']={'cue':cue,'seed':seed};atomic_json(root/'progress.json',progress);log=root/'logs'/f'{cue}-{seed}.log';log.parent.mkdir(exist_ok=True)
    with log.open('w') as f:subprocess.run([sys.executable,str(Path(__file__).resolve()),str(folder),'--cue',cue,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    progress['completed'].append(folder.name);progress['wall_seconds']=time.perf_counter()-start;atomic_json(root/'progress.json',progress);print(f"{len(progress['completed'])}/27 {folder.name}",flush=True)
  progress['status']='complete';atomic_json(root/'progress.json',progress)
 except Exception as e:progress.update(status='interrupted',error=repr(e));atomic_json(root/'progress.json',progress);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--cue',choices=CONDITIONS);p.add_argument('--seed',type=int);a=p.parse_args()
 if a.cue:worker(a.folder,a.cue,a.seed)
 else:run(a.folder)
