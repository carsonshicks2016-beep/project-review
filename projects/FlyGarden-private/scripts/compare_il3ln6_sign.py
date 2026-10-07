"""Frozen full-network two-root sign comparison with two odor exposures."""
import sys,json,argparse,subprocess,fcntl,time,resource,shutil,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json,space_check
from scripts.diagnose_downstream_persistence import DeliveryObserver
OUT=ROOT/'reports/brain-integration/recovery/il3ln6-sign-full-network-v1'
PRIOR=ROOT/'reports/brain-integration/recovery/projection-cut-factorial-v1'
SEEDS=[11701,11702];DT=.025;STEPS=240
CASES=[{'model':m,'cue':c} for m in ('original','sign_only') for c in ('none','a_left','a_right','a_both')]
SOURCES=['scripts/compare_il3ln6_sign.py','flygarden/il3ln6_sign_candidate.py','flygarden/continuous_candidate.py','flygarden/brain.py','flygarden/candidate_inputs.py','flygarden/descending.py','flygarden/neural_probe.py','scripts/diagnose_downstream_persistence.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet']+[str((PRIOR/n).relative_to(ROOT)) for n in ('protocol.json','sign-audit.json','results.json','SIGN_AUDIT.md','literature/taisz-2023.xml')]
def prepare():
    OUT.mkdir(exist_ok=True);hashes={n:file_sha(ROOT/n) for n in SOURCES}
    if (OUT/'protocol.json').exists():
        p=json.loads((OUT/'protocol.json').read_text());assert p['sources']==hashes;return p
    old=json.loads((PRIOR/'protocol.json').read_text());mapping=old['mapping'];ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy();ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('').set_index('root_id').reindex(ids).fillna('')
    indices=np.flatnonzero(ann.cell_type.eq('il3LN6'));roots=sorted(str(ids[i]) for i in indices)
    assert roots==['720575940623636701','720575940632403986'] and ann.iloc[indices].known_nt.eq('gaba').all()
    mapping['il3LN6']=[{'index':int(i),'root_id':str(ids[i]),'side':str(ann.iloc[i].side)} for i in indices]
    selected=sorted(set(old['selected']+indices.tolist()))
    con=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet');pre=con.Presynaptic_Index.to_numpy(dtype=int);post=con.Postsynaptic_Index.to_numpy(dtype=int);signed=con['Excitatory x Connectivity'].to_numpy();rows=np.flatnonzero(np.isin(pre,indices));assert len(rows)==3546 and (signed[rows]>0).all()
    edges=[]
    for row in np.flatnonzero(np.isin(post,selected)):
        i,j=pre[row],post[row];edges.append({'row':int(row),'pre':int(i),'post':int(j),'signed_synapses':float(signed[row]),'pre_root_id':str(ids[i]),'post_root_id':str(ids[j]),'pre_cell_class':str(ann.iloc[i].cell_class),'pre_cell_type':str(ann.iloc[i].cell_type)})
    sign_mapping={'root_ids':roots,'rows':rows.tolist(),'anatomical_synapses':int(con.iloc[rows].Connectivity.sum()),'annotation_sha256':hashes['data/annotations.tsv'],'connectivity_sha256':hashes['vendor/fly-brain/data/2025_Connectivity_783.parquet']}
    p={'version':1,'sources':hashes,'seeds':SEEDS,'cases':CASES,'mapping':mapping,'sign_mapping':sign_mapping,'selected':selected,'selected_local_targets':old['selected_local_targets'],'incoming_edges':edges,'clock':.0001,'window':DT,'duration':6.,'steps':STEPS,
      'pulse_windows':[[12,32],[132,152]],'recovery_windows':[[100,120],[220,240]],
      'input':'65Hz walking support;50Hz odorA at .3-.8s and3.3-3.8s;none,left,right,both antenna conditions;no vision. All matched cases consume identical owned RNG draws.',
      'change':'Only 3546 outgoing signed weights of two exact il3LN6 roots flipped at initialization; every absolute weight and all other neural/motor parameters retained. No learning, lesions, adaptation or rule-based navigator.',
      'recovery_gate':'For every non-none cue in both diagnostic seeds, first and second recovery windows must have each DM1 PN and each of four selected local neurons within5Hz of same-model same-seed support-only control, plus signed DNa02 within5Hz. First pulse meanPN increase>=10Hz over matchednone, second meanPN increase>=10Hz over its preceding recovery mean. Both response and recovery must pass; no promotion on quieting alone.',
      'contrast_gate':'Diagnostic sensory contrast prerequisite only: in each seed and pulse, (PNleft-PNright) for left cue must exceed right cue by at least10Hz, and each unilateral contrast must differ from none with opposite signs. Does not validate body direction or navigation.',
      'instrumentation':'Observer-disabled sign_only a_left11701 exact spike/count/input/motor/v/g/gate parity.',
      'checkpoint':'Both original and sign_only a_left11701 save at3.775s before second pulse ends; fresh-process next5windows exact events/counts/input/motor plus whole v/g <=1e-10mV. Record pending presynaptic delay-horizon spikes.',
      'scope':'Full-network two-seed hypothesis screen; no behavioral confidence intervals, physiological correction claim, navigation or learning validation.'}
    atomic_json(OUT/'protocol.json',p)
    for n in SOURCES:
        if n.startswith(('vendor/','data/','reports/')):continue
        dst=OUT/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    return p

def construct(p,case,seed):
    from flygarden.continuous_candidate import ContinuousCandidate
    from flygarden.il3ln6_sign_candidate import Il3LN6SignCandidate
    return ContinuousCandidate(p['mapping'],seed) if case['model']=='original' else Il3LN6SignCandidate(p['mapping'],p['sign_mapping'],seed)
def advance(cand,case,tick):
    odor=np.zeros((2,2))
    if case['cue']!='none' and (12<=tick<32 or 132<=tick<152):
        if case['cue'] in ('a_left','a_both'):odor[0,0]=1
        if case['cue'] in ('a_right','a_both'):odor[1,0]=1
    return cand.advance(DT,odor,[0,0])
def state(cand):
    import brian2 as b
    return {k:np.asarray(getattr(cand.brain.neurons,k)[:]/b.mV).copy() for k in ('v','g')}
def worker(i,seed,off=False,resume=False):
    import brian2 as b
    from flygarden.neural_probe import NeuralProbe
    p=prepare();case=CASES[i];folder=OUT/'trials'/f"{case['model']}-{case['cue']}-{seed}{'-observer-off' if off else ''}";started=time.monotonic();cand=construct(p,case,seed)
    probe=NeuralProbe(cand.brain,p['selected'],cand.generator);observer=DeliveryObserver(cand,p);observer.syn.active=not off
    if resume:
        cand.load(folder/'checkpoint')
        saved_manifest=json.loads((folder/'manifest.json').read_text());assert hashlib.sha256(np.asarray(cand.brain.synapses.w[:]/b.mV).tobytes()).hexdigest()==saved_manifest['initial_weights_sha256']
        for tick in range(151,156):
            command=advance(cand,case,tick)
            with np.load(folder/f'window-{tick:04d}.npz') as z:
                for k,v in {'counts':cand.last['spike_counts'],'spike_i':cand.last_spikes[0],'spike_t':cand.last_spikes[1],'external_i':cand.last['external_indices'],'external_t':cand.last['external_times'],'motor':command}.items():assert np.array_equal(z[k],v),(tick,k)
        with np.load(folder/'continuation-state.npz') as z:errors={k:float(np.max(np.abs(z[k]-v))) for k,v in state(cand).items()}
        assert all(e<=1e-10 for e in errors.values());atomic_json(folder/'continuation-result.json',{'status':'passed','fresh_process':True,'windows':5,'errors_mV':errors});print('Continuation passed',case['model'],flush=True);probe.close();return
    folder.mkdir(parents=True,exist_ok=False)
    original=np.asarray(cand.brain.synapses.w[:]/b.mV);sha=hashlib.sha256(original.tobytes()).hexdigest();del original
    m={'status':'running','case':case,'seed':seed,'unobserved':off,'controller':cand.manifest(),'sources':p['sources'],'initial_weights_sha256':sha,'chunks':[]};atomic_json(folder/'manifest.json',m);rows=[];last=observer.counters();checkpoint=case['cue']=='a_left' and seed==SEEDS[0] and not off
    try:
        for tick in range(STEPS):
            command=advance(cand,case,tick);data=probe.drain();counter=observer.counters();delivery=counter-last;last=counter
            assert np.array_equal(data['delivered_external_indices'],cand.last['external_indices']);assert np.array_equal(data['delivered_external_times_seconds'],cand.last['external_times'])
            assert np.array_equal(np.bincount(cand.last_spikes[0],minlength=138639),cand.last['spike_counts']);assert np.isfinite(data['voltage_mV']).all() and np.isfinite(data['net_synaptic_conductance_equivalent_mV']).all()
            file=folder/f'window-{tick:04d}.npz';space_check(folder,8*1024**2)
            with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**{k:v for k,v in data.items() if not isinstance(v,str)},counts=cand.last['spike_counts'],spike_i=cand.last_spikes[0],spike_t=cand.last_spikes[1],external_i=cand.last['external_indices'],external_t=cand.last['external_times'],motor=command,gate=np.asarray(observer.gate.not_refractory[:]).copy(),gate_t=np.asarray(observer.gate.t[:]/b.second).copy(),delivered_mV=delivery)
            file.with_suffix('.tmp').replace(file);observer.gate.resize(0)
            m['chunks'].append({'file':file.name,'sha256':file_sha(file),'end':cand.time});rows.append({'end':cand.time,'population_hz':cand.last['population_hz'],'motor':command.tolist()});atomic_json(folder/'rows.json',rows);atomic_json(folder/'manifest.json',m)
            if checkpoint and tick==150:cand.save(folder/'checkpoint')
            if checkpoint and tick==155:
                with (folder/'continuation-state.npz').open('wb') as f:np.savez_compressed(f,**state(cand))
        assert all(np.isfinite(x).all() for x in state(cand).values());final=hashlib.sha256(np.asarray(cand.brain.synapses.w[:]/b.mV).tobytes()).hexdigest();assert sha==final and cand.brain.plasticity.updates==0
        m.update(status='complete',final_weights_sha256=final,wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic_json(folder/'manifest.json',m);print(folder.name,'complete',round(m['wall_seconds'],1),'s',flush=True)
    except BaseException as e:m.update(status='interrupted',error=repr(e));atomic_json(folder/'manifest.json',m);raise
    finally:probe.close()
def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=prepare();done=[]
        jobs=[(i,s,False) for i in range(len(CASES)) for s in SEEDS]+[(5,SEEDS[0],True)]
        for i,seed,off in jobs:
            case=CASES[i];name=f"{case['model']}-{case['cue']}-{seed}{'-observer-off' if off else ''}";folder=OUT/'trials'/name;path=folder/'manifest.json'
            if path.exists():
                m=json.loads(path.read_text());assert m['status']=='complete' and m['sources']==p['sources']
            else:
                space_check(OUT,1200*1024**2);atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':name,'planned':17})
                args=[sys.executable,__file__,'--case',str(i),'--seed',str(seed)]
                if off:args.append('--unobserved')
                subprocess.run(args,cwd=ROOT,check=True)
            if case['cue']=='a_left' and seed==SEEDS[0] and not off and not (folder/'continuation-result.json').exists():subprocess.run([sys.executable,__file__,'--case',str(i),'--seed',str(seed),'--resume'],cwd=ROOT,check=True)
            done.append(name)
        atomic_json(OUT/'progress.json',{'status':'complete','completed':done,'planned':17})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--case',type=int);parser.add_argument('--seed',type=int);parser.add_argument('--unobserved',action='store_true');parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    if a.prepare:prepare()
    elif a.case is not None:worker(a.case,a.seed,a.unobserved,a.resume)
    else:run()
