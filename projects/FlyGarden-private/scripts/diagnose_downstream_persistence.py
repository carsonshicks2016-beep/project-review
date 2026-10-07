"""Preregistered late interventions; model diagnostics, never a brain repair."""
import sys,json,argparse,subprocess,fcntl,time,resource,shutil
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from flygarden.continuous_candidate import file_sha
from scripts.diagnose_recovery_interfaces import mappings
OUT=ROOT/'reports/brain-integration/recovery/downstream-persistence-v1'
SEEDS=[11401,11402]
CASES=['intact','cognate_orn_to_pn_off','positive_pn_output_off','positive_alln_to_pn_off','positive_alln_recurrence_off','all_positive_pn_input_off']
SOURCES=['scripts/diagnose_downstream_persistence.py','flygarden/brain.py','flygarden/continuous_candidate.py','flygarden/candidate_inputs.py','flygarden/descending.py','flygarden/neural_probe.py','scripts/diagnose_recovery_interfaces.py','scripts/diagnose_sensorimotor_pathway.py','scripts/diagnose_causal_timing.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet']


def prepare():
    OUT.mkdir(exist_ok=True)
    hashes={n:file_sha(ROOT/n) for n in SOURCES}
    if (OUT/'protocol.json').exists():
        p=json.loads((OUT/'protocol.json').read_text());assert p['sources']==hashes;return p
    ids,mapping=mappings()
    ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('').set_index('root_id').reindex(ids).fillna('')
    pn=np.flatnonzero(ann.cell_type.eq('DM1_lPN').to_numpy())
    orn=np.flatnonzero(ann.cell_type.eq('ORN_DM1').to_numpy())
    alln=np.flatnonzero(ann.cell_class.eq('ALLN').to_numpy())
    mapping['ALLN']=[{'index':int(i),'root_id':str(ids[i]),'side':str(ann.iloc[i]['side'])} for i in alln]
    selected=sorted(set(pn.tolist()+[n['index'] for k in ('DNa02_left','DNa02_right') for n in mapping[k]]))
    con=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet')
    pre=con.Presynaptic_Index.to_numpy(dtype=int);post=con.Postsynaptic_Index.to_numpy(dtype=int)
    pos=con['Excitatory x Connectivity'].to_numpy()>0
    pre_pn=np.isin(pre,pn);post_pn=np.isin(post,pn);pre_alln=np.isin(pre,alln);post_alln=np.isin(post,alln)
    masks=[np.zeros(len(con),dtype=bool),np.isin(pre,orn)&post_pn,pos&pre_pn,pos&pre_alln&post_pn,pos&pre_alln&post_alln,pos&post_pn]
    lesions={}
    for case,mask in zip(CASES,masks):
        rows=np.flatnonzero(mask)
        lesions[case]={'rows':rows.tolist(),'aggregated_records':len(rows),'anatomical_synapses':int(con.iloc[rows].Connectivity.sum()),
                       'source_indices':np.unique(pre[rows]).tolist(),'target_indices':np.unique(post[rows]).tolist()}
    edges=[]
    for row in np.flatnonzero(np.isin(post,selected)):
        i,j=pre[row],post[row]
        edges.append({'row':int(row),'pre':int(i),'post':int(j),'signed_synapses':float(con.iloc[row]['Excitatory x Connectivity']),
                      'pre_root_id':str(ids[i]),'post_root_id':str(ids[j]),'pre_cell_type':str(ann.iloc[i]['cell_type']),
                      'pre_cell_class':str(ann.iloc[i]['cell_class']),'pre_side':str(ann.iloc[i]['side'])})
    p={'version':1,'seeds':SEEDS,'cases':CASES,'sources':hashes,'mapping':mapping,'selected':selected,'incoming_edges':edges,'lesions':lesions,
       'clock':.0001,'window':.025,'duration':4.,'intervention_time':.8,
       'schedule':'Original model only;65Hz DNp09 support throughout;50Hz left odorA .3-.8s. At.8s zero ONLY selected transport weights, including pending arrivals; preserve v/g, inhibition, inputs, topology and decoder.',
       'observations':'All spikes/counts/input events;selected v/g at1ms;selected before-synapse refractory gates at100us;positive/negative accepted voltage-increment counters. Counters are not biological currents.',
       'annotation':'ALLN exact cell_class annotation;positive means positive signed modeled weight, not established biological excitation. Missing annotations are not assigned to ALLN.',
       'decision':'Two-seed causal localization, not population validation or a corrective controller. Report PN and DNa02 final.5s rates and intact-matched differences; strong suppression requires <=10% intact PN rate AND >=90% PN reduction in BOTH seeds. Report DNa02 separately. Odor-afferent cut is a comparator, not assumed inert. No navigation promotion.',
       'instrumentation':'An additional intact11401 replay disables the delivery observer; spikes,counts,inputs and commands must match the observed intact recording exactly.',
       'scope':'Full network; deliberately lesioned diagnostic variants, not preserved-anatomy behavior or a candidate repair.'}
    atomic_json(OUT/'protocol.json',p)
    for n in SOURCES:
        if n.startswith(('vendor/','data/')):continue
        dst=OUT/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    return p


class DeliveryObserver:
    def __init__(self,cand,p):
        import brian2 as b
        self.selected=np.array(p['selected'],dtype=int);self.edges=p['incoming_edges']
        for name in ('dp_positive','dp_negative'):
            cand.brain.neurons.variables.add_array(name,size=cand.brain.n,dimensions=b.volt.dim)
        self.syn=b.Synapses(cand.brain.neurons,cand.brain.neurons,'weight:volt',on_pre='''
            dp_positive_post += clip(weight,0*mV,inf*mV)*int(not_refractory_post)
            dp_negative_post += clip(-weight,0*mV,inf*mV)*int(not_refractory_post)''',delay=1.8*b.ms,clock=cand.brain.clock)
        self.syn.connect(i=[e['pre'] for e in self.edges],j=[e['post'] for e in self.edges])
        self.syn.weight=cand.brain.synapses.w[[e['row'] for e in self.edges]]
        self.gate=b.StateMonitor(cand.brain.neurons,'not_refractory',record=self.selected,when='before_synapses',clock=cand.brain.clock)
        cand.brain.network.add(self.syn,self.gate);self.cand=cand

    def counters(self):
        import brian2 as b
        return np.array([np.asarray(getattr(self.cand.brain.neurons,n)[self.selected]/b.mV) for n in ('dp_positive','dp_negative')])

    def update(self):
        self.syn.weight=self.cand.brain.synapses.w[[e['row'] for e in self.edges]]


def worker(case,seed,unobserved=False):
    import brian2 as b
    from flygarden.continuous_candidate import ContinuousCandidate
    from flygarden.neural_probe import NeuralProbe
    p=prepare();folder=OUT/'trials'/f"{case}-{seed}{'-observer-off' if unobserved else ''}";folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cand=ContinuousCandidate(p['mapping'],seed)
    probe=NeuralProbe(cand.brain,p['selected'],cand.generator);observer=DeliveryObserver(cand,p)
    observer.syn.active=not unobserved
    before=np.asarray(cand.brain.synapses.w[:]/b.mV).copy()
    initial_sha=__import__('hashlib').sha256(before.tobytes()).hexdigest()
    rows=np.array(p['lesions'][case]['rows'],dtype=int)
    expected=before.copy();expected[rows]=0
    expected_sha=__import__('hashlib').sha256(expected.tobytes()).hexdigest();del expected,before
    m={'status':'running','case':case,'seed':seed,'unobserved':unobserved,'controller':cand.manifest(),
       'sources':p['sources'],'initial_weights_sha256':initial_sha,'expected_lesioned_weights_sha256':expected_sha,'chunks':[]}
    atomic_json(folder/'manifest.json',m);summaries=[];last=observer.counters()
    try:
        for tick in range(160):
            if tick==32:
                cand.brain.synapses.w[rows]=0*b.mV;observer.update()
                actual=__import__('hashlib').sha256(np.asarray(cand.brain.synapses.w[:]/b.mV).tobytes()).hexdigest()
                assert actual==expected_sha;m['actual_lesioned_weights_sha256']=actual
            odor=np.zeros((2,2))
            if 12<=tick<32:odor[0,0]=1
            motor=cand.advance(.025,odor,[0,0]);data=probe.drain()
            assert np.array_equal(data['delivered_external_indices'],cand.last['external_indices'])
            counters=observer.counters();delivered=counters-last;last=counters
            assert np.isfinite(data['voltage_mV']).all() and np.isfinite(data['net_synaptic_conductance_equivalent_mV']).all()
            file=folder/f'window-{tick:04d}.npz';space_check(folder,8*1024**2)
            arrays={k:v for k,v in data.items() if not isinstance(v,str)}
            with file.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f,**arrays,counts=cand.last['spike_counts'],spike_i=cand.last_spikes[0],spike_t=cand.last_spikes[1],external_i=cand.last['external_indices'],external_t=cand.last['external_times'],motor=motor,
                                    gate=np.asarray(observer.gate.not_refractory[:]).copy(),gate_t=np.asarray(observer.gate.t[:]/b.second).copy(),delivered_mV=delivered)
            file.with_suffix('.tmp').replace(file);observer.gate.resize(0)
            m['chunks'].append({'file':file.name,'sha256':file_sha(file),'end':cand.time})
            summaries.append({'end':cand.time,'population_hz':cand.last['population_hz'],'motor':motor.tolist()})
            atomic_json(folder/'manifest.json',m);atomic_json(folder/'rows.json',summaries)
        final=__import__('hashlib').sha256(np.asarray(cand.brain.synapses.w[:]/b.mV).tobytes()).hexdigest()
        assert final==expected_sha and cand.brain.plasticity.updates==0
        m.update(status='complete',final_weights_sha256=final,wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(folder/'manifest.json',m);print(folder.name,'complete',round(m['wall_seconds'],1),'s',flush=True)
    except BaseException as e:
        m.update(status='interrupted',error=repr(e));atomic_json(folder/'manifest.json',m);raise
    finally:probe.close()


def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=prepare();done=[]
        jobs=[(c,s,False) for c in CASES for s in SEEDS]+[('intact',SEEDS[0],True)]
        for case,seed,unobserved in jobs:
            name=f"{case}-{seed}{'-observer-off' if unobserved else ''}";path=OUT/'trials'/name/'manifest.json'
            if path.exists():
                m=json.loads(path.read_text());assert m['status']=='complete' and m['sources']==p['sources']
            else:
                space_check(OUT,200*1024**2);atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':name,'planned':13})
                argv=[sys.executable,__file__,'--case',case,'--seed',str(seed)]
                if unobserved:argv.append('--unobserved')
                subprocess.run(argv,cwd=ROOT,check=True)
            done.append(name)
        atomic_json(OUT/'progress.json',{'status':'complete','completed':done,'planned':13})


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--case',choices=CASES);a.add_argument('--seed',type=int);a.add_argument('--unobserved',action='store_true');args=a.parse_args()
    if args.prepare:prepare()
    elif args.case:worker(args.case,args.seed,args.unobserved)
    else:run()
