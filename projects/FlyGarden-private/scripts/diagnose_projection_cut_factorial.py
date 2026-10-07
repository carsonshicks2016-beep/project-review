"""Frozen two-by-two transport diagnostic; not a controller candidate."""
import sys,json,argparse,time,fcntl,subprocess,shutil,copy
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from flygarden.continuous_candidate import file_sha
import scripts.diagnose_persistence_sources as prior
import scripts.diagnose_downstream_persistence as base
OUT=ROOT/'reports/brain-integration/recovery/projection-cut-factorial-v1'
SEEDS=[11601,11602]
CASES=['intact','alpn_to_alln_off','alln_recurrence_off','joint_off']
SOURCES=prior.SOURCES+['scripts/diagnose_projection_cut_factorial.py','reports/brain-integration/recovery/persistence-sources-v1/protocol.json','reports/brain-integration/recovery/persistence-sources-v1/results.json']
def prepare():
    OUT.mkdir(exist_ok=True);hashes={n:file_sha(ROOT/n) for n in SOURCES}
    if (OUT/'protocol.json').exists():
        p=json.loads((OUT/'protocol.json').read_text());assert p['sources']==hashes;return p
    p=copy.deepcopy(json.loads((prior.OUT/'protocol.json').read_text()))
    con=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet')
    pre=con.Presynaptic_Index.to_numpy(dtype=int);post=con.Postsynaptic_Index.to_numpy(dtype=int);positive=con['Excitatory x Connectivity'].to_numpy()>0
    alln=np.array([x['index'] for x in p['mapping']['ALLN']]);alpn=np.array([x['index'] for x in p['mapping']['ALPN']]);qa=np.isin(post,alln)
    rec=positive&np.isin(pre,alln)&qa;proj=positive&np.isin(pre,alpn)&qa
    lesions={}
    for case,mask in zip(CASES,[np.zeros(len(con),dtype=bool),proj,rec,proj|rec]):
        rows=np.flatnonzero(mask);lesions[case]={'rows':rows.tolist(),'aggregated_records':len(rows),'anatomical_synapses':int(con.iloc[rows].Connectivity.sum()),'source_indices':np.unique(pre[rows]).tolist(),'target_indices':np.unique(post[rows]).tolist()}
    p.update(version=1,sources=hashes,seeds=SEEDS,cases=CASES,lesions=lesions,
        schedule='Original full network;65Hz DNp09 support;50Hz left odorA .3-.8s. At .8s zero only exact positive ALPN-to-ALLN and/or ALLN-to-ALLN transport groups. State, inhibition, inputs, decoder remain fixed; learning disabled.',
        selection='Populations and four local targets inherited from previously frozen11501/11502 diagnostic;11601/11602 are new diagnostic seeds, not navigation validation.',
        decision='Two-by-two exact transport intervention. Against intact, final .5s selected-local mean <=10% reference in BOTH seeds with reference>10Hz is strong suppression. Report all four individual rates, full ALLN/ALPN, DM1PN and bilateralDNa02. Preserve failed results. No promotion.',
        instrumentation='One intact11601 observer-disabled replay must match all spikes/counts/input/commands and selected v/g/gates exactly.',scope='Conditional model circuit dependence; cuts are not biological knockouts or justified repair.')
    atomic_json(OUT/'protocol.json',p)
    for n in SOURCES:
        if n.startswith(('data/','vendor/','reports/')):continue
        dst=OUT/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    return p
base.OUT=OUT;base.SEEDS=SEEDS;base.CASES=CASES;base.prepare=prepare
def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=prepare();done=[]
        for case,seed,off in [(c,s,False) for c in CASES for s in SEEDS]+[('intact',SEEDS[0],True)]:
            name=f"{case}-{seed}{'-observer-off' if off else ''}";path=OUT/'trials'/name/'manifest.json'
            if path.exists():
                m=json.loads(path.read_text());assert m['status']=='complete' and m['sources']==p['sources']
            else:
                space_check(OUT,250*1024**2);atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':name,'planned':9})
                args=[sys.executable,__file__,'--case',case,'--seed',str(seed)]
                if off:args.append('--unobserved')
                subprocess.run(args,cwd=ROOT,check=True)
            done.append(name)
        atomic_json(OUT/'progress.json',{'status':'complete','completed':done,'planned':9})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--case',choices=CASES);a.add_argument('--seed',type=int);a.add_argument('--unobserved',action='store_true');args=a.parse_args()
    if args.prepare:prepare()
    elif args.case:base.worker(args.case,args.seed,args.unobserved)
    else:run()
