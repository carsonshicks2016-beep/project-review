"""Confirm recovery using pinned upstream construction and actual PoissonInput.

Run after the Stage 2 diagnostic worker exits; never alongside another worker.
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import brian2 as b

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_brain_reference import params,sha,REFERENCE
from flygarden.recording import atomic_json,space_check


def run(root):
    if json.loads((root/'progress.json').read_text())['status']!='complete':
        raise RuntimeError('Finish the main diagnostic worker before native confirmation')
    folder=root/'native-reference'
    space_check(root,16*1024**2);folder.mkdir(exist_ok=False)
    b.prefs.codegen.target='cython';b.defaultclock.dt=.1*b.ms;b.seed(2199)
    p=params();p['r_poi']=50*b.Hz
    parsed=ast.parse(REFERENCE.read_text())
    functions=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in ('create_network','add_poisson_inputs')]
    namespace={name:getattr(b,name) for name in ('NeuronGroup','Synapses','SpikeMonitor','PoissonInput','Network','mV','ms','Hz')}
    namespace.update(pd=pd,time=perf_counter)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(REFERENCE),'exec'),namespace)
    print('Building native upstream reference',flush=True);started=perf_counter()
    neu,syn,monitor,comp,timings=namespace['create_network'](
        ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',
        ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',p)
    ids=comp.index.to_numpy(dtype=np.int64);ordering={int(root_id):i for i,root_id in enumerate(ids)}
    ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('')
    selected=np.array([ordering[int(i)] for i in ann.loc[ann.root_id.isin(ids)&ann.cell_type.eq('ORN_DM1'),'root_id']],dtype=np.int32)
    poison=namespace['add_poisson_inputs'](neu,selected.tolist(),[],p)
    net=b.Network(neu,syn,monitor,*poison)
    manifest={'status':'running','source_revision':'a3db62f9436074e485c0278290c2164ed6150808',
              'source_sha256':sha(REFERENCE),'tool_sha256':sha(Path(__file__)),
              'neurons':len(ids),'connections':len(syn),'seed':2199,
              'clock_seconds':.0001,'baseline_seconds':.2,'pulse_seconds':.3,
              'recovery_seconds':5.5,'input_rate_hz':50,'input_neurons':len(selected),
              'alterations':'Original source factories, equations, reset and PoissonInput unchanged. Explicit clock, experimental rate, seed and on/off schedule only.',
              'chunks':[]}
    atomic_json(folder/'manifest.json',manifest)
    before=np.zeros(len(ids),dtype=np.int64);bins=[]
    for tick in range(60):
        enabled=2<=tick<5
        for obj in poison:obj.active=enabled
        net.run(.1*b.second)
        counts=np.asarray(monitor.count[:],dtype=np.int64);delta=counts-before;before=counts.copy()
        indices=np.asarray(monitor.i[:],dtype=np.int32).copy();times=np.asarray(monitor.t[:]/b.second).copy()
        assert len(indices)==int(delta.sum())
        file=folder/f'window-{tick:04d}.npz';space_check(folder,indices.nbytes+times.nbytes+delta.nbytes+1024**2)
        temporary=file.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,i=indices,t=times,counts=delta)
        temporary.replace(file)
        monitor.resize(0);monitor.variables['N'].set_value(0)
        row={'time':(tick+1)*.1,'input_enabled':enabled,'spikes':len(indices),
             'active_neurons':int((delta>0).sum()),'orn_mean_hz':float(delta[selected].mean()/.1),
             'finite':bool(np.isfinite(neu.v[:]).all() and np.isfinite(neu.g[:]).all())}
        bins.append(row);manifest['chunks'].append({'file':file.name,'sha256':sha(file),'spikes':len(indices)})
        atomic_json(folder/'bins.json',bins);atomic_json(folder/'manifest.json',manifest)
        if tick%10==9:print(f'Native reference {tick+1}/60 windows: {row["spikes"]/.1:.0f} spikes/s',flush=True)
    result={'status':'complete','tail_spikes_per_second':float(np.mean([r['spikes']/.1 for r in bins[-5:]])),
            'stimulation_spikes_per_second':float(np.mean([r['spikes']/.1 for r in bins[2:5]])),
            'total_spikes':sum(r['spikes'] for r in bins),'wall_seconds':perf_counter()-started,
            'finite_states':all(r['finite'] for r in bins),
            'scope':'Independent native-RNG recovery confirmation, not exact stochastic equivalence or sensory/body validation.'}
    manifest['status']='complete';atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    run(args.folder.resolve())
