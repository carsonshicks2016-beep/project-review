"""Verify full raw recordings and report preregistered stage 5 sensory tests."""
import json,sys,time,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_brain_reference import sha
from scripts.diagnose_reference_odor import SOURCES,SEEDS,CONDITIONS,rates_for
from scripts.diagnose_sensorimotor_pathway import mapping
from flygarden.odor_encoding import OdorReference
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/stage5-20261006'
def report():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from scripts.audit_brain_reference import params,REFERENCE
 import brian2 as b
 audit=json.loads((OUT/'parameter-audit.json').read_text());ref=params()
 pairs={'rest_reset_mV':('v_rst',b.mV),'threshold_mV':('v_th',b.mV),'membrane_ms':('t_mbr',b.ms),'refractory_ms':('t_rfc',b.ms),'synapse_ms':('tau',b.ms),'delay_ms':('t_dly',b.ms),'synaptic_gain_mV':('w_syn',b.mV)}
 actual={k:float(ref[key]/unit) for k,(key,unit) in pairs.items()}
 assert all(np.isclose(actual[k],audit[k]) for k in pairs)
 audit['verified_pinned_values']=actual;audit['reference_source_sha256']=sha(REFERENCE);atomic_json(OUT/'parameter-audit.json',audit)
 protocol=json.loads((OUT/'protocol.json').read_text());assert all(sha(ROOT/k)==v for k,v in protocol['sources'].items())
 ids,pops,_=mapping();artifact=json.loads((OUT/'odor-reference.json').read_text());encoder=OdorReference(artifact,ids)
 data={};total=0;checks=[];max_rss=0;elapsed=0
 for cue in CONDITIONS:
  for seed in SEEDS:
   folder=OUT/'trials'/f'{cue}-{seed}';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and m['neurons']==138639 and m['connections']==15091983
   assert m['learning'] is False and m['synapse_changes']==m['intrinsic_parameter_changes']==0
   assert m['sources']==protocol['sources'];assert len(m['chunks'])==30
   bins=[];raw_hashes=[]
   for tick,c in enumerate(m['chunks']):
    path=folder/c['file'];assert sha(path)==c['sha256']
    with np.load(path) as z:
     counts=z['counts'];si=z['spike_i'];st=z['spike_t'];rates=z['requested_hz'];inp=z['input_counts']
     assert np.array_equal(np.bincount(si,minlength=len(ids)),counts)
     assert len(si)==c['spikes'] and np.all(st>=tick*.1-1e-9) and np.all(st<(tick+1)*.1+1e-9)
     assert np.array_equal(rates,rates_for(cue,tick*.1+1e-7,encoder))
     assert inp.shape==rates.shape and np.all(inp[rates==0]==0)
     total+=len(si);bins.append({'count':counts.copy(),'input':inp.copy(),'requested':rates.copy(),'spike_i':si.copy() if tick<5 else None,'spike_t':st.copy() if tick<5 else None})
   data[(cue,seed)]=bins;max_rss=max(max_rss,m['peak_rss_bytes']);elapsed+=m['wall_seconds']
 checks.append('All 990 chunks: hashes, raw spike histograms, interval timestamps, exact requested rates, zero-rate input events, full-network counts and frozen learning verified.')
 for cue in CONDITIONS[:8]:
  for seed in SEEDS:
   for tick in range(5):
    a=data[(cue,seed)][tick];b=data[('baseline',seed)][tick]
    for key in ('count','input','spike_i','spike_t'):assert np.array_equal(a[key],b[key]),(cue,seed,tick,key)
 checks.append('All measured-odor and oil trials exactly match same-seed baseline raw neuron IDs/times and input counts before odor onset.')
 repeat=OUT/'repeat'/'acetate_left-5199';rm=json.loads((repeat/'manifest.json').read_text());assert rm['status']=='complete' and rm['sources']==protocol['sources']
 for c in rm['chunks']:
  rp=repeat/c['file'];op=OUT/'trials'/'acetate_left-5199'/c['file'];assert sha(rp)==c['sha256']
  with np.load(rp) as a,np.load(op) as b:
   assert set(a.files)==set(b.files)
   for key in a.files:assert np.array_equal(a[key],b[key]),key
 checks.append('Independent held-out ethyl-acetate repetition: all 30 windows exactly match raw spike IDs/times, per-neuron counts, input counts and requested rates. This does not claim full-checkpoint state equivalence.')
 dn=[pops['DNa02_left'][0]['index'],pops['DNa02_right'][0]['index']]
 def dn_rates(cue,seed,window=slice(5,10)):
  bins=data[(cue,seed)][window];return np.array([b['count'][dn] for b in bins]).sum(axis=0)/(len(bins)*.1)
 gate=[]
 for prefix in ('acetate','pentanoate'):
  for side in ('left','right'):
   for seed in SEEDS:
    rate=dn_rates(prefix+'_'+side,seed);baseline=dn_rates('baseline',seed);oil=dn_rates('oil_'+side,seed);delta=(rate[0]-rate[1])-(baseline[0]-baseline[1]);oil_delta=(rate[0]-rate[1])-(oil[0]-oil[1]);passed=bool(delta>5 if side=='left' else delta< -5)
    gate.append({'chemical':prefix,'side':side,'seed':seed,'dn_left_hz':float(rate[0]),'dn_right_hz':float(rate[1]),'steering_change_from_background_hz':float(delta),'steering_change_from_oil_hz':float(oil_delta),'passed':passed})
 recovery=[]
 for cue in CONDITIONS[:8]:
  for seed in SEEDS:
   a=sum(b['count'].sum() for b in data[(cue,seed)][10:15])/.5;b=sum(z['count'].sum() for z in data[('baseline',seed)][10:15])/.5
   recovery.append({'cue':cue,'seed':seed,'post_odor_spikes_per_second':float(a),'change_from_background_spikes_per_second':float(a-b)})
 unit_stats=[]
 for cue in ('baseline','acetate_left','acetate_right','pentanoate_left','pentanoate_right'):
  for seed in SEEDS:
   for unit in artifact['units']:
    for side in ('left','right'):
     positions=[i for i,(u,n) in enumerate(encoder.entries) if u['unit']==unit['unit'] and n['side']==side];targets=encoder.targets[positions]
     if not positions:continue
     bins=data[(cue,seed)][5:10]
     requested=np.mean([b['requested'][positions].mean() for b in bins]);delivered=sum(b['input'][positions].sum() for b in bins)/(.5*len(positions));output=sum(b['count'][targets].sum() for b in bins)/(.5*len(positions))
     unit_stats.append({'cue':cue,'seed':seed,'unit':unit['unit'],'side':side,'n':len(positions),'requested_hz':float(requested),'delivered_hz':float(delivered),'modeled_orn_hz':float(output)})
 results={'status':'complete','trials':33,'chunks':990,'simulated_seconds':99,'raw_spikes':total,'coverage':{'measured_input_neurons':artifact['covered_orn_count'],'annotated_orns':artifact['annotated_orn_count'],'full_modeled_neurons':138639},'steering_gate_passed':all(g['passed'] for g in gate),'steering_gates':gate,'recovery':recovery,'unit_activity':unit_stats,'verification':checks,'worker_wall_seconds':elapsed,'peak_rss_bytes':max_rss,'free_disk_bytes':shutil.disk_usage(OUT).free}
 atomic_json(OUT/'results.json',results)
 fig,ax=plt.subplots(2,1,figsize=(11,8),sharex=True)
 colors={'baseline':'#666666','acetate_left':'#e89525','acetate_right':'#316ecc','pentanoate_left':'#b66e20','pentanoate_right':'#145395'}
 for cue,col in colors.items():
  b=data[(cue,5199)];t=np.arange(1,31)*.1;y=np.array([z['count'][dn]/.1 for z in b]);ax[0].plot(t,y[:,0]-y[:,1],label=cue,color=col);ax[1].plot(t,[z['count'].sum()/.1 for z in b],label=cue,color=col)
 for a in ax:a.axvspan(.5,1,color='#dddddd',alpha=.5);a.grid(alpha=.2)
 ax[0].set(ylabel='DNa02 left minus right (Hz)',title='Untouched full network: held-out seed 5199');ax[0].legend(ncol=3,fontsize=8)
 ax[1].set(ylabel='Whole-network spikes / second',xlabel='Simulated seconds');fig.tight_layout();fig.savefig(OUT/'odor-steering.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5));units=[u['unit'] for u in artifact['units']]
 for key,col in [('requested_hz','#222222'),('delivered_hz','#3184bf'),('modeled_orn_hz','#c86b30')]:
  values=[next(s[key] for s in unit_stats if s['cue']=='acetate_left' and s['seed']==5199 and s['side']=='left' and s['unit']==u) for u in units];ax.plot(units,values,'o-',label=key,color=col)
 ax.tick_params(axis='x',rotation=45);ax.set(ylabel='Mean spikes / second / neuron',title='Ethyl acetate: requested input, sampled input, and modeled ORN output');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'sensory-fidelity.png',dpi=160);plt.close(fig)
 passed=sum(g['passed'] for g in gate);left=[g['dn_left_hz'] for g in gate];right=[g['dn_right_hz'] for g in gate];fidelity=np.mean([abs(s['modeled_orn_hz']-s['delivered_hz']) for s in unit_stats if s['cue']!='baseline'])
 text=f'''# Step 5 — literature-backed sensory encoding and parameter audit

The measured-rate sensory encoder is implemented and tested in an isolated full network. The preregistered steering gate {'passed' if results['steering_gate_passed'] else 'failed'}: {passed}/12 cue/seed comparisons met the criterion. No model was promoted to the arena.

## What changed in the experiment

The [DoOR Bruyne.2001.WT dataset](https://neuro.uni-konstanz.de/DoOR/content/dataset.php?dataset=Bruyne.2001.WT) supplies baseline-subtracted responses and spontaneous firing, mapped through receptor/glomerulus annotations to **751 of 2,275 annotated ORNs**. The complete brain still contains **138,639 neurons and 15,091,983 connection records**. Unmapped sensory populations remain present, with no invented response values.

Sixteen populations receive background firing. Named ethyl acetate and ethyl pentanoate stimuli, plus the oil control, use the measured rates. For example, DM1 uses 9 Hz background and 235 Hz during ethyl acetate exposure; DM2 uses 4 Hz background and 30.308 Hz during that stimulus. Negative response changes reduce background firing. Oil is measured separately because these data do not subtract the solvent response. There is no 100 Hz cap. Stimulated sensory roots use the reference zero-refractory convention; other cells retain 2.2 ms.

Linear exposure interpolation, rectangular 500 ms pulses, bilateral placement and stochastic input timing are engineered choices. These average response values do not supply biological adaptation traces, arena concentration calibration, or complete sensory coverage. Rates describe external events; recurrent modeled ORN spikes can differ.

## Outcome

Three seeds, including the committed held-out seed 5199, covered background, oil left/right, two odors left/right, side switching, quiet and legacy input controls: **33 independent full-network trials**, **99 simulated seconds**, **990 complete chunks**, **{total:,} raw spikes**. Learning was frozen. All synaptic weights and published intrinsic parameters were retained. A separate 3-second repeat verified deterministic raw recordings; its 30 chunks are retained outside the primary trial total.

Across the named-odor pulse conditions, left DNa02 rates ranged {min(left):.1f}–{max(left):.1f} Hz and right rates {min(right):.1f}–{max(right):.1f} Hz. The full comparison table in results.json includes each cue versus matched background and versus oil. Mean absolute difference between sampled external rates and modeled ORN output was {fidelity:.1f} Hz across measured-odor unit/side/seed comparisons; this is a descriptive fidelity diagnostic, not a fit or statistical validation. Recovery is reported relative to firing background, rather than assuming a silent brain.

![Steering and activity](odor-steering.png)
![Input fidelity](sensory-fidelity.png)

## Parameter audit and limits

The [published whole-brain model](https://www.nature.com/articles/s41586-024-07763-9) uses the same -52 mV reset, -45 mV threshold, 20 ms membrane constant, 2.2 ms refractory interval, 5 ms synaptic decay, 1.8 ms delay, and 0.275 mV synaptic gain. Step 1 established equation parity with the pinned release. These values were not retuned here. The paper's network release was 630; our imported data are release 783. Its gain was calibrated to feeding response, which does not establish walking calibration.

No supported DNa02- or AOTU019-specific intrinsic replacement was found. The raw archive linked by the [DNa02 physiology paper](https://elifesciences.org/articles/102230) returned HTTP 403 through both attempted public API routes; this prevents a quantitative physiological fit, not completion of this sensory audit. The local table lacks an MN9 annotation needed to reproduce the original feeding calibration.

## Verification and runtime

{chr(10).join('- '+s for s in checks)}
- Nine targeted encoder and existing steering/decoder checks passed; see tests.log.
- One neural worker at a time; total worker time {elapsed:.1f} s; maximum process memory {max_rss/1024**3:.2f} GiB; simulation throughput {99/elapsed:.3f} simulated seconds per wall second, including startup.
- Raw recordings retained. Downloads and recording checks preserve the 2 GiB reserve; no archives were deleted.
- Production brain, decoder, arena and learning rules were not changed.

## Reproduction and next decision

Run `.venv-next/bin/python scripts/prepare_odor_reference.py`, then `.venv-next/bin/python scripts/diagnose_reference_odor.py reports/brain-integration/stage5-20261006`, then `.venv-next/bin/python scripts/diagnose_reference_odor.py reports/brain-integration/stage5-20261006/repeat/acetate_left-5199 --cue acetate_left --seed 5199`, then `.venv-next/bin/python scripts/report_reference_odor.py`. Completed trials are reused only under unchanged source hashes. An incomplete trial is preserved and requires a separate retry directory; it is never overwritten. Input tables are pinned to DoOR.data {artifact['source_revision']}; hashes and access failures are in retrieval-manifest.json. Derived reference values retain **CC BY-SA 4.0**, attributed to DoOR and de Bruyne et al.; source snapshots are preserved.

{'Sensory steering now meets the committed gate, but a separate embodied closed-loop evaluation is required before integration.' if results['steering_gate_passed'] else 'Better-supported sensory input alone did not establish reliable sensory steering. The next necessary work is physiological validation of modeled sensory and descending responses, with explicit release-transfer checks; changing inhibition or neuron excitability to obtain attractive movement would not be justified by these results.'}
'''
 (OUT/'RESULTS.md').write_text(text)
 snapshot=OUT/'source-snapshot';snapshot.mkdir(exist_ok=True)
 for source in (*SOURCES,'scripts/report_reference_odor.py','tests/test_odor_encoding.py'):
  dest=snapshot/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,dest)
 atomic_json(OUT/'runtime.json',{'python':sys.version,'numpy':np.__version__,'brian2':__import__('brian2').__version__,'pandas':__import__('pandas').__version__,'source_hashes':protocol['sources'],'result_hash':sha(OUT/'results.json'),'report_script_sha256':sha(Path(__file__))})
 print('Verified',total,'spikes; steering gates',passed,'/12')
if __name__=='__main__':report()
