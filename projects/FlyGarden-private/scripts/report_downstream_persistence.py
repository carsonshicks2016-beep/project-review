"""Independent raw-spike/delay/gate delivery reconstruction and lesion report."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
from flygarden.descending import DescendingDecoder
OUT=ROOT/'reports/brain-integration/recovery/downstream-persistence-v1'
p=json.loads((OUT/'protocol.json').read_text())
assert all(file_sha(ROOT/n)==h for n,h in p['sources'].items())
import pandas as pd
import brian2 as brian
graph=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet')
ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy()
base_weights=np.asarray((graph['Excitatory x Connectivity'].to_numpy()*.275*brian.mV)/brian.mV)
baseline_sha=hashlib.sha256(base_weights.tobytes()).hexdigest();expected_hashes={};lesion_map={}
for case,lesion in p['lesions'].items():
    rows=np.array(lesion['rows'],dtype=int);weights=base_weights.copy();weights[rows]=0
    expected_hashes[case]=hashlib.sha256(weights.tobytes()).hexdigest()
    lesion_map[case]=[{'row':int(row),'pre_root_id':str(ids[int(graph.iloc[row].Presynaptic_Index)]),
                       'post_root_id':str(ids[int(graph.iloc[row].Postsynaptic_Index)]),
                       'signed_synapses':float(graph.iloc[row]['Excitatory x Connectivity']),
                       'anatomical_synapses':int(graph.iloc[row].Connectivity)} for row in rows]
del graph,weights,base_weights
atomic_json(OUT/'lesion-map.json',{'protocol_sha256':file_sha(OUT/'protocol.json'),'ordering_source_sha256':p['sources']['vendor/fly-brain/data/2025_Completeness_783.csv'],'edges':lesion_map})
atomic_json(OUT/'model-variants.json',{'scope':'Raw controller manifest identifies the common pre-intervention controller. These identities additionally identify diagnostic transport changes; no variant is promoted.',
    'variants':[{'id':'downstream-groupcut-'+case+'-v1','case':case,'neurons':138639,'anatomical_graph_records':15091983,
                 'base_neural_source_sha256':p['sources']['flygarden/brain.py'],
                 'signed_transport_weights_modified':case!='intact','intervention_seconds':.8,
                 'selected_aggregate_records':p['lesions'][case]['aggregated_records'],
                 'edge_mask_sha256':hashlib.sha256(json.dumps(p['lesions'][case]['rows']).encode()).hexdigest(),
                 'initial_weight_sha256':baseline_sha,'post_intervention_weight_sha256':expected_hashes[case],
                 'learning':False,'body_present':False,'biological_knockout_claim':False} for case in p['cases']]})
selected=np.array(p['selected']);target_col={int(n):i for i,n in enumerate(selected)}
mapping={k:np.array([n['index'] for n in v],dtype=int) for k,v in p['mapping'].items()}
edges=p['incoming_edges'];summaries=[];curves={};audits=[];rankings=[];metadata=[]
for case in p['cases']:
    zero_rows=set(p['lesions'][case]['rows'])
    matrices=[]
    for after in (False,True):
        pos=np.zeros((138639,len(selected)));neg=np.zeros_like(pos)
        for e in edges:
            w=e['signed_synapses']*.275
            if after and e['row'] in zero_rows:w=0
            col=target_col[e['post']];pos[e['pre'],col]+=max(w,0);neg[e['pre'],col]+=max(-w,0)
        matrices.append((pos,neg))
    for seed in p['seeds']:
        folder=OUT/'trials'/f'{case}-{seed}';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete'
        assert m['sources']==p['sources'] and m['actual_lesioned_weights_sha256']==m['expected_lesioned_weights_sha256']==m['final_weights_sha256'];metadata.append(m)
        assert m['initial_weights_sha256']==baseline_sha and m['final_weights_sha256']==expected_hashes[case]
        rows=json.loads((folder/'rows.json').read_text());assert len(rows)==len(m['chunks'])==160
        traces={k:[] for k in mapping};deliveries=[];previous_i=np.array([],dtype=int);previous_ticks=np.array([],dtype=int)
        rng=np.random.default_rng(seed);channels=np.array(m['controller']['input_channels']);ninputs=len(channels);decoder=DescendingDecoder()
        maxerr=0.;late_source_pos=np.zeros((138639,len(selected)));late_source_neg=np.zeros_like(late_source_pos)
        for tick,chunk in enumerate(m['chunks']):
            path=folder/chunk['file'];assert file_sha(path)==chunk['sha256']
            with np.load(path) as z:
                assert np.array_equal(np.bincount(z['spike_i'],minlength=138639),z['counts'])
                assert np.array_equal(z['counts'][selected],z['spike_counts'])
                assert np.isfinite(z['voltage_mV']).all() and np.isfinite(z['net_synaptic_conductance_equivalent_mV']).all()
                assert z['gate'].shape==(len(selected),250)
                assert np.allclose(z['gate_t'],(tick*250+np.arange(250))*.0001,rtol=0,atol=1e-12)
                assert np.allclose(z['time_seconds'],(tick*250+np.arange(0,250,10))*.0001,rtol=0,atol=1e-12)
                rates=np.array([50 if 12<=tick<32 else 0,0,0,0,65,65,0,0.])
                tt,ii=np.nonzero(rng.random((250,ninputs))<rates[channels]*.0001)
                assert np.array_equal(ii,z['external_i']) and np.array_equal(z['external_i'],z['delivered_external_indices'])
                assert np.allclose((tick*250+tt)*.0001,z['external_t'],rtol=0,atol=1e-12)
                assert np.allclose(z['external_t'],z['delivered_external_times_seconds'],rtol=0,atol=1e-12)
                counts=z['counts'];pop={k:float(counts[i].mean()/.025) for k,i in mapping.items()}
                assert all(abs(pop[k]-rows[tick]['population_hz'][k])<1e-10 for k in mapping)
                assert np.array_equal(decoder.advance(.025,pop),z['motor'])
                for k in mapping:traces[k].append(pop[k])
                current_ticks=np.rint(z['spike_t']/.0001).astype(int)
                assert np.all((current_ticks>=tick*250)&(current_ticks<(tick+1)*250))
                source_i=np.concatenate([previous_i,z['spike_i']]);arrival_ticks=np.concatenate([previous_ticks,current_ticks])+18
                within=(arrival_ticks>=tick*250)&(arrival_ticks<(tick+1)*250)
                source_i=source_i[within];arrival=arrival_ticks[within]-tick*250
                pos,neg=matrices[int(tick>=32)];actual=[]
                for matrix in (pos,neg):
                    event_values=matrix[source_i]*z['gate'][:,arrival].T
                    actual.append(event_values.sum(axis=0))
                    if tick>=140:
                        destination=late_source_pos if matrix is pos else late_source_neg
                        np.add.at(destination,source_i,event_values)
                reconstructed=np.array(actual);err=float(np.max(np.abs(reconstructed-z['delivered_mV'])))
                maxerr=max(maxerr,err);assert err<1e-6,(case,seed,tick,err)
                deliveries.append(z['delivered_mV'].copy())
                carry=current_ticks>=(tick+1)*250-18
                previous_i=z['spike_i'][carry].copy();previous_ticks=current_ticks[carry].copy()
                if case!='intact':
                    with np.load(OUT/'trials'/f'intact-{seed}'/chunk['file']) as intact:
                        assert np.array_equal(intact['external_i'],z['external_i']) and np.array_equal(intact['external_t'],z['external_t'])
                        if tick<32:
                            assert all(np.array_equal(intact[k],z[k]) for k in ('spike_i','spike_t','counts','motor','voltage_mV','net_synaptic_conductance_equivalent_mV','gate','delivered_mV'))
        traces={k:np.array(v) for k,v in traces.items()};deliveries=np.array(deliveries)
        traces['pn']=np.mean([traces[k] for k in ('DM1_lPN_left','DM1_lPN_right')],axis=0)
        traces['dna_left']=traces['DNa02_left'];traces['dna_right']=traces['DNa02_right'];curves[(case,seed)]=traces
        summary={'case':case,'seed':seed,'tail_pn_hz':float(traces['pn'][140:].mean()),
                 'tail_dna_left_hz':float(traces['dna_left'][140:].mean()),'tail_dna_right_hz':float(traces['dna_right'][140:].mean()),
                 'tail_alln_mean_hz':float(traces['ALLN'][140:].mean()),
                 'tail_orn_dm1_hz':float(np.mean([traces[k][140:].mean() for k in ('ORN_DM1_left','ORN_DM1_right')])),
                 'accepted_positive_mV_per_second':(deliveries[140:,0].sum(axis=0)/.5).tolist(),
                 'accepted_negative_mV_per_second':(deliveries[140:,1].sum(axis=0)/.5).tolist()}
        summaries.append(summary);audits.append({'case':case,'seed':seed,'chunks':160,'delivery_max_error_mV':maxerr})
        if case in ('intact','positive_alln_recurrence_off'):
            bypre={e['pre']:e for e in edges};pncols=[target_col[n] for k in ('DM1_lPN_left','DM1_lPN_right') for n in mapping[k]]
            values=late_source_pos[:,pncols].sum(axis=1)/.5
            groups={}
            for pre in np.flatnonzero(values):
                e=bypre[int(pre)];name=e['pre_cell_class'] or 'unavailable'
                groups[name]=groups.get(name,0)+float(values[pre])
            rankings.append({'case':case,'seed':seed,'target':'both DM1 lPNs','phase':'3.5-4.0s','units':'accepted summed voltage increments mV/s, not currents',
                             'positive_by_cell_class':dict(sorted(groups.items(),key=lambda kv:-kv[1])),
                             'top_positive_sources':[{'root_id':bypre[int(i)]['pre_root_id'],'cell_type':bypre[int(i)]['pre_cell_type'],'cell_class':bypre[int(i)]['pre_cell_class'],'accepted_mV_per_second':float(values[i])} for i in np.argsort(values)[-12:][::-1] if values[i]>0]})
        if case in ('intact','positive_alln_to_pn_off'):
            bypre={e['pre']:e for e in edges}
            cols=[target_col[n] for n in mapping['DNa02_left']]
            for sign,source_values in [('positive',late_source_pos),('negative',late_source_neg)]:
                values=source_values[:,cols].sum(axis=1)/.5;groups={}
                for pre in np.flatnonzero(values):
                    name=bypre[int(pre)]['pre_cell_class'] or 'unavailable'
                    groups[name]=groups.get(name,0)+float(values[pre])
                rankings.append({'case':case,'seed':seed,'target':'DNa02_left','sign':sign,'phase':'3.5-4.0s',
                    'units':'accepted summed voltage increments mV/s, not currents','exploratory_source_ranking':True,
                    'by_cell_class':dict(sorted(groups.items(),key=lambda kv:-kv[1])),
                    'top_sources':[{'root_id':bypre[int(i)]['pre_root_id'],'cell_type':bypre[int(i)]['pre_cell_type'],
                                    'cell_class':bypre[int(i)]['pre_cell_class'],'accepted_mV_per_second':float(values[i])}
                                   for i in np.argsort(values)[-12:][::-1] if values[i]>0]})
# Observer presence must not alter the neural trajectory.
parity=OUT/'trials'/f'intact-{p["seeds"][0]}-observer-off';pm=json.loads((parity/'manifest.json').read_text());assert pm['status']=='complete';metadata.append(pm)
for tick,c in enumerate(pm['chunks']):
    assert file_sha(parity/c['file'])==c['sha256']
    with np.load(parity/c['file']) as a,np.load(OUT/'trials'/f'intact-{p["seeds"][0]}'/c['file']) as b:
        assert all(np.array_equal(a[k],b[k]) for k in ('counts','spike_i','spike_t','external_i','external_t','motor','voltage_mV','net_synaptic_conductance_equivalent_mV','gate'))
        assert not a['delivered_mV'].any()
comparisons=[]
for case in p['cases'][1:]:
    for seed in p['seeds']:
        baseline=next(s for s in summaries if s['case']=='intact' and s['seed']==seed)
        s=next(s for s in summaries if s['case']==case and s['seed']==seed)
        ratio=s['tail_pn_hz']/baseline['tail_pn_hz'] if baseline['tail_pn_hz'] else None
        comparisons.append({'case':case,'seed':seed,'pn_tail_ratio':ratio,'pn_reduction_hz':baseline['tail_pn_hz']-s['tail_pn_hz'],
                            'strong_suppression':ratio is not None and ratio<=.1 and baseline['tail_pn_hz']-s['tail_pn_hz']>=.9*baseline['tail_pn_hz'],
                            'dna_left_reduction_hz':baseline['tail_dna_left_hz']-s['tail_dna_left_hz']})
strong=[case for case in p['cases'][1:] if all(c['strong_suppression'] for c in comparisons if c['case']==case)]
local_cut=[s for s in summaries if s['case']=='positive_alln_to_pn_off']
findings=[]
if all(s['tail_pn_hz']==0 and s['tail_dna_left_hz']>10 for s in local_cut):
    findings.append('Removing positive ALLN-to-DM1-PN delivery silenced both tested PNs, while left DNa02 activity persisted in both seeds. PN cessation alone does not resolve downstream persistence.')
if 'positive_pn_output_off' not in strong:
    findings.append('Removing positive output from the two DM1 PNs did not meet the strong PN-suppression threshold; their own output is not required for strong persistent PN activity in these trials.')
if 'positive_alln_recurrence_off' not in strong:
    findings.append('Removing positive connections within annotated ALLNs did not meet the strong PN-suppression threshold. The tested persistence is not explained by that local-neuron-only connection group alone.')
if 'cognate_orn_to_pn_off' not in strong:
    findings.append('Removing cognate odor-neuron delivery after the pulse did not meet the strong PN-suppression threshold; ongoing odor afferent delivery is not the sole sustaining source.')
r={'status':'complete','trials':13,'chunks_audited':2080,'instrumentation_parity_exact':True,
   'preintervention_trajectories_exact':True,'matched_external_inputs_exact':True,
   'accepted_delivery_reconstruction_passed':True,'audits':audits,'summary':summaries,'comparisons':comparisons,
   'strong_pn_suppression_both_seeds':strong,'source_rankings':rankings,'promoted':False,
   'model_findings':findings,
   'navigation_validated':False,'learning_validated':False,'biological_feedback_cause_proven':False,
   'trial_wall_seconds':sum(m['wall_seconds'] for m in metadata),'peak_rss_bytes':max(m['peak_rss_bytes'] for m in metadata),
   'auditor_sha256':file_sha(Path(__file__))}
atomic_json(OUT/'results.json',r)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
t=(np.arange(160)+1)*.025
colors=['#333333','#8b6bb3','#d97924','#287da8','#b43742','#388659']
labels=['Intact','Odor to PN cut','PN output cut','Local neuron to PN cut','Local positive recurrence cut','All positive PN input cut']
for case,color,label in zip(p['cases'],colors,labels):
    for ax,key in zip(axes,['pn','ALLN','dna_left']):
        values=np.array([curves[(case,seed)][key] for seed in p['seeds']])
        ax.plot(t,values.mean(axis=0),label=label,color=color,lw=1.4)
for ax in axes:
    ax.axvspan(.3,.8,color='gray',alpha=.1)
    ax.axvline(.8,color='gray',ls='--',alpha=.6);ax.grid(alpha=.2)
axes[0].set_ylabel('DM1 lPN mean Hz');axes[1].set_ylabel('ALLN mean Hz');axes[2].set_ylabel('Left DNa02 Hz')
axes[2].set_xlabel('Simulated seconds');axes[0].legend(fontsize=8,ncol=2)
fig.suptitle('Late diagnostic cuts at 0.8s — shaded odor pulse; two-seed means')
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
table='\n'.join(f"| {s['case']} | {s['seed']} | {s['tail_pn_hz']:.1f} | {s['tail_alln_mean_hz']:.1f} | {s['tail_dna_left_hz']:.1f} | {s['tail_dna_right_hz']:.1f} |" for s in summaries)
finding_text='\n\n'.join(findings)
(OUT/'RESULTS.md').write_text(f'''# Downstream persistence diagnostic

Completed 13 full-network runs, including a delivery-observer parity replay.
Cuts that met the prospective strong-PN-suppression threshold in both seeds:
{', '.join(strong) if strong else 'none'}.

{finding_text}

These are deliberately altered diagnostic networks, not candidate repairs.
model-variants.json distinguishes each temporary transport alteration from the
common base-controller identity in the raw trial manifests.
The application and original controller sources were left unchanged. No learning
or navigation success is claimed. All trials use a left odorA pulse .3-.8s and
constant65Hz supplied walking support. At.8s only selected synaptic transport
weights are zeroed; neural state, topology, other weights and decoder remain fixed.
Pending events encounter the new weight on arrival. Native timing is100μs;
selected membrane-state sampling is1ms.

## Final half-second responses

| Intervention | Seed | DM1 PN mean Hz | ALLN mean Hz | Left DNa02 Hz | Right DNa02 Hz |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

Strong PN suppression requires at least90% reduction relative to the matched
intact condition in both seeds. DNa02 responses are reported independently:
quieting projection neurons need not quiet a downstream recurrent network.
The ALLN mean spans all429 exactly annotated local neurons; it can hide a small
persistent subpopulation. The detailed artifact also retains actual accepted
positive/negative deliveries and the strongest active input sources into the PNs.

## Verification

All2,080 chunk hashes were checked. Raw spikes matched monitor counts. Inputs
were independently reconstructed from the owned random generator and matched
the intact condition. All intervention trajectories matched intact exactly before
the cut. The additional intact replay with the observer disabled matched spikes,
counts, inputs, commands, selected v/g and refractory gates exactly.

Positive/negative accepted input increments were independently reconstructed
from presynaptic spike ticks, the1.8ms delay, current effective weights and actual
before-synapse refractory gates, including arrivals across chunk boundaries.
All window errors were below1e-6mV. These sums are modeled voltage increments,
not physiological membrane currents or proof of real excitation/inhibition.
The positive/negative signs come from the imported model. ALLN membership comes
from exact-root cell-class annotations, not arbitrary neuron indices.

The frozen protocol identifies every selected edge position against pinned data;
lesion-map.json exports exact source/target roots and synapse counts for all cuts.
The incoming-edge table includes source annotations for the readout neurons.
Effective post-cut weights were hash-checked
against the expected vector and remained fixed afterward. Missing cell labels
are retained as unavailable, not inferred. This two-seed intervention screen
shows dependence on a modeled connection group in the tested state; it does not
identify a unique minimal loop or establish a biological cause.

Trial wall time: {r['trial_wall_seconds']:.1f}s. Maximum worker RSS:
{r['peak_rss_bytes']/1024**3:.2f}GiB.
The figure averages two seeds without confidence-interval claims.

Next: use these causal results and accepted-input rankings to choose one bounded
downstream follow-up. Do not promote a lesioned graph, adjust thresholds to force
success, or proceed to learning while sensory recovery/direction remain unpassed.
''')
progress_path=ROOT/'reports/brain-integration/recovery/progress.json';progress=json.loads(progress_path.read_text())
progress['phases']['4']['status']='downstream_causal_group_diagnostics_complete_model_revision_pending'
for name in ('RESULTS.md','results.json','protocol.json'):
    path=str((OUT/name).relative_to(ROOT))
    if path not in progress['phases']['4']['evidence']:progress['phases']['4']['evidence'].append(path)
progress['updated']=time.time();progress['next']='Choose a source-supported downstream model hypothesis from causal group cuts and actual accepted deliveries; no lesion promotion.';progress['next_action']=progress['next'];atomic_json(progress_path,progress)
ledger_path=ROOT/'reports/brain-integration/acceptance-ledger.json';ledger=json.loads(ledger_path.read_text())
for item in ledger['requirements']:
    if item['step']==8:
        item['status']=progress['phases']['4']['status'];item['completion_proven']=False
        evidence=str((OUT/'RESULTS.md').relative_to(ROOT/'reports/brain-integration'))
        if evidence not in item['evidence']:item['evidence'].append(evidence)
ledger['updated']=time.time();atomic_json(ledger_path,ledger)
print(json.dumps({'strong_suppression':strong,'summary':summaries,'rankings':rankings},indent=2))
