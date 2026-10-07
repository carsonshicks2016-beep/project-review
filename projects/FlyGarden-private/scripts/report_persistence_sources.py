"""Independent raw-spike/delay/gate delivery reconstruction and lesion report."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
from flygarden.descending import DescendingDecoder
OUT=ROOT/'reports/brain-integration/recovery/persistence-sources-v1'
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
                 'signed_transport_weights_modified':case!='steering_reference','intervention_seconds':.8,
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
                if case!='steering_reference':
                    with np.load(OUT/'trials'/f'steering_reference-{seed}'/chunk['file']) as intact:
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
        if case in ('steering_reference','local_recurrence_reference'):
            bypre={e['pre']:e for e in edges};pncols=[target_col[n] for k in ('DM1_lPN_left','DM1_lPN_right') for n in mapping[k]]
            values=late_source_pos[:,pncols].sum(axis=1)/.5
            groups={}
            for pre in np.flatnonzero(values):
                e=bypre[int(pre)];name=e['pre_cell_class'] or 'unavailable'
                groups[name]=groups.get(name,0)+float(values[pre])
            rankings.append({'case':case,'seed':seed,'target':'both DM1 lPNs','phase':'3.5-4.0s','units':'accepted summed voltage increments mV/s, not currents',
                             'positive_by_cell_class':dict(sorted(groups.items(),key=lambda kv:-kv[1])),
                             'top_positive_sources':[{'root_id':bypre[int(i)]['pre_root_id'],'cell_type':bypre[int(i)]['pre_cell_type'],'cell_class':bypre[int(i)]['pre_cell_class'],'accepted_mV_per_second':float(values[i])} for i in np.argsort(values)[-12:][::-1] if values[i]>0]})
        if case.startswith('steering_'):
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
parity=OUT/'trials'/f'steering_reference-{p["seeds"][0]}-observer-off';pm=json.loads((parity/'manifest.json').read_text());assert pm['status']=='complete';metadata.append(pm)
for tick,c in enumerate(pm['chunks']):
    assert file_sha(parity/c['file'])==c['sha256']
    with np.load(parity/c['file']) as a,np.load(OUT/'trials'/f'steering_reference-{p["seeds"][0]}'/c['file']) as b:
        assert all(np.array_equal(a[k],b[k]) for k in ('counts','spike_i','spike_t','external_i','external_t','motor','voltage_mV','net_synaptic_conductance_equivalent_mV','gate'))
        assert not a['delivered_mV'].any()
# Actual accepted external source inputs into the four residual local targets.
# Repeat just late event attribution for these targets; no new brain computation.
local_ids=[x['index'] for x in p['selected_local_targets']];local_cols=[target_col[i] for i in local_ids]
local_attribution=[]
for case in p['cases'][:3]:
    off=set(p['lesions'][case]['rows']);weights=np.zeros((138639,len(local_ids)))
    for e in edges:
        if e['post'] in local_ids and e['row'] not in off and e['signed_synapses']>0:
            weights[e['pre'],local_ids.index(e['post'])]+=e['signed_synapses']*.275
    meta={e['pre']:e for e in edges}
    for seed in p['seeds']:
        folder=OUT/'trials'/f'{case}-{seed}';totals=np.zeros((138639,len(local_ids)))
        for tick in range(140,160):
            with np.load(folder/f'window-{tick:04d}.npz') as z,np.load(folder/f'window-{tick-1:04d}.npz') as previous:
                old_ticks=np.rint(previous['spike_t']/.0001).astype(int);new_ticks=np.rint(z['spike_t']/.0001).astype(int)
                carry=old_ticks>=tick*250-18
                source=np.concatenate([previous['spike_i'][carry],z['spike_i']]);arrival=np.concatenate([old_ticks[carry],new_ticks])+18-tick*250
                valid=(arrival>=0)&(arrival<250);source=source[valid];arrival=arrival[valid]
                values=weights[source]*z['gate'][local_cols][:,arrival].T;np.add.at(totals,source,values)
        groups={}
        for i in np.flatnonzero(totals.sum(axis=1)):
            cls=meta[int(i)]['pre_cell_class'] or 'unavailable';groups[cls]=groups.get(cls,0)+float(totals[i].sum()/.5)
        local_attribution.append({'case':case,'seed':seed,'target_roots':[x['root_id'] for x in p['selected_local_targets']],
            'accepted_positive_mV_per_second_by_class':dict(sorted(groups.items(),key=lambda x:-x[1])),
            'scope':'Accepted delayed/gated modeled positive voltage increments, not biological currents.'})
comparisons=[]
for s in summaries:
    t=curves[(s['case'],s['seed'])]
    s['tail_residual_local_mean_hz']=float(t['residual_local'][140:].mean())
    s['tail_alpn_mean_hz']=float(t['ALPN'][140:].mean())
    s['tail_local_individual_hz']={x['root_id']:0. for x in p['selected_local_targets']}
    folder=OUT/'trials'/f"{s['case']}-{s['seed']}";accum=np.zeros(138639,dtype=np.int64)
    for tick in range(140,160):
        with np.load(folder/f'window-{tick:04d}.npz') as z:accum+=z['counts']
    s['tail_local_individual_hz']={x['root_id']:float(accum[x['index']]/.5) for x in p['selected_local_targets']}
    reference='local_recurrence_reference' if s['case'].startswith('local_') else 'steering_reference'
    if s['case']==reference:continue
    baseline=next(x for x in summaries if x['case']==reference and x['seed']==s['seed'])
    key='tail_residual_local_mean_hz' if reference.startswith('local') else 'tail_dna_left_hz'
    ref=baseline[key];actual=s[key];ratio=actual/ref if ref else None
    comparisons.append({'case':s['case'],'seed':s['seed'],'reference':reference,'target_metric':key,
                        'reference_hz':ref,'target_hz':actual,'ratio':ratio,'strong_suppression':ref>10 and ratio<=.1})
strong=[case for case in p['cases'] if any(c['case']==case for c in comparisons) and all(c['strong_suppression'] for c in comparisons if c['case']==case)]
r={'status':'complete','trials':15,'chunks_audited':2400,'instrumentation_parity_exact':True,'preintervention_trajectories_exact':True,
   'matched_external_inputs_exact':True,'accepted_delivery_reconstruction_passed':True,'all_nonselected_weights_match_source':True,
   'summary':summaries,'comparisons':comparisons,'strong_suppression_both_seeds':strong,'local_accepted_source_attribution':local_attribution,
   'steering_source_rankings':rankings,'audits':audits,'promoted':False,'navigation_validated':False,'learning_validated':False,
   'biological_cause_proven':False,'trial_wall_seconds':sum(m['wall_seconds'] for m in metadata),'peak_rss_bytes':max(m['peak_rss_bytes'] for m in metadata),
   'source_sha256':file_sha(Path(__file__))}
atomic_json(OUT/'results.json',r)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True);time_axis=(np.arange(160)+1)*.025
labels={'local_recurrence_reference':'Local recurrence cut reference','local_plus_alpn_to_alln_off':'Also cut projection-to-local inputs','local_plus_all_external_positive_off':'Also cut all external positive local inputs',
        'steering_reference':'Odor-processing PN silenced reference','steering_ps013_off':'Also cut PS013 steering input','steering_top4_off':'Also cut four selected steering inputs','steering_all_positive_off':'Also cut all positive left-steering inputs'}
for ax,cases,key in [(axes[0],p['cases'][:3],'residual_local'),(axes[1],p['cases'][3:],'dna_left')]:
    for case in cases:
        values=np.array([curves[(case,seed)][key] for seed in p['seeds']]);ax.plot(time_axis,values.mean(axis=0),label=labels[case])
    ax.axvspan(.3,.8,color='gray',alpha=.1);ax.axvline(.8,color='gray',ls='--');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0].set_ylabel('Four selected local neurons mean Hz');axes[1].set_ylabel('Left DNa02 Hz');axes[1].set_xlabel('Simulated seconds')
fig.suptitle('Separate late source cuts — two-seed means, model diagnostics only');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
table='\n'.join(f"| {s['case']} | {s['seed']} | {s['tail_residual_local_mean_hz']:.1f} | {s['tail_alln_mean_hz']:.1f} | {s['tail_pn_hz']:.1f} | {s['tail_dna_left_hz']:.1f} |" for s in summaries)
(OUT/'RESULTS.md').write_text(f'''# Residual-local and steering input diagnostics

Completed fifteen four-second full-network trials. Strong target suppression in
both seeds occurred for: {', '.join(strong) if strong else 'none'}.
No diagnostic variant is promoted; original application sources remain unchanged.

## Final half-second activity

| Condition | Seed | Selected local mean Hz | All local mean Hz | DM1 PN mean Hz | Left steering Hz |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

Local conditions already cut positive ALLN-to-ALLN recurrence at .8s, then compare
additional ALPN-to-ALLN or all non-ALLN positive inputs. Steering conditions already
silence DM1 PNs by cutting positive ALLN-to-PN input, then compare PS013, four
selected exact sources, or all positive input to left DNa02. All other weights,
neuron state, input events and decoder remain fixed. The two reference conditions
are different diagnostic backgrounds, not intact controls.

## Selection and claims

Four local targets and four steering sources were selected from the pooled prior
published top12 source lists, using11401/11402. This is a bounded candidate pool,
not a claim to a complete pooled all-source ranking. The new seeds11501/11502
are diagnostic probes, not held-out navigation validation. The exact selection
scores, annotations and root IDs were frozen before runs in protocol.json.

Prior local-target input ranking used nominal spike-count-weighted arrivals
because their gates had not been recorded. New data measure actual accepted
inputs into all four targets. results.json distinguishes that accepted source
attribution from the prior nominal hypothesis. ALPN and ALLN membership use
exact cell-class annotations. Missing steering-source cell classes remain
unavailable; exact cell-type labels do not establish function or physiology.

Strong suppression means final target-group mean at most10% of its matched
reference, with reference above10Hz, in both seeds. Individual local-neuron
rates are retained so a mean cannot conceal a surviving member. This screen
shows modeled dependence on an edge set, not a unique minimal biological loop.
The all-positive-input cuts are sanity controls, not proposed brain models.

## Verification

All2,400 chunk hashes were checked. Raw spike events matched monitor counts;
seeded external inputs and fixed motor decoding were independently reconstructed.
Before .8s all variants matched their common pre-intervention state exactly.
The observer-disabled steering-reference replay matched spikes, counts, inputs,
commands, selected v/g and100μs refractory gates exactly.

Accepted positive/negative increments were independently reconstructed with
1.8ms synaptic delay and actual gates, including chunk boundaries; every window
met the1e-6mV tolerance. These are modeled voltage increments, not membrane
currents. Expected whole weight-vector hashes verified that only selected
connections changed. model-variants.json distinguishes each diagnostic graph
from its common base-controller manifest. lesion-map.json retains exact root
IDs and anatomical counts; no reduced controller replaces the full network.

Wall time over all trials: {r['trial_wall_seconds']:.1f}s. Peak worker RSS:
{r['peak_rss_bytes']/1024**3:.2f}GiB. No body, reward or learning evaluation ran.
The plot averages two seeds without confidence-interval claims.

Next decisions must distinguish source dependence from physiologically justified
model revision. Keep failures and persistent activity visible; do not promote a
cut graph or infer navigation, learning or consciousness from these diagnostics.
''')
print(json.dumps({'strong_suppression':strong,'summary':summaries,'local_attribution':local_attribution},indent=2))
