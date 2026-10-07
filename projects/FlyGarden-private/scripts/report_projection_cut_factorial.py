"""Independent raw-spike/delay/gate delivery reconstruction and lesion report."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
from flygarden.descending import DescendingDecoder
OUT=ROOT/'reports/brain-integration/recovery/projection-cut-factorial-v1'
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
    'variants':[{'id':'projection-factorial-'+case+'-v1','case':case,'neurons':138639,'anatomical_graph_records':15091983,
                 'base_neural_source_sha256':p['sources']['flygarden/brain.py'],
                 'signed_transport_weights_modified':bool(p['lesions'][case]['rows']),'intervention_seconds':.8,
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
        if case in ('intact','alln_recurrence_off'):
            bypre={e['pre']:e for e in edges};pncols=[target_col[n] for k in ('DM1_lPN_left','DM1_lPN_right') for n in mapping[k]]
            values=late_source_pos[:,pncols].sum(axis=1)/.5
            groups={}
            for pre in np.flatnonzero(values):
                e=bypre[int(pre)];name=e['pre_cell_class'] or 'unavailable'
                groups[name]=groups.get(name,0)+float(values[pre])
            rankings.append({'case':case,'seed':seed,'target':'both DM1 lPNs','phase':'3.5-4.0s','units':'accepted summed voltage increments mV/s, not currents',
                             'positive_by_cell_class':dict(sorted(groups.items(),key=lambda kv:-kv[1])),
                             'top_positive_sources':[{'root_id':bypre[int(i)]['pre_root_id'],'cell_type':bypre[int(i)]['pre_cell_type'],'cell_class':bypre[int(i)]['pre_cell_class'],'accepted_mV_per_second':float(values[i])} for i in np.argsort(values)[-12:][::-1] if values[i]>0]})
        if True:
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
for x in summaries:
    t=curves[(x['case'],x['seed'])];x['tail_local_mean_hz']=float(t['residual_local'][140:].mean());x['tail_alpn_mean_hz']=float(t['ALPN'][140:].mean())
    counts=np.zeros(138639,dtype=np.int64);folder=OUT/'trials'/f"{x['case']}-{x['seed']}";m=json.loads((folder/'manifest.json').read_text())
    for chunk in m['chunks'][140:]:
        with np.load(folder/chunk['file']) as z:counts+=z['counts']
    x['tail_local_individual_hz']={str(n['root_id']):float(counts[n['index']]/.5) for n in p['selected_local_targets']}
    if x['case']!='intact':
        ref=next(v for v in summaries if v['case']=='intact' and v['seed']==x['seed']);a=x['tail_local_mean_hz'];b=ref['tail_local_mean_hz']
        comparisons.append({'case':x['case'],'seed':x['seed'],'reference_hz':b,'target_hz':a,'fraction':a/b if b else None,'strong_suppression':bool(b>10 and a<=.1*b)})
strong=[c for c in p['cases'] if c!='intact' and all(x['strong_suppression'] for x in comparisons if x['case']==c)]
r={'status':'complete','trials':9,'chunks_audited':1440,'instrumentation_parity_exact':True,'preintervention_trajectories_exact':True,'matched_external_inputs_exact':True,'accepted_delivery_reconstruction_passed':True,'all_nonselected_weights_match_source':True,'summary':summaries,'comparisons':comparisons,'strong_suppression_both_seeds':strong,'source_rankings':rankings,'audits':audits,'promoted':False,'navigation_validated':False,'learning_validated':False,'biological_cause_proven':False,'trial_wall_seconds':sum(x['wall_seconds'] for x in metadata),'peak_rss_bytes':max(x['peak_rss_bytes'] for x in metadata),'report_sha256':file_sha(Path(__file__))}
atomic_json(OUT/'results.json',r)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
labels={'intact':'Intact','alpn_to_alln_off':'Projection-to-local cut only','alln_recurrence_off':'Local recurrence cut only','joint_off':'Both cuts'}
for ax,key,title in zip(axes,['residual_local','pn','dna_left'],['Selected local neurons mean Hz','DM1 projection neurons mean Hz','Left steering neuron Hz']):
    for c in p['cases']:
        v=np.array([curves[(c,seed)][key] for seed in p['seeds']]);ax.plot(np.arange(1,161)*.025,v.mean(axis=0),label=labels[c])
    ax.axvspan(.3,.8,color='gray',alpha=.1);ax.axvline(.8,color='gray',ls='--');ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[-1].set_xlabel('Simulated seconds');fig.suptitle('Two-by-two diagnostic source cuts: two-seed means, no biological repair claim');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
table='\n'.join(f"| {x['case']} | {x['seed']} | {x['tail_local_mean_hz']:.1f} | {x['tail_alln_mean_hz']:.1f} | {x['tail_alpn_mean_hz']:.1f} | {x['tail_pn_hz']:.1f} | {x['tail_dna_left_hz']:.1f} |" for x in summaries)
(OUT/'RESULTS.md').write_text(f"""# Projection-input and local-recurrence factorial diagnostic

Nine four-second full-network runs completed. Strong selected-local suppression against intact in both seeds: {', '.join(strong) or 'none'}.

| Condition | Seed | Four local mean Hz | ALLN mean Hz | ALPN mean Hz | DM1 PN mean Hz | Left DNa02 Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Positive ALPN-to-ALLN and positive ALLN-to-ALLN transport cuts form a two-by-two comparison. They occur at .8 seconds after the same left odor pulse (.3-.8 seconds). The intact condition retains all weights. Input events, other weights, neural states and fixed decoder are unchanged at intervention. All four selected individual local rates are preserved in results.json. The frozen strong-suppression threshold is at most10% of intact final-half-second selected-local rate, with intact above10Hz, in both seeds. Diagnostic seeds11601/11602 do not establish navigation performance or biological validity.

All1,440 chunk hashes passed. Raw spikes matched counts; inputs and decoding were independently reconstructed. Before .8 seconds trajectories matched exactly. Observer-disabled intact replay matched neural events, inputs, commands and selected v/g/gates exactly. Signed accepted increments were independently reconstructed from delayed spikes and refractory gates across chunk boundaries; all errors were below1e-6mV. Whole-vector hashes verify only specified weights changed. These increments are not biological membrane currents. No learning or body evaluation ran, and no variant is promoted.

Trial wall time: {r['trial_wall_seconds']:.1f}s. Peak worker RSS: {r['peak_rss_bytes']/1024**3:.2f}GiB. The plot averages two seeds without confidence-interval claims. Original application sources remain unchanged.
""")
print(json.dumps({'strong':strong,'summary':[{k:v for k,v in x.items() if not k.startswith('accepted_')} for x in summaries]},indent=2))
