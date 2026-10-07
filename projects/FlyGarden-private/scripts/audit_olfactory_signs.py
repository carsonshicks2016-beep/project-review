"""Exact-root imported sign/annotation audit, with no model mutations."""
import sys,json,hashlib,collections
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
OUT=ROOT/'reports/brain-integration/recovery/projection-cut-factorial-v1'
p=json.loads((OUT/'protocol.json').read_text());assert all(file_sha(ROOT/n)==h for n,h in p['sources'].items())
ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64)
a=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('');assert not a.root_id.duplicated().any();a=a.set_index('root_id').reindex(ids).fillna('')
c=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet');pre=c.Presynaptic_Index.to_numpy(dtype=int);post=c.Postsynaptic_Index.to_numpy(dtype=int);sgn=c.Excitatory.to_numpy();count=c.Connectivity.to_numpy();signed=c['Excitatory x Connectivity'].to_numpy()
assert np.array_equal(ids[pre],c.Presynaptic_ID.to_numpy()) and np.array_equal(ids[post],c.Postsynaptic_ID.to_numpy());assert np.array_equal(signed,sgn*count);assert set(np.unique(sgn))<={-1,1}
positive=np.bincount(pre[sgn>0],minlength=len(ids));negative=np.bincount(pre[sgn<0],minlength=len(ids));mixed=np.flatnonzero((positive>0)&(negative>0));model_sign=np.where(positive>0,1,np.where(negative>0,-1,0))
# Latest annotation top-label comparison is not a reconstruction of the upstream
# cleft-filtered site-majority rule. Report disagreements without changing signs.
expected=a.top_nt.map({'acetylcholine':1,'gaba':-1,'glutamate':-1,'dopamine':1,'serotonin':1,'octopamine':1}).fillna(0).to_numpy(dtype=int)
conf=pd.to_numeric(a.top_nt_conf,errors='coerce').to_numpy()
def known_sign(value):
 # Negative immunostaining is absence evidence, not transmitter identity.
 tokens={t.strip().lower() for t in value.replace(';',',').split(',') if t.strip() and not t.strip().lower().endswith('-negative')}
 signs={v for k,v in {'acetylcholine':1,'gaba':-1,'glutamate':-1,'dopamine':1,'serotonin':1,'octopamine':1}.items() if k in tokens}
 return next(iter(signs)) if len(signs)==1 else 0
known=np.array([known_sign(v) for v in a.known_nt],dtype=int)
valid=(model_sign!=0)&(expected!=0);disagree=valid&(model_sign!=expected)
alln=np.array([n['index'] for n in p['mapping']['ALLN']]);alpn=np.array([n['index'] for n in p['mapping']['ALPN']]);module=np.union1d(alln,alpn)
def roots(indices):
 return [{'root_id':str(ids[i]),'index':int(i),'cell_type':str(a.iloc[i].cell_type),'cell_class':str(a.iloc[i].cell_class),'side':str(a.iloc[i].side),'model_outgoing_sign':int(model_sign[i]),'outgoing_positive_records':int(positive[i]),'outgoing_negative_records':int(negative[i]),'annotation_top_nt':str(a.iloc[i].top_nt),'annotation_confidence':None if not np.isfinite(conf[i]) else float(conf[i]),'known_nt':str(a.iloc[i].known_nt),'known_nt_source':str(a.iloc[i].known_nt_source),'top_label_rule_disagreement':bool(disagree[i]),'known_nt_rule_disagreement':bool(known[i]!=0 and model_sign[i]!=0 and known[i]!=model_sign[i])} for i in indices]
def stats(indices):
 return {'neurons':len(indices),'model_positive':int((model_sign[indices]>0).sum()),'model_negative':int((model_sign[indices]<0).sum()),'no_outgoing':int((model_sign[indices]==0).sum()),'annotation_top_nt':dict(collections.Counter(a.iloc[indices].top_nt)),'low_top_confidence_below_half':int((conf[indices]<.5).sum()),'top_label_rule_disagreements':int(disagree[indices].sum()),'known_nt_labels':dict(collections.Counter(a.iloc[indices].known_nt)),'known_nt_resolved_rule_comparisons':int(((known[indices]!=0)&(model_sign[indices]!=0)).sum()),'known_nt_rule_disagreements':int(((known[indices]!=0)&(model_sign[indices]!=0)&(known[indices]!=model_sign[indices])).sum())}
sel=np.array([n['index'] for n in p['selected_local_targets']]);masks={'ALPN_to_ALLN':np.isin(pre,alpn)&np.isin(post,alln),'ALLN_to_ALLN':np.isin(pre,alln)&np.isin(post,alln),'ALLN_to_ALPN':np.isin(pre,alln)&np.isin(post,alpn)}
edges={}
for name,mask in masks.items():
 edges[name]={}
 for label,sign in [('positive',1),('negative',-1)]:
  q=mask&(sgn==sign);edges[name][label]={'aggregated_records':int(q.sum()),'anatomical_synapses':int(count[q].sum()),'source_roots':int(len(np.unique(pre[q]))),'target_roots':int(len(np.unique(post[q])))}
r={'status':'complete','ordering_ids_match_both_graph_endpoints':True,'signed_equals_count_times_sign':True,'mixed_outgoing_sign_roots':roots(mixed),'ALLN':stats(alln),'ALPN':stats(alpn),'selected_local_targets':roots(sel),'module_top_label_disagreements':roots(module[disagree[module]]),'module_known_nt_disagreements':roots(module[(known[module]!=0)&(model_sign[module]!=0)&(known[module]!=model_sign[module])]),'module_neurons':roots(module),'edge_groups':edges,'interpretation':'A top-label comparison of a later annotation table is not the original cleft-filtered synapse-site-majority sign algorithm. Disagreements are audit leads, not sign corrections. known_nt entries are source annotations, not experimental measurements of these exact roots. No receptor-specific evidence or exact-root parameter fits supplied. No model mutation.','source_hashes':{n:file_sha(ROOT/n) for n in ('flygarden/brain.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet','vendor/fly-brain/code/paper-phil-drosophila/model.py')},'sources':[{'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/','doi':'10.1038/s41586-024-07763-9','local_xml_sha256':file_sha(OUT/'literature/shiu-2024.xml')},{'url':'https://pubmed.ncbi.nlm.nih.gov/20869598/','doi':'10.1016/j.neuron.2010.08.025','scope':'Reciprocal excitatory eLN/PN pathways; no exact FlyWire-root mapping.'},{'url':'https://pubmed.ncbi.nlm.nih.gov/20505124/','scope':'Local-neuron diversity; cannot assign all local neurons inhibitory.'}], 'audit_script_sha256':file_sha(Path(__file__))}
core=(ROOT/'flygarden/brain.py').read_text();up=(ROOT/'vendor/fly-brain/code/paper-phil-drosophila/model.py').read_text()
checks={'rest_reset_minus52':"'v_0'       : -52 * mV" in up and 'self.neurons.v=-52*b.mV' in core,'threshold_minus45':"'v_th'      : -45 * mV" in up and "threshold='v > -45*mV'" in core,'membrane20ms':"'t_mbr'     :  20 * ms" in up and '/(20*ms)' in core,'synaptic5ms':"'tau'       : 5 * ms" in up and '-g/(5*ms)' in core,'refractory2_2ms':"'t_rfc'     : 2.2 * ms" in up and 'self.neurons.rfc=2.2*b.ms' in core,'delay1_8ms':"'t_dly'     : 1.8*ms" in up and 'delay=1.8*b.ms' in core,'weight0_275mV':"'w_syn'     : .275 * mV" in up and '0.275*b.mV' in core,'both_derivatives_refractory_gated':up.count('(unless refractory)')>=2 and core.count('(unless refractory)')==2}
assert all(checks.values());r['parameter_source_checks']=checks
r['parameter_scope']='Uniform released-model constants/equations preserved for local/projection cells; no measured cell-specific parameter fit. Source undefined w reset removed; input refractory exceptions and owned Bernoulli generator are disclosed engineered adapter deviations. No adaptation, compartments or receptor-specific kinetics represented.'
r['additional_literature']=json.loads((OUT/'literature/additional-sources.json').read_text())
r['known_nt_disagreement_scope']='Compound markers parsed explicitly; negative staining excluded. Twelve lLN2P_b GABA/MIP annotations additionally conflict with positive model signs; annotation-to-physiology mapping remains qualified. The two il3LN6 GABA annotations conflict with modeled positive sign and have primary cell-type inhibitory evidence. For the two v2LN36 glutamate annotations, the disagreement is against the published-model inhibitory-glutamate assumption; transmitter identity alone does not prove every target receptor sign.'
atomic_json(OUT/'sign-audit.json',r)
print(json.dumps({k:v for k,v in r.items() if k not in ('module_neurons','module_top_label_disagreements','module_known_nt_disagreements')},indent=2))
