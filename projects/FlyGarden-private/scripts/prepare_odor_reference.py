"""Build an attributed, exact-root-ID sensory reference; never modify the live model."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_sensorimotor_pathway import mapping
from scripts.audit_brain_reference import sha
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/stage5-20261006'
def prepare():
 manifest=json.loads((OUT/'retrieval-manifest.json').read_text())
 for a in manifest['artifacts']:assert sha(OUT/'sources'/a['file'])==a['sha256']
 ids,_,ann=mapping();order={int(v):i for i,v in enumerate(ids)}
 maps=pd.read_csv(OUT/'sources/door_mappings.csv',sep=';');units=[]
 for a in manifest['artifacts']:
  name=a['file']
  if name in ('DESCRIPTION','door_mappings.csv','door_dataset_info.csv'):continue
  unit=name[:-4];m=maps[maps.receptor.eq(unit)|maps.OSN.eq(unit)]
  gloms=m.glomerulus.dropna().unique();assert len(gloms)==1 and m.adult.eq(True).any(),unit
  glom=gloms[0];rows=ann[ann.cell_type.eq('ORN_'+glom)&ann.side.isin(('left','right'))]
  assert len(rows)>0,glom
  table=pd.read_csv(OUT/'sources'/name,sep=';');sfr=float(table.loc[table.InChIKey.eq('SFR'),'Bruyne.2001.WT'].iloc[0]);responses={}
  for chemical in ('ethyl acetate','ethyl pentanoate','oil'):
   r=table[table.Name.eq(chemical)]
   assert len(r)==1,(unit,chemical)
   delta=float(r['Bruyne.2001.WT'].iloc[0]);assert np.isfinite(delta),(unit,chemical)
   responses[chemical]={'delta_hz':delta,'absolute_hz':max(0,sfr+delta),'inchikey':r.InChIKey.iloc[0]}
  units.append({'unit':unit,'glomerulus':glom,'baseline_hz':sfr,'responses':responses,'neurons':[{'root_id':str(int(r.root_id)),'index':order[int(r.root_id)],'side':r.side} for r in rows.itertuples()]})
 all_orn=ann[ann.cell_type.str.startswith('ORN')];covered=sum(len(u['neurons']) for u in units)
 result={'schema_version':1,'source_revision':manifest['door_revision'],'license':'CC BY-SA 4.0','attribution':'DoOR.data; Galizia et al. 2010; Muench and Galizia 2016; de Bruyne et al. 2001','source_study':'https://doi.org/10.1016/S0896-6273(01)00289-6','dataset':'Bruyne.2001.WT','measurement':'baseline-subtracted mean spikes/s; solvent NOT subtracted','concentration':'10^-2 lab dilution; no arena concentration calibration','encoding':'Engineered linear exposure interpolation from spontaneous baseline to measured absolute rate; rectangular 500ms pulses; no temporal adaptation claim','modeled_neurons':len(ids),'annotated_orn_count':len(all_orn),'covered_orn_count':covered,'units':units,'neuron_order_sha256':sha(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv'),'retrieval_manifest_sha256':sha(OUT/'retrieval-manifest.json')}
 atomic_json(OUT/'odor-reference.json',result);print(covered,'of',len(all_orn),'annotated ORNs mapped')
 atomic_json(OUT/'parameter-audit.json',{'published_source':'https://www.nature.com/articles/s41586-024-07763-9','rest_reset_mV':-52,'threshold_mV':-45,'membrane_ms':20,'refractory_ms':2.2,'synapse_ms':5,'delay_ms':1.8,'synaptic_gain_mV':.275,'parameter_changes':[],'reference_parity_report':'../stage1-20261005-214341/reference-comparison.json','limitations':['Published calibration used release 630; application imports 783.','Published synaptic gain calibrated to feeding response, not walking.','No evidence obtained for DNa02 or AOTU019 specific intrinsic parameter changes.','DNa02 raw recording archive API returned HTTP403; no quantitative physiological fit claimed.','MN9 annotation absent in local annotation table; feeding calibration not reproduced.']})
if __name__=='__main__':prepare()
