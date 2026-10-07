"""Recorded local-state summaries; descriptive, not causal efficacy claims."""
import json
from pathlib import Path
import numpy as np
from flygarden.recording import atomic_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/brain-integration/recovery/local-inhibition-full-network-v1'
def main():
 assert json.loads((OUT/'progress.json').read_text())['status']=='complete'
 p=json.loads((OUT/'protocol.json').read_text());routing=json.loads((ROOT/'reports/brain-integration/recovery/local-inhibition-adapter-v1/routes.json').read_text());nodes=routing['nodes'];regions={g:[i for i,key in enumerate(nodes) if key.endswith('@'+g)] for g in ['left:DM1','right:DM1']}
 targets={x['root_id']:k for k in ['DM1_lPN_left','DM1_lPN_right'] for x in p['mapping'][k]};output_counts={}
 for row in routing['routes']:
  if row['pre_node']>=0 and row['post_node']<0 and row['post_root_id'] in targets:
   target=targets[row['post_root_id']];region=nodes[row['pre_node']].split('@',1)[1];key=target+' <- '+region;output_counts[key]=output_counts.get(key,0)+row['site_count']
 summaries=[]
 for folder in sorted((OUT/'trials').glob('local-*')):
  m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete';activities=[];filtered=[]
  for chunk in m['chunks']:
   with np.load(folder/chunk['file']) as z:activities.append(z['local_activity'].copy());filtered.append(z['local_filtered_drive'].copy())
  a=np.array(activities);d=np.array(filtered);summary={'trial':folder.name,'maximum_activity':float(a.max()),'maximum_filtered_drive':float(d.max()),'nonzero_activity_nodes':int((a.max(axis=0)>0).sum()),'regional_mean_release_proxy_hz':{}}
  for region,indices in regions.items():
   summary['regional_mean_release_proxy_hz'][region]={'nodes':len(indices),'first_pulse':float(a[12:32,indices].mean()*100),'first_recovery':float(a[100:120,indices].mean()*100),'second_pulse':float(a[132:152,indices].mean()*100),'second_recovery':float(a[220:240,indices].mean()*100)}
  summaries.append(summary)
 atomic_json(OUT/'bridge-engagement.json',{'status':'recorded_descriptive_summary','state_sample_interval_s':.025,'recorded_local_state_and_filtered_drive':True,'release_units':'Engineered normalized activity times100Hz, not measured transmitter release','not_a_causal_attribution':True,'DM1_PN_local_output_site_counts':output_counts,'trials':summaries})
 print('Recorded local-state summary written.')
if __name__=='__main__':main()
