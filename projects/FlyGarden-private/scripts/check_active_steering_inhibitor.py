"""Exploratory follow-up selected from measured primary source deliveries.

Runs only after primary completion. Existing seeds are matched controls, not
unseen validation. Zeroing only the largest active negative source group into
right DNa02 is diagnostic, never a proposed controller.
"""
import sys,json,subprocess,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_brain_reference import sha
from flygarden.recording import atomic_json,space_check
from scripts.diagnose_steering_bias import worker
from scripts.report_steering_bias import inspect_trial

def run(root):
 main=json.loads((root/'progress.json').read_text());assert main['status']=='complete'
 protocol=json.loads((root/'protocol.json').read_text());right=protocol['DNa02_targets'][1]
 ranking={}
 for seed in (4101,4102):
  for cue in ('left','right'):
   file=root/'trials'/f'intact_narrow-{cue}-{seed}'/'source-deliveries.json'
   rows=json.loads(file.read_text())['contributions']
   for r in rows:
    if r['post_index']==right and r['effective_weight_mV']<0 and r['pre_cell_type']:
     ranking[r['pre_cell_type']]=ranking.get(r['pre_cell_type'],0)+r['pulse_delivered_mV']
 source=max(ranking,key=ranking.get);profile='active_right_negative_source_off';follow=root/'active-source-followup';follow.mkdir(exist_ok=False)
 edges=[r['edge_index'] for r in protocol['incoming_edges'] if r['post_index']==right and r['signed_count']<0 and r['pre_cell_type']==source]
 assert edges
 fork=copy.deepcopy(protocol);fork['sources']['scripts/check_active_steering_inhibitor.py']=sha(Path(__file__));fork['profiles']=[profile];fork['lesions'][profile]={'edge_indices':edges,'records_zeroed':len(edges),'intervention':'Exploratory diagnostic, selected from actual calibration source deliveries; right target only.'};fork['exploratory_followup']={'source_cell_type':source,'ranking_from_primary_calibration_seeds':ranking,'selected_before_followup':True,'matched_control_root':str(root),'seeds_reused':True,'held_out_validation_claim':False,'target_root_id':protocol['DNa02_root_ids'][1],'scope':'Model-specific causal follow-up; no biological knockout or correction claim.'};atomic_json(follow/'protocol.json',fork)
 results=[]
 for cue in ('left','right'):
  for seed in protocol['seeds']:
   log=follow/f'{cue}-{seed}.log';space_check(follow,32*1024**2)
   with log.open('w') as output:subprocess.run([sys.executable,str(Path(__file__)),str(follow),'--worker',profile,cue,str(seed)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
   trial=inspect_trial(follow,f'{profile}-{cue}-{seed}',fork);control=json.loads((root/'trials'/f'intact_narrow-{cue}-{seed}'/'summary.json').read_text());trial['matched_control_pulse_DNa02_hz']=control['pulse_DNa02_hz'];results.append(trial);print(source,cue,seed,trial['pulse_DNa02_hz'],flush=True)
 atomic_json(follow/'analysis.json',{'status':'complete','exploratory':True,'source_cell_type':source,'target_root_id':protocol['DNa02_root_ids'][1],'records_zeroed':len(edges),'mean_control_right_pulse_hz':float(np.mean([r['matched_control_pulse_DNa02_hz'][1] for r in results])),'mean_lesioned_right_pulse_hz':float(np.mean([r['pulse_DNa02_hz'][1] for r in results])),'source_attribution_passed':all(r['attribution_matches_observer'] for r in results),'windows':sum(r['windows'] for r in results),'raw_spikes':sum(r['spikes'] for r in results),'trials':results,'promoted':False,'independent_held_out_validation':False});print('Exploratory follow-up complete',flush=True)
if __name__=='__main__':
 if len(sys.argv)>2 and sys.argv[2]=='--worker':worker(Path(sys.argv[1]),sys.argv[3],sys.argv[4],int(sys.argv[5]))
 else:run(Path(sys.argv[1]))
