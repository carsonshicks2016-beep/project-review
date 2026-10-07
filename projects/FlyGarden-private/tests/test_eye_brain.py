import numpy as np
from scripts.diagnose_eye_brain import image_rates,populations

def test_image_stimulation_uses_completed_past_window_and_blanks():
 for tick in range(30):
  rate,times=image_rates('loom_left',tick)
  assert all(t<=tick*.1-1/30+1e-9 for t in times)
  assert rate[1]==0
  blank,_=image_rates('blanked_loom_left',tick);assert np.array_equal(blank,[0,0])
 assert image_rates('loom_left',6)[0][0]>0
 assert np.array_equal(image_rates('loom_left',0)[0],[0,0])
def test_visual_targets_are_exact_annotated_roots():
 ids,pops=populations();all_ids=[]
 for side in ('left','right'):
  neurons=pops['LPLC2_'+side];assert len(neurons)==(108 if side=='left' else 102)
  for n in neurons:
   assert n['cell_type']=='LPLC2' and n['side']==side and str(ids[n['index']])==n['root_id'];all_ids.append(n['root_id'])
 assert len(set(all_ids))==210
