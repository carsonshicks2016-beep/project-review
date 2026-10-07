"""Read preserved causal trials without fitting a new controller."""
import sys,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/recovery/odor-bias-diagnosis'
SOURCE=ROOT/'reports/brain-integration/recovery/odor-causal-v2'
KEYS=('ORN_DM1_left','ORN_DM1_right','DM1_lPN_left','DM1_lPN_right','DNa02_left','DNa02_right','DNp09_left','DNp09_right','DNg13_left','DNg13_right')
PHASES={'pre_cue':(0,.5),'cue':(.5,1.5),'recovery':(1.5,3)}
records=[];traces={};hashes={}
for seed in range(9601,9621):
 for arm in ('intact','sensory_off','steering_cut'):
  folder=SOURCE/'trials'/str(seed)/arm;manifest=json.loads((folder/'manifest.json').read_text());assert manifest['status']=='complete'
  rows=json.loads((folder/'rows.json').read_text());hashes[str(seed)+'/'+arm]=file_sha(folder/'manifest.json')
  times=np.array([r['start'] for r in rows]);rates=np.array([[r['population_hz'][k] for k in KEYS] for r in rows]);exposure=np.array([r['sensory']['antenna_odors'] for r in rows])[:,:,0]
  requested=np.array([[r['requested_hz']['ORN_DM1_'+s] for s in ('left','right')] for r in rows]);assert np.allclose(requested,exposure*50 if arm!='sensory_off' else np.zeros_like(exposure),rtol=0,atol=1e-12)
  yaw=np.unwrap([manifest['initial']['heading']]+[r['body']['heading'] for r in rows]);motors=np.array([r['applied_motor'] for r in rows]);delivered=np.zeros((len(rows),8),dtype=int)
  channels=np.array(manifest['controller']['input_channels']);perchannel=np.bincount(channels,minlength=8)
  for i,c in enumerate(manifest['chunks']):
   with np.load(folder/c['file'],allow_pickle=False) as z:delivered[i]=np.bincount(channels[z['external_i']],minlength=8)
  phases={}
  for label,(start,end) in PHASES.items():
   mask=(times>=start-1e-9)&(times<end-1e-9);idx=np.flatnonzero(mask)
   phases[label]={'population_hz':dict(zip(KEYS,rates[mask].mean(axis=0).tolist())),
    'antenna_odor_mean':exposure[mask].mean(axis=0).tolist(),'requested_orn_mean_hz':requested[mask].mean(axis=0).tolist(),
    'delivered_hz_per_root':(delivered[mask].sum(axis=0)/(len(idx)*.025*perchannel)).tolist(),
    'applied_drive_difference_mean':float(np.mean(motors[mask,1]-motors[mask,0])),
    'yaw_change_rad':float(yaw[idx[-1]+1]-yaw[idx[0]])}
  records.append({'seed':seed,'arm':arm,'cue':manifest['cue'],'phases':phases,'population_sizes':{k:len(v) for k,v in manifest['controller']['population_mapping'].items()}})
  key=(manifest['cue'],arm);traces.setdefault(key,[]).append({'rates':rates,'exposure':exposure,'motors':motors,'yaw':yaw[1:]-yaw[0]})
summary={}
for cue in ('odor_left','odor_right'):
 for arm in ('intact','sensory_off','steering_cut'):
  selected=[r for r in records if r['cue']==cue and r['arm']==arm]
  summary[cue+'/'+arm]={p:{'population_hz':{k:float(np.mean([r['phases'][p]['population_hz'][k] for r in selected])) for k in KEYS},
     'antenna_odor_mean':np.mean([r['phases'][p]['antenna_odor_mean'] for r in selected],axis=0).tolist(),
     'drive_difference':float(np.mean([r['phases'][p]['applied_drive_difference_mean'] for r in selected])),
     'yaw_change_rad':float(np.mean([r['phases'][p]['yaw_change_rad'] for r in selected]))} for p in PHASES}
atomic_json(OUT/'recorded-analysis.json',{'status':'complete','source_manifests':hashes,'analyzer_sha256':file_sha(Path(__file__)),'records':records,'summary':summary,'scope':'Retrospective localization; no new acceptance test, no decoder fitting, no biological symmetry assumption. Odor exposure at interval start, neural rates at interval end, motor interval statistics and body yaw at interval end.'})
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(5,2,figsize=(13,13),sharex=True)
for col,cue in enumerate(('odor_left','odor_right')):
 intact=traces[(cue,'intact')];t=times+.025
 for s,name in enumerate(('L','R')):
  axes[0,col].plot(times,np.mean([r['exposure'][:,s] for r in intact],axis=0),label='antenna '+name)
  axes[1,col].plot(t,np.mean([r['rates'][:,2+s] for r in intact],axis=0),label='DM1 lPN '+name)
  axes[2,col].plot(t,np.mean([r['rates'][:,4+s] for r in intact],axis=0),label='DNa02 '+name)
 for arm in ('intact','sensory_off','steering_cut'):
  group=traces[(cue,arm)];axes[3,col].plot(t,np.mean([r['motors'][:,1]-r['motors'][:,0] for r in group],axis=0),label=arm)
  axes[4,col].plot(t,np.mean([r['yaw'] for r in group],axis=0),label=arm)
 for row in range(5):
  axes[row,col].axvspan(.5,1.5,color='#d1e2bc',alpha=.3);axes[row,col].legend(fontsize=8);axes[row,col].grid(alpha=.2)
 axes[0,col].set_title(cue+' · 10 seeds');axes[4,col].set_xlabel('simulated seconds')
for row,label in enumerate(('odor concentration','projection-neuron Hz','descending-neuron Hz','right minus left drive','unwrapped yaw change, rad')):axes[row,0].set_ylabel(label)
fig.suptitle('Preserved odor trials: local exposure → neural response → movement\nSame frozen controller; retrospective diagnosis, not a new success claim')
fig.tight_layout();fig.savefig(OUT/'recorded-chain.png',dpi=140);plt.close(fig)
for k,v in summary.items():
 print(k,'cue DNa02',v['cue']['population_hz']['DNa02_left'],v['cue']['population_hz']['DNa02_right'],'odor',v['cue']['antenna_odor_mean'],'recovery yaw',v['recovery']['yaw_change_rad'])
