"""Scientific activity trace from verified saved events only."""
import sys,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flygarden.continuous_candidate import file_sha
p=ROOT/'reports/brain-integration/recovery/persistence-reference-v1';s=json.loads((p/'protocol.json').read_text());result=json.loads((p/'results.json').read_text());assert result['status']=='passed'
fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True)
for col,support in enumerate((0,65)):
 f=p/f'candidate-10101-support{support}';m=json.loads((f/'manifest.json').read_text());rates={k:[] for k in ('DM1_lPN_left','DM1_lPN_right','DNa02_left','DNa02_right')}
 for chunk in m['chunks']:
  path=f/chunk['file'];assert file_sha(path)==chunk['sha256']
  with np.load(path) as z:
   for key in rates:rates[key].append(z['counts'][[n['index'] for n in s['mapping'][key]]].mean()/.025)
 time=(np.arange(160)+1)*.025
 for row,pop in enumerate(('DM1_lPN','DNa02')):
  ax=axes[row,col]
  for side,color in [('left','#db4437'),('right','#2878b8')]:
   values=np.array(rates[pop+'_'+side]);means=np.array([values[max(0,k-3):k+1].mean() for k in range(160)]);ax.plot(time,means,color=color,label=side)
  ax.axvspan(.3,.8,color='#e8b54c',alpha=.2,label='odor input');ax.axvline(.8,color='#aaa',linestyle=':');ax.set_ylabel(pop+' mean Hz');ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2);ax.set_xlim(0,4)
 axes[0,col].set_title(f'Walking support {support} Hz; seed 10101')
 axes[1,col].set_xlabel('Simulated seconds')
fig.suptitle('Persistent activity after odor removal — reference replay matches candidate\nTrailing 100 ms averages for display; no smoothing added to the model',fontsize=12)
fig.tight_layout();fig.savefig(p/'persistent-activity.png',dpi=160);plt.close(fig)
