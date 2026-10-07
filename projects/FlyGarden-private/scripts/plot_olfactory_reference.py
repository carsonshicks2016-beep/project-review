"""Inspect independent reproduction against external published measurements."""
import sys,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'reports/brain-integration/recovery/olfactory-dynamics-reference-v1'
d=json.loads((OUT/'figure4-comparison.json').read_text());fig,ax=plt.subplots(1,2,figsize=(12,4.8));colors=['#00134a','#004991','#0083c9','#80bdd9']
for color,c in zip(colors,d['curves']):
 K=c['K']
 with np.load(OUT/f'ramp-{K:g}.npz') as z:ax[0].plot(z['time'],z['state'][:,0],color=color,label=f'{K:g} Hz/s')
 ax[0].scatter([r['time_s'] for r in c['samples']],[r['PN_hz'] for r in c['samples']],color=color,s=8,alpha=.6)
ax[0].axhline(95.2829735635,color='#b13b3b',linestyle='--',label='Equation 15 asymptote');ax[0].set_title('Published Figure 4 reproduction\nLines: independent solver; dots: source raster');ax[0].set_xlabel('Seconds');ax[0].set_ylabel('PN firing rate (Hz)');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
for color,peak in zip(['#df3d36','#dca51b','#149f7a','#34249e'],(.1,.4,1.1,1.8)):
 with np.load(OUT/f'triangle-{peak:g}.npz') as z:ax[1].plot(z['time'],z['state'][:,0],color=color,label=f'Input peak at {peak:g} s')
ax[1].set_xlim(0,4);ax[1].set_title('Triangle input and odor-free recovery\nPublished parameters; no fitted changes');ax[1].set_xlabel('Seconds');ax[1].set_ylabel('PN firing rate (Hz)');ax[1].axvline(2,color='#aaa',linestyle=':');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
fig.suptitle('Isolated mean-field reference — full connectome unchanged',fontsize=12);fig.tight_layout();fig.savefig(OUT/'reference-comparison.png',dpi=160);plt.close(fig)
