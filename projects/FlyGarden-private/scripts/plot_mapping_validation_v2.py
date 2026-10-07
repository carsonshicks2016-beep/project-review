"""Visual evidence for the fixed, failed opposite-side atlas registration."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/brain-integration/recovery'
OUT=BASE/'patchy-compartment-mapping-v2'

def main():
    native=json.loads((BASE/'patchy-compartment-mapping-v1/spatial-results.json').read_text())['anchor_side_statistics']['left']
    mirrored=json.loads((OUT/'mirror-results.json').read_text())['anchor_side_statistics']['right']
    data=json.loads((OUT/'mirror-anchor-by-glomerulus.json').read_text())
    right=[d for d in data if d['PN_side']=='right']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    x=np.arange(2);width=.34
    axes[0].bar(x-width/2,[native['unique_fraction']*100,mirrored['unique_fraction']*100],width,label='Unique enclosure')
    axes[0].bar(x+width/2,[native['correct_unique_fraction']*100,mirrored['correct_unique_fraction']*100],width,label='Correct label among enclosed')
    axes[0].set_xticks(x,['Native atlas / left anchors','Mirrored atlas / right anchors'])
    axes[0].set_ylim(0,108);axes[0].set_ylabel('Percent of fixed sensory-anchor samples')
    axes[0].legend(loc='lower left');axes[0].set_title('Whole-side validation: mirror rejected')
    axes[1].axvspan(70,100,color='green',alpha=.05)
    axes[1].axhspan(80,100,color='green',alpha=.05)
    axes[1].axvline(70,color='gray',ls='--');axes[1].axhline(80,color='gray',ls='--')
    axes[1].scatter([d['unique_fraction']*100 for d in right],[d['correct_unique_fraction']*100 for d in right],s=25,alpha=.7)
    for d in right:
        if d['glomerulus'] in ['DM1','DC3']:
            axes[1].annotate(d['glomerulus'],(d['unique_fraction']*100,d['correct_unique_fraction']*100),xytext=(6,6),textcoords='offset points')
    axes[1].set(xlim=(0,102),ylim=(0,104),xlabel='Unique enclosure (%)',ylabel='Correct label among enclosed (%)',title='Opposite-side glomeruli: failures remain visible')
    fig.suptitle('Published non-rigid FlyWire mirror — anatomical test, no controller changes')
    fig.savefig(OUT/'mirror-validation.png',dpi=160);plt.close(fig)
    a=pd.read_csv(OUT/'mirror-anchor-audit.csv');a=a[(a.PN_side=='right')&(a.glomerulus=='DM1')]
    confusion=a.groupby(['mirror_candidate','mirror_memberships']).size().rename('count').reset_index()
    confusion.to_csv(OUT/'right-DM1-confusion.csv',index=False)

if __name__=='__main__':main()
