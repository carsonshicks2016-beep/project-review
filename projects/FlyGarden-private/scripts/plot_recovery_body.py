"""Plot saved physical trials; never advances or repairs their trajectories."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.accept_recovery_body import OUT
from flygarden.body_trial import read_trial

def run():
    trials=[]
    for path in sorted((OUT/'trials').glob('*/manifest.json')):
        if json.loads(path.read_text())['status']=='complete':trials.append(read_trial(path.parent))
    if not trials:raise RuntimeError('No complete physical trial')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for m,rows,frames in trials:
        t=np.array([0.]+[r['time'] for r in rows]);p=np.array([m['initial']['position']]+[r['body']['position'] for r in rows])
        h=np.unwrap([m['initial']['heading']]+[r['body']['heading'] for r in rows])
        label=str(m['seed']);speed=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)/.025
        axes[0,0].plot(p[:,0],p[:,1],alpha=.65,label=label)
        axes[0,1].plot(t,h-h[0],alpha=.65)
        axes[1,0].plot(t[1:],speed,alpha=.5)
        slip=[r['physical_contacts']['load_weighted_tangential_speed_mm_per_second'] for r in rows]
        axes[1,1].plot(t[1:],[np.nan if s is None else s for s in slip],alpha=.5)
    axes[0,0].set(xlabel='X (mm)',ylabel='Y (mm)',title='Saved physical path',aspect='equal');axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='Simulated time (s)',ylabel='Heading change (rad)',title='Mirrored turning')
    axes[1,0].set(xlabel='Simulated time (s)',ylabel='XY speed (mm/s)',title='Starting, stopping, recovery')
    axes[1,1].set(xlabel='Simulated time (s)',ylabel='Tangential speed (mm/s)',title='Solver-load-weighted contact motion')
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        for boundary in (10,20,30,40,50):ax.axvline(boundary,color='gray',lw=.7,ls='--')
    fig.suptitle(f'Body acceptance: {len(trials)}/10 complete trials — artificial rates, supplied gait\nStop | Forward | Left | Stop | Right | Forward; contact metric excludes adhesion loads',fontsize=12)
    fig.savefig(OUT/'physical-review.png',dpi=140);plt.close(fig)
    print({'complete_trials':len(trials),'artifact':str(OUT/'physical-review.png')})

if __name__=='__main__':run()
