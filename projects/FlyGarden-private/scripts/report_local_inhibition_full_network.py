"""Finalize only a terminal comparison, preserving failed scientific gates."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/brain-integration/recovery/local-inhibition-full-network-v1'

def main():
 p=json.loads((OUT/'protocol.json').read_text());r=json.loads((OUT/'results.json').read_text());progress=json.loads((OUT/'progress.json').read_text())
 assert progress['status']=='complete' and r['status']=='complete'
 assert all(file_sha(ROOT/n)==h for n,h in p['sources'].items())
 times=np.arange(1,241)*.025;fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
 for j,cue in enumerate(['a_left','a_right','a_both']):
  for model,color in [('original','#aa5345'),('local','#187f77')]:
   pn=[];dna=[]
   for seed in p['seeds']:
    rows=json.loads((OUT/'trials'/f'{model}-{cue}-{seed}'/'rows.json').read_text());pop=[x['population_hz'] for x in rows]
    pn.append([(x['DM1_lPN_left']+x['DM1_lPN_right'])/2 for x in pop]);dna.append([x['DNa02_left']-x['DNa02_right'] for x in pop])
   for i,v in enumerate([np.array(pn),np.array(dna)]):
    axes[i,j].plot(times,v.mean(axis=0),label=model,color=color);axes[i,j].fill_between(times,v.min(axis=0),v.max(axis=0),alpha=.15,color=color)
  axes[0,j].set_title(cue);axes[0,j].legend()
  for ax in axes[:,j]:ax.axvspan(.3,.8,alpha=.12,color='gray');ax.axvspan(3.3,3.8,alpha=.12,color='gray');ax.grid(alpha=.2)
  axes[1,j].set_xlabel('Simulated seconds')
 axes[0,0].set_ylabel('Mean DM1 PN Hz');axes[1,0].set_ylabel('Left minus right DNa02 Hz');fig.suptitle('Engineered partial local inhibition: two diagnostic seeds; bands are ranges, not confidence intervals');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
 manifests=[json.loads(f.read_text()) for f in (OUT/'trials').glob('*/manifest.json')];wall=sum(x['wall_seconds'] for x in manifests);rss=max(x['peak_rss_bytes'] for x in manifests)
 table='\n'.join(f"| {x['model']} | {x['cue']} | {x['seed']} | {x['pulse']} | {x['response_gain_hz']:.2f} | {max(abs(v) for v in x['pn_difference_hz']):.2f} | {'pass' if x['passed'] else 'fail'} |" for x in r['recovery_checks'])
 (OUT/'RESULTS.md').write_text(f'''# Local-inhibition clock bridge: full-network diagnostic

All 17 six-second trials and the fresh-process continuation checks completed. Response/recovery gates: **{r['response_recovery_passed']}**. Sensory contrast gates: **{r['sensory_contrast_passed']}**. No controller promotion, body validation or learning claim.

The experiment retains all 138,639 modeled neurons and 15,091,983 original connection records. A count-preserving allocation of 64,430 sites touching the 12 qualified lLN2P_b roots replaces only eligible portions with local activity routes; unresolved portions retain original signed spiking weights. Sites are not duplicated into both mechanisms. Other cells and connections, input profile and decoder remain frozen.

The 212 local nodes use normalized bounded activity with a hypothesized 15-ms time constant. Incoming delayed spikes are converted by an engineered 5-ms exponential rate filter. All local paths and outputs have a 1.8-ms delay on the 100-microsecond brain clock. Outputs add negative increments to the original voltage-equivalent g state, not receptor conductance. Local-to-local inhibition uses the delayed local state and actual routed synapses. The partial hybrid residual, missing electrical propagation, receptor kinetics and MIP are explicit limitations. No physiological fit is claimed.

Two diagnostic seeds use the unchanged pulse schedule, response/recovery windows and contrast gates. Both pulse response gains must reach 10 Hz. Each DM1 PN, selected local neuron and signed DNa02 must recover within 5 Hz of same-model support-only trials. Contrast requires opposite left/right deviations from support-only and left-minus-right contrast at least 10 Hz. These are diagnostic gates, not population-level statistical or navigation evidence.

| Model | Cue | Seed | Pulse | Response gain Hz | Max PN recovery difference Hz | Response/recovery gate |
|---|---|---:|---:|---:|---:|---|
{table}

Raw chunk hashes, spike/count parity and paired external event equality were checked. Observer-disabled local trials match exactly. Baseline and local checkpoints resume five windows in fresh processes with exact recorded arrays and full voltage/g state error at most 1e-10 mV. Bridge tests separately exercise delayed arrival, units, pending-queue continuation and residual-count allocation.

Trial wall time: {wall:.1f} seconds. Peak worker RSS: {rss/1024**3:.2f} GiB. One native brain worker. Original source and previous checkpoints were preserved. Scientific failure remains failure even if all engineering checks pass.

![Comparison](comparison.png)
''')
 atomic_json(OUT/'report-runtime.json',{'wall_seconds_trials':wall,'peak_worker_rss_bytes':rss,'free_space_bytes':shutil.disk_usage(OUT).free,'source_hashes_reverified':True,'application_changed':False})
 print(json.dumps({'response_recovery':r['response_recovery_passed'],'contrast':r['sensory_contrast_passed'],'peak_rss_GiB':rss/1024**3},indent=2))
if __name__=='__main__':main()
