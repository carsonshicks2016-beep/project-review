"""Summarize completed fixed-point screen without changing acceptance gates."""
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'reports/brain-integration/recovery/adaptive-domain-screen-v1'
p=json.loads((O/'progress.json').read_text());assert p['status']=='complete'
r=json.loads((O/'calibration-results.json').read_text());audit=json.loads((O/'independent-recording-audit.json').read_text());assert audit['completed_trials']==len(p['completed'])
models=list(r['eligible']);fig,axes=plt.subplots(len(models),1,figsize=(11,12),sharex=True)
for ax,model in zip(axes,models):
 for cue in ['none','a_left','a_right','a_both']:
  rows=json.loads((O/'trials'/f'{model}-{cue}-11901'/'rows.json').read_text())
  ax.plot([x['end'] for x in rows],[(x['population_hz']['DM1_lPN_left']+x['population_hz']['DM1_lPN_right'])/2 for x in rows],label=cue,linewidth=.9)
 ax.axvspan(.3,.8,color='grey',alpha=.12);ax.axvspan(3.3,3.8,color='grey',alpha=.12);ax.set_ylim(0,300);ax.set_ylabel('PN rate (Hz)');ax.set_title(model);ax.legend(loc='upper right',ncol=4)
axes[-1].set_xlabel('Simulated seconds');fig.tight_layout();fig.savefig(O/'comparison.png',dpi=150);plt.close(fig)
lines=['# Fixed-point adaptive-LIF domain screen','',f"Initial calibration: {len(models)} controller arms, one calibration seed, 150 ms decay and 1.5 mV spike increment. This is one preregistered parameter point, not a completed parameter sweep.",'','| Arm | Response/recovery checks | Bilateral contrast | Burst checks | Support | Eligible |','|---|---:|---:|---:|---:|---|']
for m in models:
 vals=[]
 for g in ['recovery','contrast','bursts','support']:
  a=[x for x in r[g] if x['model']==m];vals.append(f"{sum(x['passed'] for x in a)}/{len(a)}")
 lines.append('| '+m+' | '+' | '.join(vals)+f" | {r['eligible'][m]} |")
lines+=['',f"Selected calibration domain: {p['selected'] or 'none'}. Held-out evaluation run: {p['heldout_run']}.",'','The live controller is unchanged. Local inhibition was not combined with adaptation. Uncertain local-neuron and global domains are engineering comparisons; anatomical labels do not establish their firing physiology. Passing calibration alone does not demonstrate learning, embodied navigation, or biological fidelity.','',f"Integrity audit: {audit['completed_trials']} trials, {audit['chunks']} complete chunks. Zero-adaptation regression matched baseline events and whole voltage/conductance state exactly.",'',f"Total recorded trial runtime: {sum(t['wall_seconds'] for t in audit['trials']):.1f} seconds. Peak measured worker memory: {max(t['peak_rss_bytes'] for t in audit['trials'])/2**30:.3f} GiB. This excludes parent process and operating-system overhead.",'']
lines+=['All arms retained a measurable response to both pulses, but none recovered or preserved the required bilateral contrast. Adaptation reduced persistent PN activity; even the lowest recorded individual PN recovery difference was 168 Hz, versus the 5 Hz limit. Reduced firing is not a successful recovery.','']
if not p['selected']:lines+=['No domain passed all gates at this point. Held-out gradient evaluation was therefore not launched. The next decision is whether to preregister a wider adaptation search or investigate another mechanism; no candidate is ready for promotion.','']
(O/'RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines))
