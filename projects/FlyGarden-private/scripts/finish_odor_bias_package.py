"""Publish a bounded diagnosis only after all required audits pass."""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
OUT=ROOT/'reports/brain-integration/recovery/odor-bias-diagnosis'
results=json.loads((OUT/'diagnostic-results.json').read_text());assert results['status']=='complete' and len(results['records'])==40
for name in ('conventions.json','reference-parameters.json','preservation.json','checkpoint-preservation.json','observer-parity.json','body-turn-conventions.json'):assert json.loads((OUT/name).read_text())['status']=='passed'
records=results['records'];summary=[]
for case in json.loads((OUT/'diagnostic-protocol.json').read_text())['cases']:
 group=[r for r in records if r['case']==case];assert len(group)==2
 item={'case':case,'phases':{}}
 for phase in ('pre_cue','pulse','recovery'):
  item['phases'][phase]={k:{'mean':float(np.mean([r['phases'][phase]['population_hz'][k] for r in group])),
       'range':[float(min(r['phases'][phase]['population_hz'][k] for r in group)),float(max(r['phases'][phase]['population_hz'][k] for r in group))]} for k in ('DNa02_left','DNa02_right','DM1_lPN_left','DM1_lPN_right','DM2_lPN_left','DM2_lPN_right','DNp09_left','DNp09_right')}
 summary.append(item)
# The refractory pairs share exactly the same generated events; only neural policy differs.
pairs=[]
for seed in (9801,9802):
 for support in (0,65):
  for cue in ('a_both','b_both'):
   dirs=[OUT/'diagnostics'/f'{cue}-support{support}-{policy}-{seed}' for policy in ('candidate','selected_input')]
   manifests=[json.loads((p/'manifest.json').read_text()) for p in dirs];counts=[np.zeros(138639,dtype=np.int64) for p in dirs]
   for left,right in zip(manifests[0]['chunks'],manifests[1]['chunks']):
    with np.load(dirs[0]/left['file'],allow_pickle=False) as a,np.load(dirs[1]/right['file'],allow_pickle=False) as b:
     assert np.array_equal(a['external_i'],b['external_i']) and np.array_equal(a['external_t'],b['external_t']);counts[0]+=a['counts'];counts[1]+=b['counts']
   pairs.append({'seed':seed,'support':support,'cue':cue,'input_events_exact':True,'different_neuron_counts':int(np.count_nonzero(counts[0]!=counts[1])),'total_spikes':[int(c.sum()) for c in counts]})
wall=sum(r['wall_seconds'] for r in records);peak=max(r['peak_rss_bytes'] for r in records)
packet={'status':'diagnostic_package_complete','summary':summary,'refractory_pairs':pairs,'trials':40,'seeds':2,
 'wall_seconds_trial_sum':wall,'peak_worker_rss_bytes':peak,'files':{name:file_sha(OUT/name) for name in ('diagnostic-protocol.json','diagnostic-results.json','recorded-analysis.json','conventions.json','reference-parameters.json','preservation.json','checkpoint-preservation.json','observer-parity.json','body-turn-conventions.json')},
 'navigation_validated':False,'learning_validated':False,'production_controller_changed':False,'scope':'Focused steps1–7 diagnosis. Two fresh diagnostic seeds per condition; not independent held-out behavioral acceptance.'}
atomic_json(OUT/'package-results.json',packet)
lines=['# Completed steps 1–7: odor steering diagnosis','',
 'All 40 registered full-network diagnostic trials and their independent raw-event audits completed. A fresh uninstrumented candidate exactly reproduces all 60 windows of the representative instrumented trial: all neuron spike IDs/times/counts, inputs, population rates and commands match.','',
 'Preservation, exact annotation/root joins, 60-trial physical field/sensory-clock reconstruction, eight physical turning-sign checks and bounded released-reference artifacts pass. The current production controller remains unchanged.','',
 '## Diagnostic measurements','',
 'Two fresh diagnostic seeds per condition. Values below are mean left/right DNa02 Hz; raw seed values, selected voltage/g and independently reconstructed signed nominal arrivals remain in diagnostic-results.json. These are descriptive diagnostics, not a new success gate.','',
 '| Odor condition | Support Hz | Refractory policy | Pulse L / R Hz | Recovery L / R Hz |',
 '|---|---:|---|---:|---:|']
for r in summary:
 p=r['phases']['pulse'];q=r['phases']['recovery'];c=r['case'];lines.append(f"| {c['cue']} | {c['support']} | {c['refractory']} | {p['DNa02_left']['mean']:.1f} / {p['DNa02_right']['mean']:.1f} | {q['DNa02_left']['mean']:.1f} / {q['DNa02_right']['mean']:.1f} |")
lines += ['', '## Interpretation and boundaries','',
 'The saved embodied trial preserves a small side-dependent antenna difference, but both placements recruit nearly the same downstream steering state. ORNs stop firing after odor removal while projection and descending activity persist. Physical decoding and exact replay faithfully transmit the bias. The prior held-out directional failure remains unchanged.','',
 'Exact annotation hemisphere joins validate the engineered mapping implementation, not physiological antennal receptive-field equivalence. Support-only asymmetry must remain a matched control. Neither support removal, changing odor identity, unilateral stimulation, the mirrored small gradients, nor the selected-input refractory policy resolves the strongly left-biased DNa02 response in these two diagnostic seeds. Support-only stimulation instead produces a modest right-biased readout. Odor-driven activity remains strongly left-biased after odor removal. The refractory comparison changes neural activity but does not repair directional steering. Anatomical cross-hemisphere connections and unequal cell counts do not, by themselves, establish a biological or software defect.','',
 'Previously preserved inhibitory interventions implicate model inhibition in right-DNa02 suppression, but their lesions do not establish natural steering or justify deleting inhibition in a production controller. Nominal arrivals in this package are anatomical g increments, not actual membrane currents or fully refractory-gated accepted deliveries.','',
 'Reference coverage consists of short native Poisson execution checks, identical-input full-network equation replay and exact source/data/parameter integrity. Published behavioral prediction accuracy, natural receptor responses and long-run stochastic equivalence are not reproduced by those checks.','',
 'The registered signed-yaw metric is weaker than source-bearing or actual approach: its ten positive-direction cases must not be described as ten successful food approaches. No navigation, learning or consciousness claim follows.','',
 '## Next decision','',
 'Step 8 should select a new evidence-supported sensory/state/action hypothesis, with bounded calibration on separate seeds. The present DNa02-difference candidate remains unpromoted. Do not run another large orientation or learning batch until a short frozen candidate distinguishes cue direction and recovers appropriately. Preserve the original full connectome and failed artifacts; do not install a hidden navigator, direct cue-to-turn rule, mirrored artificial wiring or inhibitory lesion as a brain repair.','',
 f'Trial computation: {wall:.1f} wall seconds; peak individual worker RSS {peak/1024**3:.2f} GiB. This excludes startup/auditors and other applications. Performance varied with other active compute.','']
(OUT/'RESULTS.md').write_text('\n'.join(lines))
print('Focused odor-bias package completed:',len(records),'diagnostics')
