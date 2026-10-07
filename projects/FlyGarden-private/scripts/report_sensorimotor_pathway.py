"""Verify Stage3 counts, evaluate preregistered cue gates and publish evidence."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from scripts.audit_brain_reference import sha
from scripts.diagnose_sensorimotor_pathway import CONDITIONS,SEEDS,inputs

def run(root):
 protocol=json.loads((root/'protocol.json').read_text());progress=json.loads((root/'progress.json').read_text());assert progress['status']=='complete' and len(progress['completed'])==36
 assert all(sha(ROOT/name)==value for name,value in protocol['sources'].items())
 rows={};windows=0;spikes=0
 for cue in CONDITIONS:
  for seed in SEEDS:
   folder=root/'trials'/f'{cue}-{seed}';manifest=json.loads((folder/'manifest.json').read_text());assert manifest['status']=='complete' and manifest['neurons']==138639 and manifest['connections']==15091983
   assert manifest['sources']==protocol['sources'] and len(manifest['chunks'])==30
   bins=json.loads((folder/'bins.json').read_text());assert len(bins)==30;rows[cue,seed]=bins
   for tick,chunk in enumerate(manifest['chunks']):
    file=folder/chunk['file'];assert sha(file)==chunk['sha256']
    with np.load(file) as data:
     counts=data['counts'];inp=data['input_counts'];assert counts.shape==(138639,) and np.all(counts>=0) and inp.shape==(len(manifest['input_root_ids']),)
     assert counts.sum()==chunk['spikes'] and np.array_equal(inp,bins[tick]['input_events'])
     population=bins[tick]['population_hz']
     for k,group in protocol['mapping'].items():
      if group:assert np.isclose(counts[[x['index'] for x in group]].mean()/.1,population[k])
     rates=inputs(cue,tick*.1+1e-7);assert np.array_equal(rates,bins[tick]['input_hz'])
     channel=np.concatenate([np.full(len(protocol['mapping'][k]),i) for i,k in enumerate(manifest['input_keys'])]);assert np.all(inp[rates[channel]==0]==0)
     assert np.isfinite(bins[tick]['candidate_motor']).all() and min(bins[tick]['candidate_motor'])>=0 and max(bins[tick]['candidate_motor'])<=1.2
    windows+=1;spikes+=chunk['spikes']
 repeat=root/'repeat-check/a_left-3199'
 if repeat.exists():
  original=root/'trials/a_left-3199'
  for file in original.glob('window-*.npz'):
   with np.load(file) as x,np.load(repeat/file.name) as y:
    for key in ('counts','input_counts'):assert np.array_equal(x[key],y[key])
  assert json.loads((repeat/'bins.json').read_text())==rows['a_left',3199]
 def rate(cue,seed,pop,start=.3,end=.5):
  return float(np.mean([r['population_hz'][pop] for r in rows[cue,seed] if start+1e-8<r['time']<=end+1e-8]))
 comparisons=[]
 for seed in SEEDS:
  for prefix in ('a','b','walking_a'):
   left=rate(prefix+'_left',seed,'DNa02_left')-rate(prefix+'_left',seed,'DNa02_right')
   right=rate(prefix+'_right',seed,'DNa02_left')-rate(prefix+'_right',seed,'DNa02_right')
   baseline_cue='walking' if prefix.startswith('walking') else 'quiet'
   bias=rate(baseline_cue,seed,'DNa02_left')-rate(baseline_cue,seed,'DNa02_right')
   l,r=left-bias,right-bias
   comparisons.append({'cue':prefix,'seed':seed,'raw_left_difference_hz':left,'raw_right_difference_hz':right,'baseline_difference_hz':bias,'left_evoked_difference_hz':l,'right_evoked_difference_hz':r,'passes':l>5 and r< -5,'recovery_left_difference_hz':rate(prefix+'_left',seed,'DNa02_left',.8,1.3)-rate(prefix+'_left',seed,'DNa02_right',.8,1.3),'recovery_right_difference_hz':rate(prefix+'_right',seed,'DNa02_left',.8,1.3)-rate(prefix+'_right',seed,'DNa02_right',.8,1.3)})
 gates={cue:all(r['passes'] for r in comparisons if r['cue']==cue) for cue in ('a','b','walking_a')}
 body=json.loads((root/'body-replay/summary.json').read_text());assert body['status']=='complete' and all(r['finite'] for r in body['results']);assert body['body_source_sha256']==protocol['sources']['flygarden/body.py'];assert body['source_sha256']==sha(ROOT/'scripts/replay_descending_body.py');assert body['supplement_source_sha256']==sha(ROOT/'scripts/check_sensorimotor_replay_controls.py');bycue={r['cue']:r for r in body['results']}
 quiet_heading=bycue['quiet']['heading_change_radians']
 direct_body=bycue['direct_left']['heading_change_radians']-quiet_heading> .05 and bycue['direct_right']['heading_change_radians']-quiet_heading< -.05 and not any(bycue[c]['flipped_samples'] for c in ('direct_left','direct_right'))
 sensory_gate=gates['a'] and gates['walking_a'];promote=sensory_gate and direct_body
 verification={'passed':True,'trials':36,'windows':windows,'total_counted_spikes':spikes,'sources_unchanged':True,'checks':['SHA256 of every count chunk','Full network size and ordering','Per-neuron counts reproduce annotated rates','External event counts and stimulus schedule','Zero events on unstimulated channels','Finite bounded motor commands','Complete matched conditions and held-out seed'],'raw_spike_times_recorded':False,'repeat_count_motor_parity':repeat.exists()}
 measurement={'neural_active_wall_seconds':progress.get('wall_seconds'), 'peak_neural_worker_rss_bytes':max(json.loads((root/'trials'/name/'manifest.json').read_text())['peak_rss_bytes'] for name in progress['completed']), 'body_replay_wall_seconds':body['wall_seconds'],'evidence_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())};atomic_json(root/'measurements.json',measurement)
 analysis={'schema_version':1,'verification':verification,'cue_comparisons':comparisons,'cue_gates':gates,'direct_body_direction_passed':direct_body,'sensory_gate_passed':sensory_gate,'eligible_for_arena_validation':promote,'promoted':False,'learning_demonstrated':False,'body_replay':body,'scope':'Isolated sensory-to-descending tests and open-loop motor replay; not closed-loop navigation.'}
 atomic_json(root/'verification.json',verification);atomic_json(root/'analysis.json',analysis)
 fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
 for cue,color in [('a_left','#167dcb'),('a_right','#dc7134'),('walking','#777777'),('walking_a_left','#178c58'),('walking_a_right','#a84ea0')]:
  bins=rows[cue,3199];t=[x['time']-.05 for x in bins]
  axes[0].plot(t,[x['population_hz']['ORN_DM1_left']-x['population_hz']['ORN_DM1_right'] for x in bins],label=cue,color=color)
  axes[1].plot(t,[x['population_hz']['DNa02_left']-x['population_hz']['DNa02_right'] for x in bins],color=color)
  axes[2].plot(t,[x['candidate_turn'] for x in bins],color=color)
 for ax in axes:ax.axvspan(.3,.8,color='grey',alpha=.12);ax.axhline(0,color='grey',lw=.5);ax.grid(alpha=.15)
 axes[0].set_ylabel('Left − right\nORN rate (Hz)');axes[1].set_ylabel('Left − right\nDNa02 rate (Hz)');axes[2].set_ylabel('Right − left\ncandidate gait drive');axes[2].set_xlabel('Simulated time (s)');axes[0].legend(ncol=3,fontsize=9);fig.suptitle('Held-out full-brain responses: lateral sensory input and measured steering output\nFrozen learning; shaded interval = odor pulse; walking inputs are engineered tonic drive',fontsize=12);fig.tight_layout();fig.savefig(root/'sensorimotor-response.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for cue in bycue:
  trace=json.loads((root/'body-replay'/f'{cue}.json').read_text())['trace'];pos=np.array([x['body']['position'] for x in trace]);head=np.unwrap([x['body']['heading'] for x in trace]);axes[0].plot(pos[:,0],pos[:,1],label=cue);axes[1].plot([x['time'] for x in trace],head,label=cue)
 axes[0].set_aspect('equal',adjustable='datalim');axes[0].set_xlabel('Body x (model units)');axes[0].set_ylabel('Body y (model units)');axes[1].set_xlabel('Simulated time (s)');axes[1].set_ylabel('Unwrapped body heading (rad)');axes[1].legend(fontsize=8);fig.suptitle('Measured neural motor commands replayed through FlyGym + supplied gait\nOpen-loop replay; direct output stimulation is a separate artificial control',fontsize=11);fig.tight_layout();fig.savefig(root/'body-replay.png',dpi=160);plt.close(fig)
 table='\n'.join(f"| {r['cue']} | {r['seed']} | {r['left_evoked_difference_hz']:.1f} | {r['right_evoked_difference_hz']:.1f} | {'pass' if r['passes'] else 'fail'} |" for r in comparisons)
 text=f'''# Fly Garden — Stage 3 sensory-to-movement pathway

## Result

Completed 36 isolated full-network trials (138,639 neurons; 15,091,983 connection records), with three matched seeds including held-out 3199. Verified {windows:,} count windows containing {spikes:,} spikes. Learning and recurrent weights remain fixed. The live application was not altered.

The tested odors caused a same-side DNa02 bias instead of reliable left/right cue tracking. The body followed that bias: walking-plus-left-odor changed heading by {bycue['walking_a_left']['heading_change_radians']:.3f} rad; walking-plus-right-odor by {bycue['walking_a_right']['heading_change_radians']:.3f} rad, both in the same direction. Disabling the steering readout reduced the matched replay to {bycue['steering_readout_disabled']['heading_change_radians']:.3f} rad.

Sensory gate: **{'PASS' if sensory_gate else 'FAIL'}**. Direct output-to-body direction control: **{'PASS' if direct_body else 'FAIL'}**. Candidate decoder: **not promoted**. {'Eligible for a separate closed-loop arena validation; these isolated tests do not establish navigation.' if promote else 'Do not reconnect this candidate as a validated sensory controller.'}

## What was implemented

A versioned exact-root-ID mapping keeps left/right ORN_DM1 and ORN_DM2 inputs separate. Synthetic 50 Hz odor activation is an engineered encoding, not a natural receptor response. The pinned reference stimulation convention sets refractory to zero only on the neurons stimulated in that condition. Quiet, left, right, bilateral, sequential switch, walking-only, walking-plus-odor, and direct steering controls retain continuous state within each 3-second trial. Ordinary trials independently reconstruct the full network.

`flygarden/descending.py` is an unpromoted candidate: forward drive uses bilateral DNp09 mean rate; steering uses DNa02 left-minus-right rate. Rate scaling, smoothing and bounds are engineered, fixed before trials, and read no odor values, positions, routes, rewards or hidden world state. The supplied walking controller coordinates the legs. Candidate commands and all annotated rates are saved per 100 ms. Counts cover every modeled neuron; raw spike timestamps were not recorded in this phase.

DNa02 was chosen before results because experimental work connects it to ipsilateral steering and lateralized olfactory responses. The research uses broader olfactory stimulation; our two ORN types are not an exact reproduction. See [steering circuit research](https://elifesciences.org/articles/102230) and [descending steering research](https://pmc.ncbi.nlm.nih.gov/articles/PMC12778575/).

## Cue gate

Fixed acceptance window: first 200 ms of the odor pulse. Positive left and negative right changes must each exceed 5 Hz relative to the matched quiet/walking baseline and hold on both calibration seeds plus the held-out seed. No fitted baseline offsets or motor gains were applied to the decoder. Recovery and later responses remain available in the raw bins; early responses alone are not navigation proof.

| Cue | Seed | Left-cue evoked left−right DNa02 (Hz) | Right-cue evoked left−right DNa02 (Hz) | Gate |
| --- | ---: | ---: | ---: | --- |
{table}

![Sensory and descending response](sensorimotor-response.png)

## Anatomical audit

`anatomical-routes.json` records exact IDs, source hashes and representative directed shortest paths from each ORN population to DNa02, DNp09 and DNg13. All imported connection records enter the topology, including inhibitory and zero signed weights. A path demonstrates anatomical reachability, not transmission, excitatory causality, or unique circuit identity. No connections were thresholded or altered in neural simulations.

## Body connection

The held-out recorded neural commands were replayed through the actual FlyGym body for seven conditions including two supplementary controls. Body simulation advances at its existing physics/gait clocks, with finite-state checks and 30 FPS pose sampling. This is open-loop motor replay, not a fly sensing and responding to its changing location. A quiet replay and a replay with DNa02 readout set to zero provide output controls; the latter disables only the decoder readout, not the recorded brain. These controls were added after observing the same-side response and do not alter the original cue gate. Artificial direct left/right DNa02 stimulation tests the neural output and gait connection separately; it cannot establish odor-driven steering. Falls are retained in the results.

![Physical motor replay](body-replay.png)

## Decision and next work

{'The cue and body gates passed; proceed to a separate controlled closed-loop arena evaluation before promotion.' if promote else 'The present sensory-to-steering configuration did not meet the gates. Keep the live controller unchanged. Inspect the documented left/right response and signed intermediate paths before choosing an additional predeclared circuit test. Do not convert odor gradients directly into turning commands or change neuron weights merely to obtain attractive movement.'}

This phase does not demonstrate learning, biological fidelity, obstacle vision, or navigation. The persistent activity diagnosed in Stage 2 remains a separate limitation.

## Runtime and verification

Neural execution: {measurement['neural_active_wall_seconds']:.1f} active wall seconds. Peak neural worker RSS: {measurement['peak_neural_worker_rss_bytes']/1024**3:.2f} GiB (macOS process accounting, excludes the live app). Initial five body replays: {measurement['body_replay_wall_seconds']:.1f} seconds; supplementary control runtime was not recorded. Raw evidence: {measurement['evidence_bytes']/1024**2:.1f} MiB before these plots/report. Body replay source identity and finite results were verified. Three decoder/schedule tests passed. {'A fresh held-out repeat matched every per-neuron 100 ms count, external event count, population rate and motor command; raw spike timing and internal-state parity are not claimed.' if repeat.exists() else 'Fresh reconstruction parity has not been tested.'}

## Artifacts and reproduction

- `protocol.json`: committed mappings, seeds, gates and source hashes.
- `trials/*/manifest.json`, `bins.json`, `window-*.npz`: intact trials and per-neuron counts.
- `anatomical-routes.json`: topology and signed representative edges.
- `body-replay/*.json`: actual physical traces and sampled poses.
- `analysis.json`, `verification.json`: machine-readable results and checks.

Run `scripts/diagnose_sensorimotor_pathway.py NEW_FOLDER`, then `scripts/map_sensorimotor_routes.py NEW_FOLDER`, then `scripts/replay_descending_body.py NEW_FOLDER`, then `scripts/check_sensorimotor_replay_controls.py NEW_FOLDER`, then `scripts/report_sensorimotor_pathway.py NEW_FOLDER`, using `.venv-next/bin/python`. Existing complete trials are retained on resume; incomplete attempts must be preserved before retry. The 2 GiB free-space reserve is enforced; no automatic deletion. Decoder/schedule tests: `tests/test_descending.py`.
'''
 (root/'RESULTS.md').write_text(text);print(json.dumps({'gates':gates,'body_direction':direct_body,'verification':verification},indent=2))
if __name__=='__main__':run(Path(sys.argv[1]))
