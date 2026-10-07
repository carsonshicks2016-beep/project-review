"""Join independently audited neural trials to raw physical command replays."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_recovery_interfaces import OUT,DT
from scripts.replay_recovery_interfaces import held_command
from flygarden.body_trial import read_trial,sha
from flygarden.recording import atomic_json

def run():
    neural=json.loads((OUT/'analysis.json').read_text());assert neural['status']=='complete' and len(neural['trials'])==28
    replay=OUT/'physical-replay';protocol=json.loads((replay/'protocol.json').read_text())
    for name,digest in protocol['sources'].items():
        path=OUT/'protocol.json' if name=='neural_protocol' else OUT/'analysis.json' if name=='neural_analysis' else ROOT/name
        assert sha(path)==digest
    results=[]
    for n in neural['trials']:
        name=f"{n['cue']}-{n['seed']}";m,rows,frames=read_trial(replay/'trials'/name)
        assert m['status']=='complete' and len(rows)==60 and len(frames)==45
        brain_folder=OUT/'trials'/name;commands=json.loads((brain_folder/'bins.json').read_text())
        assert m['sources']['neural_manifest']==sha(brain_folder/'manifest.json')
        assert m['sources']['neural_bins']==sha(brain_folder/'bins.json')
        for tick,row in enumerate(rows):
            assert row['motor']==held_command(commands,tick*DT)[1] and abs(row['time']-(tick+1)*DT)<1e-9
            for key in ('position','heading','feet','contacts','antennae'):assert np.isfinite(row['body'][key]).all()
        assert np.all(np.abs(np.asarray([f['time'] for f in frames])-np.arange(1,46)/30)<.0005+1e-9)
        p=np.asarray([m['initial']['position']]+[r['body']['position'] for r in rows])
        h=np.unwrap([m['initial']['heading']]+[r['body']['heading'] for r in rows])
        results.append({'cue':n['cue'],'seed':n['seed'],'external_events':n['external_events'],
                        'mean_pulse_motor':n['phases']['pulse']['mean_candidate_motor'],
                        'xy_displacement_mm':float(np.linalg.norm(p[-1,:2]-p[0,:2])),
                        'heading_change_rad':float(h[-1]-h[0]),'flipped_samples':sum(r['body']['flipped'] for r in rows)})
    evidence={'status':'diagnostic_complete_behavior_unvalidated','neural_trials':28,'physical_replays':28,
              'neural_analysis_sha256':sha(OUT/'analysis.json'),'replay_protocol_sha256':sha(replay/'protocol.json'),
              'auditor_sha256':sha(Path(__file__)),'trials':results,
              'finding':'Physical body and direct descending output work; selected sensory-to-walking readout is not a validated choice controller.',
              'scope':'Short diagnostic seeds; stationary visual fixtures,synthetic touch,engineered odor,state support and supplied gait. No live navigation,retreat or learning claim.'}
    atomic_json(OUT/'results.json',evidence)
    table='\n'.join(f"| {r['cue']} | {r['seed']} | {r['external_events']} | {r['xy_displacement_mm']:.4f} | {r['heading_change_rad']:.4f} | {r['flipped_samples']} |" for r in results)
    (OUT/'RESULTS.md').write_text('''# Full-brain interface diagnosis

All28 full-network trials and28 physical command replays completed. Neural raw-data audits verify root IDs,spike-count parity,input delivery,membrane timing,population rates,decoder reconstruction,and independent signed synaptic arrivals. Physical audits verify recorded commands acting in their next eligible25ms interval,finite observations,and30Hz poses. These are diagnostics,not held-out behavioral acceptance.

Quiet inputs produce no spikes or drive. Explicit DNp09 state support produces forward commands and also recruits asymmetric steering activity. Direct left/right DNa02 stimulation produces opposite command biases. Both unilateral odor cues recruit predominantly left DNa02; sensory stimulation reaches the circuit,but the readout does not establish mirrored odor choice. Looming recruits LPLC2 and DNp01 while the selected walking outputs remain weak or inactive. Translation also activates this visual encoding,so specificity remains limited. Synthetic touch reaches its target neurons without MDN retreat output.

Nominal positive/negative synaptic increments are not physical membrane currents. Their reconstruction includes delayed presynaptic events across chunk boundaries. The body replay consumes saved commands; its movement does not feed back into these neural recordings. Touch pulses are not live contact validation. All original connections remain in the modeled network; no lesion,hidden navigator,escape override or learning is introduced in this protocol.

| Cue | Seed | External events | Physical departure mm | Heading change rad | Flipped samples |
|---|---|---|---|---|---|
'''+table+'''

Next: choose a separately versioned,biologically justified candidate after reviewing these results and the archived Step4 inhibitory diagnosis. Keep calibration and evaluation seeds separate. Do not promote this matrix as autonomous navigation or treat support-only motion as sensory-guided success. See ARCHITECTURE_EVIDENCE.md in the parent recovery folder for primary motor-population references.
''')
    print({'neural_trials':28,'physical_replays':28,'status':evidence['status']})

if __name__=='__main__':run()
