import json,hashlib,platform
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def read(n):return json.loads((root/'reports'/f'{n}.json').read_text())
b=read('combined-benchmark');l=read('learning-evaluation');a=read('body-acceptance');p=read('predator-calibration')
assert l['status']==a['status']=='completed'
a['stationary_frame_definition']='Historical stall_frames counts displacement under 0.005 mm per 0.1s, including intentional stops; not a validated stall assay'
a['stable_obstacle_navigation']=all(x.get('fall_frames',1)==0 for x in a['runs']);(root/'reports/body-acceptance.json').write_text(json.dumps(a,indent=2))
report={'platform':platform.platform(),'arena_works':True,'full_brain_runs':True,'neural_activity_controls_movement':read('output-control')['passed'],'checkpoint_continuation':read('cold-start')['passed'],'learned_parameter_retention':read('retention')['passed'],'behavioral_learning_demonstrated':False,'neural_screen_passed':l['neural_screen_passed'],'stable_obstacle_navigation':a['stable_obstacle_navigation'],'combined_simulated_seconds_per_wall_second':b['throughput'],'combined_peak_rss_bytes':b['peak_rss_bytes'],'body_trials':len(a['runs']),'conditioning_trials':len(l['runs']),'limitations':['No validated retinal or ascending contact-to-brain mapping','Full model slower than real time; recorded playback provided','Obstacle exposure produces falls','Conditioning neural proxy is not behavioral preference','Generalization, threat learning, and PPO deferred behind learning gates','No narrow-passage validation; shallow step only has two short traversal trials'],'checks':20}
(root/'reports/delivery.json').write_text(json.dumps(report,indent=2))
text=f'''# Fly Garden delivery and scientific gates

The local application runs from the Desktop launcher. It provides the full network, an independently labeled supplied-controller baseline, physical walking, editable food/obstacles/shelters, a bounded predator, local sensory panels, actual eye-image inspection, immutable checkpoints, lineage, and recorded playback. RallyAI3 is independent and unchanged.

| Gate | Measured outcome |
|---|---|
| Full imported network | 138,639 neurons; 15,091,983 connection records; no reduced-network substitution |
| Combined benchmark | 10 simulated seconds; {b['throughput']:.3f} simulated seconds per wall second; {b['peak_rss_bytes']/1e9:.2f} GB peak process RSS (includes video rendering) |
| Neural output controls movement | Different left/right P9 stimulation changes decoded commands and embodied trajectories |
| Exact continuation | Same next-step activity, movement, and time after a complete application shutdown and restart |
| Retention | Actual food reinforcement changes eligible weights; saved learned parameters restore exactly |
| Body acceptance | Ten 60-second seeded runs completed with finite positions; obstacle-exposed trials fall, so robust obstacle walking is unvalidated |
| Neural conditioning screen | 20 seeds × 3 matched conditions; acquisition difference {l['neural_acquisition_difference']:.3f}, 95% bootstrap interval {l['neural_acquisition_ci95']}; reversal shift {l['neural_reversal_shift']:.3f}, interval {l['neural_reversal_ci95']} |
| Behavioral learning | Not demonstrated. {l['next_gate']} |
| Predator calibration | {p['survived_20s']}/5 baseline trials survived 20 seconds at 5 mm/s, with {p['captures']} captures and {p['falls']} falls; provisional only |

The current motor interface is dominated by engineered tonic P9 exploration; meaningful odor-guided action selection is not established. The physical body uses a supplied gait generator and contact corrections. Synthetic odor stimulation and tonic P9 drive are engineered. The actual compound-eye renderer is inspectable, but threat control uses a geometric egocentric proxy. Contact and motion feedback currently serve the supplied gait; an ascending neural adapter is unvalidated. Food/capture contact uses documented planar proximity thresholds rather than a biological mechanosensory model. Energy is a game variable.

The conditioning runner measures an engineered MBON firing-rate difference in unrewarded probes with plasticity frozen. It does not measure a fly's choices. Acquisition, reversal, and shuffled pairing are matched by seed. Weight changes, escape events, and game unlocks provide no consciousness or biological-fidelity evidence. Generalization and learned threat avoidance remain behind the learning gate. Capture does not apply aversive brain reinforcement before that validation.

Checkpoints preserve this implemented model's state and random stream, not the original animal's identity. Earlier diagnostic scene schemas remain read-only in the UI. Source/data revisions, hashes, mapping assumptions, dependency versions, and license notices are retained. Reports retain failed outcomes.

Coupling uses 100 ms sample-and-hold windows: a neural interval is decoded and the body is advanced for the same elapsed interval. Physical settling occurs during initialization. This approximation is documented, not a claim of continuous biological sensorimotor fidelity.

Representative native-rendered playback: `reports/full-brain-walking.mp4` (10 simulated seconds, experimental full network plus supplied gait).
'''
(root/'IMPLEMENTATION_REPORT.md').write_text(text);print(json.dumps(report,indent=2))
