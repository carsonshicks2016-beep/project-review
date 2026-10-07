# Fly Garden delivery and scientific gates

The local application runs from the Desktop launcher. It provides the full network, an independently labeled supplied-controller baseline, physical walking, editable food/obstacles/shelters, a bounded predator, local sensory panels, actual eye-image inspection, immutable checkpoints, lineage, and recorded playback. RallyAI3 is independent and unchanged.

| Gate | Measured outcome |
|---|---|
| Full imported network | 138,639 neurons; 15,091,983 connection records; no reduced-network substitution |
| Combined benchmark | 10 simulated seconds; 0.161 simulated seconds per wall second; 1.93 GB peak process RSS (includes video rendering) |
| Neural output controls movement | Different left/right P9 stimulation changes decoded commands and embodied trajectories |
| Exact continuation | Same next-step activity, movement, and time after a complete application shutdown and restart |
| Retention | Actual food reinforcement changes eligible weights; saved learned parameters restore exactly |
| Body acceptance | Ten 60-second seeded runs completed with finite positions; obstacle-exposed trials fall, so robust obstacle walking is unvalidated |
| Neural conditioning screen | 20 seeds × 3 matched conditions; acquisition difference -4.375, 95% bootstrap interval [-8.375, -0.375]; reversal shift -3.500, interval [-5.625, -1.25] |
| Behavioral learning | Not demonstrated. Neural response/plasticity-to-readout coupling needs investigation before embodied learning claims |
| Predator calibration | 5/5 baseline trials survived 20 seconds at 5 mm/s, with 0 captures and 0 falls; provisional only |

The current motor interface is dominated by engineered tonic P9 exploration; meaningful odor-guided action selection is not established. The physical body uses a supplied gait generator and contact corrections. Synthetic odor stimulation and tonic P9 drive are engineered. The actual compound-eye renderer is inspectable, but threat control uses a geometric egocentric proxy. Contact and motion feedback currently serve the supplied gait; an ascending neural adapter is unvalidated. Food/capture contact uses documented planar proximity thresholds rather than a biological mechanosensory model. Energy is a game variable.

The conditioning runner measures an engineered MBON firing-rate difference in unrewarded probes with plasticity frozen. It does not measure a fly's choices. Acquisition, reversal, and shuffled pairing are matched by seed. Weight changes, escape events, and game unlocks provide no consciousness or biological-fidelity evidence. Generalization and learned threat avoidance remain behind the learning gate. Capture does not apply aversive brain reinforcement before that validation.

Checkpoints preserve this implemented model's state and random stream, not the original animal's identity. Earlier diagnostic scene schemas remain read-only in the UI. Source/data revisions, hashes, mapping assumptions, dependency versions, and license notices are retained. Reports retain failed outcomes.

Coupling uses 100 ms sample-and-hold windows: a neural interval is decoded and the body is advanced for the same elapsed interval. Physical settling occurs during initialization. This approximation is documented, not a claim of continuous biological sensorimotor fidelity.

Representative native-rendered playback: `reports/full-brain-walking.mp4` (10 simulated seconds, experimental full network plus supplied gait).
