# Fly Garden

**Independent review:** start with [the plain-language review guide](docs/MUSE-REVIEW.md). This repository is a source-and-evidence snapshot; large local recordings and datasets are excluded. [Reproduction boundaries](docs/REPRODUCIBILITY.md).

A local 3D laboratory for one persistent connectome-based fly. Open **Open Fly Garden.command** on the Desktop or visit http://127.0.0.1:8794 after launching. Close it with **Close Fly Garden.command**. Everything runs locally.

Recovery implementation is tracked in [the recovery plan](reports/brain-integration/RECOVERY_PLAN.md) and [current progress](reports/brain-integration/recovery/progress.json). The live full-model option is explicitly an **engineered hybrid**, with tonic walking support and a supplied escape override. New run manifests, checkpoints and frames identify this controller and attribute each motor command; this does not promote the separate experimental controller. [Body/decoder calibration](reports/brain-integration/recovery/body-operating-range/RESULTS.md) confirms substantial motion with adequate artificial commands, while full-brain useful choice remains unresolved.

## What is implemented

- Full FlyWire v783 imported network: 138,639 neurons, 15,091,983 aggregated connection records, 54,492,922 represented anatomical synapses. No connectivity threshold or reduced-network substitution.
- Brian2 CPU runtime with retained state between windows. Annotated ORN_DM1/DM2 inputs and DNp09 left/right output populations.
- NeuroMechFly 2.1 physical body, 100 μs physics timestep, supplied hybrid leg controller updated every 500 μs. Its contact corrections and adhesion are supplied mechanisms.
- Two synthetic odor channels sampled at physical antennal sites, distance-based odor fields, obstacles, food depletion, energy, occlusion, bounded ground predator, shelter, and capture-before-food priority.
- Experimental reward-modulated depression on selected gamma KC → MBON01 / MBON11 connections; remaining neural weights fixed. This is an engineered model, not an automatically recovered learning mechanism.
- Sandbox, six challenge layouts, pause/step/reset, editor, activity panels, trials, immutable learned-state and complete-scene checkpoints, branches, recorded playback, and experiment jobs.

## Read the boundaries before interpreting behavior

The full model has tonic P9 stimulation for exploration. Meaningful odor-guided action selection is not established. P9 activity controls a supplied gait generator; individual leg control is not reconstructed from the connectome. A visible-threat proxy and escape reflex are engineered. Current visual threat inputs are egocentric geometric features, not validated retinal circuitry. Odor identities are synthetic. Energy is a game variable; predator is an engineered agent.

Behavioral learning is **not established** by weight changes or neural response differences. The conditioning runner screens MBON responses using an engineered neural valence proxy. A passing neural screen still requires embodied choice tests; failed gates remain in reports. Gameplay unlocks do not confer scientific validation.

The complete model is slower than real time on this Mac. Playback replays recorded model output. No simulation dynamics are skipped to match wall time. Obstacles can cause falls; falls end trials, and acceptance reports record them.

## Use

Begin loads the full brain. Observe shows local inputs and population responses. Learn selects layouts, toggles experimental plasticity, and runs protocols. Archive saves/branches modeled state or replays runs. New trial retains the edited arena. Restoring a learned fly applies its weights to the current arena; restoring a complete scene also restores the saved world. Save before closing, then restore through Archive on reopening. Editing pauses and resets an unscored trial, retaining learned parameters. Capture conditioning is disabled pending controlled aversive validation. Capture ends a trial; New trial resets transient neural/body state and retains learning.

The supplied-controller baseline uses only local inputs and is explicitly separate from the full model. No brain claim applies to it.

## Reproduce

Use Python 3.12 and the pinned environment in `.venv-next`. Installed dependencies are recorded in `requirements.lock`. The vendored brain, FlyGym source, and annotation hashes/revisions are recorded in `reports/provenance.json`. Neuron IDs, compartments, adapter assumptions, and sources are in `reports/neuron-mapping.json`.

```sh
.venv-next/bin/python -m pytest tests -q
.venv-next/bin/python scripts/benchmark_brain.py
.venv-next/bin/python scripts/benchmark_body.py
.venv-next/bin/python scripts/benchmark_combined.py
.venv-next/bin/python scripts/accept_body.py
.venv-next/bin/python scripts/evaluate_learning.py
```

Pause the arena before a long experiment. The web interface runs one experiment job at a time. Evaluation artifacts contain seeds and matched conditions. Each browser-launched experiment archives its source, provenance, outcomes, and learned-parameter checkpoints under `data/experiments/<id>`; later runs preserve those archives. Generated files under `data/runs` and `data/checkpoints` can be large. Complete brain checkpoints use trusted local serialization; never import arbitrary pickle files.

## Source and licenses

Brain source: https://github.com/eonsystemspbc/fly-brain (GPL-2.0-or-later; original Shiu Brian2 materials MIT; notices retained). Body source: https://github.com/NeLy-EPFL/flygym (Apache-2.0). Three.js: MIT, notice retained in `static/vendor/THREE-LICENSE.txt`. Annotation source: https://github.com/flyconnectome/flywire_annotations; retain source publication attribution. The pinned annotation README and its five requested citations are retained in `data/annotations-README.md`; a standalone annotation redistribution license was not confirmed. Application code is GPL-2.0-or-later to match the integrated brain source. Dependency licenses remain their own.

Primary learning references: Li et al., eLife 2020, https://elifesciences.org/articles/62576; Huang et al., Nature 2024, https://www.nature.com/articles/s41586-024-07819-w; computational learning model, https://www.nature.com/articles/s41467-021-22592-4.

Shallow step adds a physical 2 × 12 mm patch, 0.15 mm high. Two independent ten-second straight-walking diagnostics traversed this step without flipping. This is limited validation of that geometry, not general terrain stability. The engineered ground predator traverses these patches in its planar navigation model; they do not block line of sight. Tall walls remain obstacles.

Learning-rule audit: `reports/LEARNING_RULE_AUDIT.md` documents how the current engineered depression rule differs from the cited computational model. The separate published-equation diagnostic is GPL-3.0-or-later and never replaces the full-network arena controller.

Learn → Inspect odor pathway responses opens measured time courses from independent diagnostic states. Select a population and seed to compare odor A and B. These recordings do not advance the arena or demonstrate behavioral learning.

The supplied baseline v3 uses three engineered egocentric obstacle rays (8 mm range, 0.25 mm sampling) to turn toward local clearance. Their distances are shown under Observe. The full-brain controller does not use this rule or receive object coordinates; mapping obstacle features into biological circuitry remains unresolved.

Baseline v3 normalizes the bilateral odor difference by total odor exposure and reserves oscillatory exploration for very weak signals. Its wall reflex responds to the front ray under 6 mm or any ray under 1.5 mm. These fixed engineered choices are recorded separately from the full brain; new run manifests include the controller source hash. Earlier v2 failures remain in reports.

Playback follows recorded simulation timestamps, including sparse and irregular sampling. Browser timing checks: `node --test tests/browser/replay.test.mjs`. The baseline acceptance activity saves actual sampled poses and per-run geometry, with food, energy and failures visible in replay.

## Experimental brain integration

The staged integration sequence is in [the plan](reports/brain-integration/PLAN.md). [Step 7's measured report](reports/brain-integration/stage7-20261006/RESULTS.md) documents causal 25 ms command scheduling, continuous full-brain state, 30 Hz physical recording, an exact fresh-process checkpoint continuation, and limited engineered contact input to exact-root ascending neurons. This pipeline is separate from the unchanged live arena controller. Timing checks pass; useful neural walking and downstream touch-driven retreat remain unproven.

Run its sequential diagnostic with `.venv-next/bin/python scripts/diagnose_causal_timing.py`; verify completed artifacts with `.venv-next/bin/python scripts/report_causal_timing.py`. The checkpoint under `reports/brain-integration/stage7-20261006/trials/loom_left-7199-25ms/checkpoint-1.5` is specific to this experimental worker and its pinned sources. The completed continuation and raw recordings are retained alongside the report.

[Step 8](reports/brain-integration/stage8-20261006/RESULTS.md) completed 20 matched seeds, 60 full-brain trials and 40 physical controls. Sensory-dependent motor effects and exact physical replays pass; consistent away-directed movement and useful-response criteria fail. Selected DNp09 forward-drive cells remain silent under these recorded looming inputs. [Step 9's entry report](reports/brain-integration/stage9-20261006/RESULTS.md) therefore blocks navigation progression pending a new validated sensorimotor candidate. No navigation, predator escape or learning claim follows from these results. Run `.venv-next/bin/python scripts/check_behavioral_readiness.py` to re-audit Step 8 and regenerate the entry decision, retaining a hashed evidence snapshot.
