# Fly Garden: start here for an independent review

Fly Garden is Carson Hicks's local research and visualization project, developed with substantial AI assistance from OpenAI Codex. Carson supplied the goals, project direction and review decisions; Codex wrote and revised much of the implementation, designed diagnostics and operated experiments under Carson's authorization. The connectome data, upstream neural model, physics engine and body model were produced by their credited researchers. This is not a claim that Carson independently discovered or reconstructed a fly brain.

## What it does

The application puts a modeled fly in a small 3D arena. A Python service advances a neural simulation and a physical body; a browser displays the arena and recorded neural activity. Odor is sampled locally, neural spikes are translated into movement commands, and a supplied walking controller coordinates the legs. Food, obstacles and a ground predator provide experimental situations.

The neural graph represents 138,639 neurons, 15,091,983 aggregated connection records and 54,492,922 represented anatomical synapses from FlyWire version 783. An anatomical connection is not a full biophysical description: each modeled neuron is a simplified electrical unit. A saved state preserves model variables, not a claim of consciousness or an animal's identity.

## How it is made

1. Import upstream graph data and exact neuron IDs, preserving provenance and hashes.
2. Use Brian2 to simulate electrical units and synaptic events on a shared clock.
3. Encode engineered sensory signals into annotated input populations.
4. Decode selected descending-neuron activity into bounded forward and turning commands.
5. Use FlyGym/NeuroMechFly and MuJoCo for body mechanics and supplied leg coordination.
6. Stream sampled state to a Three.js interface; record spikes and poses in compressed chunks.
7. Preserve full checkpoints for continuation and recordings for inspection, playback and video export.
8. Test claims using fixed seeds, matched controls, frozen protocols and retained failures.

Creating the application mostly involves writing adapters and infrastructure around existing scientific tools, then checking whether the assumptions produce useful behavior. Connecting a graph to a body does not automatically make it a functioning biological brain.

## Current scientific problem

An odor pulse starts activity in the modeled smell circuit that persists after the pulse ends. Persistent activity makes subsequent cues and left/right differences difficult to use. This is a simulation failure, not a diagnosis of epilepsy in a real fly.

The previous fixed-point adaptive-LIF screen completed 21 full-network trials. All 5,040 chunks passed integrity checks; disabling adaptation exactly reproduced baseline events and voltage/conductance states. Adaptation lowered firing but failed recovery and directional contrast at the tested setting. No candidate was promoted.

The next sweep tests a temporary firing cooldown: each spike adds a voltage-equivalent adaptation term that decays over time. The fixed grid uses 50/150/450 ms decay and 1.5/4.5/13.5 mV increments, separately in projection neurons and in projection plus annotated olfactory receptor neurons. Two new calibration seeds yield 153 initial trials. Every acceptance gate must pass; only a qualifying candidate receives fresh held-out tests, including unequal bilateral odor intensity. These parameters are engineering hypotheses, not fitted calcium/potassium channels.

At publication, the sweep is in progress. The committed progress file is a snapshot, not a live feed. Do not interpret unfinished arms as final results or assume future completion from this document.

## What is established and what is not

- The full available modeled graph runs, with individual spike recordings and persistent state.
- Separate diagnostics verify selected checkpoint continuation and recording integrity.
- The application contains an arena, supplied locomotion, recording/replay and visualization machinery; the complete delivery workflow still has pending acceptance items.
- Useful sensory-dependent directional choice remains unresolved.
- Biological fidelity, successful embodied learning and a fully validated sensory-to-motor brain are not established.
- Tonic walking support, engineered escape behavior and synthetic sensory encoding must remain visible in explanations of gameplay.
- Colorful anatomy, movement, weight changes and a compelling video do not establish intelligence, learning or consciousness.

## Read these files

- `README.md`: application overview and limitations.
- `reports/provenance.json`: upstream versions, hashes and alterations.
- `reports/brain-integration/acceptance-ledger.json`: outstanding requirements.
- `reports/brain-integration/recovery/feedback-trace-v1/RESULTS.md`: persistent-activity reconstruction.
- `reports/brain-integration/recovery/adaptive-cell-reference-v1/RESULTS.md`: isolated adaptation checks.
- `reports/brain-integration/recovery/adaptive-domain-screen-v1/RESULTS.md`: completed failed fixed-point screen.
- `reports/brain-integration/recovery/adaptive-parameter-sweep-v1/README.md` and `protocol.json`: running sweep.
- `flygarden/brain.py`, `adaptive_neurons.py`, `adaptive_brain.py`, `adaptive_candidate.py`: baseline and adaptation implementation.
- `scripts/sweep_adaptive_lif.py`: frozen experiment runner and acceptance evaluation.
- `flygarden/continuous_candidate.py`, `candidate_inputs.py`, `descending.py`: sensory encoding and motor decoding.
- `flygarden/body.py`, `world.py`, `recording.py`, `server.py`: embodied application infrastructure.

## Questions for Muse

Please assess whether the explanations accurately separate Carson's project design, Codex's implementation, and upstream research. Examine whether the neural assumptions and acceptance gates support the proposed claims; identify concrete numerical, causal, reproducibility and usability problems. Suggest useful applications that fit the demonstrated capabilities. Treat missing raw data as unavailable evidence rather than inferring it from summary reports. Do not assume successful learning or consciousness.

## Repository scope

This is a source-and-evidence snapshot, not a backup of the complete local research archive. Large graph downloads, annotations with unresolved redistribution terms, morphology caches, raw spike recordings, videos, checkpoints, virtual environments, runtime state and dependency installations are excluded. No local recordings were deleted. Read `docs/REPRODUCIBILITY.md` and `docs/SHARE-MANIFEST.json` for what is included and excluded. Report links or hashes may refer to excluded artifacts; their inclusion in a report does not mean the underlying file is available here.
