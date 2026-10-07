# Supra Drift (Supra Ai 2)

A from-scratch reinforcement-learning driving simulator built in Python: custom four-wheel vehicle dynamics (Pacejka tyres, combined slip, weight transfer, turbo spooling), procedural and real-world tracks with 2.5D elevation, live synthesized engine audio, PyTorch PPO and NumPy genetic-algorithm training, a Flask web dashboard, and a browser-based 3D Observatory.

The headline program: train a **Mazda 787B** to lap a real-elevation replica of the **Nürburgring Nordschleife** at superhuman pace — beating **Stefan Bellof's 6:11.13 (371.13 s)** qualifying lap, with the outright **Porsche 919 Evo record (5:19.55 / 319.55 s)** as the horizon.

> **This file is the single entry point for the whole repository.** Every other document is indexed in [Documentation map](#documentation-map) below with a currency label, so you always know which docs are authoritative and which are historical.

---

## What's in this repo

| Area | Where | Status |
|------|-------|--------|
| Core simulator (physics, tracks, sensors, audio, viewers) | `supra/`, `run.py` | ✅ Active |
| **Fable Five** — the 787B/919 Nordschleife training pipeline | `supra/fable5.py` | ✅ Active — the main effort |
| Faithful-v2 919 record program (identity-isolated, evidence-gated) | `supra/faithful/`, `supra/record/` | 🔒 Scaffold, fail-closed |
| Command Center web dashboard | `command-center/` (port **8770**) | ✅ Active |
| Fable Observatory — 3D brain playback in the browser | `observatory/` + `supra/observatory.py` (legacy freeze: `legacy/observatory/`) | ✅ Active |
| 3D driving viewer (Three.js + WebSocket physics bridge) | `viewer3d/` (`/3d/` on the dashboard) | ✅ Active |
| Validators, gates, and ops tooling | `tools/` | ✅ Active |
| Side experiments (MuJoCo melee, 2D duels, strip folding) | `blade/`, `duel/`, `foldspace/`, `viewer/`, `foldview/`, `bladeview/` | 🗄️ Dormant (June 2026) |
| Unrelated personal homepage (Node, port 8000) | `project-dashboard/` | ⚠️ Not part of this project |

The repo root also holds dozens of experiment checkpoints (`*.pt`, `*.npz`), pipeline manifests (`*_pipeline.json`), and eval snapshots (`*_eval_latest.json`). Never assume a checkpoint name is current — check the relevant manifest or the live journals first.

---

## Quick start

```bash
# Install (macOS: always python3/pip3, never bare python/pip)
pip3 install -r requirements.txt

# Launch the Command Center — the one-stop control panel
python3 command-center/server.py        # → http://localhost:8770

# Train the 787B on the Nordschleife (Fable Five, auto stage ladder)
python3 run.py --fable 30000 --fable-stage auto

# Watch the current best brain drive
python3 run.py --watch-fable

# Drive it yourself
python3 run.py --drive --track club --car supra
python3 run.py --drive --track akina --car rx7          # tail-happy touge
python3 run.py --drive --gen technical --difficulty 0.85 --length 1400

# Build the Observatory client (needed once after checkout, or after client edits)
cd observatory && npm install && npm run validate && cd ..
# Then open http://localhost:8770/observatory/ from a running Command Center
```

Headless runs (CI, servers) need `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`.

**Driving controls:** Arrows/WASD drive · Space handbrake · Shift clutch · T auto-gearbox toggle · Q/E manual shift · C camera · V broadcast director · 1–5 director shots · P replay · I AI-vision overlay · B sensor beams · G/H/Y/O time-of-day/weather/wet/HQ toggles · F3 perf stats · F11 fullscreen · M mute · R reset · Esc quit.

---

## The simulator

All simulation code lives in `supra/`; `run.py` is the CLI for every mode.

- **Physics** (`physics.py`) — per-corner Pacejka tyres with combined-slip friction ellipse and relaxation length, load-sensitive grip, weight transfer, limited-slip diff, sub-stepped drivetrain (engine → clutch → gearbox → LSD), turbo spooling, aero and rolling drag. 2.5D elevation with slope forces and **emergent** ballistic jumps — nothing is scripted. Details: [docs/02_Physics_Model.md](docs/02_Physics_Model.md).
- **Tracks** (`track.py`) — Catmull-Rom circuits with seed-deterministic elevation.
  - Named: `club`, `national`, `coast`, `sprint`, `tech`, `oval2`, `akina`, `pass`, `endless`
  - Special: `nordschleife` (real OSM centerline + DGM1 elevation, built by `tools/build_nordschleife.py`), `speedbowl`, `superspeed`, `circuit`, `crescent`, `esses`, `hook`, `ridge`
  - Procedural: `--gen gp|technical|speedway|touge` with `--difficulty/--length/--seed`
- **Cars** (`config.py` presets) — `supra`, `rx7`, `skyline`, `lr4`, `f150`, `mazda787b`, `porsche_956`, `porsche_919evo`, `porsche_919_legacy`. The 787B carries the frozen drivetrain identity `mazda787b-5spd-ring-v1` (authentic 5-speed with a torque-aware auto-shift `RaceBox`; the AI keeps a bounded gear-offset action on top). Cleanroom 3D meshes + provenance live in `assets/vehicles/`.
- **Sensors** (`sensors.py`) — the general observation vector is **60-dim**: 9 vision beams + 25 proprioceptive + 6 look-ahead curvature + 18 hills/air + 2 mode flags. (Fable Five uses its own frozen layouts — see below.)
- **Audio** (`sound.py`) — no samples: pure additive synthesis of engine harmonics (the 787B's quad-rotor R26B firing order and beating included), turbo flutter, overrun crackle, tyre squeal, kerb rattle. **Audio V5** is the default and its Doppler is phase-continuous propagation delay at 343 m/s — a hard invariant. Details: [docs/06_Audio_Synthesis.md](docs/06_Audio_Synthesis.md).
- **Viewers** — **Viewer v2** (`viewer2.py`) is the default for drive/watch and is where all new viewer work goes. **Viewer v1** (`app.py` + `background.py` + `carart.py`) is frozen; select it with `--classic` or `SUPRA_CLASSIC_VIEWER=1`.

---

## Training paths

One simulator, several ways to train on it:

| Path | Command | What it is |
|------|---------|------------|
| Genetic algorithm | `python3 run.py --train GENS [--live] [--pop N]`, watch with `--watch` | NumPy MLP brains, elitism + tournament + crossover + mutation |
| PPO race | `python3 run.py --ppo ITERS`, watch with `--watch-ppo` | PyTorch actor-critic with GAE, curriculum, deterministic eval + keep-best |
| PPO drift | `python3 run.py --drift ITERS`, watch with `--watch-drift` | Slip-angle reward, anti-spin shaping |
| PPO hybrid | `python3 run.py --hybrid ITERS [--style ...]` | One goal-conditioned race/drift policy |
| Multi-agent race | `--opponents N`, `--multi-self-play`, watch with `--watch-race` | RaceEnv self-play |
| Ring Race (legacy) | `python3 run.py --ring-race ITERS --ring-stage auto` | Pre-Fable 787B Nordschleife pipeline; superseded by Fable Five |
| **Fable Five** | `python3 run.py --fable ITERS --fable-stage auto` | The current Nordschleife program (below) |
| Fable GA | `python3 run.py --fable-ga GENS` | Evolution baseline on the same Fable task |

Common training flags: `--resume`, `--out`, `--lr`, `--workers`, `--anneal`, `--patience`, `--max-restarts`.

**Judging results:** trust only the deterministic `[eval]` / `[eval-ring]` / `[eval-fable]` lines — rolling training-log numbers overstate reality. Always resume or watch from a `_best.pt`. Use `--fable-diag` (Brain Lab) when you need to know *why* a policy is slow or dying. Training math: [docs/03_Neural_Networks_and_Training.md](docs/03_Neural_Networks_and_Training.md).

---

## Fable Five — the Nordschleife pipeline

`supra/fable5.py` is the centrepiece: an edition-scoped, gated stage ladder that pushes a car around the full 20.8 km Nordschleife from "survive" to "superhuman". A **physics-true speed envelope** computed from the real car model feeds the AI's observations (correct pace for its position plus points down the road), so it anticipates braking zones from physics rather than trial and error.

| Stage | Goal | Gate |
|-------|------|------|
| **Foundation** | Learn the 170+ corner layout at safe pace | Basic lap completion |
| **Flow** | Connect sectors with smooth, linked inputs | Sector consistency |
| **Finish** | Close a full clean lap (30% of spawns in the final sectors) | Clean flying lap |
| **Fast** | Push toward ~96% of the physics envelope | Adaptive segments |
| **Frontier** | Exceed the envelope — racing-line gains beyond the centerline floor | Scale up to 1.15× |

An automated **Pit Wall** supervisor watches every evaluation: it rolls back collapses, reseeds from the best brain by failure mode, and decays the learning rate when refinement stalls, keeping overnight runs productive. `tools/supervise_fable5.py` adds an unattended watchdog (restarts, caffeinate, ntfy phone alerts).

**Editions.** The pipeline is per-car: the 787B (`fable5_ring_*` artifacts, target: Bellof 371.13 s) and a legacy-approximation 919 Evo (`fable5_919_ring_*`, target: 319.55 s — *not* the certifiable faithful-v2 program). **Frozen obs/action layouts:** `fable-v1` = 58 sensors + 8 pace + 2 mode = **68 dims**; hybrid-ledger cars ride `fable-v2` = fable-v1 + battery SOC + signed MGU power = **70 dims**. 3 actions (steer, longitudinal, gear-offset). Checkpoint compatibility depends on this; `PPO.load_policy` reconstructs layouts from checkpoint metadata — never hand-edit dims.

**Live status** (lap times, active runs, rollbacks) is intentionally *not* in this README — it changes nightly. Look at:
- `fable5_ring_eval_latest.json` / `fable5_919_ring_eval_latest.json` — latest deterministic eval per edition
- `fable5_ring_pipeline.json` / `fable5_919_ring_pipeline.json` — stage-ladder manifests
- [FABLE5_919_LIVE_JOURNAL.md](FABLE5_919_LIVE_JOURNAL.md) — newest-first ops journal for the 919 ladder
- [docs/legacy/HANDOFF.md](docs/legacy/HANDOFF.md) — newest-first changelog (787B/Pit Wall era)

Full design: [docs/04_Fable_Five_Pipeline.md](docs/04_Fable_Five_Pipeline.md).

---

## Faithful-v2 Porsche 919 Evo record program

A separate, identity-isolated program under `supra/faithful/` and `supra/record/` — **not** a continuation of the Fable 919 edition. Its contract is the official 20.832 km T13 flying lap strictly below **319.546 s**, with three mandatory boundaries: licensed-data physics validation, an authoritative MuJoCo replay of a CasADi/IPOPT whole-lap oracle, and independent certification of a driver-equivalent recurrent policy.

The checked-in foundation (immutable schemas, signed evidence vault, digest-pinned Linux runtime container in `containers/faithful/`, noncertifying MuJoCo baseline, adversarial validators, recurrent driver network, dimensionally anchored 919 GLB) is a **telemetry-constrained approximation scaffold**: all claim-producing APIs fail closed until licensed tyre/aero/suspension/telemetry/survey evidence is supplied and validated. The stopped legacy 919 run is preserved as `legacy_misstamped_noncertifiable` and can never be resumed or promoted.

CLI surface: `run.py --faithful-919-status / -validate / -init / -oracle / -train` plus the `--faithful-evidence-*` custodian commands. Dedicated deps: `requirements-faithful.txt` (dev) and the hash-pinned `requirements-faithful-lock.txt` (certifying container). Runs live under `runtime/fable5/editions/porsche-919evo-faithful-v2/`.

Full contract: [docs/07_Faithful_919_Record_Program.md](docs/07_Faithful_919_Record_Program.md).

---

## Interfaces & visualization

| Interface | How to reach it | Notes |
|-----------|-----------------|-------|
| **Command Center** | `python3 command-center/server.py` → http://localhost:8770 (or double-click `command-center/SUPRA.command`) | Flask + vanilla JS. Tabs: Ops, Drive, Train, Ring Race, **Fable Five** (Pit Wall), Fable GA, Watch, Race, Observatory, Garage (checkpoints), Diagnostics (Brain Lab), Tracks. SSE process streaming, launch/stop APIs. |
| **Fable Observatory** | http://localhost:8770/observatory/ | Remake Three.js client in `observatory/` (frozen rollback in `legacy/observatory/`). Python (`supra/observatory.py`) owns physics, policy inference, probes, and audio; the browser only renders streamed state. Transport: `command-center/observatory_api.py`. |
| **3D driving viewer** | http://localhost:8770/3d/ (frozen v1 at `/3d/legacy/`) | Three.js viewer with a WebSocket physics bridge (`viewer3d/`). `viewer3d/graphics_v1/` is a restore-by-copy archive; `viewer3d/assets-src/` holds source assets + credits. |
| **PyGame viewers** | `run.py --drive / --watch-*` | Viewer v2 default; v1 via `--classic`. |

Dashboard restart rules: edits to `command-center/server.py` need a server restart; edits to `supra/*.py` / `run.py` apply to newly launched runs automatically; static JS/HTML/CSS are served no-cache — just refresh the browser.

Details: [docs/05_UI_and_Visualization.md](docs/05_UI_and_Visualization.md) and [observatory/README.md](observatory/README.md).

---

## Validation & tooling (`tools/`)

Run the relevant gate before claiming a change works — not ad hoc scripts.

- **Fable Five:** `validate_fable5.py` (envelope, obs layout, env/train smoke), `validate_fable5_auto.py` (hermetic AUTO-ladder guarantees; runs in a scratch dir — use this one, and never run non-hermetic pipeline tests from the repo root, they can clobber real manifests/eval files), `validate_fable5_safeguards.py`, `validate_fable_editions.py`.
- **Faithful 919:** `validate_faithful_919.py` runs the whole foundation suite (`validate_faithful_v2_core / _program / _policy / _record_core / _av / _evidence / _runtime / _runtime_spec`).
- **Observatory:** `build_observatory_assets.py`, `validate_observatory_playback / _transport / _dashboard / _command_center_wiring.py`, plus Playwright suites. Client side: `cd observatory && npm run validate`.
- **Track & physics:** `build_nordschleife.py`, `validate_nordschleife.py`, `validate_track_elevation.py`, `validate_hills.py`, `validate_jumps.py`, `validate_787b_drivetrain.py`, `validate_919_drivetrain.py`, `regression_drivetrain.py`, `regression_baseline.py`, `validate_ring_pipeline.py`.
- **Audio:** `validate_sound.py`, `validate_audio_v4.py` (gates V5 too), `smoke_audio_v4.py`, `render_audio_v4_scenarios.py`.
- **Diagnostics & ops:** `diagnose_checkpoint.py` (deterministic telemetry export), `eval_checkpoint.py` (honest named-set eval), `supervise_fable5.py`, `notify_fable5_ntfy.py`, `repair_fable_manifests.py`, gearing optimizers (`optimize_787b/919/956_gearing.py`), lineage archivers.

---

## Repository layout

| Path | What it is |
|------|------------|
| `supra/` | The simulator: physics, tracks, sensors, PPO (`ppo.py`), Fable Five (`fable5.py`), viewers (`app.py` = frozen v1, `viewer2.py` = v2), audio (`sound.py`), diagnostics (`diagnostics.py`, `fable5_diag.py`), Observatory authority (`observatory.py`), faithful program (`faithful/`, `record/`) |
| `run.py` | CLI entry point for every mode |
| `command-center/` | Flask dashboard (`server.py`, `observatory_api.py`, `static/`) |
| `observatory/` | Remake Three.js Observatory client (npm project; built output served by the dashboard) |
| `legacy/observatory/` | Frozen pre-remake Observatory client (rollback/reference only) |
| `viewer3d/` | 3D driving viewer (active v2 in `static/`, frozen `legacy/`, archive `graphics_v1/`, sources `assets-src/`) |
| `tools/` | Validators, gates, supervisors, one-off diagnostics |
| `docs/` | The numbered documentation set (below) + dated audits + `docs/legacy/` |
| `assets/vehicles/` | Cleanroom vehicle meshes with per-car README + provenance |
| `runtime/` | Generated/ephemeral artifacts: Fable edition banks, faithful run skeletons, QA captures, logs — not source |
| `diagnostics/` | Brain Lab / eval output dumps |
| `containers/faithful/` | Digest-pinned Linux runtime for faithful-v2 certification |
| `legacy/`, `archives/`, `backups/`, `corrupted_brains/`, `scratch/` | Preserved history and scratch space |
| `.agents/` | Scratch notes from prior AI sub-agent runs — not a maintained map |
| `blade/`, `duel/`, `foldspace/`, `viewer/`, `foldview/`, `bladeview/` | **Dormant side experiments** (June 2026), self-contained: MuJoCo curriculum melee (`python3 -m blade.train/view/arena`), 2D self-play duels (`python3 -m duel.*` + `viewer/` on port 8799), PPO strip-folding (`python3 -m foldspace.*` + `foldview/` on port 8798). `bladeview/` is an empty stub. Each has its own README. |
| `project-dashboard/` | **Unrelated** personal Node homepage (port 8000) — not part of this project |
| Repo root `*.pt` / `*.npz` / `*_pipeline.json` / `*_eval_latest.json` | Experiment checkpoints, manifests, eval snapshots — check journals/manifests before trusting a name |

---

## Documentation map

**Current & authoritative**

| Doc | Contents |
|-----|----------|
| [docs/01_Overview.md](docs/01_Overview.md) | Dependencies, architecture, controls |
| [docs/02_Physics_Model.md](docs/02_Physics_Model.md) | Pacejka tyres, combined slip, 2.5D hills, emergent jumps |
| [docs/03_Neural_Networks_and_Training.md](docs/03_Neural_Networks_and_Training.md) | 60-dim general obs space, GA, PPO, race vs drift rewards |
| [docs/04_Fable_Five_Pipeline.md](docs/04_Fable_Five_Pipeline.md) | The Nordschleife program: stages, envelope, Pit Wall, editions |
| [docs/05_UI_and_Visualization.md](docs/05_UI_and_Visualization.md) | Command Center, viewers, Observatory |
| [docs/06_Audio_Synthesis.md](docs/06_Audio_Synthesis.md) | Audio V5: additive synthesis, Doppler contract |
| [docs/07_Faithful_919_Record_Program.md](docs/07_Faithful_919_Record_Program.md) | Evidence gates, authority/twin, oracle, certification |
| [observatory/README.md](observatory/README.md) | Observatory remake build/validate and clean-room contract |
| [legacy/observatory/README.md](legacy/observatory/README.md) | Frozen pre-remake Observatory client |
| [CLAUDE.md](CLAUDE.md) | Orientation + hard invariants for AI agents working in this repo |

**Live operational logs** (newest-first; the place for "what's happening right now")

| Doc | Contents |
|-----|----------|
| [FABLE5_919_LIVE_JOURNAL.md](FABLE5_919_LIVE_JOURNAL.md) | 919 ring auto-ladder ops journal |
| [docs/legacy/HANDOFF.md](docs/legacy/HANDOFF.md) | Long-running project changelog and design-decision record (787B/Pit Wall era; predates the 919 journal) |

**Point-in-time reports** (accurate for their date, not maintained)

| Doc | Contents |
|-----|----------|
| [docs/FABLE5_AUTONOMOUS_AUDIT_2026-07-12.md](docs/FABLE5_AUTONOMOUS_AUDIT_2026-07-12.md) | Audit of unattended Fable training + fixes |
| [docs/FABLE5_919_FRONTIER_PIPELINE_DOCTOR_2026-07-22.md](docs/FABLE5_919_FRONTIER_PIPELINE_DOCTOR_2026-07-22.md) | Diagnosis of the 919 frontier ladder |

**Historical / archived** (kept for provenance; do not treat as current)

| Doc | Contents |
|-----|----------|
| [docs/legacy/](docs/legacy/) | Original plans and analyses: `FABLE5_PLAN.md`, `PHYSICS_3D_PLAN.md`, `LR4_BUILD_PLAN.md`, `supra_drift_analysis.md`, the pre-Observatory project README |
| [walkthrough.md](walkthrough.md) | Audio crash/GIL debugging narrative; only its closing Audio V5 contract matches current behavior |
| [presentation_assets/ASSET_PACK.md](presentation_assets/ASSET_PACK.md) | Slide-deck data pack — its lap numbers are stale |
| Sub-project READMEs (`blade/`, `duel/`, `foldspace/`, `viewer3d/legacy/`, `viewer3d/graphics_v1/`, `legacy/*`, `assets/vehicles/*`) | Self-contained docs for their own directories; accurate locally but not project status |

---

## Conventions & hard invariants

These are enforced expectations — don't break them without an explicit decision (they mirror [CLAUDE.md](CLAUDE.md)):

1. **`python3`/`pip3` only**, never bare `python`/`pip`. Headless: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`.
2. **Fable obs/action layouts are frozen per car** (`fable-v1` 68-dim, `fable-v2` 70-dim, 3 actions). Never hand-edit dims; checkpoint compatibility depends on them.
3. **Viewer v1 is frozen** (`--classic`); all new viewer work goes in `supra/viewer2.py`.
4. **Audio V5 Doppler is required** — phase-continuous propagation delay at 343 m/s; never add independent pitch shifting after the delay; 100 ms crossfade on camera cuts; listener-local wind excluded.
5. **No envelope heat ribbon on the road surface** — `trk.fable_vref` feeds the HUD pace readout only.
6. **Trust only deterministic `[eval*]` lines**; resume/watch from `_best.pt`; prefer `--fable-diag` when diagnosing.
7. **Run the hermetic gates** (`tools/validate_fable5.py`, `tools/validate_fable5_auto.py`) before claiming a Fable change works — and never run non-hermetic pipeline tests from the repo root, they can clobber real manifests.
8. Command Center: restart the server for `server.py` edits; `supra/`/`run.py` edits apply to new runs; static assets are no-cache.
