# The Cryptographic Heist Engine

A cinematic-first multi-agent pursuit simulation with adversarial radio traffic,
spoofing, scanner confidence, and high-fidelity vehicle dynamics adapted from
the local `Supra Ai 2` project.

The current build is a hardened v1 backbone:

- 1 evader and 5 pursuers in a procedural urban grid.
- Supra-derived four-corner Pacejka physics, drivetrain, weight transfer,
  handbrake, and telemetry.
- Asymmetric heist vehicle presets: light drift evader vs heavy pursuit cars.
- Scripted chase policies for watchable baseline episodes.
- Discrete English-token radio channel with spoof/jamming events.
- PettingZoo-style parallel environment for MARL training.
- Fixed-shape observations, action spaces, rewards, and infos, including live
  decoder-accuracy, spoof-susceptibility, and counterfactual deception reward
  terms.
- Versioned JSONL replay recording and validation.
- Stable replay state/radio fingerprints plus a replay-fidelity gate for
  same-seed determinism evidence.
- Replay-only research reports and a full evidence bundle that tie together
  readiness manifests, replay report scores, replay hashes, checkpoint hashes,
  dashboard assets, and active acceptance artifacts.
- Physics acceptance gate for deterministic traces, evader/pursuer handling
  asymmetry, braking stability, collision recovery, and long-run boundedness.
- Pygame spectator viewer plus a dashboard-linked web replay cockpit with
  camera follow, transcript, scanner predictions, confidence, and procedural
  audio mapping.
- Spectator UI/audio acceptance gate for Pygame headless rendering, web replay
  cockpit contracts, dashboard replay links, and confidence-driven soundtrack
  buckets.
- PyTorch policy modules with recurrent pursuer comms and evader decoder/jammer
  heads.
- First trainable workflows: scripted evader imitation, PPO evader waypoint
  driving, PPO single-pursuer chase control, and shared PPO five-pursuer
  containment.
- First self-play bridge: PPO driving lanes can train against saved evader or
  pursuer-team checkpoint opponents instead of only scripted opponents.
- First self-play scheduler: one command alternates evader/team PPO updates,
  promotes active control checkpoints, records replay-backed control scorecards,
  and can advance the radio/scanner/jammer information stack.
- Scenario-gated self-play now warm-starts the evader candidate from the
  accepted `checkpoints/evader_ppo.pt` and keeps a validation-selected policy,
  preventing promotion cycles from discarding a repaired waypoint driver.
- First control-pair league gate: learned evader and learned five-pursuer team
  checkpoints can be evaluated together across seeds with replay evidence,
  capture/waypoint metrics, spectacle scoring, and adversarial balance scoring.
- First historical control-opponent pool: promoted evader/team checkpoint pairs
  are archived as generations and sampled by later self-play runs, moving the
  training loop beyond a single active opponent snapshot.
- First cross-generation control promotion gate: new driving candidates can be
  scored against historical evaders and pursuer teams before becoming the active
  pair, with evader and pursuer-team checkpoints promoted independently using
  separately tunable thresholds and improvement margins.
- First information-warfare workflow: supervised scanner decoder training from
  English-token radio history to future pursuer positions.
- First learned comms workflow: supervised pursuer radio-token policy that emits
  fixed-length English messages through the simulator's normal `tokens` action.
- First adversarial workflow: evader jammer policy that learns when to trigger
  spoof bursts, which English tokens to inject, and which pursuers to target.
- First counterfactual jammer workflow: paired jam/no-jam rollout labels weight
  spoof training by actual pursuer trajectory deviation.
- First combined adversarial evaluation: learned pursuer radio, evader scanner,
  and evader jammer can run together as one replayable stack with metrics.
- First authentication curriculum gate: paired jammed/no-spoof episodes produce
  counterfactual spoof-susceptibility and reward-proxy metrics.
- First checkpoint league gate: authentication metrics can be aggregated across
  seeds to score and optionally promote a radio/scanner/jammer stack.
- First train/evaluate/promote cycle: scanner, radio, and jammer candidates can
  be trained together, compared against the active stack, and promoted from one
  command.

## Setup

```bash
pip3 install -r requirements.txt
```

PettingZoo is declared in `requirements.txt`; the local environment wrapper also
has a fallback so smoke tests still run if PettingZoo is not installed yet.

## Run The Viewer

```bash
python3 run.py
# or
python3 scripts/run_viewer.py --seed 11
```

Controls:

- `Space`: pause
- `R`: reset
- `M`: mute/unmute procedural soundtrack
- `Esc`: quit

## Command Center

```bash
python3 scripts/dashboard.py
```

Open `http://127.0.0.1:8788`.

The dashboard reads `configs/commands.json`, launches registered commands,
streams job logs, indexes JSONL replays, links each replay into the cinematic
web cockpit, and edits JSON configs with a backup file written beside the saved
config. Its first screen includes a readiness board powered by `/api/readiness`,
which summarizes current acceptance status,
learned evader checkpoint acceptance, active learned-control acceptance,
learned information-stack league status, full-plan milestone progress, missing
learned-policy checkpoints, replay evidence, and the next recommended
curriculum command. It also surfaces the replay-fidelity manifest, including
state/radio checksums and same-seed scripted determinism status. Readiness is
intentionally lightweight: it indexes replay files quickly and leaves full
replay validation to explicit replay commands. The physics panel summarizes
`logs/physics_validation.json`, including handbrake yaw/radius asymmetry and
check counts. The spectator panel summarizes `logs/spectator_validation.json`,
including UI/audio checks and the active low/mid/high soundtrack buckets. The
self-play panel summarizes `logs/selfplay/selfplay_scenario_selfplay_manifest.json`,
including the latest promotion decision, scenario required-check count, and
failure reason. When the active build is green but the newest scenario-gated
self-play candidate fails, readiness routes the next command from that
candidate's scenario diagnostics instead of hiding the failure behind the
nominal curriculum plan. It also surfaces the research evidence bundle and its
live verification manifest; if the operational gates are green but the proof
bundle is missing or stale, the readiness board routes directly to
`build_evidence_bundle` and then `verify_evidence_bundle`. The Curriculum view
reads `logs/curriculum_plan.json` and can launch the plan's recommended
registered training command directly from the repair card, then launch any
listed follow-up watch/evaluation gates from the same card. Successful dashboard
acceptance jobs regenerate that plan automatically, so the repair cards follow
the latest scenario gate. The dashboard also writes
`logs/curriculum_history.json`, giving each repair card a compact runbook with
the latest repair job, follow-up job, and diagnostic trend. Replay indexing is
recursive under `replays/`, so acceptance and league subfolders show up in the
same evidence table. The dashboard also exposes `/api/replay?path=...` as a
sampled replay payload endpoint for playback-only frontends.

## Web Replay Cockpit

```bash
python3 scripts/dashboard.py --port 8788
```

Open `http://127.0.0.1:8788/viewer3d/index.html`, or use any replay row's
`3D` action in the dashboard. The cockpit loads JSONL replay data through the
dashboard API and does not step the simulator. It renders a cinematic chase
camera, tire trails, smoke cues, waypoint and event flares, pursuer containment
lines, scanner future-position vectors, a spectator-only spoof-highlighted
radio transcript, confidence meter, and browser-gated confidence-driven audio.

## Operational MARL v1 Gates

The current operational-v1 definition is evidence driven:

- `logs/acceptance_scenarios.json` and `logs/acceptance_full_targets.json` must
  pass required checks and the downtown waypoint milestone.
- `logs/evader_checkpoint_acceptance.json`,
  `logs/pursuer_team_checkpoint_acceptance.json`, and
  `logs/control_acceptance_active.json` must prove learned control gates.
- `logs/checkpoint_league_manifest.json` and `models/active/manifest.json` must
  show a passing learned radio/scanner/jammer information stack.
- `logs/replay_fidelity.json` must show valid replay checksums and a passing
  scripted determinism proof.
- `logs/physics_validation.json` must show a passing vehicle physics acceptance
  manifest.
- `logs/env_validation.json` must show a passing PettingZoo parallel API and
  MARL environment contract manifest.
- `logs/spectator_validation.json` must show passing Pygame, web replay, replay
  link, and audio mapping checks.
- `logs/operational_readiness.json` must aggregate and pass the full
  evidence set, including scenario gates, learned control, learned
  radio/scanner/jammer, self-play promotion, replay schema, dashboard commands,
  spectator/audio checks, and differentiable English-token comms.
- `/api/readiness` should report `operational` with all expected learned
  checkpoints present.

## Python Gym

```bash
python3 scripts/python_tutor.py --port 8797
```

Open `http://127.0.0.1:8797`.

Python Gym is a standalone browser program for learning Python through
randomly generated exercises. It starts with arithmetic, variables, strings,
branches, loops, and functions, then moves through data structures, JSON,
exceptions, tests, classes, generators, decorators, context managers, async
code, descriptors, and algorithms. The app generates fresh variants by topic,
checks short answers immediately, runs code exercises in a small local
subprocess checker, and stores progress in browser local storage.

## Replay And Evaluation

```bash
python3 scripts/record_replay.py replays/demo.jsonl --steps 1800
python3 scripts/render_replay.py replays/demo.jsonl
python3 scripts/verify_replay_fidelity.py replays/control_acceptance_active/downtown_chase_seed11.jsonl --json-out logs/replay_fidelity.json --scripted-determinism --scripted-seed 11 --scripted-steps 1200
python3 scripts/verify_physics.py --json-out logs/physics_validation.json
python3 scripts/verify_env.py --json-out logs/env_validation.json
python3 scripts/verify_spectator.py --json-out logs/spectator_validation.json
python3 scripts/verify_operational_readiness.py --json-out logs/operational_readiness.json
python3 scripts/play_replay.py replays/demo.jsonl
python3 scripts/eval.py --seeds 5 --steps 1800
python3 scripts/eval_acceptance_scenarios.py --seed 11 --record-dir replays/acceptance --out logs/acceptance_scenarios.json
python3 scripts/eval_acceptance_scenarios.py --seed 11 --record-dir replays/acceptance_full --out logs/acceptance_full_targets.json --downtown-min-waypoints 3 --strict
python3 scripts/eval_acceptance_scenarios.py --scenarios downtown_chase --seed 11 --evader-checkpoint checkpoints/evader_ppo.pt --record-dir replays/evader_checkpoint_acceptance --out logs/evader_checkpoint_acceptance.json
python3 scripts/eval_acceptance_scenarios.py --scenarios roadblock --seed 11 --pursuer-team-checkpoint checkpoints/pursuer_team_ppo.pt --record-dir replays/pursuer_team_checkpoint_acceptance --out logs/pursuer_team_checkpoint_acceptance.json
python3 scripts/eval_acceptance_scenarios.py --seed 11 --evader-checkpoint models/control_active/evader_ppo.pt --pursuer-team-checkpoint models/control_active/pursuer_team_ppo.pt --record-dir replays/control_acceptance_active --out logs/control_acceptance_active.json --downtown-min-waypoints 3 --strict
python3 scripts/plan_curriculum_from_manifest.py --manifest logs/acceptance_scenarios.json --out logs/curriculum_plan.json
python3 scripts/train.py --cycles 120
python3 scripts/train_imitation.py --out checkpoints/evader_imitation.pt --steps 12000 --epochs 20 --dagger-rounds 5 --dagger-steps 1200
python3 scripts/train_evader_ppo.py --out checkpoints/evader_ppo.pt --init-checkpoint checkpoints/evader_imitation.pt --bootstrap-imitation-if-missing --bootstrap-imitation-steps 12000 --bootstrap-imitation-epochs 20 --bootstrap-imitation-dagger-rounds 5 --bootstrap-imitation-dagger-steps 1200 --validation-steps 1200
python3 scripts/train_pursuer_ppo.py --out checkpoints/pursuer_ppo.pt
python3 scripts/train_pursuer_team_ppo.py --out checkpoints/pursuer_team_ppo.pt
python3 scripts/train_evader_ppo.py --out checkpoints/evader_selfplay.pt --opponent-pursuer-team-checkpoint checkpoints/pursuer_team_ppo.pt
python3 scripts/train_pursuer_team_ppo.py --out checkpoints/pursuer_team_selfplay.pt --opponent-evader-checkpoint checkpoints/evader_ppo.pt
python3 scripts/train_scanner_decoder.py --out checkpoints/scanner_decoder.pt
python3 scripts/train_radio_policy.py --out checkpoints/radio_policy.pt
python3 scripts/train_jammer_policy.py --out checkpoints/jammer_policy.pt
python3 scripts/train_counterfactual_jammer_policy.py --out checkpoints/jammer_counterfactual.pt
python3 scripts/run_information_cycle.py --name candidate --seed 11 --league-seeds 11,12,13 --promote --improvement-margin 0.01
python3 scripts/run_self_play_cycle.py --name selfplay --seed 11 --evader-control-improvement-margin 0.01 --pursuer-team-control-improvement-margin 0.01
python3 scripts/run_self_play_cycle.py --name selfplay_scenario --seed 11 --evaluate-control-scenarios --control-scenario-seed 31 --control-scenario-threshold 0.90 --evader-control-scenario-threshold 0.45 --pursuer-team-control-scenario-threshold 0.50
python3 scripts/eval_control_pair.py --name active_control --evader models/control_active/evader_ppo.pt --pursuer-team models/control_active/pursuer_team_ppo.pt --seeds 11,12 --record-prefix replays/control_pair --out logs/control_pair_manifest.json
python3 scripts/eval_control_league.py --name candidate_control --evader models/control_active/evader_ppo.pt --pursuer-team models/control_active/pursuer_team_ppo.pt --pool-dir models/control_pool --seeds 11,12 --out logs/control_league_manifest.json
python3 scripts/list_control_pool.py --pool-dir models/control_pool
python3 scripts/watch_policy.py --checkpoint checkpoints/evader_imitation.pt --record replays/evader_checkpoint.jsonl
python3 scripts/watch_policy.py --checkpoint checkpoints/evader_ppo.pt --record replays/evader_ppo.jsonl
python3 scripts/watch_pursuer_policy.py --checkpoint checkpoints/pursuer_ppo.pt --record replays/pursuer_ppo.jsonl
python3 scripts/watch_pursuer_team_policy.py --checkpoint checkpoints/pursuer_team_ppo.pt --record replays/pursuer_team_ppo.jsonl
python3 scripts/watch_scanner_decoder.py --checkpoint checkpoints/scanner_decoder.pt --record replays/scanner_decoder.jsonl
python3 scripts/watch_radio_policy.py --checkpoint checkpoints/radio_policy.pt --record replays/radio_policy.jsonl
python3 scripts/watch_jammer_policy.py --checkpoint checkpoints/jammer_policy.pt --record replays/jammer_policy.jsonl
python3 scripts/watch_adversarial_stack.py --radio checkpoints/radio_policy.pt --scanner checkpoints/scanner_decoder.pt --jammer checkpoints/jammer_policy.pt --record replays/adversarial_stack.jsonl --json-out replays/adversarial_stack_metrics.json
python3 scripts/eval_authentication_curriculum.py --radio checkpoints/radio_policy.pt --scanner checkpoints/scanner_decoder.pt --jammer checkpoints/jammer_policy.pt --record-prefix replays/auth_curriculum --json-out replays/auth_curriculum_metrics.json
python3 scripts/eval_checkpoint_league.py --name candidate --radio checkpoints/radio_policy.pt --scanner checkpoints/scanner_decoder.pt --jammer checkpoints/jammer_policy.pt --seeds 11,12,13 --out logs/checkpoint_league_manifest.json
python3 scripts/watch_policy_viewer.py --checkpoint checkpoints/evader_ppo.pt
python3 scripts/watch_pursuer_policy_viewer.py --checkpoint checkpoints/pursuer_ppo.pt
python3 scripts/watch_pursuer_team_policy_viewer.py --checkpoint checkpoints/pursuer_team_ppo.pt
python3 scripts/watch_scanner_decoder_viewer.py --checkpoint checkpoints/scanner_decoder.pt
python3 scripts/watch_radio_policy_viewer.py --checkpoint checkpoints/radio_policy.pt
python3 scripts/watch_jammer_policy_viewer.py --checkpoint checkpoints/jammer_policy.pt
python3 scripts/watch_adversarial_stack_viewer.py --radio checkpoints/radio_policy.pt --scanner checkpoints/scanner_decoder.pt --jammer checkpoints/jammer_policy.pt
```

`render_replay.py` currently validates and summarizes JSONL logs, including
stable state and radio transcript fingerprints. `verify_replay_fidelity.py`
writes a replay evidence manifest and can generate two same-seed scripted
episodes to prove deterministic replay-state checksums. The static
web cockpit in `viewer3d/static/` can load indexed replays directly from the
dashboard, deep-link to a specific replay via `?replay=replays/...jsonl`, or load
a local JSONL file by file picker/drag-and-drop.

`replay_report.py` writes replay-only research reports under
`logs/replay_reports/`, and `build_evidence_bundle.py` writes
`logs/evidence_bundle.json`, a reproducible snapshot of current readiness
manifests, active acceptance replay reports, replay fingerprints, active
checkpoint hashes, command registry hash, and dashboard/static asset hashes.
`verify_evidence_bundle.py` re-hashes those artifacts and revalidates the active
replays, catching stale or edited evidence bundles before they are treated as
proof. The clean closeout sequence is: run operational readiness, rebuild the
bundle, verify the bundle immediately, then refresh `/api/readiness`.

The v2 research-grade track is documented in `docs/v2_research_grade.md`. It
adds a longer multi-seed checkpoint league, a Three.js renderer contract, and
`build_v2_report.py`, which produces JSON and Markdown reports from the long
league, replay diagnostics, spectator validation, operational readiness, and
verified evidence bundle.

`verify_physics.py` writes `logs/physics_validation.json`, proving the current
vehicle model preserves the evader/pursuer tactical asymmetry and remains
finite under braking, collision recovery, deterministic traces, and longer
scripted chases.

`verify_spectator.py` writes `logs/spectator_validation.json`, proving the
current replay-facing spectator layer can run Pygame headlessly, serve replay
payloads to the web cockpit, deep-link dashboard replay rows, and expose
confidence-driven low/mid/high soundtrack modes.

`eval_acceptance_scenarios.py` is the current "is it working?" gate. It records
four replay-backed scenarios: downtown chase, roadblock capture, spoof burst,
and cipher shift. The default gate is tuned for the current operational
backbone, while seed `31` with `--downtown-min-waypoints 3 --strict` enforces
the original full-plan downtown waypoint target as a stricter milestone run.
Each scenario manifest includes diagnostics with owner, severity, reason,
evidence, and a suggested next action, so dashboard job logs can explain why a
candidate failed instead of only reporting pass/fail.
`plan_curriculum_from_manifest.py` converts those diagnostics into prioritized
dashboard commands, so a failure like `waypoint_milestone_gap`,
`weak_boxing`, or `jam_not_deceptive` becomes a concrete next training lane.
Self-play can also opt into `--evaluate-control-scenarios`, which runs learned
evader/team checkpoints through those scenarios and exposes a scenario
acceptance score to the promotion gate. Use
`--evader-control-scenario-threshold` and
`--pursuer-team-control-scenario-threshold` to let waypoint-escape progress and
roadblock-containment progress advance independently.

## Architecture

```text
crypt_heist/
  physics.py       Supra-derived high-fidelity vehicle model
  config.py        CarSpec presets, including evader and pursuer
  city.py          procedural grid, buildings, road sampling, raycasts
  sim.py           deterministic chase core, scripted/external actions, capture
  env.py           PettingZoo-style parallel training environment
  observations.py  fixed-shape per-agent observation vectors
  rewards.py       shaped chase and information-warfare reward components
  comms.py         English-token radio channel and spoofed jamming
  replay.py        versioned JSONL replay recording/validation
  marl.py          PyTorch policy modules for comms, scanner, and jamming
  ppo.py           PPO trainer/checkpoint paths with counterfactual info shaping
  scanner.py       trainable scanner decoder and runtime confidence updater
  radio.py         trainable pursuer radio-token policy
  jamming.py       heuristic and counterfactual evader spoof/jamming policies
  adversarial.py   combined radio/scanner/jammer evaluation and authentication gates
  league.py        checkpoint scoring and active-stack promotion gates
  cycle.py         train/evaluate/promote orchestration for the information stack
  control_league.py replay-backed learned evader/team control scoring
  control_pool.py historical learned-control opponent archive and sampler
  selfplay.py      alternating control/info self-play cycle orchestration
  scenarios.py     replay-backed acceptance gates for working-demo readiness
  physics_validation.py vehicle physics acceptance metrics and manifest writer
  spectator_validation.py spectator UI/audio acceptance metrics and manifest writer
  app.py           Pygame spectator viewer
  sound.py         confidence-driven procedural soundtrack sketch
```

Supporting folders:

- `configs/`: simulation, curriculum, and renderer defaults.
- `docs/`: architecture, reward, replay, and training notes.
- `scripts/`: viewer, replay, eval, and training smoke entrypoints.
- `viewer3d/`: dashboard-linked replay cockpit and future polished renderer.

## Tests

```bash
python3 -m unittest discover -s tests
SDL_VIDEODRIVER=dummy python3 run.py --mute --max-frames 5
```

The suite covers deterministic stepping, vehicle asymmetry, jamming events,
observation/env contracts, replay round-tripping, spectator validation, and
PyTorch policy shapes.
