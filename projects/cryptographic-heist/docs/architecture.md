# Architecture

The project is split into four layers:

1. Simulation core: deterministic physics, city, comms, rewards, observations,
   and capture logic. It must run headless and must not import renderer code.
2. Training environment: a PettingZoo-style parallel environment built around
   the simulation core.
3. Replay/event log: a versioned JSONL bridge between simulation/training and
   renderers.
4. Spectator frontends: Pygame first for iteration, Three.js later for polished
   playback from replay logs.

The Supra-derived `Vehicle` physics remains the high-fidelity truth model. Any
future JAX/MJX backend should be introduced as an accelerated proxy and checked
against this model before it is trusted.

Current training code follows the same boundary: `crypt_heist/ppo.py` trains
single-agent actor-critics and a parameter-shared five-pursuer actor-critic
through the PettingZoo-style environment, saves checkpoints, and hands those
checkpoints back to the simulation/replay/viewer layer through checkpoint
controllers for evaders, individual pursuers, and pursuer teams.
Those PPO lanes can now load saved checkpoint controllers as opponents, which
is the first bridge from scripted curriculum training toward rotating
self-play.

The environment reward contract now includes the first live authentication
pressure from the baseline spec. `crypt_heist/rewards.py` queues decoder
predictions, settles them against realized future pursuer positions, converts
that into a decoder-accuracy penalty for pursuers, and combines active spoof
offsets with jamming deception into evader information reward and pursuer
spoof-susceptibility penalties. PPO training stats expose those components so
dashboard jobs can inspect whether policies are learning the chase or the
information layer. `CryptHeistParallelEnv` can also run a low-frequency paired
counterfactual probe during PPO training: it evaluates a candidate spoof against
a no-jam branch from the same state and adds the measured deception score as
evader reward and pursuer authentication penalty.

The first trainable information-warfare component follows the same pattern:
`crypt_heist/scanner.py` trains an evader scanner decoder from radio-token
windows to future pursuer positions, then its runtime updates replay/viewer
scanner predictions and confidence without changing the simulation core.

`crypt_heist/radio.py` adds the first learned pursuer communication policy. It
imitates scripted six-token broadcasts from pursuer observations, exposes a
tested straight-through Gumbel-Softmax sequence for differentiable comms work,
and plugs back into the existing `tokens` action field. The generic recurrent
`PursuerPolicy` emits the same fixed-length relaxed token sequence alongside
continuous controls, keeping the training contract aligned with the radio
runtime.

`crypt_heist/jamming.py` adds the matching first evader jamming policy. It
predicts jam triggers, spoof token payloads, and victim masks, then feeds the
existing `jam`, `spoof_tokens`, and `target_mask` action fields so spoofed radio
events, confidence drops, and deception metrics remain owned by the simulation.
It now has two training lanes: a fast heuristic imitation lane and a paired
counterfactual lane that clones each sampled sim state, rolls out jammed and
no-jam branches, and weights jammer labels by actual pursuer trajectory
deviation plus confidence damage.

`crypt_heist/adversarial.py` is the current integration point for the
information-warfare loop. It composes learned pursuer radio, learned evader
scanner predictions, and learned evader jamming in one deterministic episode,
then emits replay-safe metrics for decoder error, unique spoof events,
confidence drops, and spoof-susceptibility proxies. It also owns the paired
authentication curriculum evaluator, which compares jammed and no-spoof replays
from the same seed to estimate counterfactual pursuer trajectory deviation and
reward proxies for both factions.

`crypt_heist/league.py` turns those paired metrics into promotion decisions. It
aggregates authentication evaluations across seeds, computes pursuer security,
evader pressure, and adversarial balance scores, and can copy a passing
radio/scanner/jammer stack into an active manifest for later self-play runs.

`crypt_heist/cycle.py` is the first automation layer above the individual
training scripts. It trains scanner, radio, and jammer candidates, sends them
through the league gate, compares against the incumbent active stack when one
exists, writes a cycle manifest, and optionally promotes the candidate as the
active information-warfare stack.

`crypt_heist/control_league.py` does the same kind of evaluation work for
learned driving checkpoints. It composes a learned evader controller and a
learned five-pursuer team controller, runs them in the high-fidelity simulator,
optionally records JSONL replay evidence, and scores capture pressure, waypoint
pressure, speed, impacts, radio activity, spectacle, and adversarial balance.
It also owns the cross-generation promotion gate that runs the candidate evader
against historical teams and historical evaders against the candidate team,
producing generalization, resilience, worst-case, and pool-depth scores that can
advance the evader and pursuer-team checkpoints independently using their own
thresholds and improvement margins.

`crypt_heist/control_pool.py` preserves those learned driving pairs as
historical self-play generations. The pool manifest stores copied evader and
pursuer-team checkpoint paths, evaluation summaries, and deterministic
generation IDs, and the self-play scheduler can sample those older checkpoints
as future opponents.

`crypt_heist/selfplay.py` is the first automation layer that links control
learning and information-stack learning. It trains a new evader against the
active or historical pursuer-team checkpoint when available, trains a new
pursuer team against either the candidate evader or a historical evader,
runs the control-pair league gate, updates active evader and team checkpoints
component-by-component when their promotion gates pass, archives accepted active
pairs into the historical control pool, and can call the information-stack cycle
so radio, scanner, and jammer checkpoints continue to advance alongside driving
policy checkpoints. When scenario acceptance is enabled, a candidate whose
scenario manifest fails is not eligible for active-control promotion.

`crypt_heist/env_validation.py` verifies the MARL training wrapper itself: fixed
agent sets, action/observation spaces, finite parallel steps, seeded
determinism, truncation, counterfactual reward-probe fields, and PettingZoo's
parallel API contract.

`crypt_heist/operational_validation.py` is the aggregate readiness layer above
those focused gates. It reads scenario, learned-control, self-play,
information-stack, replay-fidelity, environment, physics, and spectator
manifests, checks dashboard command registration and replay schema validity,
and adds a small differentiable-comms probe. The output
`logs/operational_readiness.json` gives the dashboard one auditable summary for
whether the current build satisfies the operational v1 contract.

`crypt_heist/evidence.py` sits one layer above operational readiness for
release-style proof. It snapshots the green manifests, active replay reports,
checkpoint hashes, command registry, and dashboard/static assets into
`logs/evidence_bundle.json`; `verify_evidence_bundle.py` then re-hashes and
revalidates those files so stale proof is visible. The command center readiness
API exposes both bundle status and verification status, and routes missing or
failing proof to `build_evidence_bundle` or `verify_evidence_bundle`.
