# Training Curriculum

1. Scripted sandbox: validate deterministic physics, city, replay, comms, UI.
2. Evader waypoints: train driving against scripted pursuers.
3. Single pursuer chase: train basic pursuit.
4. Multi-pursuer containment: train boxing and roadblocks.
5. Learned pursuer comms: enable token output for coordination.
6. Evader scanner: train decoder from transcript to future pursuer positions.
7. Adversarial jamming: enable limited spoof token injection.
8. Authenticated self-play: penalize decodability and spoof susceptibility.

Promote stages using evaluation scenarios, not raw rollout reward alone.

## Implemented Training Lanes

- `scripts/train_imitation.py`: supervised evader cloning from the scripted
  planner. This is the fastest checkpoint sanity check.
- `scripts/train_evader_ppo.py`: first PPO lane for the evader's continuous
  steering/throttle/handbrake controls inside `CryptHeistParallelEnv`, with
  scripted pursuers or a saved learned pursuer/pursuer-team checkpoint as the
  opponent. Its reward now includes the live information-warfare terms exposed
  by the environment, including active spoof susceptibility, evader information
  reward, and a low-frequency paired jam/no-jam counterfactual deception probe.
  It saves `evader_ppo` checkpoints that can be replayed through
  `scripts/watch_policy.py` or opened live with `scripts/watch_policy_viewer.py`.
- `scripts/train_pursuer_ppo.py`: first PPO lane for one pursuer's continuous
  chase controls inside the same environment, with the evader and remaining
  pursuers scripted or checkpoint-controlled. The reward includes decoder-
  accuracy and spoof-susceptibility penalties plus counterfactual deception
  penalties, so the first authentication pressure reaches policy updates. It
  saves `pursuer_ppo` checkpoints that can be replayed through
  `scripts/watch_pursuer_policy.py` or opened live with
  `scripts/watch_pursuer_policy_viewer.py`.
- `scripts/train_pursuer_team_ppo.py`: first shared-policy PPO lane for all
  five pursuers. Each police car contributes its own observation/action/reward
  samples to one actor-critic, so roles and relative states can begin shaping
  coordinated containment. The same live and counterfactual information-
  authentication penalties are applied to the team reward, and the team can now
  train against a saved evader checkpoint. It saves `pursuer_team_ppo`
  checkpoints for replay and live viewing.
- Dashboard self-play entries (`train_evader_ppo_self_play`,
  `train_pursuer_ppo_self_play`, and `train_pursuer_team_ppo_self_play`) are
  convenience wrappers around the PPO scripts with opponent checkpoint flags
  filled in.
- `scripts/train_scanner_decoder.py`: first evader scanner lane. It collects
  radio token windows from scripted rollouts and trains a GRU decoder to predict
  five pursuer future `(x, y)` positions. `scripts/watch_scanner_decoder.py`
  records replays with learned predictions, and the live viewer can use the
  checkpoint to drive the scanner panel and confidence meter.
- `scripts/train_radio_policy.py`: first learned pursuer comms lane. It
  imitates the scripted six-token radio protocol from pursuer observations,
  exposes a tested straight-through Gumbel-Softmax relaxed token sequence for
  differentiable comms training, and can drive the normal simulator radio
  channel while scripted vehicle control remains active.
- `scripts/train_jammer_policy.py`: first evader jamming lane. It imitates a
  pressure/cooldown heuristic that chooses when to jam, which five spoof tokens
  to inject, and which pursuers to desynchronize. It drives the normal
  `jam`/`spoof_tokens`/`target_mask` action path while scripted driving remains
  active.
- `scripts/train_counterfactual_jammer_policy.py`: stronger evader jamming
  lane. It samples scripted chase states, deep-copies each state into jammed
  and no-jam branches, rolls both forward, and weights jammer labels by the
  resulting counterfactual pursuer trajectory deviation and confidence damage.
  The saved checkpoint keeps the same `evader_jammer_policy` runtime format.
- `scripts/watch_adversarial_stack.py`: first combined information-warfare
  evaluation lane. It runs learned pursuer radio, the learned evader scanner,
  and the learned evader jammer in the same episode, recording a replay plus
  JSON metrics for decoder error, confidence collapse, spoof susceptibility,
  unique spoof events, and deception score.
- `scripts/eval_authentication_curriculum.py`: paired jammed/no-spoof
  curriculum gate. It repeats the same seeded episode with jamming disabled,
  compares pursuer trajectories frame-by-frame, and emits counterfactual
  spoof-susceptibility, pursuer authentication penalty, and evader information
  reward proxies.
- `scripts/eval_checkpoint_league.py`: first checkpoint league gate. It runs the
  authentication curriculum over multiple seeds, aggregates pursuer security,
  evader pressure, and adversarial balance scores, then can promote a passing
  radio/scanner/jammer trio into an active manifest.
- `scripts/run_information_cycle.py`: first one-command operational cycle. It
  trains scanner, radio, and jammer candidate checkpoints, runs the league gate,
  compares against the previous active stack when present, writes a cycle
  manifest, and can promote the passing stack into the active model directory.
  Pass `--jammer-mode counterfactual` to use paired jam/no-jam jammer training
  inside the cycle instead of the faster heuristic imitation lane.
- `scripts/run_self_play_cycle.py`: first alternating self-play scheduler. It
  trains an evader checkpoint against the active pursuer-team snapshot when one
  exists or against a sampled historical pursuer-team pool entry, trains a
  pursuer-team checkpoint against either the candidate evader or a sampled
  historical evader pool entry, promotes the pair into `models/control_active`,
  runs a replay-backed control-pair league evaluation, archives the pair into
  `models/control_pool`, and can run the information-stack cycle in the same
  command. The scenario-gated dashboard command warm-starts the evader candidate
  from `checkpoints/evader_ppo.pt` and uses validation selection so a short
  self-play cycle cannot silently regress the accepted waypoint driver. With
  `--evaluate-control-scenarios`, candidate learned controls also run through
  the scenario acceptance suite. A failing scenario manifest blocks
  active-control promotion even if the aggregate score is high, and passing
  manifests can still be made stricter with `--control-scenario-threshold`. The
  stricter path can split that gate with `--evader-control-scenario-threshold`
  and `--pursuer-team-control-scenario-threshold`, so waypoint escape can
  advance the evader while roadblock containment independently advances the
  team.
- `scripts/eval_control_pair.py`: first control league gate for learned driving
  checkpoints. It runs the learned evader and learned five-pursuer team together
  across seeds, optionally writes JSONL replays, aggregates captures,
  waypoints, deception, confidence, speed, and impact, then scores pursuer
  control, evader control, spectacle, and adversarial balance.
- `scripts/eval_control_league.py`: cross-generation promotion gate. It scores
  a candidate pair by running the candidate evader against historical
  pursuer-team checkpoints and historical evaders against the candidate team,
  then reports self-pair score, evader generalization, team resilience,
  worst-case score, and pool depth. Self-play uses those component scores to
  promote the candidate evader and candidate pursuer team independently, with
  separate thresholds and improvement margins when the operator wants different
  standards for escape progress versus containment progress.
- `scripts/list_control_pool.py`: dashboard-visible inspection command for the
  historical learned-control opponent pool. It reports the latest generation,
  best scored generation, and checkpoint paths that later self-play runs can
  sample as opponents.
- `scripts/eval_acceptance_scenarios.py`: replay-backed operational acceptance
  gate for the current build. It runs downtown chase, roadblock, spoof burst,
  and cipher shift scenarios, writes replays plus a JSON manifest, and separates
  required smoke checks from milestone targets. The dashboard's strict full-
  target entry uses seed 31 to require three downtown waypoints before capture.
  Each scenario also emits diagnostics that identify the likely owner, severity,
  reason, evidence fields, and next action for failed checks or missed
  milestones.
- `scripts/plan_curriculum_from_manifest.py`: diagnostic router for the
  operator loop. It reads an acceptance manifest and writes a curriculum plan
  with prioritized dashboard command IDs, arguments, follow-up gates, and the
  evidence that caused each route. The initial routes cover evader waypoint
  failures, pursuer-team roadblock failures, jammer deception failures, scanner
  confidence failures, and cipher-shift probes.

The current live training environment now feeds decoder accuracy, active spoof
susceptibility, and sampled paired counterfactual trajectory deviation into PPO
reward updates. The continuous-control PPO lanes can train against saved
checkpoint opponents, and the counterfactual jammer lane feeds jammed-vs-
baseline trajectory deviation into the jammer head. The current self-play
scheduler alternates active control and information checkpoints, archives
historical control generations, and samples older learned opponents during
later cycles. It now also gates active control promotion through a scored pool
league and allows one faction to advance without forcing the other through. The
scenario acceptance now feeds optional global and faction-specific control-
promotion gates, with a hard stop when the scenario manifest itself fails. The
dashboard readiness reads the latest scenario-gated self-play manifest and,
when that candidate fails scenario acceptance, routes the next command from the
candidate's own scenario diagnostics. The next training gap is making those
scenario thresholds progressively stricter as learned checkpoints mature, then
letting the dashboard launch longer targeted repair cycles automatically after
operator review.
