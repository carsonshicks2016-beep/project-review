# Fable Five Porsche 919 Evo pipeline doctor

> **Disposition (2026-07-23):** triaged and implemented. P0 experiment
> machinery (per-update metrics, champion/candidate grace separation,
> `--frontier-probe`) landed in the Phase 1–3 commits of 07-22; the probe was
> then hardened on 07-23 (hermetic isolated workspace under
> `runtime/fable5/probe/`, edition-aware champion loading, log_std frozen in
> the critic-only window, metrics actually written). Also landed: frontier
> failure-weighted starts (2.5/5.0), coherent frontier objective ordering,
> and reward-parts decomposition in the `--fable-diag` reports. Deliberately
> NOT changed: reward coefficients, gear/steering semantics, envelope scale
> (per this doc's own guidance — run the probe first). See the 2026-07-23
> entry in `docs/legacy/HANDOFF.md`.

**Snapshot:** 2026-07-22, about 17:22 Central  
**Scope:** read-only audit of the active `fable5_919_ring_foundation` auto-ladder run, its saved checkpoints/events, the Fable/PPO implementation, and two deterministic checkpoint diagnostic batteries. No training configuration, process, checkpoint, or source code was changed.

## Bottom line

The pipeline has a real, high-quality protected controller, but the active frontier learner is **not retaining that competence after updates**. This is no longer a case where more of the exact same frontier run is likely to discover a better lap: the current candidate has exhausted its recovery levers and is behaving as a non-finisher.

The first change should not be a blanket learning-rate or reward tweak. The evidence favors a short, instrumented **stage-transition stabilization experiment**: preserve the champion immutably, let the frontier candidate adapt in much smaller/evaluated steps, and measure whether the first loss comes from value/advantage instability, action-distribution drift, normalization drift, or one specific sector. Only then tune the responsible mechanism.

## What was observed

| Item | Evidence | Interpretation |
| --- | --- | --- |
| Trainer health | PID 1789 was live, CPU-active, and still updating the pipeline manifest. | This is a learning failure, not a dead worker or hung trainer. |
| Protected champion | `fable5_919_ring_best.pt`: deterministic 437.73 s clean lap; 16/16 standard sector starts clean; 5/5 closure drills clean. | There is a legitimate finisher to protect and use as the control. |
| Active candidate | `fable5_919_ring_foundation_frontier.pt` at frontier scale 0.93: no clean line lap, 0/16 sector starts clean, 0/5 closure drills; farthest line run only 11.5%. | The active policy has genuinely fallen far off the protected behavior. |
| Recent online evaluation | Latest score 0.078, 1/16 clean sectors, 18.3% progress, 0.935 pace ratio, 43.75% terminal rate. | It is neither completing nor maintaining a safe useful partial-lap state. |
| Recovery state | 51 evaluations since the best; 12 rollbacks, 6 reseeds, and 1 consolidation already consumed; last-five mean score 3.46 versus best 228.45. | The current recovery policy has exhausted its designed interventions without restoring a viable candidate. |
| Curriculum transition | Foundation, flow, finish, and fast were revalidated at zero updates from the inherited clean policy; frontier segment 1 used 1,408 updates and then advanced scale 0.90 -> 0.93. | The ladder verified the inherited policy, but did not give each reward regime an adaptation period before frontier learning. |
| Storage headroom | About 19 GiB free (98% of the data volume used). | Not the cause of the current collapse, but too little margin for an unattended evidence-heavy run. |

**Post-snapshot state:** after the diagnostic work completed, the original direct trainer process was no longer present. Its last persisted frontier evaluation was the 0.078 result at update 6,092; its checkpoint was last written at update 6,124. There is no matching live replacement or durable supervisor log, so the exact terminal cause cannot be established from the retained local artifacts. This does not alter the diagnosis above, but it reinforces the observability recommendation below.

## Direct checkpoint control test

I ran the built-in deterministic Fable diagnostic battery at reduced CPU priority against the two saved checkpoints. It reads checkpoints and writes only a separate JSON report.

| Test | Protected champion | Current frontier candidate |
| --- | --- | --- |
| Checkpoint | `fable5_919_ring_best.pt` | `fable5_919_ring_foundation_frontier.pt` |
| Flying line | Clean 437.73 s at race drop-in | Both race and cautious drop-ins die at 11.5% |
| Standard sector battery | 16/16 clean | 0/16 clean |
| Late-lap closure drills | 5/5 clean | 0/5 clean |
| Main failures | One traction-break in sector 7 | Four sector-1 ran-wide failures; sector-11 traction breaks; sector-2 overspeed entries |
| On-track pace | 92.4% of envelope | 85.9% of envelope |
| Result | `FINISHER` | `NON-FINISHER` |

Raw reports:

- `diagnostics/fable5/fable5_919_ring_best_20260722_172141.json`
- `diagnostics/fable5/fable5_919_ring_foundation_frontier_20260722_171834.json`

This control matters. It rules out a blanket claim that the track, evaluator, or 919 model is currently incapable of producing the 437.73 s lap. It also shows that the active candidate is not merely missing a fragile final sector; it has lost broad driving competence.

## Confirmed failure loop

The current frontier behavior is consistent with a repeated destructive-update/recovery loop:

1. A 437.73 s, 16/16 clean evaluation is preserved as the banked best.
2. The next frontier windows produce no-lap evaluations far below the lap-scale metric.
3. The PitWall identifies a collapse and restores the banked checkpoint. The restore reloads the entire `ActorCritic` state (shared trunk, policy mean, value head, and action noise) and the observation-normalization state, then replaces Adam with a fresh optimizer.
4. It mildly lowers learning rate/entropy and narrows noise, but another update window destroys the driving behavior again.
5. After 12 rollbacks, 6 reseeds, and one consolidation, the pit levers are exhausted. The current candidate continues to receive weak evaluations.

That safety system did its primary job: the global champion is still safe. Its secondary job—rehabilitating a frontier learner in place—has not worked in this run.

The source supports this mechanism directly:

- `supra/ppo.py` defines one shared-trunk actor/critic. A checkpoint restore therefore restores both policy and value behavior, not just steering/throttle outputs.
- `PPO.reseed_from_best()` restores network and normalizer state, then drops stale Adam moments even in calm recovery.
- `PitWall.note()` treats an unhealthy evaluation below 25% of the banked lap metric as a collapse and invokes the calm restore until its rollback budget is used.

This is a strong hypothesis about *why adaptation cannot accumulate*, not a claim that restoring full state is intrinsically wrong. It is appropriate for emergency protection; the question is whether it is too aggressive during a deliberate reward/stage transition.

## Where the candidate is failing

The durable frontier sector heatmap is highest in sectors 3, 7, and 1 (heat 13.96, 12.18, and 10.72 respectively). The latest online evaluation failed sectors 1, 3, 7, 7, 7, 9, 12, and 1. The independent current-checkpoint diagnostic puts the immediate killer at sector 1, then sees sector 11 traction breaks and sector 2 overspeed entries.

The practical conclusion is to target **sectors 1, 3, and 7 first**, while recording sector 11 and 2 as secondary signatures. Do not target only the historic sector-3 Aremberg/Fuchsröhre issue: its local reference-speed derate is already implemented, and the present evidence is broader.

## Ranked changes to try

### P0 — add a controlled frontier-transition stabilization experiment

**Why:** The policy is clean before the first frontier update and catastrophically bad shortly after it. The existing 32-update evaluation interval is too coarse to identify which early update first causes loss.

**Experiment, not a permanent change:** start from an immutable copy of the 437.73 s champion and run a separate frontier-candidate prefix at the same 0.90 scale. For the first 32–64 updates:

1. Evaluate every 4 or 8 updates, using the existing deterministic evaluation plus the fixed diagnostic sector battery.
2. Log per-update policy KL, clip fraction, value loss/explained variance, advantage mean/std, entropy/log-std by action dimension, normalizer count/mean/variance deltas, and action means for steering/throttle/gear.
3. First run a critic-only calibration window (freeze the shared trunk and policy output; train the value head only) to test whether the new frontier return scale invalidates value estimates without changing driving actions.
4. Then run very small actor updates with the same logging. Stop the probe immediately on the first loss of a clean lap; retain both the last clean and first failed checkpoints for an A/B diagnostic.

**Pass criterion:** three consecutive full evaluations retain a clean lap and at least 14/16 clean sectors.  
**What it tells us:**

- Clean critic-only but failing actor update: policy/action update is the culprit.
- Failing critic-only with no action change: reward/return/value-scale transition is the culprit.
- Sudden normalizer or action-distribution movement before failure: target that subsystem rather than reducing every PPO parameter blindly.

### P0 — separate “protect the champion” from “allow a candidate to learn”

**Why:** The global champion is correctly immutable, but full rollback of the only active candidate makes every recovery attempt restart from the same state. The live evidence is 12 rollbacks plus 6 reseeds with no retained improvement.

**Candidate design:** retain the current global/Hall-of-Fame checkpoint as an untouchable evaluator baseline. Train a named `frontier_candidate` branch with a small bounded grace budget after a stage/scale change. During that grace budget, score every micro-window but do not overwrite the champion or call the candidate an improvement. If behavior becomes unsafe, restore only the candidate branch; if it becomes healthy, require repeated independent evaluation before promotion.

The safety invariant must remain: no experimental candidate may overwrite `fable5_919_ring_best.pt` unless it passes the full deterministic battery and improves the selected metric. This proposal changes *candidate experimentation*, not champion protection.

### P1 — make frontier starts failure-weighted and prove local repair before another full-lap push

**Why:** The frontier heatmap says the policy repeatedly dies in sectors 1/3/7. `flow` explicitly increases failure-start weighting (`sector_fail_gain=2.5`, `seed_bias_cap=5.0`), while `frontier` currently falls back to the generic values. The frontier candidate has lost broad competence, so a targeted recovery curriculum is more informative than a full-lap-only reward chase.

**Experiment:** in a separate candidate run, allocate roughly half of training starts to the current high-heat sectors and their approach zones; retain the remaining starts as ordinary line/closure starts. Promote only if the fixed sector battery reaches 14/16, then 16/16, *and* the race-drop-in lap remains clean. Keep the evaluator's starts and seeds fixed so this cannot be gamed by the curriculum.

**Do not do yet:** increase envelope scale or unlock more speed pressure. The candidate is currently slower than the envelope and still failing; this is a stability/line-holding problem first.

### P1 — instrument reward decomposition before retuning reward coefficients

**Why:** Frontier increases pace reward to 0.16, retains unscaled progress-per-m reward, reduces overspeed coefficient to 0.008, and moves its overspeed onset to 1.10x the raw envelope. That looks permissive on paper, but the latest failing candidate is only at 0.935 pace ratio and the diagnostic averages 0.859. Therefore the evidence does **not** support the simple story “it is currently failing because it is too fast.”

**Experiment:** record cumulative reward parts by sector and termination type (`pace`, `progress_m`, `overspeed`, `offtrack`, `crash`, alignment/slip terms) for the clean champion rollout and first failed micro-window. Then run one ablation at fixed scale and starts:

- current frontier reward;
- no unscaled progress-per-m term, or a progress term scaled by target pace;
- a more protective overspeed onset/weight copied from fast.

Choose the smallest change that retains the full-lap controller across the P0 gate. Do not change pace cap, progress reward, overspeed weight, learning rate, and entropy all at once; that would erase causal evidence.

### P1 — correct the frontier goal/reporting semantics

The manifest currently says frontier should “beat 319.55 s (919 Evo record), then chase the theoretical 415.5 s.” A 415.5 s time is slower than 319.55 s, so that ordering cannot describe a coherent post-record objective. The 415.5 s value is also a centerline envelope estimate, not automatically a race-line lower bound.

This does not explain the immediate PPO collapse, but it can lead to bad stop/go decisions. Define explicit, ordered objectives instead:

1. retain a repeatable clean lap;
2. improve 437.73 s under fixed certification;
3. cross an agreed simulation benchmark;
4. treat a real-world-record comparison as a separately labeled external benchmark, not a later target that is numerically slower than the model's “theoretical” value.

### P2 — investigate gear and steering only with an A/B checkpoint comparison

Both diagnostics flag gear activity on 100% of steps. The failed candidate has mean gear offset -1.23 versus -1.03 for the champion. Both also flag steering sawing, but the champion's measured reversals are actually higher (109/km versus 33/km for the failed candidate). Those observations are useful instrumentation targets, not evidence that either control head is the root cause.

Before changing action semantics, compare gear histograms, actual shift events, throttle/brake overlap, and action means between the saved clean and first-failed micro-window. A gear-policy clamp or steering-smoothness penalty without that comparison would be guesswork.

### P2 — make unattended experiments observable and restartable

The live command is running directly under the command center. `tools/supervise_fable5.py` already provides persistent stdout capture, `caffeinate`, checkpoint-freshness monitoring, a 10 GiB free-space floor, and bounded restart accounting. Use it for the next long experiment, with a distinct candidate prefix and the P0 metrics written to durable files.

The present 19 GiB free is above that threshold but leaves limited room. Free space before an overnight candidate run; do not let diagnostics/checkpoints turn an otherwise interpretable training result into an operational stop.

## Recommended decision sequence

1. **Do not promote or overwrite the existing 437.73 s champion.** It is the crucial control and safety floor.
2. **Do not launch another identical 49,901-update frontier run.** This run has already shown the same recovery mechanism 18 times (12 rollbacks + 6 reseeds) and has exhausted its intervention budget.
3. Implement the P0 telemetry and 4–8-update transition probe in an isolated candidate prefix.
4. Run the critic-only calibration followed by tiny actor-update probes at scale 0.90, not 0.93 or above.
5. If the policy holds, add targeted starts for sectors 1/3/7 and require repeated full-battery passes before raising scale.
6. If the policy fails in the first micro-window, use the captured clean/failed pair to fix the responsible subsystem; only then consider reward-coefficient changes.

## Evidence locations

- Live run state: `fable5_919_ring_pipeline.json`
- Per-evaluation event stream: `runtime/fable5/fable5_919_ring_foundation_frontier/events.jsonl`
- Protected checkpoint: `fable5_919_ring_best.pt`
- Current candidate: `fable5_919_ring_foundation_frontier.pt`
- Pipeline/PitWall/reward implementation: `supra/fable5.py`
- PPO restore and optimizer behavior: `supra/ppo.py`
- Deterministic diagnostic implementation: `supra/fable5_diag.py`
- Durable supervision: `tools/supervise_fable5.py`

## Confidence notes

High confidence: the champion/candidate behavioral contrast, current plateau state, exhausted recovery levers, sector priorities, and objective-string inconsistency.  
Medium confidence: full checkpoint restoration plus cold optimizer is preventing useful frontier adaptation; it fits the evidence but needs the P0 micro-window experiment to establish causality.  
Low confidence until instrumented: whether the first destabilizer is value learning, policy KL/action shift, normalizer drift, reward shaping, or drivetrain control.
