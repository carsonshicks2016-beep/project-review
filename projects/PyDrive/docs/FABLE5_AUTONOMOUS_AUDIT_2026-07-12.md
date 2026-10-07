# Fable Five Autonomous Training Audit — 2026-07-12

## Outcome

The old `12345678` run was stopped cleanly after live evidence showed foundation
and flow gating, followed by a finish-stage collapse. It was not safe to leave
unattended: its pace reference contained impossible acceleration jumps, its
evaluated checkpoint could be mutated before saving, its gear exploration was
too broad, and it had no durable process supervisor.

The pipeline is now guarded for unattended execution. It can preserve and
recertify genuine gains, recover from bounded failures, and stop loudly instead
of silently burning compute. No reinforcement-learning system can guarantee
constant improvement; plateauing remains possible.

## Critical findings and corrections

1. **Invalid speed envelope (P0).** The forward solver skipped constraints when
   net acceleration was negative. The Ring profile had 1,495 reachability
   violations and speed jumps up to about 60 m/s over one 3 m sample. The fixed
   both-sign reachability pass has no violations, a maximum positive jump of
   about 1.49 m/s, and a reachable centerline reference of 462.22 s.
2. **Checkpoint/evaluation mismatch (P0).** PitWall could roll back or reseed the
   policy inside evaluation before PPO saved it. Eval is now two-phase: save the
   exact evaluated weights and protected best first, then mutate and save only
   the recovered latest state. A SHA-256 policy binding is mandatory for current
   protocol certification.
3. **Observation normalization corruption (P0).** Running statistics were being
   updated from already-normalized observations. New policies commit raw sensor
   samples only after an accepted PPO transaction. Legacy coordinate systems are
   frozen rather than silently migrated.
4. **Destructive PPO overshoot (P0).** KL was checked before the next minibatch,
   after an overshooting step had already landed. Fable updates are now
   transactional: policy and Adam state roll back on full-batch KL rejection,
   the normalizer remains unchanged, and the safety LR backs off.
5. **Unbounded discrete gear noise (P1).** The rounded gear-offset action shared
   continuous-action noise and remained at standard deviation 1.0. Per-stage
   caps now tighten from log-std -0.90 to -1.45 and raw parameters are clamped
   after every optimizer step/load/reseed.
6. **Cross-stage metric and optimizer leakage (P1).** Unlike stage metrics no
   longer seed PitWall. Every destination stage runs a baseline evaluation, and
   cross-stage warm starts reset Adam, schedule caps, and excessive source noise.
7. **Certification gaps (P1).** Current evidence now requires matching stage,
   eval protocol, source and Nordschleife asset fingerprint, policy hash,
   architecture, car, track, finite network/optimizer/normalizer state, and
   whole-body on-track validity at every diagnostic physics substep.
8. **Unattended operations (P0/P1).** Training now has a global owner lock,
   atomic/durable best and champion writes, immutable champion archives, bounded
   worker reply timeouts, chronic environment-error escalation, signal-safe
   checkpoints, crash-budget accounting, persistent event logs, disk pause and
   recovery, checkpoint-stall recovery, `caffeinate`, and bounded supervisor
   restarts. The notification sidecar survives ntfy outages and reports stalls,
   restarts, low disk, stage changes, milestones, completion, and failures.

## Stage tuning at population 64

| Stage | Scale | LR | Target KL | Value clip | Entropy | Initial log-std | Gear cap | Eval updates | Patience |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| foundation | 0.78 | 3.0e-4 | 0.080 | 6 | 0.0080 | -0.30 | -0.90 | 8 | 63 |
| flow | 0.86 | 3.0e-4 | 0.060 | 8 | 0.0050 | -0.45 | -1.00 | 8 | 75 |
| finish | 0.90 | 2.5e-4 | 0.040 | 10 | 0.0040 | -0.55 | -1.15 | 10 | 88 |
| fast | adaptive to 0.96 | 1.8e-4 | 0.020 | 10 | 0.0035 | -0.60 | -1.30 | 8 | 113 |
| frontier | adaptive 0.90–1.15 | 1.2e-4 | 0.015 | 10 | 0.0025 | -0.70 | -1.45 | 8 | 88 |

The requested 64 environments remain intact, but worker processes are capped to
8 on this 12-core Mac. This avoids oversubscription while each worker owns eight
environments. Evaluation cadence and patience scale by transitions rather than
blindly retaining small-population update counts.

## Monotonic-growth safeguards

- A gated proof outranks an ungated metric; any certified lap outranks a lapless
  fragment; faster certified laps rank higher.
- Protected best and Hall-of-Fame checkpoints are atomic, policy-bound records.
- A stale global champion cannot block the first current-protocol champion; it
  is archived before replacement.
- AUTO budget and in-flight stage state persist with protocol/code fingerprints.
  Source changes deliberately force baseline recertification.
- Finish retains the ladder until it banks a lap or clears the explicit near-lap
  quality bar. Fast/frontier adapt target scale in bounded segments.
- PitWall rollback, reseed, consolidation, weak-sector replay, numerical
  recovery, eval retry, and plateau handling are independently bounded.

## Validation evidence

All three executable suites must pass before launch:

```bash
python3 tools/validate_fable5.py
python3 tools/validate_fable5_auto.py
python3 tools/validate_fable5_safeguards.py
```

The safeguards suite additionally launches synthetic completion and hung-child
process trees, proves full process-group cleanup, and verifies that a second
trainer exits with the dedicated duplicate-owner code instead of starting.

## Remaining scientific limitation

The corrected reachable centerline envelope is about 8:42, while Bellof's target
is 6:11.13. A racing line can improve on a centerline profile, but the present
audit does not prove that the configured 787B/track/tyre model has enough margin
to close a 91-second gap. The unattended system is ready to search without losing
certified progress; the superhuman target itself remains an empirical research
question, not a guaranteed outcome.
