# Replay Schema

Replay logs are JSONL.

Line 1 is metadata:

```json
{"type":"meta","version":1,"seed":11,"dt":0.0083333333}
```

Each following line is a frame:

```json
{
  "type": "frame",
  "i": 120,
  "time": 1.0,
  "agents": [],
  "actions": {},
  "radio": [],
  "waypoint": [0.0, 0.0],
  "confidence": 0.5,
  "metrics": {},
  "predictions": {},
  "collisions": [],
  "rewards": {},
  "reward_components": {},
  "camera": {}
}
```

Renderers must consume replay frames without stepping the simulation.

Replay validation is intentionally stricter than basic JSON parsing. A replay
must have schema version `1`, positive `dt`, monotonic frame times, contiguous
frame indexes, 6 finite vehicle states per frame, finite scanner predictions,
radio event metadata, confidence, waypoint, metrics, and optional enhanced
payloads for actions, collisions, rewards, reward components, and camera hints.
This catches malformed training/rendering artifacts before they are treated as
evidence.

Enhanced frames written by the active simulator include:

- `actions`: one row per vehicle with applied controls, source
  (`scripted`/`external`), requested controls, token IDs, spoof token IDs, jam
  intent, and target masks when supplied by a policy.
- `collisions`: impact events with event time, impact magnitude, last collision
  impulse, and suggested camera shake.
- `rewards` and `reward_components`: per-agent reward evidence keyed by
  `EVADER` and `P1` through `P5`. Component rows repeat shared chase terms and
  include role-specific information-warfare terms such as `information_reward`,
  `decoder_accuracy`, `spoof_susceptibility`, and pursuer `auth_penalty`.
- `camera`: deterministic replay-side hints containing focus, lead, target,
  subject, event type, shake, and zoom.

`crypt_heist.replay.replay_fingerprint()` produces stable SHA-256 fingerprints
over canonicalized replay content:

- `full_sha256`: complete replay payload, excluding only JSON whitespace.
- `state_sha256`: vehicle state, waypoint, confidence, scanner predictions,
  metrics, actions, collisions, rewards, reward components, and camera hints.
- `radio_sha256`: radio transcript and spoof markers.

`scripts/verify_replay_fidelity.py` writes `logs/replay_fidelity.json` with a
summary, fingerprints, and an optional same-seed scripted determinism proof.
That manifest is surfaced by the dashboard readiness API so a replay can be
tracked as a deterministic research artifact instead of just a playable file.

Replay summaries also roll up the reward evidence into dashboard-friendly
analytics:

- `avg_evader_reward`, `avg_pursuer_reward`, `total_evader_reward`, and
  `total_pursuer_reward`.
- `avg_decoder_accuracy`, `avg_spoof_susceptibility`,
  `avg_evader_information_reward`, and `avg_pursuer_auth_penalty`.
- `max_pursuer_auth_penalty`, `max_spoof_susceptibility`, and
  `reward_agent_coverage`.

Replay summaries also include radio/authentication analytics computed from
de-duplicated radio events:

- `radio_word_entropy`, `pursuer_word_entropy`, `spoof_word_entropy`, and
  normalized entropy variants for protocol-diversity tracking.
- `radio_spoof_ratio`, `jam_events`, `cipher_rotations`,
  `avg_jam_confidence_drop`, and `max_jam_confidence_drop`.
- `avg_cipher_confidence_drop`, `avg_cipher_confidence_recovery`,
  `confidence_min`, `confidence_max`, and `confidence_range`.

Acceptance scenario manifests reference replay files rather than embedding the
full frame stream. `scripts/eval_acceptance_scenarios.py` writes one replay per
scenario and a JSON manifest containing checks, metrics, replay summaries, and
pass/fail state. This keeps the scenario gate usable from the dashboard while
preserving the replay boundary between simulation/training and rendering.

`scripts/replay_report.py` writes a compact replay-only research report. The
report contains the full replay summary plus chase, information-warfare, reward,
and replay-contract sections; deterministic fingerprints; notable replay events;
and diagnostics with severity, area, reason, message, and evidence fields. The
dashboard exposes the same command through the `R` replay-row action and writes
reports under `logs/replay_reports/`.
