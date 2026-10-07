# Audio V4 rollout walkthrough

Audio V4 is the default in this worktree. Use `SUPRA_AUDIO_ENGINE=v3` for the
one-release fallback. Run `PYTHONPATH=. python3 tools/validate_audio_v4.py`
before integration. A live viewer reports callback health through
`SpatialAudioMixer.metrics`.

The audio subprocess chooses the active output device's native sample rate and
accepts variable callback frame counts. Viewer telemetry is adapted into a
versioned, sequence-locked `AudioFrame`; missing fields receive neutral values.
Reset, shift, landing, collision, damage, and surface events use monotonic
generation counters.

Audio identity is entirely procedural. The checked-in
`mazda787b_procedural_identity_v1.json` contains engineering constants and
subjective goals only. Real recordings and the NFS mod are neither decoded nor
used by automated acceptance. The reference analyzer remains optional,
unsupported tooling and is not called by release validation.

Doppler is mandatory and is generated only by fractional propagation delay at
343 m/s. Never apply another pitch transform to a delayed source. Camera cuts
use a 100 ms transition, and listener-local wind bypasses propagation.
