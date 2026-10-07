# supra training files — snapshot before the drift/race/hybrid audit (2026-09-10)

config.py · ppo.py · ppo_env.py exactly as they were before the lineage audit.

Restore:
    cp -p supra/legacy/pre_lineage_audit_20260910/{config,ppo,ppo_env}.py supra/

Changes made after this snapshot:
  1. PPOSpec: hybrid gets its own curriculum promotion gate (was silently using
     the pure-race promote_at=0.72, which its own reward fights against).
  2. ppo_env: hybrid drift_frac now uses the same corner gate the style reward
     uses, so the eval measures what training actually pays for.
  3. DriftReward.scale was dead code; it is now applied, default 1.0 (no-op for
     existing tunings) so the knob is real.
  4. Hybrid style bonus ramps in with curriculum difficulty and starts at a
     lower slip angle, so there is a continuous gradient from grip into slide.
