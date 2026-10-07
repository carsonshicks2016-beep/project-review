# Physics Validation

The physics gate verifies the current high-fidelity vehicle model as a research
artifact before training or replay evidence is trusted.

Run:

```bash
python3 scripts/verify_physics.py --json-out logs/physics_validation.json
```

The manifest includes:

- `deterministic_trace`: two same-seed scripted simulations must produce the
  same vehicle trace and jamming budget.
- `handbrake_asymmetry`: the evader must rotate tighter than the pursuers under
  the same high-speed handbrake command, while shedding substantially more
  speed.
- `hard_braking_stability`: heavy pursuer braking must stay finite, bounded,
  and non-explosive.
- `collision_recovery`: building contact must damp velocity, stay finite, and
  push the vehicle clear of padded geometry.
- `long_run_stability`: a longer scripted chase must avoid NaNs, runaway speed,
  and runaway impact energy.

Dashboard readiness treats `logs/physics_validation.json` as a gate. If the
manifest is missing or failing, `/api/readiness` recommends the registered
`verify_physics` command.
