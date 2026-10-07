# V2 Research-Grade Track

V2 is measured separately from the shippable v1 gate. V1 proves the simulator is
operational; v2 proves the learned information-warfare stack and cinematic
renderer are strong enough to study, compare, and show.

## Evidence Loop

1. Run the long checkpoint league:

   ```bash
   python3 scripts/eval_checkpoint_league.py \
     --name v2_candidate \
     --radio checkpoints/radio_policy.pt \
     --scanner checkpoints/scanner_decoder.pt \
     --jammer checkpoints/jammer_policy.pt \
     --seeds 11,12,13,14,15 \
     --steps 1800 \
     --record-prefix logs/v2_checkpoint_league \
     --out logs/v2_checkpoint_league_manifest.json
   ```

2. Refresh spectator validation:

   ```bash
   python3 scripts/verify_spectator.py --json-out logs/spectator_validation.json
   ```

3. Build the v2 report:

   ```bash
   python3 scripts/build_v2_report.py \
     --league-manifest logs/v2_checkpoint_league_manifest.json \
     --out logs/v2_research_report.json \
     --markdown-out logs/v2_research_report.md
   ```

4. Refresh v1 release evidence after any source or static-asset changes:

   ```bash
   python3 scripts/verify_operational_readiness.py --json-out logs/operational_readiness.json
   python3 scripts/build_evidence_bundle.py --out logs/evidence_bundle.json
   python3 scripts/verify_evidence_bundle.py logs/evidence_bundle.json --json-out logs/evidence_bundle_verification.json
   ```

## V2 Report Gates

The v2 report currently checks:

- Long multi-seed league depth: at least 5 seeds and 1800 steps.
- Learned information-stack scores: overall, pursuer security, evader pressure,
  adversarial balance, and spoof susceptibility.
- Counterfactual jamming effect: deception lift, confidence damage, spoof-event
  lift, and cross-seed susceptibility variance.
- Cinematic replay proof: nominal replay diagnostic report with waypoint and jam
  evidence.
- Spectator proof: audio mapping, dashboard replay payload, replay deep links,
  and the Three.js replay renderer contract.
- Release integrity: operational readiness plus verified evidence bundle.

## Renderer Track

The web replay cockpit now uses a Three.js WebGL scene as the primary chase
renderer and keeps the older 2D canvas as fallback. Spectator validation checks
for the Three.js renderer contract, but final v2 acceptance should also include
browser screenshot and canvas-pixel checks across desktop and mobile viewports.
