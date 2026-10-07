"""Step 9 entry check. Audit Step 8 before evaluating its frozen gates."""
import json
import sys
import subprocess
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flygarden.recording import atomic_json

PREVIOUS = ROOT / 'reports/brain-integration/stage8-20261006'
OUT = ROOT / 'reports/brain-integration/stage9-20261006'
REQUIRED = ('causal_motor_effect', 'directional_avoidance',
            'useful_response', 'exact_motor_replay')


def readiness(result):
    """Fail closed on partial, missing, inconsistent or unsuccessful evidence."""
    if result.get('status') != 'complete' or result.get('seed_count') != 20:
        raise ValueError('All twenty matched seeds must be complete and audited')
    gates = result.get('gates', {})
    if any(type(gates.get(key)) is not bool for key in REQUIRED):
        raise ValueError('Missing or invalid audited gates')
    failed = [key for key in REQUIRED if not gates[key]]
    if gates.get('progression_ready') is not (not failed):
        raise ValueError('Inconsistent progression decision')
    return failed


def run():
    progress = json.loads((PREVIOUS / 'progress.json').read_text())
    if progress['status'] != 'complete':
        raise RuntimeError('Step 8 is still running; no navigation trials launched')
    # Reconstruct the evidence rather than trusting an old results.json.
    subprocess.run([sys.executable, str(ROOT / 'scripts/report_causal_behavior.py')],
                   cwd=ROOT, check=True)
    file = PREVIOUS / 'results.json'
    raw = file.read_bytes()
    result = json.loads(raw)
    failed = readiness(result)
    OUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    snapshot = OUT / 'evidence' / f'{digest}.json'
    snapshot.parent.mkdir(exist_ok=True)
    if snapshot.exists():
        assert snapshot.read_bytes() == raw
    else:
        temporary = snapshot.with_suffix('.tmp')
        temporary.write_bytes(raw)
        temporary.replace(snapshot)
    state = {
        'schema_version': 1,
        'status': 'blocked_by_measured_response' if failed else 'entry_ready',
        'step8_results_sha256': digest,
        'audited_result_snapshot': str(snapshot.relative_to(OUT)),
        'failed_gates': failed,
        'navigation_trials_launched': 0,
        'production_controller_modified': False,
        'learning_enabled': False,
        'note': 'Entry readiness is not a completed navigation evaluation.',
    }
    atomic_json(OUT / 'readiness.json', state)
    gates = '\n'.join(f"| {key.replace('_', ' ')} | {'FAIL' if key in failed else 'PASS'} |"
                      for key in REQUIRED)
    counts = result['useful_response_counts']
    intact = [record for record in result['records'] if record['arm'] == 'intact']
    forward_spikes = sum(record['population_spikes']['DNp09_left'] +
                         record['population_spikes']['DNp09_right'] for record in intact)
    steering_spikes = sum(record['population_spikes']['DNa02_left'] +
                          record['population_spikes']['DNa02_right'] for record in intact)
    text = f'''# Step 9 — behavioral progression entry

Status: **{state['status']}**. The complete Step 8 evidence was independently audited before this decision. No navigation, predator or foraging trials were launched and the live controller remains unchanged.

| Required Step 8 gate | Result |
|---|---|
{gates}

Measured useful-response counts: {counts['away_at_least_0_05_rad_count']}/20 gain at least 0.05 radians away heading; {counts['departure_at_least_0_1_mm_count']}/20 depart at least 0.1 mm from their matched input-off body; {counts['no_flip_count']}/20 avoid flips. The required counts remain 16, 16 and 18, respectively.

Across the intact trials, the selected DNp09 forward-drive cells emitted **{forward_spikes} spikes**, and the two DNa02 steering cells emitted **{steering_spikes} spikes**. The tested decoder's forward-rate term therefore remains zero; small steering-derived commands still reach the body. This is a measured limitation of the current stimulus/network/readout combination, not evidence that the spike recorder or complete brain is inactive. Maximum physical departure from the matched input-off body was **{max(effect['final_xy_departure_mm'] for effect in result['paired_effects']):.6f} mm**.

## Next dependent work

If an entry gate fails, first diagnose the sensory-to-descending response and its conversion to actual body motion. Keep existing failure records and thresholds. Any revised sensory encoding or decoder becomes a new, versioned candidate, calibrated on separate seeds and then tested on held-out matched trials. Adding a walking drive or escape controller is an explicit engineered intervention and must never be hidden behind the brain label.

After entry readiness, the progression is: live local cue response → orientation → obstacle navigation → unfamiliar layouts → approaching threats → food collection under danger. Each level needs fresh body-generated sensory observations, matched sensory-off controls, recorded collisions, falls and stalls, and a committed evaluation protocol before trials. The Step 8 stationary-camera recordings do not substitute for this live feedback. A failed level stops progression to the next; game unlocks do not certify scientific success.

The full measured evidence and motor-population diagnosis are in [Step 8](../stage8-20261006/RESULTS.md). `readiness.json` pins that audited result by hash. Rerun this check with `.venv-next/bin/python scripts/check_behavioral_readiness.py`.
'''
    (OUT / 'RESULTS.md').write_text(text)
    print(json.dumps(state))


if __name__ == '__main__':
    run()
