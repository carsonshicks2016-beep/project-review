"""Explicit identity and motor attribution for existing live controllers.

This describes the legacy application's actual behavior. It does not promote
the experimental causal controller or change neural/gait dynamics.
"""
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TONIC_P9 = (.65, .65)
SOURCE_HASHES = {f'flygarden/{name}': hashlib.sha256((ROOT / 'flygarden' / name).read_bytes()).hexdigest()
                 for name in ('engine.py','controller_contract.py','body.py','world.py','brain.py','baseline.py')}

def controller_contract(mode, learning=False):
    if mode not in ('full', 'baseline'):
        raise ValueError('Unknown controller mode')
    full = mode == 'full'
    names = ['engine.py', 'controller_contract.py', 'body.py', 'world.py',
             'brain.py' if full else 'baseline.py']
    sources = {f'flygarden/{name}': SOURCE_HASHES[f'flygarden/{name}'] for name in names}
    return {
        'schema_version': 1,
        'id': 'engineered-hybrid-legacy-v1' if full else 'supplied-local-baseline-v3',
        'name': 'Engineered hybrid · full connectome' if full else 'Supplied local-sensory baseline v3',
        'kind': 'engineered_hybrid' if full else 'supplied_baseline',
        'brain_enabled': full,
        'learning_active': bool(full and learning),
        'behavioral_learning_demonstrated': False,
        'validated_neural_navigation': False,
        'tonic_descending_stimulation': list(TONIC_P9) if full else [],
        'sensory_encoding': 'Synthetic ORN_DM1/DM2 odor; geometric egocentric threat proxy' if full else 'Local odor, obstacle rays and threat proxy',
        'leg_controller': 'Supplied NeuroMechFly HybridTurningController',
        'supplied_escape': True,
        'command_interval_seconds': .1,
        'command_timing': 'Legacy same-window neural output application' if full else 'Local-sensory supplied motor policy',
        'experimental_causal_scheduler_integrated': False,
        'sources': sources,
    }

def attribute_motor(mode, sensory, proposed):
    """Return unchanged valid motor behavior with its actual causal source."""
    if mode not in ('full', 'baseline'):
        raise ValueError('Unknown controller mode')
    raw = np.asarray(proposed, dtype=float)
    if raw.shape != (2,) or not np.isfinite(raw).all():
        raise ValueError('Expected two finite proposed motor commands')
    motor = raw.copy()
    override = mode == 'full' and sensory['threat'] > .25
    if override:
        turn = .6 if sensory['threat_bearing'] > 0 else -.6
        motor = np.array([1. + turn, 1. - turn])
    return {
        'proposed_motor': raw.tolist(),
        'applied_motor': np.clip(motor, 0, 1.2).tolist(),
        'source': 'supplied_escape_override' if override else 'brain_adapter_with_tonic_support' if mode == 'full' else 'supplied_local_policy',
        'escape_override_active': bool(override),
        'controller_id': 'engineered-hybrid-legacy-v1' if mode == 'full' else 'supplied-local-baseline-v3',
    }
