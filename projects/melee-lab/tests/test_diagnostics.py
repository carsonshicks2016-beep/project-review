"""Comprehensive verification for the Extreme Diagnostic & Stress-Testing Lab."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from melee_lab import server
from melee_lab.diagnostics import (
    STATE_CONDITIONED,
    action_probabilities,
    list_targets,
    probe_checkpoint,
    state_battery,
    state_sensitivity,
    probe_dataset,
    probe_replay,
    probe_target,
)


@pytest.fixture
def client():
    return TestClient(server.app)


def test_list_targets():
    root = Path('.')
    targets = list_targets(root)
    assert isinstance(targets, list)
    assert len(targets) > 0

    types = {t['type'] for t in targets}
    assert 'checkpoint' in types
    assert 'dataset' in types

    sample = targets[0]
    assert 'path' in sample
    assert 'name' in sample
    assert 'type' in sample
    assert 'size_bytes' in sample


def test_probe_checkpoint():
    root = Path('.')
    # Find any existing checkpoint in runs
    ckpts = list(root.glob('runs/*/*.zip'))
    valid_ckpts = [p for p in ckpts if p.with_suffix('.json').exists()]
    assert len(valid_ckpts) > 0, "No valid checkpoint found for testing"

    target = valid_ckpts[0]
    res = probe_checkpoint(target, root=root, stress_level='standard')

    assert res['file_type'] == 'checkpoint'
    assert 'composite_score' in res
    assert 0.0 <= res['composite_score'] <= 100.0
    assert 'rank' in res
    assert 'health_grade' in res
    assert res['health_grade'] in ('EXCELLENT', 'WARNING', 'CRITICAL')
    assert 'total_parameters' in res
    assert res['total_parameters'] > 0
    assert 'layer_stats' in res
    assert len(res['layer_stats']) > 0

    radar = res['radar']
    for axis in ('awareness', 'reaction', 'recovery', 'confirms', 'discipline', 'robustness'):
        assert axis in radar
        assert 0.0 <= radar[axis] <= 100.0

    stress = res['stress_tests']
    assert 'threat_reaction_score' in stress
    assert 'deep_recovery_score' in stress
    assert 'lethal_confirm_score' in stress
    assert 'discipline_score' in stress
    assert 'noise_robustness_pct' in stress

    assert 'recommendations' in res
    assert len(res['recommendations']) > 0


def test_probe_dataset():
    root = Path('.')
    datasets = list(root.glob('datasets/*.npz'))
    assert len(datasets) > 0, "No dataset found in datasets/"

    target = datasets[0]
    res = probe_dataset(target, root=root)

    assert res['file_type'] == 'dataset'
    assert res['samples'] > 0
    assert res['games_covered'] > 0
    assert 0.0 <= res['composite_score'] <= 100.0
    assert 'action_distribution' in res
    assert 'returns' in res
    assert 'recommendations' in res


def test_probe_replay():
    root = Path('.')
    replays = sorted(root.glob('runs/**/Game_20260916T181333.slp'))
    if not replays:
        # Fallback to any completed replay
        candidates = sorted(root.glob('runs/**/*.slp'), key=lambda p: p.stat().st_mtime)
        replays = [p for p in candidates if p.stat().st_size > 50000]

    if replays:
        target = replays[0]
        res = probe_replay(target, root=root)
        assert res['file_type'] == 'replay'
        assert res['duration_seconds'] > 0
        assert res['total_frames'] > 0
        assert 'apm' in res
        assert 'rest_telemetry' in res
        assert 'composite_score' in res
        assert 0.0 <= res['composite_score'] <= 100.0


def test_probe_target_dispatcher():
    root = Path('.')
    ckpts = [p for p in root.glob('runs/*/*.zip') if p.with_suffix('.json').exists()]
    if ckpts:
        res = probe_target(ckpts[0], root=root, stress_level='standard')
        assert res['file_type'] == 'checkpoint'

    datasets = list(root.glob('datasets/*.npz'))
    if datasets:
        res = probe_target(datasets[0], root=root)
        assert res['file_type'] == 'dataset'

    with pytest.raises(ValueError, match="Unsupported file format"):
        probe_target('dummy.txt', root=root)


def test_api_targets(client):
    resp = client.get('/api/diagnostics/targets')
    assert resp.status_code == 200
    data = resp.json()
    assert 'targets' in data
    assert isinstance(data['targets'], list)
    assert len(data['targets']) > 0


def test_api_probe(client):
    root = Path('.')
    ckpts = [p for p in root.glob('runs/*/*.zip') if p.with_suffix('.json').exists()]
    assert len(ckpts) > 0

    rel_path = str(ckpts[0].relative_to(root))
    resp = client.post('/api/diagnostics/probe', json={
        'target': rel_path,
        'stress_level': 'standard'
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data['file_type'] == 'checkpoint'
    assert 'composite_score' in data
    assert 'radar' in data


def test_api_probe_not_found(client):
    resp = client.post('/api/diagnostics/probe', json={
        'target': 'runs/nonexistent/latest.zip',
        'stress_level': 'standard'
    })
    assert resp.status_code == 404


def test_api_upload(client):
    resp = client.post(
        '/api/diagnostics/upload?filename=test_fixture.npz',
        content=b'\x93NUMPY\x01\x00',
        headers={'Content-Type': 'application/octet-stream'}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 'path' in data
    assert data['filename'] == 'test_fixture.npz'
    assert Path(data['path']).exists()
    # Clean up fixture
    Path(data['path']).unlink(missing_ok=True)


def test_probe_gamecube_save(tmp_path):
    # Create mock .gci file
    gci_file = tmp_path / '01-GALE-melee.gci'
    dummy_payload = b'GALE01\x00\x00' + b'\x12' * 2048
    gci_file.write_bytes(dummy_payload)

    from melee_lab.diagnostics import probe_gamecube_save
    res = probe_gamecube_save(gci_file, root=tmp_path)

    assert res['file_type'] == 'gamecube_save'
    assert res['game_code'] == 'GALE01'
    assert res['file_size_bytes'] == len(dummy_payload)
    assert 'installed_path' in res
    assert res['composite_score'] == 100.0
    assert (tmp_path / '.runtime/dolphin-user/GC/USA/Card A/01-GALE-melee.gci').exists()


def test_api_import_checkpoint_and_gci(client, tmp_path):
    root = Path('.')
    ckpts = list(root.glob('runs/*/*.zip'))
    valid_ckpts = [p for p in ckpts if p.with_suffix('.json').exists()]
    assert len(valid_ckpts) > 0

    # 1. Test importing checkpoint via raw bytes
    ckpt_bytes = valid_ckpts[0].read_bytes()
    resp = client.post(
        '/api/checkpoints/import?filename=test_custom_save.zip&name=Test%20Pro%20Puff&character=JIGGLYPUFF',
        content=ckpt_bytes,
        headers={'Content-Type': 'application/octet-stream'}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data['ok'] is True
    assert data['type'] == 'checkpoint'
    assert data['name'] == 'Test Pro Puff'
    assert data['character'] == 'JIGGLYPUFF'
    imported_path = Path(data['path'])
    assert imported_path.exists()
    assert imported_path.with_suffix('.json').exists()

    # Clean up imported run folder
    import shutil
    shutil.rmtree(imported_path.parent, ignore_errors=True)

    # 2. Test importing GameCube memory card save (.gci)
    gci_bytes = b'GALE01\x00\x00' + b'\x55' * 1024
    resp_gci = client.post(
        '/api/checkpoints/import?filename=test_card.gci',
        content=gci_bytes,
        headers={'Content-Type': 'application/octet-stream'}
    )
    assert resp_gci.status_code == 200
    gci_data = resp_gci.json()
    assert gci_data['ok'] is True
    assert gci_data['type'] == 'gamecube_save'
    assert Path(gci_data['path']).exists()
    Path(gci_data['path']).unlink(missing_ok=True)


def test_api_import_via_source_path(client):
    root = Path('.')
    ckpts = [p for p in root.glob('runs/*/*.zip') if p.with_suffix('.json').exists()]
    assert len(ckpts) > 0

    src = str(ckpts[0].relative_to(root))
    resp = client.post(f'/api/checkpoints/import?source_path={src}&name=Cloned%20Target&character=FOX')
    assert resp.status_code == 200
    data = resp.json()
    assert data['ok'] is True
    assert data['name'] == 'Cloned Target'
    assert data['character'] == 'FOX'
    imported_path = Path(data['path'])
    assert imported_path.exists()

    # Clean up
    import shutil
    shutil.rmtree(imported_path.parent, ignore_errors=True)


def test_scan_fleet():
    from melee_lab.diagnostics import scan_fleet
    root = Path('.')
    rankings = scan_fleet(root=root, scope='distinct', stress_level='standard')
    assert isinstance(rankings, list)
    assert len(rankings) > 0
    # Verify sorted descending by composite_score
    scores = [r.get('composite_score', 0.0) for r in rankings]
    assert scores == sorted(scores, reverse=True)
    first = rankings[0]
    assert 'rank_position' in first
    assert first['rank_position'] == 1
    assert 'composite_score' in first
    assert 'radar' in first


def test_api_rankings_and_status(client):
    # Test GET rankings
    resp = client.get('/api/diagnostics/rankings')
    assert resp.status_code == 200
    data = resp.json()
    assert 'rankings' in data
    assert 'is_scanning' in data

    # Test GET status
    status_resp = client.get('/api/diagnostics/rankings/status')
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert 'is_scanning' in status_data

    # Test trigger scan endpoint
    scan_resp = client.post('/api/diagnostics/rankings/scan?scope=distinct')
    assert scan_resp.status_code == 200
    assert scan_resp.json()['ok'] is True




def _blind_model():
    """A policy whose logits ignore the observation entirely -- the failure this probe
    exists to name. Zeroing the action head makes every state produce one distribution."""
    import numpy as np
    import torch
    import gymnasium as gym
    from stable_baselines3 import PPO
    from melee_lab.controller import DIMENSIONS
    from melee_lab.state import OBS_SIZE

    class Fixture(gym.Env):
        observation_space = gym.spaces.Box(-5, 5, shape=(OBS_SIZE,), dtype=np.float32)
        action_space = gym.spaces.MultiDiscrete(np.array(DIMENSIONS, dtype=np.int64))
        def reset(self, seed=None, options=None): return np.zeros(OBS_SIZE, np.float32), {}
        def step(self, action): return np.zeros(OBS_SIZE, np.float32), 0.0, True, False, {}

    model = PPO('MlpPolicy', Fixture(), n_steps=16, batch_size=8, seed=0,
                policy_kwargs={'net_arch': [8]}, device='cpu')
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    return model


def test_state_sensitivity_names_a_state_blind_policy():
    model = _blind_model()
    result = state_sensitivity(model, 'FOX')
    # Identical logits everywhere, so every pairwise distance collapses to zero.
    assert result['mean_tv'] == pytest.approx(0.0, abs=1e-6)
    assert result['max_tv'] == pytest.approx(0.0, abs=1e-6)
    assert result['verdict'] == 'STATE-BLIND'
    assert result['score'] == pytest.approx(0.0, abs=1e-3)
    assert result['gate'] == STATE_CONDITIONED
    assert result['states'] == len(state_battery('FOX'))


def test_state_sensitivity_is_a_bounded_distance():
    model = _blind_model()
    for character in ('FOX', 'JIGGLYPUFF'):
        result = state_sensitivity(model, character)
        # Total variation is a probability distance: it cannot leave [0, 1].
        assert 0.0 <= result['min_tv'] <= result['mean_tv'] <= result['max_tv'] <= 1.0
        assert result['closest_pair'][0] != result['closest_pair'][1]
        assert result['onstage_vs_dying_tv'] is not None


def test_state_battery_adapts_to_the_character():
    fox, puff = state_battery('FOX'), state_battery('JIGGLYPUFF')
    assert set(fox) == set(puff)
    # Jump count is what makes an offstage state readable, and it is character-specific.
    assert fox['deep offstage'][0]['jumps_left'] == 2
    assert puff['deep offstage'][0]['jumps_left'] == 5
    # Every scenario must name an agent and an opponent state.
    assert all(len(v) == 2 for v in fox.values())


def test_action_probabilities_cover_every_controller_axis():
    from melee_lab.controller import DIMENSIONS
    from melee_lab.diagnostics import _build_synthetic_obs
    import melee, numpy as np

    obs = _build_synthetic_obs(
        p1_kwargs=dict(character=melee.Character.FOX, pos_x=0.0, pos_y=0.0),
        p2_kwargs=dict(character=melee.Character.FOX, pos_x=20.0, pos_y=0.0))
    axes = action_probabilities(_blind_model(), obs)
    assert [len(a) for a in axes] == list(DIMENSIONS)
    for a in axes:
        assert np.isclose(a.sum(), 1.0)
