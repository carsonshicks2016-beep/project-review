"""Tests for tournament replay parsing and behavioral cloning imitation learning."""
import glob
import json
from pathlib import Path
import numpy as np
import pytest
import melee

from melee_lab import controller, imitation
from melee_lab.config import Config
from melee_lab.state import OBS_SIZE
from melee_lab.storage import validate_checkpoint


@pytest.fixture
def sample_slp():
    matches = glob.glob('runs/**/replays/*.slp', recursive=True)
    clean = []
    for p in matches:
        with open(p, 'rb') as f:
            data = f.read()
        if data.rfind(b'metadata') != -1 and len(data) > 100000:
            clean.append(p)
    if not clean:
        pytest.skip('No completed .slp replays available in runs/ for testing.')
    return clean[0]


def test_parse_replay_extracts_observations_and_factored_actions(sample_slp):
    obs, acts, returns = imitation.parse_replay(sample_slp, character='FOX', stage='FINAL_DESTINATION')
    assert len(obs) > 0
    assert len(acts) == len(obs)
    assert len(returns) == len(obs)
    assert returns.dtype == np.float32
    assert np.isfinite(returns).all()
    assert obs.shape[1] == OBS_SIZE
    assert acts.shape[1] == len(controller.DIMENSIONS)
    assert obs.dtype == np.float32
    assert acts.dtype == np.int64

    # Every action component must fall within valid axis dimensions
    for col, limit in enumerate(controller.DIMENSIONS):
        assert np.all(acts[:, col] >= 0)
        assert np.all(acts[:, col] < limit)


def test_parse_replay_skips_mismatched_character(sample_slp):
    obs, acts, returns = imitation.parse_replay(sample_slp, character='BOWSER', stage='FINAL_DESTINATION')
    assert len(obs) == 0
    assert len(acts) == 0
    assert len(returns) == 0


def test_discounted_returns_accumulate_backwards():
    out = imitation.discounted([1.0, 0.0, 2.0], 0.5)
    assert out[2] == pytest.approx(2.0)
    assert out[1] == pytest.approx(1.0)
    assert out[0] == pytest.approx(1.5)


def test_validation_split_holds_out_whole_replays():
    groups = np.repeat(np.arange(10), 100)
    train, val = imitation.split(len(groups), groups, 0.2, seed=7)
    assert len(train) + len(val) == len(groups)
    # No replay may contribute frames to both sides.
    assert not set(groups[train]) & set(groups[val])
    assert len(set(groups[val])) == 2


def test_extract_action_vector():
    class MockCtrl:
        main_stick = (1.0, 0.5)
        c_stick = (0.5, 0.5)
        l_shoulder = 0.0
        r_shoulder = 0.0
        button = {melee.Button.BUTTON_A: True, melee.Button.BUTTON_B: False}

    vec = imitation.extract_action_vector(MockCtrl())
    assert len(vec) == len(controller.DIMENSIONS)
    assert vec[2] == 1  # A button is index 2


def test_train_imitation_reduces_loss_and_saves_valid_checkpoint(sample_slp, tmp_path):
    obs, acts, returns = imitation.parse_replay(sample_slp, character='FOX', stage='FINAL_DESTINATION')
    # Train on first 500 samples for quick test execution
    sub_obs, sub_acts, sub_returns = obs[:500], acts[:500], returns[:500]

    out_dir = tmp_path / 'imitation-run'
    model, report = imitation.train_imitation((sub_obs, sub_acts, sub_returns), out_dir, epochs=2, batch_size=128)

    assert report['epochs'] == 2
    assert report['samples'] == 500
    assert report['value_head_fitted'] is True
    assert report['final_value_explained_variance'] is not None
    assert len(report['history']) == 2
    assert report['history'][1]['train_loss'] < report['history'][0]['train_loss']

    ckpt_path = Path(report['checkpoint'])
    assert ckpt_path.exists()
    assert (ckpt_path.with_suffix('.json')).exists()
    assert (out_dir / 'imitation.json').exists()

    # Must be valid for controller action space
    config = Config(action_set='controller', action_frames=1, character='FOX', stage='FINAL_DESTINATION', randomize_opponent=True)
    meta = validate_checkpoint(ckpt_path, config)
    assert meta['actions']['space'] == 'controller-v1'


def test_value_head_learns_the_demonstrated_returns(tmp_path):
    """A random critic is what destroys a cloned policy on PPO's first update, so the
    value head must actually come out of imitation fitted to something."""
    # Comfortably more samples than the 1,836 observed features: below that the critic
    # can fit any target on spurious directions and generalise to none of them.
    rng = np.random.default_rng(0)
    obs = rng.normal(0, .3, (8000, OBS_SIZE)).astype(np.float32)
    acts = np.zeros((8000, len(controller.DIMENSIONS)), dtype=np.int64)
    # A return the critic can only get right by reading the observation.
    returns = (5 * obs[:, 0]).astype(np.float32)

    _, report = imitation.train_imitation((obs, acts, returns), tmp_path / 'value', epochs=15, batch_size=128)
    assert report['final_value_explained_variance'] > 0.5


def test_anchor_pulls_a_policy_toward_the_demonstrations(tmp_path):
    """The anchor is the whole point of running imitation alongside PPO: applied to a
    policy that has drifted, its supervised batches must move it back."""
    import torch
    from melee_lab.worker import Anchor, new_model

    rng = np.random.default_rng(0)
    obs = rng.normal(0, .3, (400, OBS_SIZE)).astype(np.float32)
    acts = np.zeros((400, len(controller.DIMENSIONS)), dtype=np.int64)
    acts[:, 0] = 42
    path = tmp_path / 'demo.npz'
    np.savez_compressed(path, observations=obs, actions=acts)

    class Runtime:
        directory = tmp_path
        def publish(self, **kw): pass
        def control(self): pass
        def idle(self): pass

    model = new_model(imitation.DummyImitationEnv(), seed=3)
    anchor = Anchor(path, Runtime(), coefficient=1.0)
    anchor.model = model
    anchor.total = 1000

    x = torch.as_tensor(obs, dtype=torch.float32)
    y = torch.as_tensor(acts, dtype=torch.long)
    with torch.no_grad():
        before = float(-model.policy.get_distribution(x).log_prob(y).mean())
    for _ in range(25):
        anchor._on_rollout_start()
    with torch.no_grad():
        after = float(-model.policy.get_distribution(x).log_prob(y).mean())
    assert after < before


def test_anchor_rejects_demonstrations_from_another_controller(tmp_path):
    from melee_lab.worker import Anchor
    path = tmp_path / 'wrong.npz'
    np.savez_compressed(path, observations=np.zeros((4, OBS_SIZE), np.float32),
                        actions=np.full((4, len(controller.DIMENSIONS)), 999, np.int64))
    with pytest.raises(ValueError, match='different controller'):
        Anchor(path, None)


def test_anchor_rejects_demonstrations_from_another_observation_encoding(tmp_path):
    from melee_lab.worker import Anchor
    path = tmp_path / 'stale.npz'
    np.savez_compressed(path, observations=np.zeros((4, 64), np.float32),
                        actions=np.zeros((4, len(controller.DIMENSIONS)), np.int64))
    with pytest.raises(ValueError, match='this build observes'):
        Anchor(path, None)


def test_packed_round_trip_is_exact_and_much_smaller(sample_slp):
    from melee_lab.imitation import Demonstrations
    obs, acts, returns = imitation.parse_replay(sample_slp, character='FOX', stage='FINAL_DESTINATION')
    packed = Demonstrations.from_dense(obs[:2000], acts[:2000], returns[:2000])
    assert np.array_equal(packed.observations(), obs[:2000])
    assert np.array_equal(packed.observations(np.arange(10)), obs[:10])
    # Reused buffers must not leak the previous minibatch into the next one.
    first = packed.observations(np.arange(5), reuse=True).copy()
    packed.observations(np.arange(100, 110), reuse=True)
    assert np.array_equal(packed.observations(np.arange(5), reuse=True), first)
    assert packed.nbytes < obs[:2000].nbytes / 10


def test_shards_join_with_replays_kept_distinct():
    from melee_lab.imitation import Demonstrations, concatenate
    def shard(n, groups):
        return Demonstrations.from_dense(
            np.zeros((n, OBS_SIZE), np.float32),
            np.zeros((n, len(controller.DIMENSIONS)), np.int64),
            np.zeros(n, np.float32), np.asarray(groups, np.int32))
    joined = concatenate([shard(4, [0, 0, 1, 1]), shard(3, [0, 1, 2])])
    assert len(joined) == 7
    # The second shard's replays must not be merged into the first shard's.
    assert list(joined.groups) == [0, 0, 1, 1, 2, 3, 4]


def test_strided_sampling_keeps_the_two_observed_frames_adjacent(sample_slp):
    """Stride selects which decisions to keep; it must not stretch the frame stack,
    which the live environment always fills with consecutive frames."""
    dense, _, _ = imitation.parse_replay(sample_slp, character='FOX', stage='FINAL_DESTINATION', stride=1)
    strided, _, _ = imitation.parse_replay(sample_slp, character='FOX', stage='FINAL_DESTINATION', stride=4)
    assert 0 < len(strided) < len(dense)
    half = OBS_SIZE // 2
    # Each strided observation is a (previous frame, current frame) pair that also
    # appears, adjacent, in the unstrided parse.
    lookup = {dense[i].tobytes(): i for i in range(len(dense))}
    found = sum(1 for row in strided[:200] if row.tobytes() in lookup)
    assert found > 150


def test_anchor_weight_decays_to_zero_across_the_budget(tmp_path):
    from melee_lab.worker import Anchor
    path = tmp_path / 'demo.npz'
    np.savez_compressed(path, observations=np.zeros((4, OBS_SIZE), np.float32),
                        actions=np.zeros((4, len(controller.DIMENSIONS)), np.int64))
    anchor = Anchor(path, None, coefficient=0.5)
    anchor.total = 100
    anchor.num_timesteps = 0
    assert anchor.weight() == pytest.approx(0.5)
    anchor.num_timesteps = 50
    assert anchor.weight() == pytest.approx(0.25)
    anchor.num_timesteps = 100
    assert anchor.weight() == pytest.approx(0.0)


def test_holdout_is_capped_so_a_large_archive_keeps_its_replays():
    groups = np.repeat(np.arange(5000), 20)
    train, val = imitation.split(len(groups), groups, 0.15, seed=7)
    assert len(set(groups[val])) == imitation.MAX_HELD_OUT_REPLAYS
    assert not set(groups[train]) & set(groups[val])
    assert len(set(groups[train])) == 5000 - imitation.MAX_HELD_OUT_REPLAYS


def test_actor_and_critic_keep_their_own_best_epoch(tmp_path):
    """The two branches share no weights, so composing the best of each is exact --
    not an average, and not a compromise epoch that is best for neither."""
    from melee_lab.imitation import DummyImitationEnv, branch
    from melee_lab.worker import new_model

    early = new_model(DummyImitationEnv(), seed=1).policy.state_dict()
    late = new_model(DummyImitationEnv(), seed=2).policy.state_dict()
    composed = dict(late)
    composed.update(branch(early, True))       # critic from the early model

    model = new_model(DummyImitationEnv(), seed=3)
    model.policy.load_state_dict(composed)
    restored = model.policy.state_dict()
    for key in restored:
        source = early if key.startswith(imitation.VALUE_BRANCH) else late
        assert np.array_equal(restored[key].numpy(), source[key].numpy()), key


def test_controller_transfer_refuses_a_policy_that_already_has_the_controller():
    """Reshaping a factored head by move name leaves every row past the vocabulary at its
    random initialisation -- which is the whole of the C-stick, A, B, Y, Z and trigger."""
    from melee_lab.transfer import transfer_policy
    from melee_lab.config import Config
    from melee_lab.imitation import DummyImitationEnv
    env = DummyImitationEnv()
    env.config = Config(action_set='controller', action_frames=1)
    with pytest.raises(ValueError, match='already uses the full controller'):
        transfer_policy('runs/tournament-fox/latest.zip', env, seed=1)


def test_manager_treats_full_controller_on_a_controller_policy_as_a_resume(tmp_path):
    import subprocess, shutil
    from melee_lab.manager import Manager
    class Fake:
        pid = 1
        def poll(self): return 0
    original, subprocess.Popen = subprocess.Popen, lambda cmd, **kw: Fake()
    try:
        m = Manager()
        # Isolate from whatever is running on this machine right now.
        m.active = lambda: None
        m.availability = lambda: dict(can_start=True, reason='ready', message='')
        run = m.start(mode='train', steps=2048, bootstrap=0, envs=1,
                      checkpoint='runs/tournament-fox/latest.zip', fast_inputs=True)
        request = json.loads((m.runs / run / 'request.json').read_text())
        assert request['fast_inputs'] is False, 'must not transfer a controller policy'
        assert request['config']['action_set'] == 'controller'
        shutil.rmtree(m.runs / run)
    finally:
        subprocess.Popen = original
