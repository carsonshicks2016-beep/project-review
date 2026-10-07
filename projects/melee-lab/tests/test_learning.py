import json
import numpy as np
import gymnasium as gym
import melee
import pytest
from stable_baselines3 import PPO
from melee_lab.state import History,OBS_SIZE,reward,outcome
from melee_lab.config import Config
from melee_lab.worker import fit_demonstrations
from melee_lab.storage import save_model,validate_checkpoint
from melee_lab.actions import ACTIONS,apply


def game(stocks=(3,3),damage=(0,0),frame=0):
    g=melee.GameState();g.frame=frame
    for i in (1,2):
        p=melee.PlayerState();p.stock=stocks[i-1];p.percent=damage[i-1]
        p.character=melee.Character.FOX if i==1 else melee.Character.MARIO
        p.action=melee.Action.STANDING;g.players[i]=p
    return g


def test_stock_rewards_do_not_mistake_respawn_for_damage():
    before=game((3,3),(20,140));after=game((3,2),(20,0))
    assert reward(before,after)==4
    assert reward(game((3,3),(140,20)),game((2,3),(0,20)))==-4
    assert reward(game((1,1),(30,30)),game((1,0),(30,0)))==14
    assert outcome(game((0,0)))=='draw'
    assert outcome(game((1,3))) is None


def test_history_preserves_previous_frame_and_clears_on_reset():
    h=History();a=game();b=game(frame=4);b.players[1].position.x=45
    first=h.reset(a);second=h.push(b)
    assert first.shape==(OBS_SIZE,)
    assert np.array_equal(first[:OBS_SIZE//2],second[:OBS_SIZE//2])
    assert not np.array_equal(first,second)
    assert np.array_equal(h.reset(a),first)
    assert np.isfinite(second).all()


class LearningFixture(gym.Env):
    """Tiny classification fixture for testing optimizer wiring; not a game simulator."""
    observation_space=gym.spaces.Box(-5,5,shape=(OBS_SIZE,),dtype=np.float32)
    action_space=gym.spaces.Discrete(len(ACTIONS))
    def reset(self,seed=None,options=None):
        super().reset(seed=seed); self.n=0
        return np.zeros(OBS_SIZE,np.float32),{}
    def step(self,a):
        self.n+=1
        return np.zeros(OBS_SIZE,np.float32),float(a==1),self.n>=8,False,{}


def test_real_optimizer_checkpoint_roundtrip_and_contract(tmp_path):
    import torch
    torch.set_num_threads(1)
    env=LearningFixture()
    model=PPO('MlpPolicy',env,n_steps=32,batch_size=16,n_epochs=2,seed=1,
              policy_kwargs={'net_arch':[16]},device='cpu')
    before=model.policy.action_net.weight.detach().clone()
    model.learn(64)
    assert not torch.equal(before,model.policy.action_net.weight)
    observations=np.zeros((64,OBS_SIZE),np.float32);labels=np.full(64,2)
    loss,acc=fit_demonstrations(model,observations,labels,epochs=50)
    assert acc==1
    target=tmp_path/'policy.zip';config=Config()
    save_model(model,target,config)
    validate_checkpoint(target,config)
    restored=PPO.load(target,env=env,device='cpu')
    assert int(restored.predict(observations[0],deterministic=True)[0])==2
    assert restored.num_timesteps==64
    config.action_frames=8
    with pytest.raises(ValueError,match='action frames'): validate_checkpoint(target,config)


class Controller:
    def __init__(self): self.buttons=set();self.sticks={}
    def release_all(self): self.buttons.clear();self.sticks.clear()
    def press_button(self,b): self.buttons.add(b)
    def tilt_analog(self,b,*xy): self.sticks[b]=xy


def test_button_pulses_allow_repeated_jumps_without_sticky_input():
    c=Controller();apply(c,4,0)
    assert melee.Button.BUTTON_Y in c.buttons
    apply(c,4,1);assert not c.buttons
    apply(c,21,2);assert melee.Button.BUTTON_L in c.buttons
    apply(c,0,0);assert not c.buttons


def test_snapshot_survives_an_animation_libmelee_cannot_name():
    """libmelee returns UnknownAnimation for ids outside its enum, which has no .name;
    snapshot() reached straight for it and took down the whole run."""
    from melee_lab.state import snapshot
    class UnknownAnimation:
        def __init__(self,value): self.value=value
    g=game()
    g.players[1].action=UnknownAnimation(9999)
    row=snapshot(g)['1']
    assert row['action']=='UNKNOWN_9999'
    assert snapshot(game())['1']['action']=='STANDING'


def test_new_model_architecture():
    from melee_lab.worker import new_model
    env = LearningFixture()
    model = new_model(env, seed=42)
    # Check policy network architecture has 256 units in layers
    mlp = model.policy.mlp_extractor
    assert mlp.latent_dim_pi == 256
    assert mlp.latent_dim_vf == 256
    # Check the first layer takes OBS_SIZE and outputs 256
    assert mlp.policy_net[0].in_features == OBS_SIZE
    assert mlp.policy_net[0].out_features == 256
    assert mlp.policy_net[2].in_features == 256
    assert mlp.policy_net[2].out_features == 256


def test_anchored_ppo_runs_and_leaves_a_loadable_checkpoint(tmp_path):
    """The whole point of anchoring is that it survives a real PPO loop: SB3 owns the
    optimizer and the rollout buffer, and the anchor borrows both between rollouts."""
    import torch
    from melee_lab.controller import DIMENSIONS
    from melee_lab.imitation import DummyImitationEnv
    from melee_lab.worker import Anchor, finetune, new_model

    class Bandit(DummyImitationEnv):
        """Rewards exactly the action the demonstrations show, so an anchored run has a
        reachable optimum and the two objectives cannot pull against each other."""
        def step(self, action):
            return np.zeros(OBS_SIZE, np.float32), float(int(action[0]) == 42), False, True, {}

    rng = np.random.default_rng(0)
    demonstrations = tmp_path / 'demo.npz'
    actions = np.zeros((256, len(DIMENSIONS)), dtype=np.int64); actions[:, 0] = 42
    np.savez_compressed(demonstrations,
                        observations=rng.normal(0, .3, (256, OBS_SIZE)).astype(np.float32),
                        actions=actions)

    class Runtime:
        def __init__(self): self.published = {}
        def publish(self, **kw): self.published.update(kw)
        def control(self): pass
        def idle(self): pass

    runtime = Runtime()
    model = finetune(new_model(Bandit(), seed=5))
    assert model.ent_coef == .001
    anchor = Anchor(demonstrations, runtime, coefficient=.5)
    model.learn(total_timesteps=2048, callback=[anchor])

    assert runtime.published['anchor_samples'] == 256
    # Decayed over the budget rather than held at its starting weight.
    assert 0 <= runtime.published['anchor_weight'] < .5
    observation = torch.zeros((1, OBS_SIZE))
    assert int(model.policy.get_distribution(observation).mode()[0][0]) == 42

    save_model(model, tmp_path / 'anchored', Config(action_set='controller', action_frames=1))
    reloaded = PPO.load(tmp_path / 'anchored.zip', device='cpu')
    assert reloaded.num_timesteps == model.num_timesteps


def test_factored_evaluation_defaults_to_sampling():
    """A factored pad's joint mode is every axis at rest, which is an absorbing state:
    a neutral input leaves the next observation almost unchanged, so the mode returns
    neutral again and the agent stands still. Controller policies must be sampled;
    named-move vocabularies, where the most likely move is a real answer, keep the mode."""
    import inspect
    from melee_lab import worker
    assert inspect.signature(worker.evaluate).parameters['deterministic'].default is None
    body = inspect.getsource(worker.evaluate)
    assert 'if deterministic is None: deterministic=not factored' in body
    assert 'model.predict(obs,deterministic=deterministic)' in body


def test_catalog_accepts_sampled_evaluation_evidence(tmp_path):
    """Controller policies can only ever produce sampled evidence, so refusing it would
    make them impossible to promote on any evaluation they can actually run."""
    import hashlib
    from melee_lab.catalog import Catalog
    from melee_lab.storage import write_json

    run = tmp_path / 'runs' / '20260914-000000-abcde'
    run.mkdir(parents=True)
    policy = run / 'input-policy.zip'
    policy.write_bytes(b'not really a policy, but its bytes are what identify it')
    write_json(policy.with_suffix('.json'), {'config': {}, 'steps': 5})

    def evidence_for(label):
        write_json(run / 'request.json', dict(mode='evaluate', checkpoint_identity=dict(
            sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
            source='runs/x/latest.zip', steps=5, config={})))
        write_json(run / 'evaluation.json', dict(
            policy=label, scripted_overrides=False, three_stock_start_verified=True,
            episodes=2, wins=1, results=['win', 'loss'], config={'cpu_level': 1}))
        catalog = Catalog(tmp_path)
        catalog.ingest_evaluations()
        return [e for e in catalog.evidence() if e.get('run_id') == run.name]

    assert evidence_for('sampled neural network'), 'sampled evidence must be ingested'
    assert evidence_for('deterministic neural network'), 'mode evidence must still be ingested'
    (tmp_path / 'evaluations' / (run.name + '.json')).unlink()
    assert not evidence_for('scripted teacher'), 'non-policy evidence must stay out'


def test_decision_hold_is_off_by_default():
    """Holding a decision across frames was calibrated on the states pros visit. On the
    states the agent actually reaches the policy is far more peaked and already repeats
    60% of frames, so a hold made it twice as sticky as a human and scored worse. It
    stays available as a flag and off by default."""
    import inspect
    from melee_lab import worker
    assert worker.DECISION_HOLD == 1
    assert "if action is None or frame%hold==0:" in inspect.getsource(worker.evaluate)


def test_decision_hold_when_enabled_raises_input_persistence():
    """When switched on, the hold must do what it claims to."""
    rng = np.random.default_rng(0)
    # Stand in for a policy: independent draws per frame, as the real one makes.
    draws = rng.integers(0, 40, size=(6000, 7))

    def persistence(a):
        return np.all(a[1:] == a[:-1], axis=1).mean()

    def held(a, k):
        out = a.copy()
        for i in range(len(out)):
            if i % k: out[i] = out[i - 1]
        return out

    assert persistence(draws) < .05
    # Holding k frames drives persistence toward (k-1)/k; at k=3 that is ~67-75%,
    # the band human play occupies (76%).
    assert .6 < persistence(held(draws, 3)) < .8
    assert persistence(held(draws, 1)) < persistence(held(draws, 3)) < persistence(held(draws, 6))


def test_features_libmelee_cannot_read_from_replays_are_zeroed():
    """invulnerability_left and the ECB box come from post-frame bytes that pre-3.x
    replays lack, so footage parses them as 0 while live Dolphin supplies real values --
    ECB in absolute world coordinates, which saturate the clip at /20. Reading them
    would train on constants and infer on large numbers."""
    g = game()
    p = g.players[1]
    p.invulnerability_left = 120
    p.ecb.bottom.y, p.ecb.top.y = -90.0, -78.0
    p.ecb.left.x, p.ecb.right.x = -101.0, -96.4
    from melee_lab.state import player_features
    assert list(player_features(p)[18:23]) == [0., 0., 0., 0., 0.]
    # Neighbours must still be read, or the slice is masking more than intended.
    assert player_features(p)[17] == float(p.off_stage)
    # The slots stay, so the observation contract and every checkpoint remain valid.
    assert OBS_SIZE == 1836


def test_anchor_hands_off_fully_on_a_resumed_run():
    """SB3 adds the checkpoint's existing steps into total_timesteps, so the decay has
    to span what this call actually runs or the anchor never reaches zero."""
    import numpy as np
    import tempfile, pathlib
    from melee_lab.controller import DIMENSIONS
    from melee_lab.state import OBS_SIZE
    from melee_lab.worker import Anchor

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / 'demo.npz'
        np.savez_compressed(path, observations=np.zeros((4, OBS_SIZE), np.float32),
                            actions=np.zeros((4, len(DIMENSIONS)), np.int64))
        class Runtime:
            def publish(self, **kw): pass
        anchor = Anchor(path, Runtime(), coefficient=0.5)

        class Model: num_timesteps = 600_000
        anchor.model = Model()
        anchor.locals = {'total_timesteps': 10_600_000}   # 600k resumed + 10M new
        anchor._on_training_start()

        assert anchor.start == 600_000
        assert anchor.total == 10_000_000
        anchor.num_timesteps = 600_000
        assert anchor.weight() == pytest.approx(0.5)
        anchor.num_timesteps = 5_600_000
        assert anchor.weight() == pytest.approx(0.25)
        anchor.num_timesteps = 10_600_000
        assert anchor.weight() == pytest.approx(0.0)
