"""Readable PPO driver: parallel MuJoCo, paired checkpoints, honest evaluation.

--steps means NEW decisions, also on resume. Budgets round up to one rollout.
SIGINT/SIGTERM request a checkpoint at the next safe callback boundary.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import pickle
import shutil
import signal
import time
import uuid
from pathlib import Path
import gymnasium as gym
import numpy as np
import psutil
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs import make_parkour_env
from eval import evaluate

ROOT = Path(__file__).resolve().parent
PPO_KEYS = ['n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma',
            'gae_lambda', 'clip_range', 'ent_coef', 'max_grad_norm', 'vf_coef', 'target_kl']


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    """Readers see either the old complete file or the new complete file."""
    path = Path(path)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective_ppo(model):
    return {key: value(1.0) if callable(value) else value
            for key in PPO_KEYS if (value := getattr(model, key, None)) is not None}


def checkpoint_metadata(directory):
    """Resolve aliases once; verify both files before loading trusted local files."""
    directory = Path(directory).resolve(strict=True)
    metadata = json.loads((directory / 'metadata.json').read_text())
    for name in ('policy.zip', 'vecnormalize.pkl'):
        expected = metadata.get('sha256', {}).get(name)
        if expected is None or digest(directory / name) != expected:
            raise ValueError(f'Checkpoint checksum missing or mismatched: {directory / name}')
    return directory, metadata


def make_stock_env(env_id, env_kwargs):
    def create():
        return Monitor(gym.make(env_id, **env_kwargs))
    return create


def environment_contract(config, vec, difficulty):
    parkour = config['env_id'] == 'ParkourHumanoid'
    physics = vec.env_method('get_contract')[0] if parkour else {
        'version': config['env_id'], 'gymnasium_version': gym.__version__}
    return dict(env_id=config['env_id'], env_kwargs=config.get('env_kwargs', {}),
                difficulty=float(difficulty) if parkour else None,
                replay_prob=float(config.get('curriculum', {}).get('replay_prob', 0.)),
                physics=physics, observation_shape=list(vec.observation_space.shape),
                action_shape=list(vec.action_space.shape), action_low=vec.action_space.low.tolist(),
                action_high=vec.action_space.high.tolist())


def validate_resume(metadata, contract):
    if metadata['env_id'] != contract['env_id']:
        raise ValueError('Resume environment differs; use --warm-start for a deliberate transfer.')
    old = metadata.get('environment')
    if old is None:
        # Upstream v5 has versioned semantics. Legacy parkour did not.
        if contract['env_id'] == 'ParkourHumanoid':
            raise ValueError('Legacy parkour physics changed. Use --warm-start and fresh evaluation.')
        if contract['env_kwargs']:
            raise ValueError('Legacy checkpoint cannot establish custom env kwargs; use --warm-start.')
        return
    for key in ('env_id', 'env_kwargs', 'physics', 'observation_shape', 'action_shape', 'action_low', 'action_high'):
        if old.get(key) != contract.get(key):
            raise ValueError(f'Resume contract changed ({key}); use --warm-start for a new experiment.')


def transfer_policy(source, target):
    """Transfer every layer; allow only appended observation columns at input.

    Architecture mismatches fail instead of silently copying only two layers.
    New sensor columns start at zero, preserving the source controller's actions.
    """
    if source.action_space.shape != target.action_space.shape:
        raise ValueError('Warm-start action shapes differ.')
    for attr in ('low', 'high'):
        if not np.array_equal(getattr(source.action_space, attr), getattr(target.action_space, attr)):
            raise ValueError('Warm-start action bounds differ.')
    old, new = source.policy.state_dict(), target.policy.state_dict()
    if old.keys() != new.keys():
        raise ValueError('Warm-start network depth differs; match the source net_arch.')
    source_dim, target_dim = source.observation_space.shape[0], target.observation_space.shape[0]
    for key, value in old.items():
        if value.shape == new[key].shape:
            new[key] = value.clone()
        elif (key in ('mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight')
              and value.shape[0] == new[key].shape[0] and value.shape[1] == source_dim
              and new[key].shape[1] == target_dim and source_dim < target_dim):
            new[key] = torch.zeros_like(new[key])
            new[key][:, :source_dim] = value
        else:
            raise ValueError(f'Warm-start architecture differs at {key}; match the source net_arch.')
    # Activation functions have no tensors, so compare separately.
    if source.policy.activation_fn != target.policy.activation_fn:
        raise ValueError('Warm-start activation differs; match the source activation.')
    target.policy.load_state_dict(new, strict=True)


def transfer_normalizer(source_path, norm, source_dim, target_dim, max_count=10000):
    with open(source_path, 'rb') as stream:
        previous = pickle.load(stream)  # trusted local SB3 checkpoint
    if previous.obs_rms.mean.shape != (source_dim,) or source_dim > target_dim:
        raise ValueError('Observation normalizer does not match the source policy.')
    norm.obs_rms.mean[:source_dim] = previous.obs_rms.mean
    norm.obs_rms.var[:source_dim] = previous.obs_rms.var
    # RMS shares one count across features. Cap historical weight on extension
    # so new sensors adapt within a few rollouts rather than millions of steps.
    norm.obs_rms.count = (previous.obs_rms.count if source_dim == target_dim
                         else min(float(previous.obs_rms.count), max_count))
    norm.clip_obs, norm.epsilon = previous.clip_obs, previous.epsilon
    return dict(shared_features=source_dim, new_features=target_dim-source_dim,
                source_count=float(previous.obs_rms.count), effective_count=float(norm.obs_rms.count),
                reward_statistics='reset for new task')


class Curriculum:
    """Advance only on binary course success across consecutive reviews."""
    def __init__(self, config, saved=None):
        self.config = config
        self.level = float(config.get('initial_level', 1.))
        self.success_streak = self.failure_streak = 0
        if saved:
            self.level = float(saved['level'])
            self.success_streak = int(saved.get('success_streak', 0))
            self.failure_streak = int(saved.get('failure_streak', 0))

    def state(self):
        return dict(level=self.level, success_streak=self.success_streak,
                    failure_streak=self.failure_streak, config=self.config)

    def observe(self, result, can_advance=True):
        old = self.level
        if not can_advance:
            return None
        success = float(result.get('success_rate', 0.))
        enough = len(result.get('episodes', [])) >= int(self.config.get('min_eval_episodes', 5))
        self.success_streak = self.success_streak + 1 if enough and success >= self.config.get('advance_threshold', .8) else 0
        self.failure_streak = self.failure_streak + 1 if enough and success < self.config.get('demote_threshold', .2) else 0
        required = int(self.config.get('consecutive_reviews', 2))
        if self.success_streak >= required:
            self.level = min(float(self.config.get('max_level', 5)), self.level + float(self.config.get('level_step', 1)))
        elif self.failure_streak >= required:
            self.level = max(float(self.config.get('min_level', 0)), self.level - float(self.config.get('level_step', 1)))
        if old != self.level:
            self.success_streak = self.failure_streak = 0
            return dict(previous_level=old, level=self.level, success_rate=success)
        return None


class RunState:
    def __init__(self, run, config):
        self.run, self.config = run, config
        self.started = time.monotonic()
        process = psutil.Process()
        self.data = dict(schema_version=2, run=str(run.resolve()), state='starting',
                         pid=os.getpid(), process_create_time=process.create_time(),
                         command=process.cmdline(), cwd=os.getcwd(), started_at=utc_now(),
                         stage=config['stage'], env_id=config['env_id'], requested_steps=config['total_timesteps'])
        self.write()

    def write(self, state=None, model=None, **updates):
        if state:
            self.data['state'] = state
        self.data.update(updates)
        if model is not None:
            steps = int(model.num_timesteps)
            start = self.data.get('start_timesteps', 0)
            target = self.data.get('target_timesteps', start+self.config['total_timesteps'])
            self.data.update(timesteps=steps, new_steps=steps-start,
                             progress=min(1., (steps-start)/max(1, target-start)),
                             steps_per_second=(steps-start)/max(.001, time.monotonic()-self.started))
        self.data.update(updated_at=utc_now(), elapsed_seconds=time.monotonic()-self.started)
        atomic_json(self.run/'status.json', self.data)
        atomic_json(self.run/'run_state.json', self.data)


class ReviewCallback(BaseCallback):
    def __init__(self, config, run, state, curriculum, contract, source_hashes, stop_request, benchmark=False):
        super().__init__()
        self.config, self.run, self.state = config, run, state
        self.curriculum, self.contract = curriculum, contract
        self.source_hashes, self.stop_request, self.benchmark = source_hashes, stop_request, benchmark
        self.rows, self.best_by_level = [], {}
        self.last_update, self.last_heartbeat = -1, 0.
        self.initial_steps = state.data['start_timesteps']
        self.latest_checkpoint = None

    def _on_step(self):
        if self.stop_request:
            self.state.write('stopping', self.model, stop_reason=self.stop_request['reason'])
            return False
        if time.monotonic()-self.last_heartbeat >= 5:
            self.state.write('training', self.model, curriculum_level=self.curriculum.level)
            self.last_heartbeat = time.monotonic()
        return True

    def save_pair(self, reason, evaluation=None):
        """Publish an immutable pair; metadata is written before the directory appears."""
        base = self.run/'checkpoints'
        base.mkdir(exist_ok=True)
        name = f'step-{self.model.num_timesteps:012d}-{reason}-{uuid.uuid4().hex[:6]}'
        temp = base/('.'+name)
        temp.mkdir()
        self.model.save(temp/'policy.zip')
        self.training_env.save(temp/'vecnormalize.pkl')
        contract = copy.deepcopy(self.contract)
        if self.config['env_id'] == 'ParkourHumanoid':
            contract['difficulty'] = self.curriculum.level
        metadata = dict(schema_version=2, env_id=self.config['env_id'], timesteps=self.model.num_timesteps,
                        created_at=utc_now(), environment=contract, curriculum_state=self.curriculum.state(),
                        effective_ppo=effective_ppo(self.model), source_sha256=self.source_hashes,
                        sha256={n: digest(temp/n) for n in ('policy.zip', 'vecnormalize.pkl')})
        if evaluation is not None:
            metadata['evaluation'] = evaluation
            atomic_json(temp/'selection.json', evaluation)
        atomic_json(temp/'metadata.json', metadata)
        final = base/name
        os.replace(temp, final)
        self.latest_checkpoint = str(final.resolve())
        return final

    def alias(self, name, target):
        temp = self.run/('.'+name+'.'+uuid.uuid4().hex)
        temp.symlink_to(os.path.relpath(target, self.run), target_is_directory=True)
        os.replace(temp, self.run/name)

    def review(self, update):
        cfg, level = self.config, self.curriculum.level
        self.state.write('evaluating', self.model, curriculum_level=level)
        video = self.run/'videos'/f'update-{update:05d}.mp4' if cfg.get('record_video', True) else None
        seed_start = int(cfg.get('eval_seed', 10000))
        if cfg.get('rotate_eval_seeds', False):
            seed_start += len(self.rows)*cfg['eval_episodes']
        result = evaluate(self.model, self.training_env, cfg['env_id'],
                          range(seed_start, seed_start+cfg['eval_episodes']), video,
                          difficulty=level if cfg['env_id'] == 'ParkourHumanoid' else None,
                          env_kwargs=cfg.get('env_kwargs', {}),
                          max_episode_steps=cfg.get('eval_max_episode_steps'))
        result.update(update=update, timesteps=self.model.num_timesteps,
                      new_steps=self.model.num_timesteps-self.initial_steps,
                      curriculum_level=level, evidence_type='training_selection',
                      video=str(video.relative_to(self.run)) if video else None)
        self.rows.append(result)
        transition = self.curriculum.observe(result, can_advance=cfg['stage'] == 4 and result['new_steps'] > 0)
        if transition:
            self.training_env.env_method('set_difficulty', self.curriculum.level)
            result['curriculum_transition'] = transition
        atomic_json(self.run/'evaluations.json', self.rows)
        checkpoint = self.save_pair('review', result)
        level_key = str(level) if cfg['env_id'] == 'ParkourHumanoid' else 'flat'
        # Rewards across different terrain levels are not comparable. Select by
        # true success within each level, with progress/reward only as tie-breakers.
        score = ((result.get('success_rate', 0.), result.get('progress_fraction', 0.), result['mean_reward'])
                 if cfg['env_id'] == 'ParkourHumanoid' else (result.get('survival_fraction', 0.), result['mean_reward']))
        if score > self.best_by_level.get(level_key, (-float('inf'),)):
            self.best_by_level[level_key] = score
            self.alias('best', checkpoint)
            if cfg['env_id'] == 'ParkourHumanoid':
                self.alias(f'best-level-{level:g}', checkpoint)
        self.alias('last', checkpoint)
        self.logger.record('eval/raw_reward', result['mean_reward'])
        self.logger.record('eval/mean_length', np.mean([e['length'] for e in result['episodes']]))
        for key in ('speed', 'distance', 'survival_fraction', 'upright', 'facing', 'clearance_rate', 'success_rate', 'progress_fraction'):
            if key in result:
                self.logger.record('eval/'+key, result[key])
        for key, value in result.get('mean_reward_terms', {}).items():
            self.logger.record('eval_terms/'+key, value)
        self.logger.record('curriculum/level', self.curriculum.level)
        self.logger.dump(self.model.num_timesteps)
        self.state.write('training', self.model, latest_evaluation=result,
                         curriculum_level=self.curriculum.level, checkpoint=self.latest_checkpoint)
        print(f"REVIEW update={update} steps={self.model.num_timesteps} reward={result['mean_reward']:.1f} "
              f"success={result.get('success_rate', 0.):.2f} level={level:g}", flush=True)
        self.last_update = update

    def _on_rollout_start(self):
        update = (self.model.num_timesteps-self.initial_steps)//(self.model.n_steps*self.training_env.num_envs)
        if not self.benchmark and not self.stop_request and update % self.config['eval_every_updates'] == 0 and update != self.last_update:
            self.review(update)

    def _on_training_end(self):
        update = (self.model.num_timesteps-self.initial_steps)//(self.model.n_steps*self.training_env.num_envs)
        if not self.stop_request and not self.benchmark and update != self.last_update:
            self.review(update)
        checkpoint = self.save_pair('stopped' if self.stop_request else 'final')
        self.alias('last', checkpoint)
        self.logger.dump(self.model.num_timesteps)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/01-warmup.yaml')
    p.add_argument('--run', default='runs/stage1')
    p.add_argument('--steps', type=int, help='NEW decisions, including on resume (rounds to a rollout)')
    p.add_argument('--n-envs', type=int)
    p.add_argument('--benchmark', action='store_true', help='Pipeline speed test; no evaluation or learned-skill claim')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--resume', type=Path, help='Continue a paired checkpoint into a NEW run')
    source.add_argument('--warm-start', type=Path, help='Transfer compatible weights/observation statistics to a new task')
    p.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--apply-config-on-resume', action='store_true', help='Apply configured PPO settings, otherwise retain saved settings')
    a = p.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if cfg['stage'] not in (1, 2, 3, 4):
        raise ValueError('Only Stages 1 through 4 are implemented.')
    for argument, key in ((a.steps, 'total_timesteps'), (a.n_envs, 'n_envs')):
        if argument is not None:
            cfg[key] = argument
    for key in ('total_timesteps', 'n_envs', 'n_steps', 'eval_every_updates', 'eval_episodes'):
        if int(cfg[key]) <= 0:
            raise ValueError(f'{key} must be positive')
    if a.apply_config_on_resume and not a.resume:
        raise ValueError('--apply-config-on-resume requires --resume')
    run = Path(a.run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run/'videos').mkdir()
    cfg['device'] = a.device
    (run/'config.yaml').write_text(yaml.safe_dump(cfg))
    state = RunState(run, cfg)
    stop_request = {}
    def request_stop(signum, frame):
        stop_request['reason'] = signal.Signals(signum).name
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    vec = norm = model = callback = None
    try:
        source_hashes = {}
        paths = [ROOT/'train.py', ROOT/'eval.py', ROOT/'requirements.lock'] + sorted((ROOT/'envs').glob('*.py'))
        for path in paths:
            destination = run/'source'/path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            source_hashes[str(path.relative_to(ROOT))] = digest(destination)
        atomic_json(run/'source-manifest.json', source_hashes)
        parent = metadata = None
        if a.resume or a.warm_start:
            parent, metadata = checkpoint_metadata(a.resume or a.warm_start)
        curriculum = Curriculum(cfg.get('curriculum', {}), metadata.get('curriculum_state') if a.resume else None)
        torch.set_num_threads(1)
        vec_cls = SubprocVecEnv if cfg['n_envs'] > 1 else DummyVecEnv
        options = {'start_method': 'spawn'} if cfg['n_envs'] > 1 else {}
        if cfg['env_id'] == 'ParkourHumanoid':
            env_kwargs = dict(cfg.get('env_kwargs', {}))
            if set(env_kwargs) & {'difficulty', 'replay_prob'}:
                raise ValueError('Set difficulty and replay_prob in curriculum, not env_kwargs.')
            factory = make_parkour_env(difficulty=curriculum.level,
                         replay_prob=cfg.get('curriculum', {}).get('replay_prob', 0.), **env_kwargs)
        else:
            factory = make_stock_env(cfg['env_id'], cfg.get('env_kwargs', {}))
        vec = vec_cls([factory for _ in range(cfg['n_envs'])], **options)
        vec.seed(cfg['seed'])
        contract = environment_contract(cfg, vec, curriculum.level)
        if a.resume:
            validate_resume(metadata, contract)
            norm = VecNormalize.load(parent/'vecnormalize.pkl', vec)
            norm.training, norm.norm_reward = True, True
            overrides = {k: cfg[k] for k in PPO_KEYS if k in cfg} if a.apply_config_on_resume else {}
            model = PPO.load(parent/'policy.zip', env=norm, device=a.device,
                             tensorboard_log=str(run/'tensorboard'), **overrides)
            if norm.gamma != model.gamma:
                norm.gamma = model.gamma
                norm.ret_rms = RunningMeanStd(shape=())
            norm.returns = np.zeros(vec.num_envs)
            atomic_json(run/'resume.json', dict(parent=str(parent), parent_steps=model.num_timesteps,
                        applied_config=a.apply_config_on_resume, policy_sha256=digest(parent/'policy.zip'),
                        requested_new_steps=cfg['total_timesteps'], restored_curriculum=curriculum.state()))
        else:
            norm = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10., gamma=cfg['gamma'])
            arch = cfg.get('net_arch', [64, 64])
            policy_kwargs = dict(net_arch=dict(pi=arch, vf=arch),
                                 activation_fn=torch.nn.ReLU if cfg.get('activation') == 'relu' else torch.nn.Tanh,
                                 ortho_init=cfg.get('ortho_init', True), log_std_init=cfg.get('log_std_init', 0.))
            model = PPO('MlpPolicy', norm, device=a.device, seed=cfg['seed'], tensorboard_log=str(run/'tensorboard'),
                        verbose=0, policy_kwargs=policy_kwargs, **{k: cfg[k] for k in PPO_KEYS if k in cfg})
            if a.warm_start:
                previous = PPO.load(parent/'policy.zip', device=a.device)
                old_dim, new_dim = previous.observation_space.shape[0], model.observation_space.shape[0]
                shared = (metadata['env_id'] == cfg['env_id'] and old_dim == new_dim) or (
                    metadata['env_id'] == 'Humanoid-v5' and cfg['env_id'] == 'ParkourHumanoid' and old_dim == 348 and new_dim == 366)
                if not shared:
                    raise ValueError('Warm-start observation semantics are not a supported shared layout.')
                transfer_policy(previous, model)
                transfer = transfer_normalizer(parent/'vecnormalize.pkl', norm, old_dim, new_dim,
                                               cfg.get('warm_start_normalizer_count_cap', 10000))
                atomic_json(run/'warm_start.json', dict(source=str(parent), source_steps=previous.num_timesteps,
                            sha256=digest(parent/'policy.zip'), normalizer_transfer=transfer,
                            optimizer='fresh', environment_changed=metadata.get('environment') != contract))
        effective = effective_ppo(model)
        atomic_json(run/'effective_ppo.json', effective)
        atomic_json(run/'environment.json', contract)
        initial_steps = int(model.num_timesteps)
        rollout_size = model.n_steps*norm.num_envs
        rounded_steps = int(np.ceil(cfg['total_timesteps']/rollout_size))*rollout_size
        state.write('training', model, start_timesteps=initial_steps,
                    target_timesteps=initial_steps+rounded_steps, rollout_size=rollout_size,
                    budget_rounding_steps=rounded_steps-cfg['total_timesteps'], effective_ppo=effective,
                    environment=contract, evidence_type='pipeline_benchmark' if a.benchmark else 'training')
        callback = ReviewCallback(cfg, run, state, curriculum, contract, source_hashes, stop_request, a.benchmark)
        # SB3 adds num_timesteps internally on resume. Pass NEW steps exactly once.
        model.learn(cfg['total_timesteps'], callback=callback, reset_num_timesteps=not bool(a.resume))
        elapsed = time.monotonic()-state.started
        result = dict(steps=model.num_timesteps, new_steps=model.num_timesteps-initial_steps,
                      requested_new_steps=cfg['total_timesteps'], start_timesteps=initial_steps,
                      seconds=elapsed, steps_per_second=(model.num_timesteps-initial_steps)/elapsed,
                      device=str(model.device), mps_available=torch.backends.mps.is_available(),
                      cuda_available=torch.cuda.is_available(), outcome='stopped' if stop_request else 'completed',
                      evidence_type='pipeline_benchmark' if a.benchmark else 'training')
        atomic_json(run/'performance.json', result)
        state.write(result['outcome'], model, finished_at=utc_now(), checkpoint=callback.latest_checkpoint,
                    stop_reason=stop_request.get('reason'))
        print(json.dumps(result), flush=True)
    except BaseException as error:
        recovery = None
        if callback is not None and model is not None:
            try:
                recovery = callback.save_pair('failed')
                callback.alias('last', recovery)
            except Exception as save_error:
                state.data['checkpoint_error'] = str(save_error)
        state.write('failed', model, error=f'{type(error).__name__}: {error}', finished_at=utc_now(),
                    checkpoint=str(recovery) if recovery else state.data.get('checkpoint'))
        raise
    finally:
        if norm is not None:
            norm.close()
        elif vec is not None:
            vec.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
