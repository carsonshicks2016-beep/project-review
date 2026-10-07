"""Slippi tournament replay parser and behavioral cloning imitation learner.

Parses professional .slp replays, translates human inputs into the factored
controller space, and trains policy weights offline via supervised log-likelihood.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import melee

from .config import ROOT, Config
from .controller import nearest, space, DIMENSIONS, NAMES, contract
from .state import History, encode, reward, compress, expand, CONTINUOUS_PER_SAMPLE, SCHEMA, OBS_SIZE
from .storage import save_model, write_json
from .worker import new_model, new_recurrent_model, GAMMA


# X and Y are the same jump on a GameCube pad, but the controller space only carries Y.
# Measured over 96,784 tournament player-frames, X is pressed on 2.72% of frames and is
# 30.4% of every jump input: dropping it labelled a third of demonstrated jumps as "no
# button pressed", so cloning taught the policy not to jump exactly where a human did.
# Folding X onto Y recovers those labels without adding an axis, which would invalidate
# every existing checkpoint.
BUTTON_ALIASES = {'BUTTON_X': 'BUTTON_Y'}


def extract_action_vector(ctrl):
    """Extract a 7-axis factored controller vector from a melee.PlayerState.controller_state."""
    stick = (float(ctrl.main_stick[0]), float(ctrl.main_stick[1]))
    cstick = (float(ctrl.c_stick[0]), float(ctrl.c_stick[1]))
    # A digital L/R click reads analog 1.0, so the shoulders carry every shield and
    # air-dodge input already; only the jump buttons need aliasing.
    trigger = float(max(getattr(ctrl, 'l_shoulder', 0.0), getattr(ctrl, 'r_shoulder', 0.0)))
    buttons = []
    for b, on in ctrl.button.items():
        if not on: continue
        name = f'BUTTON_{b.name.replace("BUTTON_", "")}'
        buttons.append(BUTTON_ALIASES.get(name, name))
    return nearest(stick=stick, cstick=cstick, buttons=buttons, trigger=trigger)


EMPTY = (np.empty((0, OBS_SIZE), dtype=np.float32),
         np.empty((0, len(DIMENSIONS)), dtype=np.int64),
         np.empty(0, dtype=np.float32))


def discounted(rewards, discount):
    """Returns-to-go. The final transition is not bootstrapped: a replay that ends in
    a KO is genuinely terminal, and one cut short by a quit simply under-counts its
    last few seconds."""
    out = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + discount * running
        out[i] = running
    return out


def parse_replay(slp_path, character='FOX', stage='FINAL_DESTINATION', stride=1, min_frame=120):
    """Parse one .slp replay file into (observations, actions, returns) arrays.

    Extracts samples from the perspective of any player matching `character`.
    If both players match (e.g. Fox mirror), extracts from both perspectives.

    `returns` scores each demonstrated decision with the same reward PPO optimises,
    so the value head can be fitted on the footage instead of starting random.
    """
    slp_path = Path(slp_path)
    if not slp_path.is_file():
        return EMPTY

    try:
        console = melee.Console(path=str(slp_path), is_dolphin=False, allow_old_version=True)
        console.connect()
    except Exception as exc:
        sys.stderr.write(f"Failed to open {slp_path.name}: {exc}\n")
        return EMPTY

    target_char = getattr(melee.Character, character, None)
    target_stage = getattr(melee.Stage, stage, None) if stage else None

    # Step until we determine players and stage
    gamestate = None
    while gamestate is None or not gamestate.players:
        gamestate = console.step()
        if gamestate is None:
            return EMPTY

    if target_stage is not None and gamestate.stage != target_stage:
        return EMPTY

    active_ports = [p for p in (1, 2, 3, 4) if p in gamestate.players]
    if len(active_ports) != 2:
        return EMPTY

    p1_port, p2_port = active_ports[0], active_ports[1]
    pairings = []
    if target_char is None or gamestate.players[p1_port].character == target_char:
        pairings.append((p1_port, p2_port))
    if target_char is None or gamestate.players[p2_port].character == target_char:
        pairings.append((p2_port, p1_port))

    if not pairings:
        return EMPTY

    histories = {pair: History(pair[0], pair[1]) for pair in pairings}
    samples = {pair: ([], [], []) for pair in pairings}
    previous = {pair: None for pair in pairings}

    frame_count = 0
    while True:
        if gamestate is None:
            break
        if gamestate.menu_state == melee.Menu.IN_GAME and gamestate.frame >= min_frame:
            frame_count += 1
            # The history is pushed on EVERY frame and only sampled every `stride`.
            # Pushing on sampled frames alone would stack two frames `stride` apart --
            # an observation the live environment, which steps one frame at a time,
            # never produces.
            keep = frame_count % stride == 0
            for pair in pairings:
                agent_port, opp_port = pair
                if agent_port in gamestate.players and opp_port in gamestate.players:
                    obs = histories[pair].push(gamestate)
                    if not keep: continue
                    act_vec = extract_action_vector(gamestate.players[agent_port].controller_state)
                    obs_list, act_list, reward_list = samples[pair]
                    # The reward earned by the PREVIOUS sample's action, so rewards
                    # trail observations by one and the last sample is dropped below.
                    # Stock and damage deltas telescope, so measuring straight across a
                    # strided window is the same as summing the frames inside it.
                    if previous[pair] is not None:
                        reward_list.append(reward(previous[pair], gamestate, agent_port, opp_port))
                    obs_list.append(obs)
                    act_list.append(act_vec)
                    previous[pair] = gamestate
        gamestate = console.step()

    all_obs = []
    all_acts = []
    all_returns = []
    discount = GAMMA ** max(1, int(stride))
    for pair in pairings:
        o_list, a_list, r_list = samples[pair]
        if not r_list:
            continue
        # Each perspective of a replay is its own episode, so returns never run across
        # the boundary between them.
        all_obs.append(np.asarray(o_list[:len(r_list)], dtype=np.float32))
        all_acts.append(np.asarray(a_list[:len(r_list)], dtype=np.int64))
        all_returns.append(discounted(r_list, discount))

    if not all_obs:
        return EMPTY

    return (np.concatenate(all_obs, axis=0), np.concatenate(all_acts, axis=0),
            np.concatenate(all_returns, axis=0))


def build_dataset(slp_paths, output_path=None, character='FOX', stage='FINAL_DESTINATION',
                  stride=1, max_samples=None, seen=None, quiet=False):
    """Parse replays into one packed demonstration shard.

    Also records which replay each sample came from, so validation can hold out whole
    games. Splitting 60-per-second frames at random puts a frame's near-identical
    neighbours on both sides of the split and reports an accuracy nobody can reproduce
    on an unseen match.

    `seen` is a caller-owned {sha256: name} map, so a streamed ingest can carry
    duplicate detection across shards instead of only within one."""
    shards = []
    total_samples = 0

    paths = []
    for path in slp_paths:
        path = Path(path)
        if path.is_dir():
            paths.extend(sorted(path.rglob('*.slp')))
        elif path.is_file() and path.suffix == '.slp':
            paths.append(path)

    if not quiet: print(f"Discovered {len(paths)} .slp replay files.")

    if seen is None: seen = {}
    group = 0
    for i, path in enumerate(paths):
        # The same game saved under two names would otherwise land in both the training
        # set and the holdout.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            if not quiet: print(f"[{i+1}/{len(paths)}] {path.name}: skipped (identical to {seen[digest]})")
            continue
        seen[digest] = path.name

        obs, acts, returns = parse_replay(path, character=character, stage=stage, stride=stride)
        if len(obs) > 0:
            shards.append(Demonstrations.from_dense(
                obs, acts, returns, np.full(len(obs), group, dtype=np.int32)))
            group += 1
            total_samples += len(obs)
            if not quiet: print(f"[{i+1}/{len(paths)}] {path.name}: +{len(obs)} samples (total: {total_samples})")
        elif not quiet:
            print(f"[{i+1}/{len(paths)}] {path.name}: skipped (mismatched stage/character or empty)")

        if max_samples and total_samples >= max_samples:
            if not quiet: print(f"Reached sample limit ({max_samples}).")
            break

    if not shards:
        raise ValueError("No valid demonstration frames were parsed from the provided replays.")

    dataset = concatenate(shards)
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.subset(slice(0, max_samples))

    if not quiet:
        print(f"\nDataset compiled: {len(dataset)} samples from {group} replays, "
              f"{dataset.nbytes/1e6:.0f} MB packed.")
        print(f"Returns: mean {dataset.returns.mean():.2f}, std {dataset.returns.std():.2f}, "
              f"range [{dataset.returns.min():.1f}, {dataset.returns.max():.1f}].")

    if output_path:
        out = dataset.save(output_path)
        if not quiet:
            print(f"Saved compressed dataset to: {out.resolve()} ({out.stat().st_size / 1024 / 1024:.2f} MB)")

    return dataset



class Demonstrations:
    """A demonstration dataset held in packed form, expanded a minibatch at a time.

    Fifty million frames of tournament footage is 367 GB dense and 21 GB packed, so
    everything downstream -- the split, the training loop, the PPO anchor -- indexes
    this rather than a dense array, and only the rows in the current minibatch are
    ever expanded. Datasets written before packing existed still load: their dense
    observations are packed on the way in."""

    KEYS = ('continuous', 'globals', 'ids', 'actions', 'returns', 'groups')

    def __init__(self, continuous, globals_, ids, actions, returns=None, groups=None):
        self.continuous, self.globals, self.ids = continuous, globals_, ids
        self.actions, self.returns, self.groups = actions, returns, groups
        self._buffer = None

    @classmethod
    def from_dense(cls, observations, actions, returns=None, groups=None):
        observations = np.asarray(observations)
        if observations.ndim != 2 or observations.shape[1] != OBS_SIZE:
            raise ValueError(f'Demonstrations encode {observations.shape[-1]} features; '
                             f'this build observes {OBS_SIZE}.')
        return cls(*compress(observations), actions, returns, groups)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        if 'observations' in data:      # pre-packing dataset
            return cls.from_dense(data['observations'], data['actions'],
                                  data['returns'] if 'returns' in data else None,
                                  data['groups'] if 'groups' in data else None)
        if data['continuous'].shape[1] != CONTINUOUS_PER_SAMPLE:
            raise ValueError(f"Demonstrations pack {data['continuous'].shape[1]} continuous "
                             f'features per sample; this build packs {CONTINUOUS_PER_SAMPLE}.')
        return cls(data['continuous'], data['globals'], data['ids'], data['actions'],
                   data['returns'] if 'returns' in data else None,
                   data['groups'] if 'groups' in data else None)

    def __len__(self):
        return len(self.ids)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.continuous, self.globals, self.ids, self.actions)
                   if a is not None)

    def observations(self, index=None, reuse=False):
        """Dense observations for `index`. With reuse=True the same buffer is returned
        each call, which is safe for one minibatch at a time and not for anything held."""
        c, g, i = (self.continuous, self.globals, self.ids) if index is None else \
                  (self.continuous[index], self.globals[index], self.ids[index])
        if not reuse:
            return expand(c, g, i)
        self._buffer = expand(c, g, i, self._buffer)
        return self._buffer

    def subset(self, index):
        return Demonstrations(self.continuous[index], self.globals[index], self.ids[index],
                              self.actions[index],
                              None if self.returns is None else self.returns[index],
                              None if self.groups is None else self.groups[index])

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(continuous=self.continuous, globals=self.globals, ids=self.ids,
                      actions=self.actions)
        if self.returns is not None: arrays['returns'] = self.returns
        if self.groups is not None: arrays['groups'] = self.groups
        np.savez_compressed(path, **arrays)
        return path


def concatenate(parts):
    """Join shards into one dataset, renumbering groups so held-out replays stay whole.

    Preallocated rather than np.concatenate-ed: a list of shards plus the concatenation
    of that list is two copies of the whole dataset resident at once."""
    parts = [p for p in parts if len(p)]
    if not parts: raise ValueError('No demonstration shards to join.')
    total = sum(len(p) for p in parts)
    has_returns = all(p.returns is not None for p in parts)
    has_groups = all(p.groups is not None for p in parts)
    out = Demonstrations(
        np.empty((total, parts[0].continuous.shape[1]), np.float32),
        np.empty((total, parts[0].globals.shape[1]), np.float32),
        np.empty((total, parts[0].ids.shape[1]), np.int16),
        np.empty((total, parts[0].actions.shape[1]), np.int64),
        np.empty(total, np.float32) if has_returns else None,
        np.empty(total, np.int32) if has_groups else None)
    at = group = 0
    for part in parts:
        n = len(part)
        out.continuous[at:at+n] = part.continuous
        out.globals[at:at+n] = part.globals
        out.ids[at:at+n] = part.ids
        out.actions[at:at+n] = part.actions
        if has_returns: out.returns[at:at+n] = part.returns
        if has_groups:
            out.groups[at:at+n] = part.groups + group
            group += int(part.groups.max()) + 1
        at += n
    return out


class DummyImitationEnv(gym.Env):
    """Lightweight dummy env to construct a PPO model with the correct action/observation spaces."""
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(-5, 5, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = space()
        self.config = Config(action_set='controller', action_frames=1, character='FOX', stage='FINAL_DESTINATION', randomize_opponent=True)
    def reset(self, **kwargs):
        return np.zeros(OBS_SIZE, dtype=np.float32), {}
    def step(self, action):
        return np.zeros(OBS_SIZE, dtype=np.float32), 0.0, False, False, {}


MAX_HELD_OUT_REPLAYS = 150

# net_arch keeps pi and vf as separate towers, so a policy's two branches share no
# weights at all and can be stopped at different epochs. They want different ones:
# the actor is still improving long after the critic has started memorising returns.
VALUE_BRANCH = ('mlp_extractor.value_net.', 'value_net.')


def branch(state, value):
    return {k: v.detach().clone() for k, v in state.items()
            if k.startswith(VALUE_BRANCH) is value}


def split(n_samples, groups, val_split, seed):
    """Hold out whole replays when we know which replay each frame came from.

    The holdout is also capped: a fixed fraction of five thousand replays would retire
    hundreds of games to measure something a hundred already measure tightly."""
    rng = np.random.default_rng(seed)
    if groups is None or len(np.unique(groups)) < 2:
        order = rng.permutation(n_samples)
        cut = int(n_samples * val_split)
        return order[cut:], (order[:cut] if cut else order)
    unique = np.unique(groups)
    count = min(max(1, round(len(unique) * val_split)), MAX_HELD_OUT_REPLAYS)
    held = rng.permutation(unique)[:count]
    mask = np.isin(groups, held)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def batches(n, size, generator=None):
    """Index batches over n samples, shuffled when a generator is given."""
    order = generator.permutation(n) if generator is not None else np.arange(n)
    for at in range(0, n, size):
        yield order[at:at+size]


def sequence_batches(data, batch_size=256, seq_len=32, generator=None):
    """Index contiguous intra-replay sequence slices of length seq_len."""
    T = seq_len
    B = max(1, batch_size // T)
    groups = data.groups if data.groups is not None else np.zeros(len(data), dtype=np.int32)
    n = len(data) - T
    if n <= 0:
        yield np.arange(len(data))
        return
    stride = T
    valid = np.flatnonzero(groups[:n:stride] == groups[T-1:n+T-1:stride]) * stride
    if len(valid) == 0:
        valid = np.arange(0, n, stride)
    order = generator.permutation(valid) if generator is not None else valid
    for at in range(0, len(order), B):
        chunk = order[at:at+B]
        if len(chunk) == 0: continue
        yield (chunk[:, None] + np.arange(T)).reshape(-1)


def evaluate_batched(model, data, vf_coef, size=4096):
    """Held-out metrics without ever expanding the whole validation set at once."""
    import torch
    total_nll = axis_hits = exact_hits = 0.
    value_error = 0.
    predictions = [] if data.returns is not None else None
    targets = [] if data.returns is not None else None
    axis_hits = np.zeros(data.actions.shape[1])
    is_recurrent = hasattr(model.policy, 'lstm_actor') or 'Recurrent' in type(model.policy).__name__
    total_samples = 0
    with torch.no_grad():
        if not is_recurrent:
            for index in batches(len(data), size):
                x = torch.as_tensor(data.observations(index, reuse=True))
                y = torch.as_tensor(data.actions[index], dtype=torch.long)
                distribution = model.policy.get_distribution(x)
                total_nll += float(-distribution.log_prob(y).sum())
                match = distribution.mode() == y
                exact_hits += float(match.all(dim=-1).sum())
                axis_hits += match.float().sum(dim=0).cpu().numpy()
                total_samples += len(index)
                if predictions is not None:
                    predictions.append(model.policy.predict_values(x).reshape(-1).cpu().numpy())
                    targets.append(data.returns[index])
        else:
            T = 32
            n_layers = getattr(model.policy, 'lstm_actor', None).num_layers if hasattr(model.policy, 'lstm_actor') else 1
            hidden_size = getattr(model.policy, 'lstm_output_dim', 128)
            for index in sequence_batches(data, batch_size=size, seq_len=T):
                B = len(index) // T
                x = torch.as_tensor(data.observations(index, reuse=True))
                y = torch.as_tensor(data.actions[index], dtype=torch.long)
                episode_starts = torch.zeros(B * T, dtype=torch.float32)
                episode_starts[0::T] = 1.0
                lstm_states = (torch.zeros(n_layers, B, hidden_size), torch.zeros(n_layers, B, hidden_size))
                distribution, _ = model.policy.get_distribution(x, lstm_states, episode_starts)
                total_nll += float(-distribution.log_prob(y).sum())
                match = distribution.mode() == y
                exact_hits += float(match.all(dim=-1).sum())
                axis_hits += match.float().sum(dim=0).cpu().numpy()
                total_samples += len(index)
                if predictions is not None:
                    v = model.policy.predict_values(x, lstm_states, episode_starts)
                    predictions.append(v.reshape(-1).cpu().numpy())
                    targets.append(data.returns[index])
    n = max(1, total_samples)
    report = dict(val_loss=total_nll/n, val_accuracy=exact_hits/n,
                  axis_accuracy=(axis_hits/n).tolist(), value_explained_variance=None, value_mse=None)
    report['val_held_out_objective'] = report['val_loss']
    if predictions is not None and len(predictions) > 0:
        predicted = np.concatenate(predictions)
        target_vals = np.concatenate(targets)
        variance = float(target_vals.var())
        report['value_explained_variance'] = float(1 - ((target_vals-predicted).var()/variance)) if variance > 0 else 0.
        report['value_mse'] = float(((target_vals-predicted)**2).mean())
        report['val_held_out_objective'] += vf_coef * report['value_mse']
    return report


def train_imitation(dataset, output_dir, epochs=15, batch_size=256, lr=1e-3, val_split=0.15, seed=7, config=None, vf_coef=.5, progress_seconds=5, recurrent=False):
    """Clone the demonstrated policy and fit the value head to the demonstrated returns.

    Both heads matter. Cloning alone hands PPO a sharp policy behind a randomly
    initialised critic, and the first advantages it computes are noise -- which is
    exactly the update that destroys the clone. The value branch is separate from the
    policy branch, so the two losses share a step without competing for weights.

    Held-out replays decide when to stop. With a few hundred games the fit turns over
    from learning to memorising well before the last epoch, so the epoch with the best
    held-out score is the one kept."""
    owned = isinstance(dataset, (str, Path))
    if owned:
        print(f"Loading dataset from {dataset}...")
        data = Demonstrations.load(dataset)
    elif isinstance(dataset, Demonstrations):
        data = dataset
    else:
        observations, actions, *rest = dataset
        data = Demonstrations.from_dense(observations, actions, *rest)

    n_samples = len(data)
    if n_samples == 0:
        raise ValueError("Dataset is empty.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_idx, val_idx = split(n_samples, data.groups, val_split, seed)
    train, validation = data.subset(train_idx), data.subset(val_idx)
    held = len(np.unique(validation.groups)) if validation.groups is not None else 0
    replays = int(data.groups.max()) + 1 if data.groups is not None else 0
    has_returns = data.returns is not None
    packed_bytes = data.nbytes
    # The split copies every row, so holding the unsplit dataset as well doubles a
    # multi-gigabyte resident set for the rest of training.
    if owned: data = None
    print(f"Training set: {len(train):,} samples | Validation set: {len(validation):,} samples"
          + (f" ({held} held-out replays)." if held else " (random frames; no replay grouping).")
          + f" {packed_bytes/1e6:.0f} MB packed.", flush=True)
    if not has_returns:
        print("Dataset has no returns; the value head will start random. Re-parse the replays to fit it.")

    env = DummyImitationEnv()
    if config is not None:
        env.config = config
    model = new_recurrent_model(env, seed=seed) if recurrent else new_model(env, seed=seed)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    generator = np.random.default_rng(seed)
    history = []
    best_policy = (float('inf'), 0, None)
    best_value = (float('inf'), 0, None)
    start_time = time.time()

    print("\nStarting supervised behavioral cloning:", flush=True)
    total_batches = max(1, -(-len(train) // batch_size))
    for epoch in range(1, epochs + 1):
        model.policy.set_training_mode(True)
        train_loss = 0.0
        count = 0
        # Millions of frames make an epoch minutes long, so report inside it: a run
        # that prints once every few minutes is indistinguishable from a hung one.
        epoch_start = last_report = time.time()
        seen = 0

        batch_iter = sequence_batches(train, batch_size, seq_len=32, generator=generator) if recurrent else batches(len(train), batch_size, generator)
        for index in batch_iter:
            x = torch.as_tensor(train.observations(index, reuse=True))
            y = torch.as_tensor(train.actions[index], dtype=torch.long)
            if not recurrent:
                loss = -model.policy.get_distribution(x).log_prob(y).mean()
                if train.returns is not None:
                    loss = loss + vf_coef * torch.nn.functional.mse_loss(
                        model.policy.predict_values(x),
                        torch.as_tensor(train.returns[index]).reshape(-1, 1))
            else:
                T = 32
                B = len(index) // T
                episode_starts = torch.zeros(B * T, dtype=torch.float32)
                episode_starts[0::T] = 1.0
                n_layers = getattr(model.policy, 'lstm_actor', None).num_layers if hasattr(model.policy, 'lstm_actor') else 1
                hidden_size = getattr(model.policy, 'lstm_output_dim', 128)
                lstm_states = (torch.zeros(n_layers, B, hidden_size), torch.zeros(n_layers, B, hidden_size))
                dist, _ = model.policy.get_distribution(x, lstm_states, episode_starts)
                loss = -dist.log_prob(y).mean()
                if train.returns is not None:
                    v = model.policy.predict_values(x, lstm_states, episode_starts)
                    loss = loss + vf_coef * torch.nn.functional.mse_loss(
                        v, torch.as_tensor(train.returns[index]).reshape(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach())
            count += 1
            seen += len(index)
            now = time.time()
            if now - last_report >= progress_seconds:
                rate = seen / max(1e-6, now - epoch_start)
                remaining = (len(train) - seen) / max(1.0, rate)
                print(f"  epoch {epoch:2d}/{epochs} {100*count/total_batches:5.1f}% · "
                      f"{count:,}/{total_batches:,} batches · loss {train_loss/count:.4f} · "
                      f"{rate/1000:.0f}k samples/s · {remaining:.0f}s left in epoch", flush=True)
                last_report = now

        avg_train_loss = train_loss / max(1, count)

        model.policy.set_training_mode(False)
        print(f"  epoch {epoch:2d}/{epochs} scoring {len(validation):,} held-out frames ...", flush=True)
        scores = evaluate_batched(model, validation, vf_coef)

        state = model.policy.state_dict()
        marks = ''
        if scores['val_loss'] < best_policy[0]:
            best_policy = (scores['val_loss'], epoch, branch(state, False)); marks += ' *actor'
        if scores['value_mse'] is not None and scores['value_mse'] < best_value[0]:
            best_value = (scores['value_mse'], epoch, branch(state, True)); marks += ' *critic'

        elapsed = time.time() - start_time
        explained = scores['value_explained_variance']
        print(f"Epoch {epoch:2d}/{epochs} [{elapsed:.0f}s] - Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {scores['val_loss']:.4f} | Val Exact Acc: {scores['val_accuracy']*100:.2f}%"
              + (f" | Value R2: {explained:.3f}" if explained is not None else "") + marks, flush=True)
        history.append(dict(epoch=epoch, train_loss=avg_train_loss, **scores))

    restored = dict(model.policy.state_dict())
    for _, _, weights in (best_policy, best_value):
        if weights: restored.update(weights)
    model.policy.load_state_dict(restored)
    policy_epoch = best_policy[1] or epochs
    value_epoch = best_value[1] or policy_epoch
    print(f"\nKeeping the actor from epoch {policy_epoch} and the critic from epoch {value_epoch}.")
    kept = dict(history[policy_epoch - 1])
    kept['value_explained_variance'] = history[value_epoch - 1]['value_explained_variance']

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_model(
        model,
        out_dir / 'latest',
        env.config,
        imitation_epochs=policy_epoch,
        imitation_samples=n_samples,
        imitation_val_accuracy=kept['val_accuracy'],
        imitation_value_fitted=has_returns,
        imitation_held_out_replays=held,
        imitation_replays=replays,
        training_cpu_level=1
    )

    report = {
        'samples': n_samples,
        'replays': replays,
        'train_samples': len(train),
        'val_samples': len(validation),
        'held_out_replays': held,
        'value_head_fitted': has_returns,
        'epochs': epochs,
        'kept_policy_epoch': policy_epoch,
        'kept_value_epoch': value_epoch,
        'batch_size': batch_size,
        'learning_rate': lr,
        'final_val_loss': kept['val_loss'],
        'final_val_accuracy': kept['val_accuracy'],
        'final_value_explained_variance': kept['value_explained_variance'],
        'axis_names': list(NAMES),
        'final_axis_accuracy': kept['axis_accuracy'],
        'history': history,
        'duration_seconds': round(time.time() - start_time, 2),
        'checkpoint': checkpoint_path
    }
    write_json(out_dir / 'imitation.json', report)
    print(f"\nModel and report saved to {out_dir.resolve()}.")
    return model, report


def main():
    p = argparse.ArgumentParser(description="Slippi Tournament Replay Imitation Learning")
    sub = p.add_subparsers(dest='command', required=True)

    # Parse subcommand
    p_parse = sub.add_parser('parse', help='Extract dataset from .slp replays')
    p_parse.add_argument('--input', nargs='+', required=True, help='Paths or directories containing .slp files')
    p_parse.add_argument('--output', required=True, help='Output path for .npz dataset file')
    p_parse.add_argument('--character', default='FOX', help='Character to learn from (default: FOX)')
    p_parse.add_argument('--stage', default='FINAL_DESTINATION', help='Stage filter (default: FINAL_DESTINATION)')
    p_parse.add_argument('--stride', type=int, default=1, help='Frame sampling stride (default: 1)')
    p_parse.add_argument('--max-samples', type=int, default=None, help='Max frames to extract')

    # Train subcommand
    p_train = sub.add_parser('train', help='Train policy from an existing .npz dataset')
    p_train.add_argument('--dataset', required=True, help='Path to .npz dataset file')
    p_train.add_argument('--output', required=True, help='Output directory for checkpoint')
    p_train.add_argument('--epochs', type=int, default=15, help='Training epochs (default: 15)')
    p_train.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    p_train.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    p_train.add_argument('--recurrent', action='store_true', help='Train recurrent LSTM policy')

    # Run subcommand (Parse + Train in one go)
    p_run = sub.add_parser('run', help='Parse replays and train in one step')
    p_run.add_argument('--replays', nargs='+', required=True, help='Paths or directories containing .slp files')
    p_run.add_argument('--output', required=True, help='Output directory for checkpoint')
    p_run.add_argument('--character', default='FOX', help='Character to learn from (default: FOX)')
    p_run.add_argument('--stage', default='FINAL_DESTINATION', help='Stage filter (default: FINAL_DESTINATION)')
    p_run.add_argument('--epochs', type=int, default=15, help='Training epochs (default: 15)')
    p_run.add_argument('--recurrent', action='store_true', help='Train recurrent LSTM policy')

    args = p.parse_args()

    if args.command == 'parse':
        build_dataset(args.input, args.output, character=args.character, stage=args.stage, stride=args.stride, max_samples=args.max_samples)
    elif args.command == 'train':
        train_imitation(args.dataset, args.output, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, recurrent=getattr(args, 'recurrent', False))
    elif args.command == 'run':
        dataset = build_dataset(args.replays, character=args.character, stage=args.stage)
        train_imitation(dataset, args.output, epochs=args.epochs, recurrent=getattr(args, 'recurrent', False))


if __name__ == '__main__':
    main()
