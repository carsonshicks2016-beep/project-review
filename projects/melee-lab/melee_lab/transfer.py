"""Explicit controller-head transfer. Preserves learned features, resets optimizer state.

A transferred policy is a new, unevaluated policy. It does not inherit old scores.
"""
import torch
from stable_baselines3 import PPO
from .catalog import read_json
from .actions import vocabulary
from .config import Config
from pathlib import Path

def transfer_policy(path, env, seed, n_envs=1):
    from .worker import new_model
    source=PPO.load(path,device='cpu')
    meta=read_json(Path(path).with_suffix('.json'))
    old_config=Config(**meta['config'])
    new_config=env.get_attr('config')[0] if hasattr(env,'get_attr') else env.unwrapped.config
    old_type=getattr(old_config,'action_set','legacy')
    new_type=getattr(new_config,'action_set','legacy')
    # Discrete↔MultiDiscrete transfers have incompatible action head shapes.
    old_discrete=old_type in ('legacy','expanded')
    new_discrete=new_type in ('legacy','expanded')
    if old_discrete!=new_discrete:
        raise ValueError(f'Cannot transfer between {old_type} and {new_type} action spaces. '
                         f'Train a fresh checkpoint with the target action set instead.')
    # Everything below remaps the head by MOVE NAME. A factored controller head has no
    # moves in it -- its rows are axis values (97 stick angles, then the C-stick, the
    # four buttons, the trigger) -- so name-matching would leave every row past the
    # vocabulary at its random initialisation. That is the whole action head for every
    # button. A controller policy is already in the target space: resume it directly.
    if not new_discrete:
        raise ValueError('This policy already uses the full controller. Resume it directly '
                         'rather than transferring; there is no action head to reshape.')
    old_actions=vocabulary(old_config)
    new_actions=vocabulary(new_config)
    # Rebuild the source's feature extractor. This transfer only reshapes the
    # controller head; adopting a different hidden width would discard every
    # learned feature and could not be copied across anyway.
    target=new_model(env,seed,n_envs,net_arch=getattr(source.policy,'net_arch',None))
    state=target.policy.state_dict(); old=source.policy.state_dict()
    for name,weights in old.items():
        if name.startswith('action_net.'): continue
        if name not in state or state[name].shape!=weights.shape:
            raise ValueError('Policy architecture cannot be transferred to this controller.')
        state[name]=weights.clone()
    names={a.name:i for i,a in enumerate(old_actions)}
    for i,action in enumerate(new_actions):
        j=names.get(action.name)
        if j is None:
            # Seed raw actions from the closest known controller input.
            step=action.steps[0]
            j=min(range(len(old_actions)),key=lambda k:
                sum(abs(a-b) for a,b in zip(old_actions[k].steps[0].stick,step.stick))+
                2*len(set(old_actions[k].steps[0].buttons)^set(step.buttons))+
                sum(abs(a-b) for a,b in zip(old_actions[k].steps[0].cstick,step.cstick)))
        state['action_net.weight'][i]=old['action_net.weight'][j]
        state['action_net.bias'][i]=old['action_net.bias'][j]-(2 if action.name not in names else 0)
    target.policy.load_state_dict(state)
    target.num_timesteps=source.num_timesteps
    return target
