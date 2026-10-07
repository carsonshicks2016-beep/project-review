import json
import os
from pathlib import Path
from dataclasses import asdict
from .state import SCHEMA,OBS_SIZE
from .actions import ACTIONS, vocabulary


def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False))
    os.replace(temp,path)


def action_contract(config):
    """What a checkpoint's controller must match. A named-move list for the discrete
    vocabularies; the pad's shape for the factored one."""
    if getattr(config,'action_set','legacy')=='controller':
        from .controller import contract
        return contract()
    return [a.name for a in vocabulary(config)]


def is_recurrent_checkpoint(path):
    path = Path(path)
    meta_path = path.with_suffix('.json')
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get('architecture') == 'recurrent' or meta.get('recurrent'):
                return True
        except Exception:
            pass
    zip_path = path.with_suffix('.zip') if path.suffix != '.zip' else path
    if zip_path.exists():
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                if 'data' in z.namelist():
                    data_bytes = z.read('data')
                    if b'Recurrent' in data_bytes or b'sb3_contrib' in data_bytes or b'lstm' in data_bytes:
                        return True
        except Exception:
            pass
    return False


def load_model(path, env=None, device='cpu'):
    if is_recurrent_checkpoint(path):
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path, env=env, device=device)
    from stable_baselines3 import PPO
    try:
        model = PPO.load(path, env=env, device=device)
        if hasattr(model.policy, 'lstm_actor') or type(model.policy).__name__ == 'RecurrentActorCriticPolicy':
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO.load(path, env=env, device=device)
        return model
    except Exception:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path, env=env, device=device)


def save_model(model,path,config,**extra):
    path=Path(path).with_suffix('.zip')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.tmp.zip')
    model.save(tmp); os.replace(tmp,path)
    is_rec = extra.get('architecture') == 'recurrent' or type(model).__name__ == 'RecurrentPPO'
    if is_rec and 'architecture' not in extra:
        extra['architecture'] = 'recurrent'
    write_json(path.with_suffix('.json'),dict(schema=SCHEMA,observation_size=OBS_SIZE,
        actions=action_contract(config),config=asdict(config),steps=int(model.num_timesteps),**extra))
    return str(path)


def validate_checkpoint(path, config):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('schema') != SCHEMA:
        raise ValueError(f"Checkpoint schema '{meta.get('schema')}' is incompatible with current '{SCHEMA}'.")
    if meta.get('observation_size') != OBS_SIZE:
        raise ValueError(f"Checkpoint observation size ({meta.get('observation_size')}) does not match current model ({OBS_SIZE}).")

    expected_actions = action_contract(config)
    meta_actions = meta.get('actions')
    is_controller_cfg = getattr(config, 'action_set', 'legacy') == 'controller'
    is_controller_meta = (meta_actions == 'controller') or (
        isinstance(meta_actions, dict) and meta_actions.get('space') == 'controller-v1'
    )

    if is_controller_cfg and is_controller_meta:
        pass
    elif meta_actions != expected_actions:
        if isinstance(meta_actions, list) and is_controller_cfg:
            raise ValueError(f"Checkpoint was trained with a discrete {len(meta_actions)}-action space, but current setup uses the 7-axis controller. Choose a controller policy or start a fresh agent.")
        elif not isinstance(meta_actions, list) and not is_controller_cfg:
            raise ValueError("Checkpoint was trained with the full controller, but current setup uses discrete actions.")
        else:
            raise ValueError('Checkpoint observation/action contract is incompatible.')

    check_keys = ['character', 'stage', 'action_frames', 'stocks']
    if not getattr(config, 'randomize_opponent', False):
        check_keys.append('opponent')
    for key in check_keys:
        checkpoint_val = meta.get('config', {}).get(key)
        config_val = getattr(config, key, None)
        if checkpoint_val != config_val:
            raise ValueError(f"Checkpoint {key.replace('_', ' ')} ('{checkpoint_val}') does not match current configuration ('{config_val}').")
    return meta
