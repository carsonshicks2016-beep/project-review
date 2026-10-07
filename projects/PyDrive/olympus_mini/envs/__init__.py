from olympus_mini.envs.base_athlete_env import BaseAthleteEnv
from olympus_mini.envs.sprint_env import SprintEnv
from olympus_mini.envs.hurdle_env import HurdleEnv
from olympus_mini.envs.vault_env import VaultEnv
from olympus_mini.envs.multi_task_env import MultiTaskOlympusEnv

def make_env(task="sprint", frame_skip=10, max_steps=600):
    if task == "sprint":
        return SprintEnv(frame_skip=frame_skip, max_steps=max_steps)
    elif task in ("hurdle", "hurdles"):
        return HurdleEnv(frame_skip=frame_skip, max_steps=max_steps)
    elif task in ("vault", "pole_vault"):
        return VaultEnv(frame_skip=frame_skip, max_steps=max_steps)
    elif task in ("multi", "all", "random"):
        return MultiTaskOlympusEnv(task_mode="random", frame_skip=frame_skip, max_steps=max_steps)
    else:
        raise ValueError(f"Unknown task: {task}. Choose from 'sprint', 'hurdle', 'vault', 'multi'.")
