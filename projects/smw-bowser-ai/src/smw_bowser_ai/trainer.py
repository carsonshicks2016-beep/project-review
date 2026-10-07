from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 100_000
    seed: int = 0
    output_dir: Path = Path("artifacts/training")


def train_ppo(config: TrainingConfig) -> None:
    try:
        import gymnasium  # noqa: F401  # type: ignore[import-not-found]
        import stable_baselines3  # noqa: F401  # type: ignore[import-not-found]
    except Exception as error:
        raise RuntimeError(
            "PPO training dependencies are not installed. Run: python3 -m pip install -e '.[rl]'"
        ) from error
    raise NotImplementedError(
        "The PPO environment wrapper should be enabled after the bridge calibration trace exists. "
        "Use smw-ai serve and smw-ai calibrate-log first."
    )

