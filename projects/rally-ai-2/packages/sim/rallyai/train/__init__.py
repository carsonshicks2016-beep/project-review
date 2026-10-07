"""Training and evaluation package.

Phase B owns vectorisation / normaliser. Phase C owns PPO, checkpoints,
curriculum, and the train CLI. Phase D owns ``evaluate``.

Import submodules directly (``rallyai.train.ppo``) rather than relying on
re-exports here — keeps parallel workstreams from fighting over this file.
"""
