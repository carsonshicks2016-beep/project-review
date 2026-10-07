# Parkour Lab

A staged MuJoCo + Gymnasium + Stable-Baselines3 project. **Stages 1 and 2 are implemented.** Stage 1 is complete; Stage 2 was authorized after review. Stop after Stage 2 for video review before custom terrain. Stage 3–5 configurations are planning documents, not implemented capabilities.

## Local setup

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python train.py --config configs/01-warmup.yaml --run runs/stage1
.venv/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006
.venv/bin/python eval.py --checkpoint runs/stage1/best --video runs/stage1/heldout.mp4
.venv/bin/python scripts/plot.py runs/stage1
```

Run names must be new: existing experiments cannot be silently overwritten. Each best/last checkpoint contains both `policy.zip` and `vecnormalize.pkl`; keep the pair together. Evaluation freezes observation statistics, disables reward normalization, uses deterministic actions, and reports environment reward. The evaluation seed set used for selection is separate from the final held-out set. Video contains the first complete episode of each evaluation, including early falls, at the environment's native playback rate.

## Humanoid (Stage 2)

```sh
.venv/bin/python train.py --config configs/02-humanoid.yaml --run runs/stage2
.venv/bin/tensorboard --logdir runs/stage2/tensorboard --host 127.0.0.1 --port 6007
.venv/bin/python eval.py --checkpoint runs/stage2/best --video runs/stage2/heldout.mp4
.venv/bin/python scripts/plot.py runs/stage2
.venv/bin/python scripts/progress.py runs/stage2
```

The real run already occupies `runs/stage2`; use a new run name for a new experiment. Periodic videos and complete policy/normalization pairs are saved every 100 updates (409600 transitions), plus the initial and final policy. Metadata infers the evaluation environment. `runs/<run>/source/` preserves the training source; `checkpoints/update-xxxxx/` preserves periodic artifacts. Best and last each carry hashes of both policy and normalization.

To recover a checkpoint, use the **same stage configuration** and a new output directory:

```sh
.venv/bin/python train.py --config configs/02-humanoid.yaml --resume runs/stage2/last --steps 1000000 --run runs/stage2-continuation
```

`--steps` is additional transitions on resume. The saved optimizer and normalization are reused; this is recovery, not a hyperparameter migration mechanism. Environment RNG and partial episode states restart, so resume is not a bitwise continuation. Parent policy hash and step count are recorded. See [Stage 2 notes](stages/02-humanoid/NOTES.md) for observations, actuation, reward incentives, and measured results.

## Why SB3

The user permits SB3 for speed of iteration. It supplies tested PPO timeout bootstrapping, subprocess environments, logging, and normalization, letting this repository focus on simulation and inspectable training/evaluation code. `train.py` is a small driver, not a claim that the PPO implementation is reproduced here. Read PPO in `.venv/lib/python3.12/site-packages/stable_baselines3/ppo/ppo.py` and rollout handling in `common/on_policy_algorithm.py`.

## Machine and speed

Apple M2 Pro, 12 CPU cores, 19 GPU cores, 16 GiB unified memory. Standard MuJoCo steps on CPU. PyTorch MPS availability does not imply GPU physics. CPU PPO with one learner thread avoids oversubscription; benchmark 1, 4, and 8 environments using the complete rollout-plus-update loop. Hopper is cheap enough that subprocess communication may cost more than the physics; repeat this benchmark on Humanoid before choosing its worker count.

MJX-JAX/Brax use a different batched training path; SB3 SubprocVecEnv cannot automatically exploit thousands of GPU simulations. For a later massive-parallel experiment, an NVIDIA GPU Linux machine with MJX is the more direct route, subject to benchmarking and validating environment/reward parity. MJX-Warp specifically targets NVIDIA. No cloud resources are provisioned by this project.

## Stage gates

1. Hopper warmup: train, inspect raw rewards, validate held-out episodes and video. Stop for review.
2. Humanoid-v5 walking: benchmark physics, tune walking, inspect falls/shuffling/one-legged local optima. Stop for review.
3. Custom MJCF and terrain-aware environment: boxes, actual gaps, stairs, wall, body-relative terrain measurements, one-time obstacle bonuses. Stop for review.
4. Curriculum: advance on measured course completion, keep terrain-level history and evaluate fixed levels. Stop for review.
5. Push recovery, friction/mass randomization, and held-out perturbation tests. Stop for review.

Stretch work follows these gates. Motion clips require skeleton mapping, joint coordinate conventions, unit/time alignment, joint-limit checks and foot-contact validation before imitation. DeepMimic adds reference phase and pose/velocity imitation rewards. AMP adds a discriminator over motion transitions, reference sampling, normalization and separate diagnostics. None is implemented or represented by the warmup checkpoint.

## Sources

- [Hopper specification](https://gymnasium.farama.org/environments/mujoco/hopper/)
- [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/latest/mjx.html)


## Stability experiments after the initial Humanoid review

The initial 10M-step Humanoid policy runs forward but fails full-episode survival. A separate parkour run was stopped with its checkpoints preserved so CPU time could focus on that prerequisite.

```sh
.venv/bin/python train.py --config configs/02-humanoid-stability.yaml --resume runs/stage2/best --apply-config-on-resume --run runs/new-stability-experiment
```

`--apply-config-on-resume` explicitly replaces saved PPO hyperparameters with the new configuration. Without that flag, resume retains the saved settings. Effective parameters and the parent policy hash are recorded in the new run. Changing gamma resets discounted-return normalization while retaining observation statistics. Check `stages/02-humanoid/NOTES.md` for the result of each experiment before deciding which checkpoint to use.
