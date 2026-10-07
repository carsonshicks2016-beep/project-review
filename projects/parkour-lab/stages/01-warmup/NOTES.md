# Stage 1 — Hopper warmup

## Scope and implementation

Hopper-v5 is a planar, one-legged torque-controlled robot. Success means sustained forward hopping, not human walking. Gymnasium supplies the stock MJCF; we do not modify its geometry, contacts, actuators or rewards. PPO is from Stable-Baselines3, with observation/reward normalization and subprocess simulation. The best checkpoint is chosen by mean raw return on five fixed evaluation seeds. A separate ten-seed evaluation is reserved for final verification.

## What the robot senses and controls

The default observation has 11 values in this order:

- `qpos[1:]`: root height z, torso pitch, thigh angle, leg angle, foot angle (5).
- `qvel`: root x/z velocities, torso angular velocity, three joint velocities (6), clipped to [-10, 10] by the environment.

The global root x position is omitted, so translating the robot does not change what the policy sees. Height and orientation remain essential for fall detection and balance. Angles use radians and velocities use simulation units (meters/second or radians/second). Positions are generalized coordinates; in later 3D models quaternion orientation means qpos and qvel need not have equal lengths.

Hopper does **not** include measured contact forces in its default observation. Ground contact still exists in MuJoCo dynamics. Contact-force features, when added to a humanoid observation, can indicate support/load but can be spiky; check frame, scale, clipping and physics-step aggregation. A terrain sensor will be needed for anticipating obstacles rather than reacting only after impact.

The three bounded actions in [-1, 1] drive joint motors; these are actuator controls scaled through the MJCF gears, not target joint angles. There is no hidden gait controller or reference motion. Joint torques, gravity, inertia, friction and ground reaction forces produce movement.

MuJoCo advances at 0.002 seconds per physics step; Gymnasium holds an action for 4 steps, so each control decision represents 0.008 seconds (125 Hz). A full 1,000-decision episode is 8 simulated seconds. Playback uses that rate rather than an arbitrary 30 fps. Changing timestep or frame skip changes contact integration and the control problem, so it is not a free speed knob.

## Default reward and termination

At each control decision:

`r = (x_after - x_before)/0.008 + 1[healthy] - 0.001 * sum(action**2)`

Forward velocity rewards speed. The +1 healthy reward makes survival valuable. The control penalty discourages unnecessarily large actuator commands but is weak: it is a torque-effort proxy, not physical energy consumption (which involves torque times joint velocity). A later energy objective should state which quantity is penalized and integrate consistently over time.

Health requires height above 0.7, torso pitch strictly between -0.2 and 0.2 radians, and the environment's bounded-state checks. An unhealthy state terminates the episode. The 1,000-step time limit truncates it. These differ for value bootstrapping; SB3 handles time-limit bootstrapping. The v5 healthy bonus is awarded only while healthy.

A standing agent can earn survival reward without learning locomotion; a fast but unstable agent can collect progress then fall. The reward contains no natural-motion objective, so an awkward gait can score well. Hopper's hopping is intended morphology, while one-legged hopping in a humanoid can be an unwanted local optimum under a similar forward reward.

## Normalization and evaluation

Training normalizes observations and discounted returns. Curves and checkpoint selection use **raw environment reward**, so a changing normalization scale cannot masquerade as learning. Evaluation calls normalization without updating its statistics. Reloading `vecnormalize.pkl` with `training=False` and `norm_reward=False` reproduces this behavior. Evaluation seeds 10000–10004 select checkpoints; seeds 20000–20009 are held out from selection. This is a one-training-seed warmup, not evidence of seed-robust convergence.

Videos are recorded at update zero, every 50 PPO updates, and after the final update. With 8 environments and 512 decisions per rollout, one update collects 4096 transitions. Each video shows one deterministic episode; five episodes contribute to each selection score. Later videos must include course-level success and failures, not only reward.

## Hardware and benchmark

M2 Pro / 12 CPU cores / 19 GPU cores / 16 GiB unified memory; MPS available, CUDA unavailable. Learner uses one CPU thread. A 32768-transition rollout-plus-PPO benchmark (excluding worker startup and videos) measured:

| Environments | Steps/second |
|---|---:|
| 1 | 3093 |
| 4 | 4840 |
| 8 | 5760 |

Eight SubprocVecEnv workers were selected. These are short single-run measurements with different rollout sizes, so do not interpret small differences as a rigorous scaling law. Rebenchmark Humanoid, whose physics is more expensive. Local MPS policy optimization has not been benchmarked; it does not accelerate CPU MuJoCo physics. A Linux/NVIDIA MJX experiment is a separate batched pipeline and should be compared at matched task/physics settings.

## Results

Completed 1,503,232 transitions (367 full updates; the requested 1.5 million is rounded up to a rollout boundary) in 289.85 seconds including evaluation/video overhead: 5,186 transitions/second. No later stage was run.

- Initial five-seed deterministic return: 63.17.
- Selected checkpoint: update 150, 614,400 transitions; five-seed return 3,138.08, with four of five episodes surviving to the time limit.
- Final checkpoint: update 367; five-seed return 1,063.1. It is preserved as `runs/stage1/last`, not substituted for the best policy.
- Selected policy on ten new held-out seeds: mean return **3520.08**, **10/10** complete 1,000-step episodes without unhealthy termination, mean distance **20.17 m** in 8 simulated seconds (about 2.52 m/s).
- Video: `runs/stage1/heldout.mp4`, seed 20000, decoded and checked: 1000 frames, 640×480, 125 fps, 8 seconds. Sampled frames show repeated leg compression/extension with the torso remaining upright. Frame inspection is not a biomechanics validation.
- Curve: `runs/stage1/reward-curve.png`; JSON stores every episode, including falls. All nine periodic/final videos are retained.

The policy learned sustained forward hopping. The training curve then regressed substantially, with shorter deterministic episodes. This is real performance loss, not just normalized-reward drift. PPO's local updates do not guarantee monotonic performance, and a fixed learning rate plus a changing on-policy distribution can destabilize a contact-sensitive gait. This run does not isolate the precise cause; do not claim a proven optimizer or reward defect. A follow-up could test a decaying learning rate and target-KL limit across multiple training seeds. The preserved early best checkpoint satisfies the warmup pipeline goal without concealing the regression.

The held-out 10/10 result covers only the stock reset noise and an eight-second horizon. One selection seed still failed; this is not robustness to pushes, different friction, terrain, or long durations.

Pipeline checks passed: raw reward equals its component sum; observation/action dimensions match; repeated seeded evaluation is deterministic; observation statistics remain frozen during evaluation; checkpoint reload works; TensorBoard serves the expected evaluation/training tags; the MP4 decodes at the intended duration. A final logging flush was added after this run so subsequent runs also write the terminal evaluation immediately to TensorBoard; this run's terminal result is preserved in JSON and the curve.

## References

- https://gymnasium.farama.org/environments/mujoco/hopper/
- Installed `gymnasium/envs/mujoco/hopper_v5.py` inspected for observation, reward, and health checks.
- https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- https://mujoco.readthedocs.io/en/latest/mjx.html
