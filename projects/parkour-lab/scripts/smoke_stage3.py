"""Smoke test for Stage 3 parkour environment."""
import sys
sys.path.insert(0, '.')
import numpy as np
from envs.parkour_env import ParkourHumanoidEnv
from envs import terrain

print("=== Testing terrain generation ===")
rng = np.random.default_rng(42)
segments, length = terrain.generate_course(rng)
print(f"  Segments: {len(segments)}")
print(f"  Course length: {length:.1f}m")
for s in segments[:5]:
    print(f"    {s.kind}: x=[{s.x_start:.1f}, {s.x_end:.1f}] h={s.height:.2f} geoms={len(s.geom_specs)}")
positions, sizes, n_active = terrain.course_to_geom_arrays(segments)
print(f"  Active geoms: {n_active} / {terrain.MAX_GEOMS}")

print("\n=== Testing geom arrays ===")
assert positions.shape == (terrain.MAX_GEOMS, 3)
assert sizes.shape == (terrain.MAX_GEOMS, 3)
# Inactive geoms should be at z=-10
assert positions[n_active, 2] == -10.0
print("  Array shapes OK")

print("\n=== Testing segment_at_x ===")
seg = terrain.segment_at_x(segments, 2.5)
assert seg is not None
assert seg.kind == 'flat'
print(f"  x=2.5 -> {seg.kind} (correct: start zone)")

print("\n=== Creating ParkourHumanoidEnv ===")
env = ParkourHumanoidEnv()
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape}")
expected_obs = 22 + 23 + 13*10 + 13*6 + 17 + 13*6 + 18  # qpos + qvel + cinert + cvel + actuator + cfrc + heights
assert env.observation_space.shape == (expected_obs,), f"Expected ({expected_obs},), got {env.observation_space.shape}"
assert env.action_space.shape == (17,), f"Expected (17,), got {env.action_space.shape}"
print("  Spaces OK")

print("\n=== Testing reset ===")
obs, info = env.reset(seed=42)
print(f"  Obs shape: {obs.shape}")
print(f"  Obs range: [{obs.min():.2f}, {obs.max():.2f}]")
print(f"  Height scan (last 18): {obs[-18:]}")
print(f"  Info: course_length={info.get('course_length', 'N/A')}, n_segments={info.get('n_segments', 'N/A')}")

print("\n=== Testing step ===")
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"  Obs shape: {obs.shape}")
print(f"  Reward: {reward:.3f}")
print(f"  Terminated: {terminated}")
print(f"  Info keys: {sorted(info.keys())}")

print("\n=== Running 100 steps ===")
total_reward = 0.0
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated:
        print(f"  Terminated at step {i+1}")
        break
print(f"  Total reward after {min(i+1, 100)} steps: {total_reward:.1f}")
print(f"  Final position: x={info['x_position']:.2f}, y={info['y_position']:.2f}")

env.close()
print("\n=== ALL SMOKE TESTS PASSED ===")
