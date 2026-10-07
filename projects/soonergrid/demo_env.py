from soonergrid.gym import NormanTrafficEnv
import time

print("Loading environment...")
env = NormanTrafficEnv(dt_s=15.0, episode_duration_s=3600.0) # 1 hour
print("Resetting...")
obs, infos = env.reset()
print(f"Agents initialized: {env.agents}")
start_t = time.time()
print("Stepping 10 times with random actions...")
for _ in range(10):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    if all(terms.values()) or all(truncs.values()):
        break

print(f"Done in {time.time() - start_t:.2f}s")
for agent in env.agents:
    print(f"{agent} reward: {rewards[agent]:.3f}")
print("Global State:", env.state())
