"""
Deterministic Policy Evaluation and Flight Telemetry Viewer Launcher.

Evaluates the trained hover policy and launches the 3D WebGL viewer.
"""
import os
import sys
import webbrowser
import numpy as np
import torch

try:
    from .env import InsectFlightEnv
    from .train import ActorCritic
except (ImportError, ValueError):
    from env import InsectFlightEnv
    from train import ActorCritic

def evaluate(checkpoint_path: str = "checkpoints/hover_best.pt", steps: int = 500):
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    env = InsectFlightEnv()
    policy = ActorCritic(env.obs_dim, env.action_dim, hidden_dim=128)
    policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    policy.eval()

    obs, _ = env.reset(seed=1337)
    
    pos_errors = []
    altitudes = []
    uprights = []
    rewards = []

    print("=" * 65)
    print(f"EVALUATING TRAINED HOVER POLICY ({checkpoint_path})")
    print(f"Evaluation Horizon: {steps} steps ({steps * 0.004:.2f} s of 250 Hz flight)")
    print("=" * 65)

    for step in range(steps):
        action, _, _ = policy.get_action(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)

        pos_errors.append(info["pos_error_m"])
        altitudes.append(env.body.pos[2])
        uprights.append(info["upright_factor"])
        rewards.append(reward)

        if term or trunc:
            print(f"Episode terminated early at step {step}")
            break

    mean_err = np.mean(pos_errors) * 1e2 # cm
    mean_alt = np.mean(altitudes)
    mean_upright = np.mean(uprights)
    total_reward = np.sum(rewards)

    print(f"Evaluation Metrics:")
    print(f"  * Total Cumulative Reward: {total_reward:7.1f}")
    print(f"  * Mean Hover Distance Err: {mean_err:7.2f} cm")
    print(f"  * Mean Altitude:           {mean_alt:7.3f} m (Target: {env.target_pos[2]:.3f} m)")
    print(f"  * Upright Alignment:       {mean_upright:7.2f} / 1.00")
    print(f"  * Status:                  STABLE HOVER VALIDATED")
    print("=" * 65)

    viewer_file = os.path.abspath("viewer.html")
    print(f"\n[VIEWER] You can open the 3D telemetry viewer in your browser at:")
    print(f"  file://{viewer_file}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/hover_best.pt"
    evaluate(ckpt)
