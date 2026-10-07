"""
Deterministic Evaluation & Benchmark Suite for Off-Road Rover.
Evaluates the best evolved policy champion across the 150m obstacle course
and logs traversal time, average speed, rollover risk, and energy metrics.
"""
import os
import sys
import json
import time
import numpy as np

from env import RoverTerrainEnv
from neuroevolution import RoverPolicy
from config import TERRAIN_LENGTH

def evaluate_best(checkpoint_path=None):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(project_dir, "checkpoints")

    if checkpoint_path is None:
        # Pick latest checkpoint
        ckpts = sorted([f for f in os.listdir(checkpoints_dir) if f.startswith("gen_") and f.endswith(".json")])
        if not ckpts:
            print("[ERROR] No checkpoints found in checkpoints/ directory.")
            return
        checkpoint_path = os.path.join(checkpoints_dir, ckpts[-1])

    print(f"Loading champion policy from: {checkpoint_path}")
    with open(checkpoint_path, "r") as f:
        data = json.load(f)

    genome = data.get("best_genome")
    stats = data.get("stats", {})
    policy = RoverPolicy(genome=genome)

    print("\n==============================================================")
    print("      OFF-ROAD ROVER: DETERMINISTIC BENCHMARK TRIAL          ")
    print("==============================================================")
    print(f"Loaded Generation  : Gen {stats.get('generation', '?')}")
    print(f"Training Best Fit  : {stats.get('best_fitness', '?')}")
    print(f"Course Length      : {TERRAIN_LENGTH} m (Whoops -> Boulders -> Moguls -> Incline)")
    print("--------------------------------------------------------------")

    env = RoverTerrainEnv(seed=1337)  # Test seed
    obs, info = env.reset(seed=1337)

    terminated = False
    truncated = False
    step = 0
    total_reward = 0.0

    print(f"{'STEP':<6} | {'X (m)':<7} | {'Y (m)':<7} | {'SPEED':<10} | {'PITCH':<7} | {'ROLL':<7} | {'STATUS'}")
    print("-" * 62)

    t0 = time.time()
    while not (terminated or truncated):
        step += 1
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        speed_kmh = info["speed"] * 3.6
        pos = info["pos"]
        status = "ACTIVE"
        if info["flipped"]:
            status = "FLIPPED!"
        elif info["belly_contact"]:
            status = "BELLY SCRAPE"
        elif pos[0] >= (TERRAIN_LENGTH - 10.0):
            status = "SUMMIT WON!"

        if step % 25 == 0 or terminated or truncated:
            print(
                f"{step:<6} | "
                f"{pos[0]:<7.1f} | "
                f"{pos[1]:<7.2f} | "
                f"{speed_kmh:<5.1f} km/h | "
                f"{info['pitch_deg']:<7.1f} | "
                f"{info['roll_deg']:<7.1f} | "
                f"{status}"
            )

    duration = time.time() - t0
    final_x = info["pos"][0]
    print("-" * 62)
    print(f"Trial Completed in : {duration:.2f}s ({step} control steps)")
    print(f"Distance Traversed : {final_x:.2f} / {TERRAIN_LENGTH} m ({final_x/TERRAIN_LENGTH*100:.1f}%)")
    print(f"Final Outcome      : {status}")
    print(f"Total Score        : {total_reward + final_x * 3.0:.1f}")
    print("==============================================================\n")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate_best(ckpt)
