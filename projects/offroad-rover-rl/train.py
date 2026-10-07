"""
Training Pipeline for Off-Road Rover Neuroevolution.
Runs population evaluation, logs generation statistics, exports 3D terrain and
generational champion replay trajectories for the interactive WebGL dashboard.
"""
import os
import sys
import json
import time
import argparse

from env import RoverTerrainEnv
from neuroevolution import NeuroevolutionEngine
from terrain import ProceduralTerrain

def main():
    parser = argparse.ArgumentParser(description="Train Off-Road Rover with Neuroevolution.")
    parser.add_argument("--generations", type=int, default=25, help="Number of generations to evolve.")
    parser.add_argument("--pop-size", type=int, default=40, help="Population size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for procedural terrain.")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(project_dir, "checkpoints")
    static_dir = os.path.join(project_dir, "dashboard", "static")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    print("==================================================================")
    print("      OFF-ROAD ROVER: 3D TERRAIN TRAVERSAL NEUROEVOLUTION        ")
    print("==================================================================")
    print(f"Target Generations : {args.generations}")
    print(f"Population Size    : {args.pop_size}")
    print(f"Course Length      : 150.0 meters with boulders & moguls")
    print(f"Checkpoints Dir    : {checkpoints_dir}")
    print("------------------------------------------------------------------")

    # Export terrain data for Three.js viewer
    print("Exporting 3D terrain heightmap and boulder coordinates...")
    terrain = ProceduralTerrain(seed=args.seed)
    terrain_data = terrain.export_terrain_data(nx=120, ny=40)
    terrain_json_path = os.path.join(static_dir, "terrain.json")
    with open(terrain_json_path, "w") as f:
        json.dump(terrain_data, f)
    print(f"Saved terrain map to {terrain_json_path}")

    env = RoverTerrainEnv(seed=args.seed)
    engine = NeuroevolutionEngine(pop_size=args.pop_size, seed=args.seed)

    print("\nStarting Neuroevolution Loop...")
    print(f"{'GEN':<5} | {'BEST FIT':<9} | {'AVG FIT':<8} | {'MAX DIST (m)':<12} | {'FLIP %':<7} | {'STEPS':<6} | {'TIME'}")
    print("-" * 68)

    t0_start = time.time()
    for gen in range(args.generations):
        t0 = time.time()
        stats, champ = engine.evolve_generation(env, save_dir=checkpoints_dir)
        dt = time.time() - t0

        print(
            f"{stats['generation']:<5} | "
            f"{stats['best_fitness']:<9.1f} | "
            f"{stats['avg_fitness']:<8.1f} | "
            f"{stats['max_distance']:<12.1f} | "
            f"{int(stats['flip_rate']*100):>5}% | "
            f"{stats['survived_steps']:<6} | "
            f"{dt:.2f}s"
        )

    total_time = time.time() - t0_start
    print("-" * 68)
    print(f"Evolution completed in {total_time:.1f}s.")

    # Save summary history
    history_path = os.path.join(checkpoints_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(engine.history, f, indent=2)
    print(f"Saved training history to {history_path}")

if __name__ == "__main__":
    main()
