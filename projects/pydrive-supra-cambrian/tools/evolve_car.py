#!/usr/bin/env python3
import os
import sys
import copy
import uuid
import random
import torch
import numpy as np
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from supra.config import porsche_919evo, PPOSpec
from supra.car_genome import CarGenome
from supra.ppo import PPO
from supra.track import named_track

# We will override these in main() with argparse
POP_SIZE = 4
GENERATIONS = 3
PPO_ITERS = 25
TEST_TRACK = "club"
OUT_FILE = "evolution_log.json"

class Individual:
    def __init__(self, genome: CarGenome, weights=None, parent_id=None):
        self.id = str(uuid.uuid4())[:8]
        self.parent_id = parent_id
        self.genome = genome
        self.weights = weights
        self.fitness = -np.inf
        
def evaluate_individual(ind: Individual, generation: int):
    print(f"\n--- Evaluating Genome {ind.id} (Gen {generation}) ---")
    spec = ind.genome.spec
    print(f"Physics: Mass {spec.mass:.1f}kg | Downforce {spec.downforce_ClA:.2f} | Drag {spec.drag_area:.2f} | Grip {spec.mu:.2f}")
    
    # Configure PPO for a short sprint on the test track
    # We use 4 workers to speed up the rollout
    ppo_cfg = PPOSpec(n_envs=4, n_workers=4, rollout=256)
    track = named_track(TEST_TRACK)
    
    ppo = PPO(mode="race", car=spec, ppo=ppo_cfg, fixed_track=track)
    
    # Warm-start brain if we have inherited weights
    if ind.weights is not None:
        ppo.net.load_state_dict(ind.weights)
        print("  -> Brain warm-started from parent.")
        
    # Train for a few iterations
    tmp_ckpt = f"/tmp/supra_coevo_{ind.id}.pt"
    # PPO.train returns (average return, best lap, etc) but doesn't explicitly return a tuple.
    # We will just run train and then evaluate the policy
    ppo.train(iterations=PPO_ITERS, checkpoint=tmp_ckpt, log_every=5)
    
    # Save the updated brain back to the individual
    ind.weights = copy.deepcopy(ppo.net.state_dict())
    
    # Evaluate fitness (run 1 deterministic lap using the first env in vec)
    print("  -> Running fitness lap...")
    obs_batch = ppo.vec.reset()
    total_reward = 0.0
    for _ in range(1000):
        s = ppo.vec.sensor_dim
        # We only care about the first env for evaluation
        obs = obs_batch[0]
        nobs = ppo.norm.normalize(obs[:s])
        nobs = np.concatenate([nobs, obs[s:]]).astype(np.float32)
        with torch.no_grad():
            a = ppo.net.act_mean(torch.as_tensor(nobs).unsqueeze(0)).squeeze(0).numpy()
        a = np.clip(a, -1, 1)
        # Step all envs with the same action just to keep vec moving
        actions = [a for _ in range(ppo_cfg.n_envs)]
        obs_batch, r_batch, term_batch, trunc_batch, final_batch, info_batch = ppo.vec.step(actions)
        total_reward += r_batch[0]
        if term_batch[0] or trunc_batch[0]:
            break
            
    # Fitness is the total reward (further + faster)
    ind.fitness = total_reward
    print(f"  -> Fitness (Reward): {ind.fitness:.2f}")
    return ind

def main():
    global POP_SIZE, GENERATIONS, PPO_ITERS, TEST_TRACK, OUT_FILE
    parser = argparse.ArgumentParser(description="Supra Cambrian Co-Evolution")
    parser.add_argument("--pop", type=int, default=4, help="Population size")
    parser.add_argument("--gens", type=int, default=3, help="Number of generations")
    parser.add_argument("--iters", type=int, default=25, help="PPO inner loop iterations")
    parser.add_argument("--track", type=str, default="club", help="Track name")
    parser.add_argument("--out", type=str, default="evolution_log.json", help="Output JSON log")
    args = parser.parse_args()
    
    POP_SIZE = args.pop
    GENERATIONS = args.gens
    PPO_ITERS = args.iters
    TEST_TRACK = args.track
    OUT_FILE = args.out

    print(f"Initializing Supra-Cambrian Co-Evolution Loop")
    print(f"Track: {TEST_TRACK} | Pop: {POP_SIZE} | Gens: {GENERATIONS}")
    
    base_spec = porsche_919evo()
    
    # Gen 0: Mutate from base spec
    population = []
    for _ in range(POP_SIZE):
        g = CarGenome(base_spec).mutate(sigma=0.1)
        population.append(Individual(g))
        
    evolution_log = []
        
    for gen in range(1, GENERATIONS + 1):
        print(f"\n================ GENERATION {gen} ================")
        
        # Train and evaluate all individuals
        for ind in population:
            evaluate_individual(ind, gen)
            
        # Rank by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        print("\n--- Generation Leaderboard ---")
        
        gen_data = {
            "generation": gen,
            "population": [],
            "max_fitness": float(population[0].fitness),
            "avg_fitness": float(np.mean([ind.fitness for ind in population]))
        }
        
        for i, ind in enumerate(population):
            s = ind.genome.spec
            print(f"#{i+1}: {ind.id} | Fitness: {ind.fitness:.2f} | Mass: {s.mass:.1f}kg | Grip: {s.mu:.2f} | DF: {s.downforce_ClA:.2f}")
            gen_data["population"].append({
                "id": ind.id,
                "parent_id": ind.parent_id,
                "fitness": float(ind.fitness),
                "mass": float(s.mass),
                "grip": float(s.mu),
                "downforce": float(s.downforce_ClA),
                "drag": float(s.drag_area)
            })
            
        evolution_log.append(gen_data)
        log_path = os.path.join(os.path.dirname(__file__), OUT_FILE)
        with open(log_path, "w") as f:
            json.dump(evolution_log, f, indent=2)
            
        # Selection: Keep top half
        survivors = population[:POP_SIZE // 2]
        
        # Reproduction: Clone and mutate
        next_gen = list(survivors) # Elitism: keep parents
        while len(next_gen) < POP_SIZE:
            parent = random.choice(survivors)
            # Inherit physical genome with mutation
            child_genome = CarGenome(parent.genome.spec).mutate(sigma=0.05)
            # Inherit neural weights EXACTLY
            child = Individual(child_genome, weights=copy.deepcopy(parent.weights), parent_id=parent.id)
            next_gen.append(child)
            
        population = next_gen
        
    print("\nEvolution complete.")
    best = population[0]
    print(f"Champion: {best.id} (Fitness: {best.fitness:.2f})")
    
if __name__ == "__main__":
    main()
