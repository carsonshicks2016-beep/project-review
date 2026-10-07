"""
Neuroevolution Genetic Algorithm for Off-Road Rover Terrain Traversal.
Implements vectorized MLP neural networks, tournament selection, BLX-alpha crossover,
adaptive Gaussian mutation, elitism, and generation trajectory logging for 3D playback.
"""
import os
import json
import numpy as np
from config import (
    POPULATION_SIZE, ELITISM_COUNT, MUTATION_RATE,
    MUTATION_STRENGTH, TOURNAMENT_SIZE
)

class RoverPolicy:
    """Compact 2-hidden-layer MLP Neural Network Controller (30 -> 32 -> 16 -> 2)."""
    def __init__(self, genome=None):
        self.in_dim = 30
        self.h1_dim = 32
        self.h2_dim = 16
        self.out_dim = 2

        self.num_params = (
            (self.in_dim * self.h1_dim + self.h1_dim) +
            (self.h1_dim * self.h2_dim + self.h2_dim) +
            (self.h2_dim * self.out_dim + self.out_dim)
        )

        if genome is None:
            # Xavier / He uniform initialization
            self.genome = np.random.uniform(-0.4, 0.4, size=self.num_params).astype(np.float32)
        else:
            self.genome = np.array(genome, dtype=np.float32)

        self._unpack_weights()

    def _unpack_weights(self):
        idx = 0
        w1_size = self.in_dim * self.h1_dim
        self.w1 = self.genome[idx:idx + w1_size].reshape(self.in_dim, self.h1_dim)
        idx += w1_size
        self.b1 = self.genome[idx:idx + self.h1_dim]
        idx += self.h1_dim

        w2_size = self.h1_dim * self.h2_dim
        self.w2 = self.genome[idx:idx + w2_size].reshape(self.h1_dim, self.h2_dim)
        idx += w2_size
        self.b2 = self.genome[idx:idx + self.h2_dim]
        idx += self.h2_dim

        w3_size = self.h2_dim * self.out_dim
        self.w3 = self.genome[idx:idx + w3_size].reshape(self.h2_dim, self.out_dim)
        idx += w3_size
        self.b3 = self.genome[idx:idx + self.out_dim]

    def set_genome(self, genome):
        self.genome = np.array(genome, dtype=np.float32)
        self._unpack_weights()

    def act(self, obs):
        """Forward pass with tanh activations producing action in [-1, 1]."""
        h1 = np.tanh(obs @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        out = np.tanh(h2 @ self.w3 + self.b3)
        return out


class NeuroevolutionEngine:
    def __init__(self, pop_size=POPULATION_SIZE, seed=42):
        self.pop_size = pop_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.population = [RoverPolicy() for _ in range(pop_size)]
        self.generation = 0
        self.history = []

    def evaluate_individual(self, policy, env, record_trajectory=False):
        """Runs one full evaluation episode of a policy in the environment."""
        obs, info = env.reset(seed=self.seed)
        total_reward = 0.0
        max_dist = info["pos"][0]
        trajectory = []

        if record_trajectory:
            trajectory.append(info)

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            max_dist = max(max_dist, info["pos"][0])

            if record_trajectory:
                trajectory.append(info)

        fitness = total_reward + (max_dist * 3.0)
        return {
            "fitness": float(fitness),
            "distance": float(max_dist),
            "steps": int(env.current_step),
            "flipped": bool(info["flipped"]),
            "trajectory": trajectory if record_trajectory else None
        }

    def evolve_generation(self, env, save_dir="checkpoints"):
        """
        Evaluates the current population, tracks champion, logs telemetry,
        and produces the next generation through selection, crossover, and mutation.
        """
        self.generation += 1
        results = []

        # 1. Evaluate all candidates (record trajectory for the very first one to seed)
        for i, ind in enumerate(self.population):
            res = self.evaluate_individual(ind, env, record_trajectory=False)
            res["index"] = i
            results.append(res)

        # Sort descending by fitness
        results.sort(key=lambda r: r["fitness"], reverse=True)

        best_idx = results[0]["index"]
        best_policy = self.population[best_idx]

        # Re-run best policy with full telemetry recording for 3D viewer
        champion_eval = self.evaluate_individual(best_policy, env, record_trajectory=True)

        # Calculate population statistics
        fitnesses = [r["fitness"] for r in results]
        distances = [r["distance"] for r in results]
        flip_count = sum(1 for r in results if r["flipped"])

        gen_stats = {
            "generation": self.generation,
            "best_fitness": round(float(results[0]["fitness"]), 2),
            "avg_fitness": round(float(np.mean(fitnesses)), 2),
            "max_distance": round(float(max(distances)), 2),
            "avg_distance": round(float(np.mean(distances)), 2),
            "flip_rate": round(float(flip_count / self.pop_size), 2),
            "survived_steps": champion_eval["steps"]
        }
        self.history.append(gen_stats)

        population_summary = [
            {
                "rank": rank + 1,
                "id": r["index"],
                "fitness": round(float(r["fitness"]), 2),
                "distance": round(float(r["distance"]), 2),
                "steps": int(r["steps"]),
                "flipped": bool(r["flipped"])
            }
            for rank, r in enumerate(results)
        ]

        # Save checkpoint if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"gen_{self.generation:03d}.json")
            ckpt_data = {
                "stats": gen_stats,
                "population": population_summary,
                "best_genome": [float(x) for x in best_policy.genome],
                "trajectory": champion_eval["trajectory"]
            }
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)

            with open(ckpt_path, "w") as f:
                json.dump(ckpt_data, f, cls=NumpyEncoder)

        # 2. Reproduction: Elitism
        new_population = []
        for i in range(ELITISM_COUNT):
            elite_idx = results[i]["index"]
            new_population.append(RoverPolicy(self.population[elite_idx].genome.copy()))

        # 3. Tournament Selection & Crossover for remainder
        while len(new_population) < self.pop_size:
            # Select parent 1
            t1 = self.rng.choice(self.pop_size, size=TOURNAMENT_SIZE, replace=False)
            best_t1 = min(t1, key=lambda idx: [r["index"] for r in results].index(idx))
            p1 = self.population[best_t1].genome

            # Select parent 2
            t2 = self.rng.choice(self.pop_size, size=TOURNAMENT_SIZE, replace=False)
            best_t2 = min(t2, key=lambda idx: [r["index"] for r in results].index(idx))
            p2 = self.population[best_t2].genome

            # BLX-alpha blend crossover
            alpha = 0.35
            gamma = (1.0 + 2.0 * alpha) * self.rng.uniform(size=p1.shape) - alpha
            child_genome = p1 + gamma * (p2 - p1)

            # Gaussian Mutation
            mutate_mask = self.rng.uniform(size=p1.shape) < MUTATION_RATE
            noise = self.rng.normal(0.0, MUTATION_STRENGTH, size=p1.shape)
            child_genome[mutate_mask] += noise[mutate_mask]

            new_population.append(RoverPolicy(child_genome))

        self.population = new_population
        return gen_stats, champion_eval
