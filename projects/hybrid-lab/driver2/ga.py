"""Curriculum-aware Genetic Algorithm trainer.

Evolves a population of LSTM brains through progressive track condensation.
One population is evaluated on ALL segments at the current curriculum level;
fitness = weighted mean across segments.  When mastery is reached, segments
condense and the same population continues on longer episodes.

Parallel evaluation uses ProcessPoolExecutor, matching the existing
supra.evolution pattern.
"""
from __future__ import annotations

import os
import sys
import time
import json
import concurrent.futures
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .brain import LSTMBrain
from .config import Driver2Config, BrainConfig, GAConfig
from .curriculum import SegmentCurriculum, SegmentResult
from .manifest import ManifestWriter, RunManifest, create_run_id


# ------------------------------------------------------------------ #
# Top-level worker (must be picklable for multiprocessing)
# ------------------------------------------------------------------ #
def _eval_one_segment(args: tuple) -> dict:
    """Evaluate a single genome on a single segment.  Runs in a worker process."""
    genome, seg_start, seg_end, entry_speed, time_budget, cfg_dict = args

    # Lazy import to avoid re-importing in the main process
    from .config import Driver2Config, BrainConfig
    from .env import Driver2Env

    cfg = Driver2Config(**{
        k: v for k, v in cfg_dict.items()
        if k in Driver2Config.__dataclass_fields__
    })
    # Reconstruct nested configs
    cfg.brain = BrainConfig(**cfg_dict["brain"])
    cfg.ga = GAConfig(**cfg_dict["ga"])

    env = Driver2Env(cfg)
    brain = LSTMBrain.from_genome(genome, cfg.brain)

    n_pts = len(env.track.curvature)
    if seg_end > seg_start:
        idxs = np.arange(seg_start, seg_end) % n_pts
    else:
        idxs = np.concatenate([np.arange(seg_start, n_pts), np.arange(0, seg_end)])
    v_ref_seg = env.v_ref[idxs] if len(idxs) > 0 else None

    result = env.evaluate_segment(
        brain, seg_start, seg_end, entry_speed, time_budget, v_ref_seg
    )
    return {
        "progress_frac": result.progress_frac,
        "mean_speed": result.mean_speed,
        "pace_ratio": result.pace_ratio,
        "clean": result.clean,
        "terminal": result.terminal,
        "offtrack_seconds": result.offtrack_seconds,
        "time_elapsed": result.time_elapsed,
        "fitness": result.fitness,
    }


class CurriculumGA:
    """Population of LSTM brains with curriculum-driven evaluation."""

    def __init__(self, cfg: Driver2Config | None = None, run_id: str | None = None):
        cfg = cfg or Driver2Config()
        self.cfg = cfg
        self.run_id = run_id or create_run_id()

        # Brain template
        self.brain_cfg = cfg.brain
        tmpl = LSTMBrain(cfg.brain)
        self.genome_size = tmpl.genome_size
        print(f"[driver2] LSTM brain: {self.genome_size:,} parameters")

        # GA state
        ga = cfg.ga
        self.pop_size = ga.pop_size
        self.rng = np.random.default_rng(42)
        self.genomes = [
            LSTMBrain(cfg.brain).get_genome() for _ in range(ga.pop_size)
        ]
        self.best_genome = self.genomes[0].copy()
        self.best_fitness = -1e9
        self.generation = 0

        # Curriculum (deferred until train() — needs env for track/v_ref)
        self.curriculum: SegmentCurriculum | None = None

        # Manifest
        base = Path(cfg.artifact_dir)
        self.writer = ManifestWriter(str(base), self.run_id)

        # Workers
        self.workers = ga.workers if ga.workers > 0 else (os.cpu_count() or 4)

    # ------------------------------------------------------------------ #
    # Population genetics
    # ------------------------------------------------------------------ #
    def _tournament(self, fitnesses: np.ndarray) -> int:
        """Tournament selection: best of k random candidates."""
        k = self.cfg.ga.tournament_k
        candidates = self.rng.choice(len(fitnesses), size=k, replace=False)
        return int(candidates[np.argmax(fitnesses[candidates])])

    def _breed(self, fitnesses: np.ndarray) -> list[np.ndarray]:
        """Create next generation via elitism + tournament + crossover + mutation."""
        ga = self.cfg.ga
        n = len(self.genomes)
        ranked = np.argsort(fitnesses)[::-1]

        next_gen: list[np.ndarray] = []

        # 1. Elitism — top performers carry over unchanged
        elite_n = max(1, int(n * ga.elite_frac))
        for i in range(elite_n):
            next_gen.append(self.genomes[ranked[i]].copy())

        # 2. Fill the rest with tournament-selected, crossed-over, mutated children
        while len(next_gen) < n:
            p1 = self._tournament(fitnesses)
            p2 = self._tournament(fitnesses)

            # Uniform crossover
            mask = self.rng.random(self.genome_size) < 0.5
            child = np.where(mask, self.genomes[p1], self.genomes[p2]).copy()

            # Gaussian mutation
            mut_mask = self.rng.random(self.genome_size) < ga.mutation_rate
            child[mut_mask] += self.rng.normal(0, ga.mutation_sigma, mut_mask.sum())

            next_gen.append(child.astype(np.float32))

        return next_gen

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_generation(self) -> tuple[np.ndarray, list[list[SegmentResult]]]:
        """Score every genome on every segment.  Returns (fitnesses, all_results)."""
        cur = self.curriculum
        segments = cur.segments
        n_seg = len(segments)

        # Serialise the config for worker processes
        cfg_dict = {
            "car": self.cfg.car,
            "track": self.cfg.track,
            "brain": asdict(self.cfg.brain),
            "ga": asdict(self.cfg.ga),
            "dt": self.cfg.dt,
            "artifact_dir": self.cfg.artifact_dir,
        }

        # Build work items: (genome, seg_start, seg_end, entry_speed, budget, cfg)
        work = []
        for gi, genome in enumerate(self.genomes):
            for si, (s_start, s_end) in enumerate(segments):
                entry_v = cur.entry_speed(si)
                budget = cur.time_budget(si)
                work.append((genome, s_start, s_end, entry_v, budget, cfg_dict))

        # Execute in parallel
        results_flat: list[dict] = []
        if self.workers > 1 and len(work) > 4:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers
            ) as pool:
                results_flat = list(pool.map(_eval_one_segment, work))
        else:
            # Sequential fallback (debugging / small runs)
            results_flat = [_eval_one_segment(w) for w in work]

        # Reshape: results_flat[gi * n_seg + si]
        all_results: list[list[SegmentResult]] = []
        fitnesses = np.zeros(self.pop_size)

        for gi in range(self.pop_size):
            genome_results = []
            seg_fitnesses = []
            for si in range(n_seg):
                r = results_flat[gi * n_seg + si]
                sr = SegmentResult(
                    segment_idx=si,
                    progress_frac=r["progress_frac"],
                    mean_speed=r["mean_speed"],
                    pace_ratio=r["pace_ratio"],
                    clean=r["clean"],
                    terminal=r["terminal"],
                    terminal_reason="",
                    offtrack_seconds=r["offtrack_seconds"],
                    time_elapsed=r["time_elapsed"],
                    fitness=r["fitness"],
                )
                genome_results.append(sr)
                seg_fitnesses.append(sr.fitness)

            all_results.append(genome_results)

            # Weighted mean across segments
            weights = cur.fitness_weights(seg_fitnesses)
            fitnesses[gi] = float(np.dot(seg_fitnesses, weights))

        return fitnesses, all_results

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(
        self,
        max_generations: int = 10_000,
        on_generation: Callable | None = None,
        verbose: bool = True,
    ):
        """Main training loop.

        Args:
            max_generations: Hard cap on total generations.
            on_generation:   Optional callback(ga, gen, fitnesses, results).
            verbose:         Print progress to stdout.
        """
        # Initialise environment + curriculum
        from .env import Driver2Env

        env = Driver2Env(self.cfg)
        self.curriculum = SegmentCurriculum(
            env.track, env.v_ref, self.cfg.curriculum
        )

        if verbose:
            print(f"[driver2] Run {self.run_id}")
            print(f"[driver2] Car: {self.cfg.car}, Track: {self.cfg.track}")
            print(f"[driver2] Population: {self.pop_size}, Genome: {self.genome_size}")
            print(f"[driver2] Curriculum: {self.curriculum.n_segments} segments "
                  f"(level {self.curriculum.level})")
            print(f"[driver2] Workers: {self.workers}")
            print()

        for gen in range(max_generations):
            t0 = time.time()

            # Evaluate
            fitnesses, all_results = self._evaluate_generation()

            # Update best
            best_idx = int(np.argmax(fitnesses))
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_genome = self.genomes[best_idx].copy()
                self.writer.save_genome(self.best_genome, tag="best")

            # Stats
            gen_time = time.time() - t0
            best_results = all_results[best_idx]
            n_clean = sum(1 for r in best_results if r.clean)
            n_complete = sum(1 for r in best_results if r.progress_frac >= 0.99)
            mean_pace = float(np.mean([r.pace_ratio for r in best_results]))

            if verbose:
                lvl = self.curriculum.level
                n_seg = self.curriculum.n_segments
                print(
                    f"[gen {gen:4d}] L{lvl} ({n_seg:2d} seg) | "
                    f"best {fitnesses[best_idx]:+.4f} "
                    f"mean {fitnesses.mean():+.4f} | "
                    f"clean {n_clean}/{n_seg} complete {n_complete}/{n_seg} "
                    f"pace {mean_pace:.3f} | "
                    f"{gen_time:.1f}s"
                )

            # Log generation
            self.writer.save_generation_log(gen, {
                "level": self.curriculum.level,
                "n_segments": self.curriculum.n_segments,
                "best_fitness": float(fitnesses[best_idx]),
                "mean_fitness": float(fitnesses.mean()),
                "worst_fitness": float(fitnesses.min()),
                "best_clean": n_clean,
                "best_complete": n_complete,
                "best_pace": mean_pace,
                "time_seconds": gen_time,
            })

            # Callback
            if on_generation:
                on_generation(self, gen, fitnesses, all_results)

            # Breed next generation
            self.genomes = self._breed(fitnesses)
            self.generation = gen + 1

            # Curriculum condensation check
            self.curriculum.tick_generation()
            if self.curriculum.should_condense(best_results):
                old_n = self.curriculum.n_segments
                if self.curriculum.condense():
                    new_n = self.curriculum.n_segments
                    if verbose:
                        print(
                            f"\n  >>> CONDENSE: {old_n} → {new_n} segments "
                            f"(level {self.curriculum.level}) <<<\n"
                        )
                    # Save condensation checkpoint
                    self.writer.save_genome(
                        self.best_genome,
                        tag=f"condense_L{self.curriculum.level}",
                    )

        # Final save
        self.writer.save_genome(self.best_genome, tag="final")
        if verbose:
            print(f"\n[driver2] Training complete. Best fitness: {self.best_fitness:.4f}")
            print(f"[driver2] Artifacts: {self.writer.run_dir}")
