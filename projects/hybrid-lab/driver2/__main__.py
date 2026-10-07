from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from .config import Driver2Config, BrainConfig, GAConfig, CurriculumConfig, EvalConfig
from .ga import CurriculumGA
from .brain import LSTMBrain
from .evaluator import Driver2Evaluator

def main() -> None:
    """CLI entry point for driver2 package."""

    parser = argparse.ArgumentParser(description="Driver2: Progressive Curriculum GA Trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a driver model")
    train_parser.add_argument("--car", type=str, default="porsche_919evo", help="Car name to use for training")
    train_parser.add_argument("--pop", type=int, default=200, help="Population size for GA")
    train_parser.add_argument("--segments", type=int, default=64, help="Number of curriculum segments")
    train_parser.add_argument("--generations", type=int, default=10000, help="Maximum number of generations")
    train_parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 for auto)")
    train_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained driver model")
    eval_parser.add_argument("--genome", type=str, required=True, help="Path to genome .npz file")
    eval_parser.add_argument("--car", type=str, default="porsche_919evo", help="Car name to use for evaluation")
    eval_parser.add_argument("--laps", type=int, default=3, help="Number of laps to evaluate")

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch a driver model run in a viewer")
    watch_parser.add_argument("--run", type=str, required=True, help="Run ID to watch")

    args = parser.parse_args()

    if True:
        if args.command == "train":
            # Headless operation for training
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            os.environ['SDL_AUDIODRIVER'] = 'dummy'
            
            # 1. Parse args into a Driver2Config
            cfg = Driver2Config(
                car=args.car,
                brain=BrainConfig(),
                ga=GAConfig(pop_size=args.pop, workers=args.workers),
                curriculum=CurriculumConfig(n_initial_segments=args.segments),
                eval=EvalConfig()
            )
            
            # 2. Print the config summary
            print("--- Driver2 Training Config ---")
            print(f"Car:         {cfg.car}")
            print(f"Population:  {cfg.ga.pop_size}")
            print(f"Segments:    {cfg.curriculum.n_initial_segments}")
            print(f"Workers:     {cfg.ga.workers}")
            print("-------------------------------")

            # 3. Create CurriculumGA(cfg)
            ga = CurriculumGA(cfg)
            
            # 4. Call ga.train
            print(f"Starting training for up to {args.generations} generations...")
            ga.train(max_generations=args.generations, verbose=args.verbose)

        elif args.command == "eval":
            # Headless operation for evaluation
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            os.environ['SDL_AUDIODRIVER'] = 'dummy'
            
            # 1. Load genome from args.genome
            genome_path = Path(args.genome)
            if not genome_path.exists():
                raise FileNotFoundError(f"Genome file not found: {genome_path}")
                
            print(f"Loading genome from {genome_path}...")
            data = np.load(genome_path)
            if 'genome' not in data:
                raise KeyError("Key 'genome' not found in the npz file.")
            genome_data = data['genome']
            
            # 2. Create a BrainConfig and LSTMBrain.from_genome()
            brain_cfg = BrainConfig()
            brain = LSTMBrain.from_genome(genome_data, brain_cfg)
            
            # 3. Create a Driver2Evaluator(cfg)
            cfg = Driver2Config(
                car=args.car,
                brain=brain_cfg,
                ga=GAConfig(),
                curriculum=CurriculumConfig(),
                eval=EvalConfig()
            )
            evaluator = Driver2Evaluator(cfg)
            
            # 4. Call evaluator.evaluate()
            print(f"Evaluating for {args.laps} laps...")
            results = evaluator.evaluate(brain, n_laps=args.laps)
            
            # 5. Print results formatted nicely
            print("\n--- Evaluation Results ---")
            if isinstance(results, dict):
                for k, v in results.items():
                    print(f"{k}: {v}")
            else:
                print(results)
            print("--------------------------")

        elif args.command == "watch":
            from .viewer import main as viewer_main
            sys.argv = [sys.argv[0], "--run", args.run]
            viewer_main()

    # Removed try-except block to allow tracebacks to bubble up

if __name__ == "__main__":
    main()
