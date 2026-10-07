"""
neat_trader.py — NEAT evolution loop for the trading agent.

This is directly analogous to your Mario TAS evolve.py:
    
    Mario:                          Trading:
    ─────                           ───────
    env = retro.make(game)     →    env = TradingEnv(states, prices)
    obs = env.reset()          →    obs = env.reset()
    action = nn.activate(obs)  →    action = nn.activate(obs)
    obs, reward = env.step()   →    obs, reward = env.step()
    genome.fitness = x_pos     →    genome.fitness = total_return

The NEAT algorithm evolves the network topology and weights to maximize
portfolio returns, exactly like it evolved Mario networks to maximize
rightward progress.
"""

import neat
import numpy as np
import pickle
import os
from pathlib import Path
from env.trading_env import TradingEnv


class NEATTrader:
    """
    NEAT-based trading agent with evolution loop.
    
    Supports multi-ticker training: each genome is evaluated across
    all provided assets and receives the average fitness. This forces
    the neural network to learn universal trading patterns rather than
    memorizing a single chart.
    """
    
    def __init__(
        self,
        state_matrix: np.ndarray = None,
        prices: np.ndarray = None,
        dates: list = None,
        config_path: str = None,
        checkpoint_dir: str = None,
        initial_balance: float = 10000.0,
    ):
        """
        Args:
            state_matrix: (T, features) from state_builder (single-ticker legacy)
            prices: (T,) close prices (single-ticker legacy)
            dates: Date strings
            config_path: Path to NEAT config file
            checkpoint_dir: Where to save checkpoints
            initial_balance: Starting balance for each genome
        """
        # Multi-ticker datasets: {ticker: (state_matrix, prices, dates)}
        self.datasets = {}
        
        # Legacy single-ticker support
        self.state_matrix = state_matrix
        self.prices = prices
        self.dates = dates
        
        if state_matrix is not None and prices is not None:
            self.datasets["default"] = (state_matrix, prices, dates)
        
        self.initial_balance = initial_balance
        
        # Config
        if config_path is None:
            config_path = str(Path(__file__).parent / "config-trader")
        self.config_path = config_path
        
        # Checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = str(Path(__file__).parent.parent / "checkpoints")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Tracking
        self.best_genome = None
        self.best_fitness = float("-inf")
        self.generation_stats = []
    
    def add_dataset(self, ticker: str, state_matrix: np.ndarray,
                    prices: np.ndarray, dates: list = None):
        """Add a ticker dataset for multi-asset training."""
        self.datasets[ticker] = (state_matrix, prices, dates)
        # Keep legacy pointers updated to first dataset
        if self.state_matrix is None:
            self.state_matrix = state_matrix
            self.prices = prices
            self.dates = dates
    
    def _score_single(self, net, state_matrix, prices, dates) -> tuple:
        """
        Run a neural network on a single asset and return (fitness, metrics, env).
        """
        env = TradingEnv(
            state_matrix, prices, dates,
            initial_balance=self.initial_balance,
        )
        obs, _ = env.reset()
        
        for step in range(len(prices) - 1):
            output = net.activate(obs.tolist())
            action = int(np.argmax(output))
            obs, reward, done, _, info = env.step(action)
            if done:
                break
        
        metrics = env.get_final_metrics()
        
        # Multi-objective fitness:
        fitness = metrics["total_return_pct"]
        fitness += min(20, max(0, metrics["sharpe_ratio"] * 5))  # Sharpe bonus
        fitness -= metrics["max_drawdown_pct"] * 0.3              # Drawdown penalty
        if metrics["total_trades"] < 3:
            fitness -= 50  # Must trade
        
        return fitness, metrics, env
    
    def eval_genome(self, genome, config) -> float:
        """
        Evaluate a single genome across ALL datasets (multi-ticker).
        
        The genome's fitness is the AVERAGE across all assets.
        This forces the network to learn universal patterns.
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        if not self.datasets:
            raise ValueError("No datasets loaded! Call add_dataset() or pass data to __init__.")
        
        fitnesses = []
        for ticker, (states, prices, dates) in self.datasets.items():
            fitness, metrics, env = self._score_single(net, states, prices, dates)
            fitnesses.append(fitness)
        
        # Average fitness across all assets (forces generalization)
        return float(np.mean(fitnesses))
    
    def eval_genomes(self, genomes, config):
        """
        Evaluate all genomes in a generation across all tickers.
        """
        gen_fitnesses = []
        
        for genome_id, genome in genomes:
            fitness = self.eval_genome(genome, config)
            genome.fitness = fitness
            gen_fitnesses.append(fitness)
            
            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = genome
        
        # Generation stats
        num_tickers = len(self.datasets)
        ticker_names = list(self.datasets.keys())
        stats = {
            "best": max(gen_fitnesses),
            "avg": np.mean(gen_fitnesses),
            "worst": min(gen_fitnesses),
            "std": np.std(gen_fitnesses),
        }
        self.generation_stats.append(stats)
        
        gen_num = len(self.generation_stats)
        ticker_str = f" ({num_tickers} assets)" if num_tickers > 1 else ""
        print(f"  Gen {gen_num:3d}{ticker_str} │ "
              f"Best: {stats['best']:+8.2f} │ "
              f"Avg: {stats['avg']:+8.2f} │ "
              f"Worst: {stats['worst']:+8.2f} │ "
              f"All-Time Best: {self.best_fitness:+8.2f}")
    
    def evolve(self, num_generations: int = 100, resume_from: str = None) -> neat.DefaultGenome:
        """
        Run the NEAT evolution loop.
        
        Args:
            num_generations: Number of generations to evolve
            resume_from: Path to checkpoint file to resume from
            
        Returns:
            The best genome found
        """
        # Load NEAT config
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path,
        )
        
        # Create or restore population
        if resume_from and os.path.exists(resume_from):
            print(f"📂 Resuming from checkpoint: {resume_from}")
            pop = neat.Checkpointer.restore_checkpoint(resume_from)
        else:
            pop = neat.Population(config)
        
        # Add reporters
        pop.add_reporter(neat.StdOutReporter(False))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        
        # Save checkpoints every 10 generations
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "neat-checkpoint-")
        pop.add_reporter(neat.Checkpointer(
            generation_interval=10,
            filename_prefix=checkpoint_prefix
        ))
        
        # ── EVOLVE ──
        print(f"\n{'='*70}")
        print(f"🧬 NEAT EVOLUTION — {num_generations} Generations")
        print(f"   Population: {config.pop_size}")
        print(f"   Inputs: {config.genome_config.num_inputs}")
        print(f"   Outputs: {config.genome_config.num_outputs}")
        print(f"   Data: {len(self.prices)} candles")
        print(f"   Balance: ${self.initial_balance:,.2f}")
        print(f"{'='*70}\n")
        
        winner = pop.run(self.eval_genomes, num_generations)
        
        # Save the best genome
        best_path = os.path.join(self.checkpoint_dir, "best_genome.pkl")
        with open(best_path, "wb") as f:
            pickle.dump(winner, f)
        print(f"\n💾 Best genome saved to {best_path}")
        
        return winner
    
    def run_best(self, genome=None, config_path=None) -> dict:
        """
        Run the best genome and return detailed results.
        
        Args:
            genome: Genome to evaluate (defaults to self.best_genome)
            config_path: Path to config (defaults to self.config_path)
            
        Returns:
            Dict with metrics, trade log, and portfolio history
        """
        if genome is None:
            genome = self.best_genome
        if genome is None:
            raise ValueError("No genome to evaluate! Run evolve() first.")
        
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path or self.config_path,
        )
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = TradingEnv(
            self.state_matrix,
            self.prices,
            self.dates,
            initial_balance=self.initial_balance,
        )
        
        obs, _ = env.reset()
        
        for step in range(len(self.prices) - 1):
            output = net.activate(obs.tolist())
            action = int(np.argmax(output))
            obs, reward, done, _, info = env.step(action)
            if done:
                break
        
        metrics = env.get_final_metrics()
        
        return {
            "metrics": metrics,
            "trade_log": env.trade_log,
            "portfolio_history": env.portfolio_history,
            "action_history": env.action_history,
            "generation_stats": self.generation_stats,
        }


def load_best_genome(checkpoint_dir: str = None) -> tuple:
    """Load the best genome from a previous training run."""
    if checkpoint_dir is None:
        checkpoint_dir = str(Path(__file__).parent.parent / "checkpoints")
    
    path = os.path.join(checkpoint_dir, "best_genome.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved genome at {path}")
    
    with open(path, "rb") as f:
        genome = pickle.load(f)
    
    print(f"📂 Loaded best genome from {path}")
    return genome
