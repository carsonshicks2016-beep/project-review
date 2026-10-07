"""
Training Pipeline for RAVE (VAE & Prior Models)
Runs training on preprocessed audio datasets with MPS (Metal) GPU acceleration on macOS.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

class Trainer:
    def __init__(self, db_path: str = "data/processed/rave_db", run_dir: str = "runs"):
        self.db_path = Path(db_path)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def train_vae(
        self,
        run_name: str = "aphex_bukem_run",
        config: str = "v2",
        batch_size: int = 4,
        max_steps: int = 50000,
        val_every: int = 2500,
        workers: int = 4
    ) -> subprocess.Popen:
        """
        Launches RAVE VAE training process.
        """
        if not (self.db_path / "data.mdb").exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}. Run preprocessing first!")

        run_name = run_name.strip().replace(" ", "_")
        cmd = [
            "rave", "train",
            "--config", config,
            "--db_path", str(self.db_path),
            "--name", run_name,
            "--channels", "1",
            "--batch", str(batch_size),
            "--max_steps", str(max_steps),
            "--val_every", str(val_every)
        ]

        print("Starting RAVE VAE training with command:")
        print(" ".join(cmd))
        process = subprocess.Popen(cmd)
        return process

    def train_prior(
        self,
        model_checkpoint_dir: str,
        run_name: str = "aphex_bukem_prior",
        batch_size: int = 8,
        max_steps: int = 50000,
        workers: int = 4
    ) -> subprocess.Popen:
        """
        Launches RAVE Prior training on latent space.
        """
        cmd = [
            "rave", "train_prior",
            "--model", model_checkpoint_dir,
            "--db_path", str(self.db_path),
            "--name", run_name,
            "--batch", str(batch_size),
            "--max_steps", str(max_steps),
            "--workers", str(workers),
            "--out_path", str(self.run_dir)
        ]

        print("Starting RAVE Prior training with command:")
        print(" ".join(cmd))
        process = subprocess.Popen(cmd)
        return process

    def export_model(self, run_dir_path: str, out_name: str = "aphex_bukem_model.ts") -> str:
        """
        Exports a trained checkpoint into a standalone TorchScript model.
        """
        out_dir = Path("models/exported")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name

        cmd = [
            "rave", "export",
            "--run", run_dir_path,
            "--output", str(out_path)
        ]

        print(f"Exporting model from {run_dir_path} to {out_path}...")
        subprocess.run(cmd, check=True)
        print("Export complete!")
        return str(out_path)

if __name__ == "__main__":
    trainer = Trainer()
    print("Trainer module ready!")
