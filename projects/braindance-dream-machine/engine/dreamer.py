"""
Neural Audio Dream Machine
Generates unconditional audio hallucinations and latent space explorations
from trained RAVE models and pre-trained checkpoints.
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import soundfile as sf

class Dreamer:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.model_path = None
        self.sr = 48000
        self.latent_dim = 8
        self.hop_size = 2048

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Loads a TorchScript RAVE model or PyTorch checkpoint."""
        print(f"Loading neural audio model from: {model_path} onto {self.device}...")
        self.model_path = model_path
        
        # We run TorchScript models on CPU or MPS depending on kernel support
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
        except Exception as e:
            print(f"Loading on {self.device} failed, falling back to CPU: {e}")
            self.device = torch.device("cpu")
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()

        # Inspect model params if available
        if hasattr(self.model, "decode_params"):
            params = self.model.decode_params
            if isinstance(params, torch.Tensor) and len(params) >= 2:
                self.latent_dim = int(params[0].item())
                self.hop_size = int(params[1].item())
        
        # Check sample rate convention from filename or default
        if "44100" in model_path:
            self.sr = 44100
        else:
            self.sr = 48000

        print(f"Model loaded! Latent Dim: {self.latent_dim}, Hop: {self.hop_size}, SR: {self.sr} Hz")

    def dream(
        self,
        duration_sec: float = 8.0,
        temperature: float = 1.0,
        momentum: float = 0.92,
        glitch_probability: float = 0.15,
        stutter_len: int = 4,
        seed: Optional[int] = None,
        out_path: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Hallucinates a new audio track from scratch.
        - duration_sec: Length of generated track in seconds.
        - temperature: Energy / variance of latent space (0.5 = chill, 1.0 = normal, 1.8 = wild/noisy).
        - momentum: Latent trajectory memory (0.0 = white noise, 0.95 = smooth liquid evolution).
        - glitch_probability: Chance of Aphex-style micro-stutters and temporal repeats.
        - stutter_len: Number of latent frames repeated during a glitch burst.
        """
        if self.model is None:
            raise ValueError("No model loaded! Call load_model() first.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Number of latent steps
        total_samples = int(self.sr * duration_sec)
        num_latent_steps = max(1, total_samples // self.hop_size)

        with torch.no_grad():
            # Generate continuous latent trajectory (Ornstein-Uhlenbeck process)
            z_curr = torch.randn(1, self.latent_dim, 1, device=self.device) * temperature
            trajectory = [z_curr]

            in_glitch = 0
            glitch_frame = None

            for step in range(num_latent_steps - 1):
                if in_glitch > 0 and glitch_frame is not None:
                    # Stutter loop
                    trajectory.append(glitch_frame)
                    in_glitch -= 1
                    continue

                # Check if we should trigger an Aphex micro-glitch
                if np.random.rand() < glitch_probability:
                    in_glitch = stutter_len
                    # Either repeat current or pitch-skewed noise
                    glitch_frame = z_curr.clone()
                    if np.random.rand() < 0.5:
                        glitch_frame = glitch_frame * (1.5 + np.random.rand())
                    trajectory.append(glitch_frame)
                    continue

                # Normal smooth latent evolution
                innovation = torch.randn(1, self.latent_dim, 1, device=self.device) * temperature
                sigma = np.sqrt(max(0.01, 1.0 - momentum ** 2))
                z_curr = momentum * z_curr + sigma * innovation
                trajectory.append(z_curr)

            z_full = torch.cat(trajectory, dim=-1)

            # Decode latent vector to raw audio
            audio_tensor = self.model.decode(z_full)
            audio = audio_tensor.squeeze().cpu().numpy()

            # Normalize audio with headroom
            max_amp = np.max(np.abs(audio))
            if max_amp > 1e-4:
                audio = (audio / max_amp) * 0.88

            # Save to disk
            if out_path is None:
                os.makedirs("dreams", exist_ok=True)
                timestamp = int(time.time())
                out_path = f"dreams/dream_{timestamp}.wav"

            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, audio, self.sr)
            print(f"Hallucinated track saved to: {out_path} ({len(audio)/self.sr:.2f}s)")

            return audio, out_path

if __name__ == "__main__":
    import glob
    # Test with cached model if available
    cached_models = glob.glob("/Users/REVIEW_USER/.cache/huggingface/hub/**/magnets*.ts", recursive=True)
    if cached_models:
        dreamer = Dreamer(cached_models[0])
        dreamer.dream(duration_sec=6.0, temperature=1.2, glitch_probability=0.2)
