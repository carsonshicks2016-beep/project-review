"""
Neural Timbre Transfer & Latent Morphing Engine
Transforms any external audio into the learned sonic vocabulary of the RAVE model,
or morphs input rhythm/harmony with generative dream states.
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import soundfile as sf
import librosa

class TimbreTransformer:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_path
        self.sr = 48000 if "48000" in model_path else 44100
        
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
        except Exception as e:
            print(f"Loading on {self.device} failed, falling back to CPU: {e}")
            self.device = torch.device("cpu")
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()

        self.latent_dim = 8
        self.hop_size = 2048
        if hasattr(self.model, "decode_params"):
            params = self.model.decode_params
            if isinstance(params, torch.Tensor) and len(params) >= 2:
                self.latent_dim = int(params[0].item())
                self.hop_size = int(params[1].item())

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Loads and formats audio to 1-channel at target SR."""
        y, orig_sr = sf.read(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1) # Mono
        if orig_sr != self.sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
        
        # Pad to multiple of hop_size
        pad_len = (self.hop_size - (len(y) % self.hop_size)) % self.hop_size
        if pad_len > 0:
            y = np.pad(y, (0, pad_len))

        tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        return tensor

    def transform(
        self,
        audio_path: str,
        dream_blend: float = 0.0,
        temperature: float = 1.0,
        latent_boost: float = 1.0,
        out_path: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Re-synthesizes audio through the model.
        - dream_blend: 0.0 = pure timbre transfer; 1.0 = completely dissolved into generative dream.
        - temperature: amount of random variation added to latent features.
        - latent_boost: scale factor on latent activations.
        """
        x = self.load_audio(audio_path)

        with torch.no_grad():
            z = self.model.encode(x) * latent_boost

            if dream_blend > 0.0 or temperature != 1.0:
                # Generate random dream trajectory matching length
                dream_noise = torch.randn_like(z) * temperature
                z = (1.0 - dream_blend) * z + dream_blend * dream_noise

            # Decode
            y_tensor = self.model.decode(z)
            y = y_tensor.squeeze().cpu().numpy()

            # Normalize
            max_amp = np.max(np.abs(y))
            if max_amp > 1e-4:
                y = (y / max_amp) * 0.88

            if out_path is None:
                os.makedirs("dreams", exist_ok=True)
                timestamp = int(time.time())
                out_path = f"dreams/morphed_{timestamp}.wav"

            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, y, self.sr)
            print(f"Morphed audio saved to: {out_path}")

            return y, out_path

if __name__ == "__main__":
    import glob
    cached_models = glob.glob("/Users/REVIEW_USER/.cache/huggingface/hub/**/magnets*.ts", recursive=True)
    if cached_models and os.path.exists("data/demo/aphex_braindance_breakbeat.wav"):
        transformer = TimbreTransformer(cached_models[0])
        transformer.transform(
            "data/demo/aphex_braindance_breakbeat.wav",
            dream_blend=0.35,
            temperature=1.1
        )
