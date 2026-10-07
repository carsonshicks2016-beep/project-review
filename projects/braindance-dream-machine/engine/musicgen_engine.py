"""
MusicGen Autoregressive Neural Music Engine
Generates actual structured musical compositions (drums, chords, basslines, melodies)
using Meta's MusicGen Transformer architecture with Apple Silicon Metal acceleration.
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple
import torch
import soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor

class MusicGenComposer:
    def __init__(self, model_id: str = "facebook/musicgen-small", device: Optional[str] = None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading MusicGen ({model_id}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.sampling_rate = self.model.config.audio_encoder.sampling_rate
        print(f"MusicGen loaded! SR: {self.sampling_rate} Hz")

    def compose(
        self,
        prompt: str,
        duration_sec: float = 10.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        out_path: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Generates full musical audio from text prompt.
        50 tokens = ~1 second of audio.
        """
        tokens_to_generate = int(duration_sec * 50)

        t0 = time.time()
        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=tokens_to_generate,
                temperature=temperature,
                guidance_scale=guidance_scale,
                do_sample=True
            )

        elapsed = time.time() - t0
        audio = audio_values[0, 0].cpu().numpy()

        if out_path is None:
            os.makedirs("dreams", exist_ok=True)
            timestamp = int(time.time())
            out_path = f"dreams/musicgen_{timestamp}.wav"

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, self.sampling_rate)
        print(f"Musical composition generated in {elapsed:.2f}s -> {out_path}")

        return out_path, elapsed

if __name__ == "__main__":
    composer = MusicGenComposer()
    out, el = composer.compose(
        "165 bpm atmospheric drum and bass, lush ambient rhodes chords, deep sub bass, rolling amen breakbeats",
        duration_sec=6.0
    )
    print("Test finished:", out)
