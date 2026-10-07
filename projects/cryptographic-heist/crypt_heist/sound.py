"""Small confidence-driven procedural soundtrack for the spectator app."""

from __future__ import annotations

import numpy as np


SOUNDTRACK_BUCKETS = {
    "low": {
        "label": "dissonant jazz",
        "description": "low confidence: dissonant jazz arrangement",
        "frequencies": [92.0, 119.0, 151.0, 233.0],
        "waveform": "noisy",
    },
    "mid": {
        "label": "hybrid tension",
        "description": "mid confidence: hybrid tension arrangement",
        "frequencies": [110.0, 146.83, 174.61, 220.0],
        "waveform": "sine",
    },
    "high": {
        "label": "structured classical",
        "description": "high confidence: structured classical arrangement",
        "frequencies": [130.81, 164.81, 196.00, 261.63],
        "waveform": "gated_sine",
    },
}


def confidence_bucket(confidence: float) -> str:
    value = float(np.clip(confidence, 0.0, 1.0))
    if value < 0.3:
        return "low"
    if value > 0.7:
        return "high"
    return "mid"


def soundtrack_plan(confidence: float) -> dict:
    bucket = confidence_bucket(confidence)
    plan = dict(SOUNDTRACK_BUCKETS[bucket])
    plan["bucket"] = bucket
    plan["confidence"] = float(np.clip(confidence, 0.0, 1.0))
    return plan


class HarmonicTensionSoundtrack:
    def __init__(self, pygame, enabled: bool = True):
        self.pygame = pygame
        self.enabled = enabled
        self.channel = None
        self.last_bucket = None
        self.timer = 0.0
        if not enabled:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self.channel = pygame.mixer.Channel(1)
        except Exception:
            self.enabled = False

    def set_enabled(self, enabled: bool):
        self.enabled = enabled and self.channel is not None
        if not self.enabled and self.channel is not None:
            self.channel.stop()

    def update(self, confidence: float, dt: float):
        if not self.enabled or self.channel is None:
            return
        self.timer -= dt
        bucket = confidence_bucket(confidence)
        if self.timer > 0.0 and bucket == self.last_bucket and self.channel.get_busy():
            return
        self.last_bucket = bucket
        self.timer = 0.42
        sound = self._make_sound(bucket, confidence)
        self.channel.play(sound)

    def _make_sound(self, bucket: str, confidence: float):
        sr = 44100
        dur = 0.42
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
        if bucket == "low":
            freqs = SOUNDTRACK_BUCKETS["low"]["frequencies"]
            wave = sum(np.sin(2 * np.pi * f * t) for f in freqs)
            wave += 0.20 * np.random.default_rng().normal(size=t.shape)
            amp = 0.10
        elif bucket == "high":
            freqs = SOUNDTRACK_BUCKETS["high"]["frequencies"]
            gate = (np.sin(2 * np.pi * 3.0 * t) > -0.2).astype(float)
            wave = sum(np.sin(2 * np.pi * f * t) for f in freqs) * (0.55 + 0.45 * gate)
            amp = 0.085
        else:
            freqs = SOUNDTRACK_BUCKETS["mid"]["frequencies"]
            wave = sum(np.sin(2 * np.pi * f * t) for f in freqs)
            amp = 0.075
        env = np.minimum(1.0, t / 0.04) * np.minimum(1.0, (dur - t) / 0.08)
        data = np.asarray(wave * env * amp * (0.7 + 0.3 * confidence), dtype=np.float32)
        data = np.clip(data, -1.0, 1.0)
        return self.pygame.sndarray.make_sound((data * 32767).astype(np.int16))
