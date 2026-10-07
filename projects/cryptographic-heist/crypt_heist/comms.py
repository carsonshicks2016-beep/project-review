"""Discrete English-word radio channel, scanner, and jamming events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VOCABULARY = [
    "alpha", "bravo", "delta", "echo", "falcon", "iron", "river", "switch",
    "north", "south", "east", "west", "close", "wide", "fast", "slow",
    "box", "gate", "net", "hook", "shadow", "mirror", "cut", "hold",
    "left", "right", "center", "front", "rear", "seal", "open", "clear",
    "red", "blue", "white", "black", "orange", "violet", "silver", "green",
    "tower", "market", "bridge", "tunnel", "harbor", "station", "plaza",
    "vector", "orbit", "pulse", "anchor", "needle", "cipher", "mask",
    "ghost", "spark", "glass", "marble", "ember", "signal", "phase",
    "crown", "lattice", "keystone", "ramp", "lock", "wheel", "arc",
]
TOKEN_TO_WORD = {i: word for i, word in enumerate(VOCABULARY)}
WORD_TO_TOKEN = {word: i for i, word in TOKEN_TO_WORD.items()}

ROLE_WORDS = {
    "front": ["front", "gate", "anchor"],
    "left": ["left", "hook", "wide"],
    "right": ["right", "hook", "wide"],
    "rear": ["rear", "shadow", "mirror"],
    "cut": ["cut", "switch", "net"],
}

AUTH_WORDS = [
    ("alpha", "seal"),
    ("bravo", "lock"),
    ("delta", "crown"),
    ("echo", "lattice"),
]


@dataclass
class RadioEvent:
    time: float
    speaker: str
    word: str
    spoofed: bool = False
    meaning: str = ""


class CommsChannel:
    """Shared radio transcript with spectator-only spoof labels."""

    def __init__(self, rng: np.random.Generator | None = None, max_events: int = 180):
        self.rng = rng or np.random.default_rng()
        self.max_events = max_events
        self.events: list[RadioEvent] = []
        self.cipher_epoch = 0
        self.cipher_timer = 0.0
        self.confidence = 0.35
        self.jamming_budget = 7
        self.last_cipher_change = 0.0
        self.last_jam_time = -999.0
        self.last_spoof_words: list[str] = []

    def tick(self, dt: float, now: float):
        self.cipher_timer += dt
        if self.cipher_timer >= 18.0:
            self.cipher_timer = 0.0
            self.cipher_epoch = (self.cipher_epoch + 1) % len(AUTH_WORDS)
            self.last_cipher_change = now
            self.confidence = max(0.18, self.confidence * 0.42)
            self._append(now, "dispatch", "cipher", False, "pursuers rotated auth phrase")
            self._append(now, "dispatch", AUTH_WORDS[self.cipher_epoch][0], False, "new auth prefix")

    def broadcast_pursuer(self, now: float, pursuer_id: int, role: str, target: tuple[float, float]):
        role_words = ROLE_WORDS.get(role, ["vector", "hold", "net"])
        auth = AUTH_WORDS[self.cipher_epoch]
        x, y = target
        quadrant = ("east" if x >= 0 else "west", "south" if y >= 0 else "north")
        words = [
            auth[0],
            role_words[0],
            quadrant[0],
            quadrant[1],
            role_words[1],
            auth[1],
        ]
        for w in words:
            self._append(now, f"P{pursuer_id + 1}", w, False, role)
        self.confidence = min(0.98, self.confidence + 0.018)

    def broadcast_tokens(self, now: float, speaker: str, token_ids, spoofed: bool = False, meaning: str = "learned"):
        for token_id in list(token_ids)[:8]:
            word = TOKEN_TO_WORD.get(int(token_id) % len(VOCABULARY), "alpha")
            self._append(now, speaker, word, spoofed, meaning)
        if not spoofed:
            self.confidence = min(0.98, self.confidence + 0.008)

    def inject_jam(self, now: float, payload=None) -> bool:
        if self.jamming_budget <= 0:
            return False
        self.jamming_budget -= 1
        if payload is None:
            payload = list(self.rng.choice(
                ["alpha", "gate", "east", "switch", "ghost", "hold", "west", "seal", "cut"],
                size=5,
                replace=False,
            ))
        else:
            payload = [TOKEN_TO_WORD.get(int(x) % len(VOCABULARY), str(x)) for x in payload][:5]
        self.last_jam_time = now
        self.last_spoof_words = [str(w) for w in payload]
        for w in payload:
            self._append(now, "EVADER", w, True, "spoofed command")
        self.confidence = max(0.08, self.confidence - 0.22)
        return True

    def recent(self, n: int = 48) -> list[RadioEvent]:
        return self.events[-n:]

    def _append(self, now: float, speaker: str, word: str, spoofed: bool, meaning: str):
        self.events.append(RadioEvent(now, speaker, str(word), spoofed, meaning))
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]
