from __future__ import annotations

import random


NAMES = ["Ada", "Lin", "Tao", "Bea", "Kai", "Mia", "Rex", "Zoe", "Ola", "Ivy", "Sam", "Ned", "Uma", "Gil"]
WORDS = ["python", "variable", "function", "iterator", "package", "module", "keyword", "lambda", "socket", "kernel"]
SIMPLE_WORDS = ["dojo", "python", "spark", "loop", "logic", "quest", "focus", "byte", "shell", "stack"]
FRUITS = ["apple", "banana", "cherry", "mango", "kiwi", "plum", "pear", "fig", "grape", "peach"]
COLORS = ["red", "blue", "green", "amber", "violet", "teal", "coral", "gray"]
ANIMALS = ["otter", "falcon", "gecko", "panda", "lynx", "heron", "moose", "viper"]
CITIES = ["oslo", "kyoto", "lima", "cairo", "perth", "quito", "turin", "salem"]
SKUS = ["A1", "B2", "C3", "D4", "E5", "F6"]


def pick(rng: random.Random, pool: list[str], k: int) -> list[str]:
    """k distinct items from a pool, order randomized."""
    return rng.sample(pool, k)


def money(rng: random.Random, low: int = 1, high: int = 20) -> float:
    return rng.randint(low * 4, high * 4) / 4
