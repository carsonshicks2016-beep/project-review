from __future__ import annotations

import random
import uuid
from typing import Callable

from ..curriculum import TOPIC_BY_ID, TOPICS
from ..models import Exercise


Generator = Callable[[random.Random, int], Exercise]
_ARCHETYPES: dict[str, dict[str, Generator]] = {}


def archetype(topic_id: str, name: str) -> Callable[[Generator], Generator]:
    """Register one problem form for a topic. A topic can hold many forms."""

    def decorator(func: Generator) -> Generator:
        def wrapped(rng: random.Random, seed: int) -> Exercise:
            exercise = func(rng, seed)
            exercise.archetype = name
            return exercise

        wrapped.__name__ = func.__name__
        _ARCHETYPES.setdefault(topic_id, {})[name] = wrapped
        return wrapped

    return decorator


def _id(topic_id: str, seed: int) -> str:
    stable = uuid.uuid5(uuid.NAMESPACE_DNS, f"python-dojo:{topic_id}:{seed}").hex[:10]
    return f"pydojo-{topic_id}-{seed}-{stable}"


def parse_exercise_id(exercise_id: str) -> tuple[str, int] | None:
    if not exercise_id.startswith("pydojo-"):
        return None
    body = exercise_id.removeprefix("pydojo-")
    parts = body.rsplit("-", 2)
    if len(parts) != 3:
        return None
    topic_id, seed_text, _fingerprint = parts
    try:
        return topic_id, int(seed_text)
    except ValueError:
        return None


def make_exercise(topic_id: str, seed: int, mode: str, title: str, prompt: str, concept: str, **kwargs: object) -> Exercise:
    topic = TOPIC_BY_ID[topic_id]
    return Exercise(
        id=_id(topic_id, seed),
        seed=seed,
        topic_id=topic.id,
        topic_label=topic.label,
        stage=topic.stage,
        difficulty=topic.difficulty,
        mode=mode,
        title=title,
        prompt=prompt,
        concept=concept,
        **kwargs,
    )


def generate_exercise(
    topic_id: str | None = None,
    seed: int | None = None,
    stage: str | None = None,
    archetype_name: str | None = None,
) -> Exercise:
    """Deterministically build an exercise: the same (topic, seed) always
    reproduces the same problem, including which archetype it uses."""
    if seed is None:
        seed = random.randint(1, 2_000_000_000)
    rng = random.Random(seed)
    if not topic_id:
        candidates = [topic for topic in TOPICS if topic.id in _ARCHETYPES and (stage is None or topic.stage == stage)]
        if not candidates:
            raise ValueError("No exercise generators match the requested filters.")
        topic_id = rng.choice(candidates).id
    forms = _ARCHETYPES.get(topic_id)
    if not forms:
        raise ValueError(f"No generator exists for topic {topic_id!r}.")
    if archetype_name is None:
        archetype_name = rng.choice(sorted(forms))
    elif archetype_name not in forms:
        raise ValueError(f"Topic {topic_id!r} has no archetype {archetype_name!r}.")
    return forms[archetype_name](rng, seed)


def pick_exercise(
    topic_id: str,
    seen_fingerprints: dict[str, float] | None = None,
    archetype_last_seen: dict[str, float] | None = None,
    tries: int = 24,
) -> Exercise:
    """Generate several candidates and keep the most novel one: an unseen
    problem fingerprint first, then the archetype practiced longest ago."""
    seen = seen_fingerprints or {}
    arch_seen = archetype_last_seen or {}
    best: Exercise | None = None
    best_key: tuple[int, float, float] | None = None
    for _ in range(max(1, tries)):
        exercise = generate_exercise(topic_id=topic_id)
        fingerprint = exercise.fingerprint()
        key = (
            1 if fingerprint in seen else 0,
            arch_seen.get(exercise.archetype, 0.0),
            seen.get(fingerprint, 0.0),
        )
        if best is None or key < best_key:
            best, best_key = exercise, key
        if key == (0, 0.0, 0.0):
            break
    assert best is not None
    return best


def all_generator_topic_ids() -> list[str]:
    return [topic.id for topic in TOPICS if topic.id in _ARCHETYPES]


def archetype_names(topic_id: str) -> list[str]:
    return sorted(_ARCHETYPES.get(topic_id, {}))


def archetype_count(topic_id: str) -> int:
    return len(_ARCHETYPES.get(topic_id, {}))


def total_archetypes() -> int:
    return sum(len(forms) for forms in _ARCHETYPES.values())
