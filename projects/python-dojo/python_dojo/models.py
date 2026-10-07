from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Topic:
    id: str
    label: str
    stage: str
    order: int
    summary: str
    difficulty: int
    prerequisites: tuple[str, ...] = ()


@dataclass(frozen=True)
class Lesson:
    topic_id: str
    title: str
    body: str
    example: str
    pitfall: str = ""


@dataclass(frozen=True)
class CodeTest:
    expression: str
    expected: Any = None
    raises: str | None = None
    note: str = ""


@dataclass
class Exercise:
    id: str
    seed: int
    topic_id: str
    topic_label: str
    stage: str
    difficulty: int
    mode: str
    title: str
    prompt: str
    concept: str
    archetype: str = ""
    code: str = ""
    starter: str = ""
    choices: list[str] = field(default_factory=list)
    expected: Any = None
    accepted: list[str] = field(default_factory=list)
    tests: list[CodeTest] = field(default_factory=list)
    solution: str = ""
    solution_note: str = ""
    hints: list[str] = field(default_factory=list)
    wrong_answer: str = "__definitely_wrong__"

    def fingerprint(self) -> str:
        raw = "|".join(
            (
                self.topic_id,
                self.archetype,
                self.prompt,
                self.code,
                self.starter,
                repr(self.choices),
                repr(self.expected),
            )
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("expected", "accepted", "tests", "solution", "solution_note", "hints", "wrong_answer"):
            data.pop(key, None)
        return data


@dataclass(frozen=True)
class AttemptResult:
    correct: bool
    message: str
    details: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MasteryScore:
    topic_id: str
    attempts: int
    correct: int
    streak: int
    mastery: float
    unlocked: bool
    needs_review: bool


@dataclass(frozen=True)
class ProgressState:
    total_attempts: int
    total_correct: int
    streak: int
    current_topic_id: str
    next_topic_id: str
    unlocked_topics: list[str]
    review_queue: list[str]
    mastery: list[MasteryScore]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "total_correct": self.total_correct,
            "streak": self.streak,
            "current_topic_id": self.current_topic_id,
            "next_topic_id": self.next_topic_id,
            "unlocked_topics": self.unlocked_topics,
            "review_queue": self.review_queue,
            "mastery": [asdict(item) for item in self.mastery],
        }


@dataclass(frozen=True)
class LearnerSettings:
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    use_ollama: bool = True
    interface: str = "web"
