from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from .curriculum import TOPIC_BY_ID, TOPICS
from .models import LearnerSettings, MasteryScore, ProgressState


DEFAULT_DB = Path.home() / ".pythondojo" / "dojo.sqlite3"
LEGACY_PROGRESS = Path("/Users/REVIEW_USER/Documents/antigravity/fearless-hubble/progress.json")


class ProgressStore:
    def __init__(self, path: str | Path | None = None) -> None:
        override = os.environ.get("PYTHON_DOJO_DB")
        self._uses_default_path = path is None and override is None
        self.path = Path(path or override or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate_legacy_progress()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS topic_stats (
                    topic_id TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    correct INTEGER NOT NULL DEFAULT 0,
                    streak INTEGER NOT NULL DEFAULT 0,
                    mastery REAL NOT NULL DEFAULT 0,
                    unlocked INTEGER NOT NULL DEFAULT 0,
                    last_seen REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    exercise_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    correct INTEGER NOT NULL
                );
                """
            )
            for index, topic in enumerate(TOPICS):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO topic_stats(topic_id, unlocked)
                    VALUES (?, ?)
                    """,
                    (topic.id, 1 if index == 0 else 0),
                )
            conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1')")
            conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES ('ollama_host', ?)", ("http://127.0.0.1:11434",))
            conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES ('ollama_model', ?)", ("qwen2.5-coder:7b",))
            conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES ('use_ollama', 'true')")
            conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES ('interface', 'web')")

    def _migrate_legacy_progress(self) -> None:
        if not self._uses_default_path:
            return
        if not LEGACY_PROGRESS.exists():
            return
        with self._connect() as conn:
            migrated = conn.execute("SELECT value FROM meta WHERE key = 'legacy_progress_migrated'").fetchone()
            attempts = conn.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()["count"]
            if migrated or attempts:
                return
            try:
                data = json.loads(LEGACY_PROGRESS.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('legacy_progress_migrated', 'failed')")
                return
            solved = int(data.get("total_solved", 0) or 0)
            legacy_seen = data.get("seen_concepts", []) or []
            for index, topic in enumerate(TOPICS[: max(1, min(len(TOPICS), solved + 1))]):
                attempts_for_topic = 3 if index < solved else 0
                correct_for_topic = attempts_for_topic
                mastery = self._mastery(attempts_for_topic, correct_for_topic)
                conn.execute(
                    """
                    UPDATE topic_stats
                    SET attempts = ?, correct = ?, streak = ?, mastery = ?, unlocked = 1, last_seen = ?
                    WHERE topic_id = ?
                    """,
                    (attempts_for_topic, correct_for_topic, correct_for_topic, mastery, time.time(), topic.id),
                )
            note = json.dumps({"source": str(LEGACY_PROGRESS), "total_solved": solved, "seen_concepts": legacy_seen})
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('legacy_progress_migrated', ?)", (note,))

    def settings(self) -> LearnerSettings:
        with self._connect() as conn:
            values = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM settings")}
        return LearnerSettings(
            ollama_host=values.get("ollama_host", "http://127.0.0.1:11434"),
            ollama_model=values.get("ollama_model", "qwen2.5-coder:7b"),
            use_ollama=values.get("use_ollama", "true").lower() == "true",
            interface=values.get("interface", "web"),
        )

    def update_settings(self, **settings: Any) -> LearnerSettings:
        allowed = {"ollama_host", "ollama_model", "use_ollama", "interface"}
        with self._connect() as conn:
            for key, value in settings.items():
                if key not in allowed:
                    continue
                if isinstance(value, bool):
                    stored = "true" if value else "false"
                else:
                    stored = str(value)
                conn.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)", (key, stored))
        return self.settings()

    def record_attempt(self, exercise_id: str, topic_id: str, seed: int, mode: str, correct: bool) -> ProgressState:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM topic_stats WHERE topic_id = ?", (topic_id,)).fetchone()
            attempts = int(row["attempts"]) + 1
            correct_count = int(row["correct"]) + (1 if correct else 0)
            streak = int(row["streak"]) + 1 if correct else 0
            mastery = self._mastery(attempts, correct_count)
            conn.execute(
                """
                UPDATE topic_stats
                SET attempts = ?, correct = ?, streak = ?, mastery = ?, unlocked = 1, last_seen = ?
                WHERE topic_id = ?
                """,
                (attempts, correct_count, streak, mastery, now, topic_id),
            )
            conn.execute(
                "INSERT INTO attempts(created_at, exercise_id, topic_id, seed, mode, correct) VALUES (?, ?, ?, ?, ?, ?)",
                (now, exercise_id, topic_id, seed, mode, 1 if correct else 0),
            )
            self._unlock_available_topics(conn)
        return self.state()

    def state(self) -> ProgressState:
        with self._connect() as conn:
            rows = {row["topic_id"]: row for row in conn.execute("SELECT * FROM topic_stats")}
            totals = conn.execute("SELECT COUNT(*) AS total, COALESCE(SUM(correct), 0) AS correct FROM attempts").fetchone()
            latest = conn.execute("SELECT correct FROM attempts ORDER BY id DESC LIMIT 25").fetchall()
        mastery = []
        unlocked = []
        review = []
        for topic in TOPICS:
            row = rows[topic.id]
            score = MasteryScore(
                topic_id=topic.id,
                attempts=int(row["attempts"]),
                correct=int(row["correct"]),
                streak=int(row["streak"]),
                mastery=round(float(row["mastery"]), 3),
                unlocked=bool(row["unlocked"]),
                needs_review=bool(row["attempts"]) and float(row["mastery"]) < 0.7,
            )
            mastery.append(score)
            if score.unlocked:
                unlocked.append(topic.id)
            if score.needs_review:
                review.append(topic.id)
        streak = 0
        for row in latest:
            if row["correct"]:
                streak += 1
            else:
                break
        current = self._first_unmastered_unlocked(mastery)
        next_topic = self.next_topic_id(mastery)
        return ProgressState(
            total_attempts=int(totals["total"]),
            total_correct=int(totals["correct"]),
            streak=streak,
            current_topic_id=current,
            next_topic_id=next_topic,
            unlocked_topics=unlocked,
            review_queue=review[:5],
            mastery=mastery,
        )

    def next_topic_id(self, mastery: list[MasteryScore] | None = None) -> str:
        scores = mastery or self.state().mastery
        for score in scores:
            if score.unlocked and score.attempts > 0 and score.mastery < 0.5:
                return score.topic_id
        for score in scores:
            if score.unlocked and score.mastery < 0.82:
                return score.topic_id
        return scores[-1].topic_id

    def reset(self) -> ProgressState:
        with self._connect() as conn:
            conn.execute("DELETE FROM attempts")
            conn.execute("UPDATE topic_stats SET attempts = 0, correct = 0, streak = 0, mastery = 0, unlocked = 0, last_seen = 0")
            conn.execute("UPDATE topic_stats SET unlocked = 1 WHERE topic_id = ?", (TOPICS[0].id,))
        return self.state()

    def _first_unmastered_unlocked(self, mastery: list[MasteryScore]) -> str:
        for score in mastery:
            if score.unlocked and score.mastery < 0.82:
                return score.topic_id
        return mastery[-1].topic_id

    def _unlock_available_topics(self, conn: sqlite3.Connection) -> None:
        rows = {row["topic_id"]: row for row in conn.execute("SELECT * FROM topic_stats")}
        for index, topic in enumerate(TOPICS):
            if index == 0:
                continue
            previous = TOPICS[index - 1]
            previous_mastery = float(rows[previous.id]["mastery"])
            previous_correct = int(rows[previous.id]["correct"])
            previous_attempts = int(rows[previous.id]["attempts"])
            if previous_mastery >= 0.74 or previous_correct >= 3 or previous_attempts >= 5:
                conn.execute("UPDATE topic_stats SET unlocked = 1 WHERE topic_id = ?", (topic.id,))

    def _mastery(self, attempts: int, correct: int) -> float:
        if attempts <= 0:
            return 0.0
        accuracy = correct / attempts
        practice_depth = min(1.0, correct / 3)
        return max(0.0, min(1.0, accuracy * 0.7 + practice_depth * 0.3))


def topic_label(topic_id: str) -> str:
    return TOPIC_BY_ID[topic_id].label
