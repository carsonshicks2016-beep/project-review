from __future__ import annotations

from python_dojo.generators import generate_exercise
from python_dojo.progress import ProgressStore


def test_progress_unlocks_and_recommends(tmp_path):
    store = ProgressStore(tmp_path / "dojo.sqlite3")
    state = store.state()
    assert state.next_topic_id == "print_output"
    exercise = generate_exercise("print_output", seed=1)
    for _ in range(3):
        state = store.record_attempt(exercise.id, exercise.topic_id, exercise.seed, exercise.mode, True)
    assert "numbers" in state.unlocked_topics
    assert state.total_correct == 3


def test_reset_progress(tmp_path):
    store = ProgressStore(tmp_path / "dojo.sqlite3")
    exercise = generate_exercise("print_output", seed=2)
    store.record_attempt(exercise.id, exercise.topic_id, exercise.seed, exercise.mode, True)
    state = store.reset()
    assert state.total_attempts == 0
    assert state.unlocked_topics == ["print_output"]
