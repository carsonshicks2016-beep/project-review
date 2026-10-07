from __future__ import annotations

from python_dojo.terminal import collect_answer


class DummyExercise:
    mode = "choice"
    choices = ["alpha", "beta"]


def test_choice_collection_accepts_number(monkeypatch):
    monkeypatch.setattr("python_dojo.terminal.prompt", lambda *_args, **_kwargs: "2")
    assert collect_answer(DummyExercise()) == "beta"
