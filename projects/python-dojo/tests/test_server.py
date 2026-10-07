from __future__ import annotations

from python_dojo.progress import ProgressStore
from python_dojo.server import create_app


def test_api_flow_generates_checks_and_persists_progress(tmp_path):
    app = create_app(ProgressStore(tmp_path / "dojo.sqlite3"))
    client = app.test_client()
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["ok"] is True

    curriculum = client.get("/api/curriculum")
    assert curriculum.status_code == 200
    assert curriculum.get_json()["topic_count"] >= 30

    generated = client.post("/api/exercise", json={"topic_id": "numbers", "seed": 5})
    assert generated.status_code == 200
    exercise = generated.get_json()
    assert "expected" not in exercise
    assert "tests" not in exercise

    wrong = client.post("/api/check", json={"id": exercise["id"], "answer": "not a number"})
    assert wrong.status_code == 200
    assert wrong.get_json()["correct"] is False

    fresh_progress = client.get("/api/progress").get_json()
    assert fresh_progress["total_attempts"] == 1


def test_expired_exercise_can_be_regenerated_from_id(tmp_path):
    app = create_app(ProgressStore(tmp_path / "dojo.sqlite3"))
    client = app.test_client()
    generated = client.post("/api/exercise", json={"topic_id": "numbers", "seed": 11}).get_json()
    # A second app simulates a server restart; the deterministic id should still recover the exercise.
    restarted = create_app(ProgressStore(tmp_path / "dojo.sqlite3")).test_client()
    checked = restarted.post("/api/check", json={"id": generated["id"], "answer": "999"})
    assert checked.status_code == 200
    assert checked.get_json()["correct"] is False
