from __future__ import annotations

import argparse
import random
import threading
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory

from .curriculum import curriculum_payload
from .generators import generate_exercise, parse_exercise_id
from .grader import check_answer
from .progress import ProgressStore
from .tutor import TutorCoach


STATIC = Path(__file__).resolve().parent / "static"
_lock = threading.Lock()
_exercises: dict[str, Any] = {}


def create_app(store: ProgressStore | None = None) -> Flask:
    progress_store = store or ProgressStore()
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="")

    @app.get("/")
    def index() -> Response:
        return Response((STATIC / "index.html").read_text(encoding="utf-8"), mimetype="text/html")

    @app.get("/api/health")
    def api_health():
        coach = TutorCoach(progress_store.settings())
        return jsonify(
            {
                "ok": True,
                "stored_exercises": len(_exercises),
                "db_path": str(progress_store.path),
                "ollama_available": coach.ollama_available(),
            }
        )

    @app.get("/api/curriculum")
    def api_curriculum():
        return jsonify(curriculum_payload())

    @app.get("/api/progress")
    def api_progress():
        return jsonify(progress_store.state().as_dict())

    @app.post("/api/exercise")
    def api_exercise():
        payload: dict[str, Any] = request.get_json(force=True) or {}
        topic_id = payload.get("topic_id") or progress_store.state().next_topic_id
        stage = payload.get("stage") or None
        seed = payload.get("seed")
        if seed in ("", None):
            seed = random.randint(1, 2_000_000_000)
        try:
            exercise = generate_exercise(topic_id=topic_id, stage=stage, seed=int(seed))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        with _lock:
            _exercises[exercise.id] = exercise
            if len(_exercises) > 500:
                for key in list(_exercises)[:100]:
                    _exercises.pop(key, None)
        return jsonify(exercise.public_dict())

    @app.post("/api/check")
    def api_check():
        payload: dict[str, Any] = request.get_json(force=True) or {}
        exercise_id = str(payload.get("id", ""))
        answer = str(payload.get("answer", ""))
        exercise = _find_exercise(exercise_id)
        if not exercise:
            return jsonify({"correct": False, "message": "Exercise expired. Generate a new one.", "details": []}), 404
        result = check_answer(exercise, answer)
        progress = progress_store.record_attempt(exercise.id, exercise.topic_id, exercise.seed, exercise.mode, result.correct)
        data = result.as_dict()
        data["progress"] = progress.as_dict()
        return jsonify(data)

    @app.post("/api/hint")
    def api_hint():
        payload: dict[str, Any] = request.get_json(force=True) or {}
        exercise_id = str(payload.get("id", ""))
        answer = str(payload.get("answer", ""))
        message = str(payload.get("message", ""))
        exercise = _find_exercise(exercise_id)
        if not exercise:
            return jsonify({"hint": "Generate a fresh exercise and try again."}), 404
        result = check_answer(exercise, answer)
        if message and result.correct:
            result = result.__class__(False, message, result.details)
        hint = TutorCoach(progress_store.settings()).hint(exercise, answer, result)
        return jsonify({"hint": hint})

    @app.post("/api/progress/reset")
    def api_progress_reset():
        return jsonify(progress_store.reset().as_dict())

    @app.post("/api/settings")
    def api_settings():
        payload: dict[str, Any] = request.get_json(force=True) or {}
        settings = progress_store.update_settings(**payload)
        return jsonify(settings.__dict__)

    @app.get("/<path:path>")
    def static_files(path: str):
        return send_from_directory(STATIC, path)

    return app


def _find_exercise(exercise_id: str):
    with _lock:
        exercise = _exercises.get(exercise_id)
    if exercise:
        return exercise
    parsed = parse_exercise_id(exercise_id)
    if not parsed:
        return None
    topic_id, seed = parsed
    try:
        return generate_exercise(topic_id=topic_id, seed=seed)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Python Dojo browser app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    args = parser.parse_args(argv)
    create_app().run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
