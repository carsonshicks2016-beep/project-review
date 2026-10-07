from __future__ import annotations

import argparse
import random
import threading
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory

from python_tutor.engine import Exercise, check_answer, curriculum_payload, generate_exercise


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

_lock = threading.Lock()
_exercises: dict[str, Exercise] = {}


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="")

    @app.get("/")
    def index():
        return Response((STATIC / "index.html").read_text(encoding="utf-8"), mimetype="text/html")

    @app.get("/api/health")
    def api_health():
        return jsonify({"ok": True, "stored_exercises": len(_exercises)})

    @app.get("/api/curriculum")
    def api_curriculum():
        return jsonify(curriculum_payload())

    @app.post("/api/exercise")
    def api_exercise():
        payload: dict[str, Any] = request.get_json(force=True) or {}
        topic_id = payload.get("topic_id") or None
        level = payload.get("level") or None
        seed = payload.get("seed")
        if seed in ("", None):
            seed = random.randint(1, 2_000_000_000)
        try:
            seed = int(seed)
            exercise = generate_exercise(topic_id=topic_id, level=level, seed=seed)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        with _lock:
            _exercises[exercise.id] = exercise
            if len(_exercises) > 500:
                for key in list(_exercises)[:100]:
                    _exercises.pop(key, None)
        data = exercise.public_dict()
        data["seed"] = seed
        return jsonify(data)

    @app.post("/api/check")
    def api_check():
        payload: dict[str, Any] = request.get_json(force=True) or {}
        exercise_id = payload.get("id")
        answer = payload.get("answer", "")
        with _lock:
            exercise = _exercises.get(str(exercise_id))
        if not exercise:
            return jsonify({"correct": False, "message": "Exercise expired. Generate a new one.", "details": []}), 404
        return jsonify(check_answer(exercise, str(answer)))

    @app.get("/<path:path>")
    def static_files(path: str):
        return send_from_directory(STATIC, path)

    return app


app = create_app()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the interactive Python tutor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    args = parser.parse_args(argv)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

