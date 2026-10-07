from __future__ import annotations

import unittest

from python_tutor.engine import LEVELS, TOPICS, check_answer, curriculum_payload, generate_exercise
from python_tutor.server import create_app


class PythonTutorTests(unittest.TestCase):
    def test_curriculum_spans_foundations_to_expert(self):
        payload = curriculum_payload()
        levels = [level["name"] for level in payload["levels"]]
        self.assertEqual(levels, list(LEVELS))
        self.assertIn("Foundations", levels)
        self.assertIn("Expert", levels)
        self.assertGreaterEqual(payload["topic_count"], 18)

    def test_generators_produce_public_payloads_without_answers(self):
        for topic in TOPICS:
            exercise = generate_exercise(topic_id=topic.id, seed=42)
            public = exercise.public_dict()
            self.assertEqual(public["topic_id"], topic.id)
            self.assertNotIn("expected", public)
            self.assertNotIn("accepted", public)
            self.assertNotIn("tests", public)
            self.assertIn(public["mode"], {"predict", "short", "choice", "code"})

    def test_short_answer_checker(self):
        exercise = generate_exercise(topic_id="arithmetic", seed=7)
        self.assertTrue(check_answer(exercise, str(exercise.expected))["correct"])
        self.assertFalse(check_answer(exercise, "not a number")["correct"])

    def test_code_checker_accepts_function_solution(self):
        exercise = generate_exercise(topic_id="functions", seed=3)
        solution = """
def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value
"""
        result = check_answer(exercise, solution)
        self.assertTrue(result["correct"], result)

    def test_code_checker_accepts_advanced_solutions(self):
        descriptor = generate_exercise(topic_id="descriptors", seed=2)
        descriptor_solution = """
class PositiveNumber:
    def __init__(self):
        self.values = {}

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.values.get(instance)

    def __set__(self, instance, value):
        if value <= 0:
            raise ValueError("expected positive")
        self.values[instance] = value
"""
        self.assertTrue(check_answer(descriptor, descriptor_solution)["correct"])

        context = generate_exercise(topic_id="context", seed=4)
        start_word = context.tests[0].expected[3][0]
        finish_word = context.tests[0].expected[3][-1]
        context_solution = f"""
class ListSession:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self.log.append({start_word!r})
        return self

    def add(self, value):
        self.log.append(value)

    def __exit__(self, exc_type, exc, tb):
        self.log.append({finish_word!r})
        return False
"""
        self.assertTrue(check_answer(context, context_solution)["correct"])

    def test_code_checker_blocks_unsafe_import(self):
        exercise = generate_exercise(topic_id="functions", seed=3)
        result = check_answer(exercise, "import os\n\ndef clamp(value, low, high):\n    return value\n")
        self.assertFalse(result["correct"])
        self.assertIn("blocked", result["message"])

    def test_flask_api_generates_and_checks(self):
        client = create_app().test_client()
        curriculum = client.get("/api/curriculum")
        self.assertEqual(curriculum.status_code, 200)
        generated = client.post("/api/exercise", json={"topic_id": "arithmetic", "seed": 5})
        self.assertEqual(generated.status_code, 200)
        payload = generated.get_json()
        self.assertNotIn("expected", payload)
        checked = client.post("/api/check", json={"id": payload["id"], "answer": "wrong"})
        self.assertEqual(checked.status_code, 200)
        self.assertFalse(checked.get_json()["correct"])


if __name__ == "__main__":
    unittest.main()

