from __future__ import annotations

from python_dojo.curriculum import TOPICS, curriculum_payload
from python_dojo.generators import all_generator_topic_ids, generate_exercise, parse_exercise_id
from python_dojo.grader import check_answer, lint_student_code


def test_curriculum_spans_basics_to_professional():
    payload = curriculum_payload()
    stages = [stage["name"] for stage in payload["stages"]]
    assert stages[0] == "Absolute Basics"
    assert stages[-1] == "Professional Projects"
    assert payload["topic_count"] >= 30


def test_every_topic_has_a_generator_and_hidden_answers():
    assert set(all_generator_topic_ids()) == {topic.id for topic in TOPICS}
    for topic in TOPICS:
        exercise = generate_exercise(topic_id=topic.id, seed=42)
        public = exercise.public_dict()
        assert public["topic_id"] == topic.id
        assert public["seed"] == 42
        assert public["id"] == generate_exercise(topic_id=topic.id, seed=42).id
        assert "expected" not in public
        assert "tests" not in public
        assert "solution" not in public
        assert parse_exercise_id(exercise.id) == (topic.id, 42)


def test_canonical_solutions_pass_and_wrong_answers_fail():
    for topic in TOPICS:
        exercise = generate_exercise(topic_id=topic.id, seed=7)
        answer = exercise.solution if exercise.mode == "code" else str(exercise.expected)
        assert check_answer(exercise, answer).correct, topic.id
        wrong = exercise.wrong_answer
        if wrong == answer:
            wrong = "__wrong__"
        assert not check_answer(exercise, wrong).correct, topic.id


def test_sandbox_blocks_unsafe_code():
    unsafe_snippets = [
        "open('x.txt', 'w')",
        "eval('1 + 1')",
        "exec('print(1)')",
        "import os",
        "import subprocess",
        "().__class__",
        "while True:\n    pass",
    ]
    exercise = generate_exercise(topic_id="functions", seed=3)
    for snippet in unsafe_snippets[:-1]:
        assert lint_student_code(snippet)
    result = check_answer(exercise, unsafe_snippets[-1])
    assert not result.correct
    assert "too long" in result.message
