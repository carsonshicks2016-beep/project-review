from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import json
import math
import random
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any, Callable


LEVELS = (
    "Foundations",
    "Control Flow",
    "Data Structures",
    "Practical Python",
    "Advanced",
    "Expert",
)


@dataclass(frozen=True)
class Topic:
    id: str
    label: str
    level: str
    order: int
    summary: str


@dataclass
class CodeTest:
    expression: str
    expected: Any = None
    raises: str | None = None
    note: str = ""


@dataclass
class Exercise:
    id: str
    topic_id: str
    topic_label: str
    level: str
    mode: str
    title: str
    prompt: str
    concept: str
    code: str = ""
    starter: str = ""
    choices: list[str] = field(default_factory=list)
    expected: Any = None
    accepted: list[str] = field(default_factory=list)
    tests: list[CodeTest] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("expected", None)
        data.pop("accepted", None)
        data.pop("tests", None)
        return data


TOPICS: tuple[Topic, ...] = (
    Topic("arithmetic", "Arithmetic and precedence", "Foundations", 10, "numbers, operators, precedence"),
    Topic("variables", "Variables and names", "Foundations", 20, "assignment, updating values, simple state"),
    Topic("strings", "Strings and slicing", "Foundations", 30, "indexes, slices, f-strings, methods"),
    Topic("conditionals", "Conditionals", "Control Flow", 40, "if, elif, else, booleans"),
    Topic("loops", "Loops", "Control Flow", 50, "for, while, ranges, accumulation"),
    Topic("functions", "Functions", "Control Flow", 60, "parameters, return values, composition"),
    Topic("lists", "Lists and tuples", "Data Structures", 70, "indexing, mutation, sorted data"),
    Topic("dicts", "Dictionaries and sets", "Data Structures", 80, "lookup tables, counts, uniqueness"),
    Topic("comprehensions", "Comprehensions", "Data Structures", 90, "compact loops over collections"),
    Topic("json", "JSON and parsing", "Practical Python", 100, "structured data from strings"),
    Topic("exceptions", "Exceptions", "Practical Python", 110, "try, except, defensive boundaries"),
    Topic("testing", "Testing instincts", "Practical Python", 120, "edge cases and small contracts"),
    Topic("classes", "Classes", "Advanced", 130, "objects, methods, state"),
    Topic("generators", "Generators", "Advanced", 140, "yield, lazy sequences, iteration"),
    Topic("decorators", "Decorators and closures", "Advanced", 150, "wrappers, inner functions, closure state"),
    Topic("context", "Context managers", "Advanced", 160, "enter/exit protocols and cleanup"),
    Topic("asyncio", "Async Python", "Expert", 170, "coroutines, await, task orchestration"),
    Topic("descriptors", "Descriptors", "Expert", 180, "attribute access hooks and validation"),
    Topic("algorithms", "Algorithms", "Expert", 190, "search, invariants, complexity"),
)


TOPIC_BY_ID = {topic.id: topic for topic in TOPICS}
Generator = Callable[[random.Random, str], Exercise]
_GENERATORS: list[tuple[str, Generator]] = []


def register(topic_id: str):
    def decorator(func: Generator) -> Generator:
        _GENERATORS.append((topic_id, func))
        return func

    return decorator


def curriculum_payload() -> dict[str, Any]:
    return {
        "levels": [
            {
                "name": level,
                "topics": [
                    {
                        "id": topic.id,
                        "label": topic.label,
                        "summary": topic.summary,
                        "order": topic.order,
                    }
                    for topic in TOPICS
                    if topic.level == level
                ],
            }
            for level in LEVELS
        ],
        "topic_count": len(TOPICS),
        "exercise_modes": ["predict", "short", "choice", "code"],
    }


def generate_exercise(topic_id: str | None = None, level: str | None = None, seed: int | None = None) -> Exercise:
    rng = random.Random(seed)
    options = []
    for candidate_topic_id, func in _GENERATORS:
        topic = TOPIC_BY_ID[candidate_topic_id]
        if topic_id and topic.id != topic_id:
            continue
        if level and topic.level != level:
            continue
        options.append((candidate_topic_id, func))
    if not options:
        raise ValueError("No exercise generators match the requested filters.")
    selected_topic_id, func = rng.choice(options)
    exercise_id = f"{selected_topic_id}-{uuid.uuid4().hex[:12]}"
    return func(rng, exercise_id)


def check_answer(exercise: Exercise, answer: str) -> dict[str, Any]:
    if exercise.mode == "code":
        return _check_code_answer(exercise, answer)
    if exercise.mode == "choice":
        ok = _normalize(answer) == _normalize(str(exercise.expected))
        return _feedback(ok, "Correct." if ok else f"Not quite. Expected {exercise.expected!r}.")
    if _numeric_like(exercise.expected):
        try:
            actual = float(answer.strip())
        except ValueError:
            return _feedback(False, f"Expected a number close to {exercise.expected}.")
        ok = math.isclose(actual, float(exercise.expected), rel_tol=1e-9, abs_tol=1e-9)
        return _feedback(ok, "Correct." if ok else f"Not quite. Expected {exercise.expected}.")
    accepted = [_normalize(str(item)) for item in ([exercise.expected] + exercise.accepted)]
    ok = _normalize(answer) in accepted
    return _feedback(ok, "Correct." if ok else f"Not quite. Expected {exercise.expected!r}.")


def _feedback(ok: bool, message: str, details: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "correct": ok,
        "message": message,
        "details": details or [],
    }


def _normalize(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _numeric_like(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _topic(topic_id: str) -> Topic:
    return TOPIC_BY_ID[topic_id]


def _exercise(
    topic_id: str,
    exercise_id: str,
    mode: str,
    title: str,
    prompt: str,
    concept: str,
    **kwargs: Any,
) -> Exercise:
    topic = _topic(topic_id)
    return Exercise(
        id=exercise_id,
        topic_id=topic.id,
        topic_label=topic.label,
        level=topic.level,
        mode=mode,
        title=title,
        prompt=prompt,
        concept=concept,
        **kwargs,
    )


def _check_code_answer(exercise: Exercise, answer: str) -> dict[str, Any]:
    if not answer.strip():
        return _feedback(False, "Add some Python first.")
    lint_error = _lint_student_code(answer)
    if lint_error:
        return _feedback(False, lint_error)
    tests = [asdict(test) for test in exercise.tests]
    result = _run_student_code(answer, tests)
    if not result.get("ok"):
        return _feedback(False, result.get("message", "The checker could not run the code."))
    details = result.get("details", [])
    failed = [row for row in details if not row.get("ok")]
    if failed:
        first = failed[0]
        if first.get("raises"):
            msg = f"{first['expression']} should raise {first['raises']}, but got {first.get('actual', 'no exception')}."
        else:
            msg = f"{first['expression']} returned {first.get('actual')}; expected {first.get('expected')}."
        return _feedback(False, msg, details)
    return _feedback(True, "All checks passed.", details)


def _lint_student_code(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"SyntaxError on line {exc.lineno}: {exc.msg}"
    banned_calls = {
        "breakpoint",
        "compile",
        "dir",
        "eval",
        "exec",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "vars",
    }
    allowed_imports = {"asyncio", "collections", "dataclasses", "functools", "itertools", "json", "math", "statistics", "typing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_imports:
                    return f"Import {alias.name!r} is blocked in this local practice sandbox."
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in allowed_imports:
                return f"Import from {node.module!r} is blocked in this local practice sandbox."
        if isinstance(node, ast.Name) and node.id in banned_calls:
            return f"{node.id} is blocked in this local practice sandbox."
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"system", "popen", "spawn"}:
            return f"{node.func.attr} is blocked in this local practice sandbox."
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "Dunder attribute access is blocked in this local practice sandbox."
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            return "Dunder string access is blocked in this local practice sandbox."
    return None


def _run_student_code(source: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    runner = textwrap.dedent(
        """
        from __future__ import annotations

        import asyncio
        import builtins
        import collections
        import dataclasses
        import functools
        import itertools
        import json
        import math
        import statistics
        import sys
        import traceback
        import typing

        SOURCE = __SOURCE__
        TESTS = __TESTS__
        ALLOWED_MODULES = {
            "asyncio": asyncio,
            "collections": collections,
            "dataclasses": dataclasses,
            "functools": functools,
            "itertools": itertools,
            "json": json,
            "math": math,
            "statistics": statistics,
            "typing": typing,
        }

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".", 1)[0]
            if root not in ALLOWED_MODULES:
                raise ImportError(f"Import {name!r} is blocked")
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins = {
            "__build_class__": builtins.__build_class__,
            "__import__": safe_import,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "classmethod": classmethod,
            "dict": dict,
            "enumerate": enumerate,
            "Exception": Exception,
            "filter": filter,
            "float": float,
            "hasattr": hasattr,
            "int": int,
            "isinstance": isinstance,
            "iter": iter,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "object": object,
            "pow": pow,
            "print": print,
            "property": property,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "setattr": setattr,
            "sorted": sorted,
            "staticmethod": staticmethod,
            "str": str,
            "sum": sum,
            "super": super,
            "tuple": tuple,
            "type": type,
            "ValueError": ValueError,
            "zip": zip,
        }
        globals_box = {
            "__builtins__": safe_builtins,
            "__name__": "__python_gym_student__",
            "asyncio": asyncio,
            "collections": collections,
            "dataclass": dataclasses.dataclass,
            "functools": functools,
            "itertools": itertools,
            "json": json,
            "math": math,
            "statistics": statistics,
            "typing": typing,
        }

        def payload(ok, message="", details=None):
            print(json.dumps({"ok": ok, "message": message, "details": details or []}, default=repr))

        try:
            exec(compile(SOURCE, "<student>", "exec"), globals_box, globals_box)
            details = []
            for test in TESTS:
                expression = test["expression"]
                expected = test.get("expected")
                raises = test.get("raises")
                try:
                    actual = eval(expression, globals_box, globals_box)
                    if raises:
                        details.append({
                            "expression": expression,
                            "ok": False,
                            "raises": raises,
                            "actual": repr(actual),
                        })
                    else:
                        details.append({
                            "expression": expression,
                            "ok": actual == expected,
                            "actual": repr(actual),
                            "expected": repr(expected),
                            "note": test.get("note", ""),
                        })
                except Exception as exc:
                    if raises and type(exc).__name__ == raises:
                        details.append({
                            "expression": expression,
                            "ok": True,
                            "actual": type(exc).__name__,
                            "expected": raises,
                            "note": test.get("note", ""),
                        })
                    else:
                        details.append({
                            "expression": expression,
                            "ok": False,
                            "actual": f"{type(exc).__name__}: {exc}",
                            "expected": repr(expected) if not raises else raises,
                            "note": test.get("note", ""),
                        })
            payload(True, details=details)
        except Exception as exc:
            payload(False, f"{type(exc).__name__}: {exc}")
        """
    )
    runner = runner.replace("__SOURCE__", repr(source))
    runner = runner.replace("__TESTS__", repr(tests))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "runner.py"
        path.write_text(runner, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.5,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "message": "The code took too long to finish."}
    if completed.returncode != 0:
        return {"ok": False, "message": completed.stderr.strip() or "The checker process failed."}
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "message": "The checker produced no result."}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"ok": False, "message": "The checker produced invalid output."}


@register("arithmetic")
def arithmetic_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    a = rng.randint(2, 9)
    b = rng.randint(3, 12)
    c = rng.randint(2, 7)
    d = rng.randint(1, 5)
    value = a + b * c - d
    code = f"result = {a} + {b} * {c} - {d}\nprint(result)"
    return _exercise(
        "arithmetic",
        exercise_id,
        "short",
        "Predict the number",
        "What number is printed?",
        "Python evaluates multiplication before addition and subtraction unless parentheses say otherwise.",
        code=code,
        expected=value,
    )


@register("variables")
def variables_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    start = rng.randint(4, 20)
    boost = rng.randint(2, 9)
    cost = rng.randint(1, 7)
    expected = (start + boost) * 2 - cost
    code = f"score = {start}\nscore = score + {boost}\nscore = score * 2\nscore = score - {cost}\nprint(score)"
    return _exercise(
        "variables",
        exercise_id,
        "short",
        "Track the variable",
        "What value does score hold at the end?",
        "A variable name points to the latest value assigned to it. Read updates top to bottom.",
        code=code,
        expected=expected,
    )


@register("strings")
def strings_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    words = ["cipher", "python", "variable", "function", "iterator", "package"]
    word = rng.choice(words)
    start = rng.randint(0, max(0, len(word) - 4))
    stop = rng.randint(start + 2, len(word))
    expected = word[start:stop].upper()
    code = f'word = "{word}"\nprint(word[{start}:{stop}].upper())'
    return _exercise(
        "strings",
        exercise_id,
        "short",
        "Slice the string",
        "What exact text is printed?",
        "Slices include the start index and stop before the end index. String methods return new strings.",
        code=code,
        expected=expected,
    )


@register("conditionals")
def conditionals_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    youth = rng.randint(12, 17)
    senior = rng.randint(61, 70)
    standard = rng.randint(16, 30)
    discount = rng.randint(4, 9)
    return _exercise(
        "conditionals",
        exercise_id,
        "code",
        "Write a branching function",
        f"Define ticket_price(age). Return {youth} for ages under 18, {senior} for ages 65 and up, and {standard} otherwise. If the age is under 5, subtract {discount} from the youth price.",
        "Conditionals let one function handle several cases. Put the most specific cases before broader ones.",
        starter="def ticket_price(age):\n    # return the right price\n    pass\n",
        tests=[
            CodeTest("ticket_price(4)", youth - discount),
            CodeTest("ticket_price(17)", youth),
            CodeTest("ticket_price(30)", standard),
            CodeTest("ticket_price(65)", senior),
        ],
    )


@register("loops")
def loops_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    multiple = rng.randint(2, 6)
    limit = rng.randint(18, 38)
    return _exercise(
        "loops",
        exercise_id,
        "code",
        "Accumulate with a loop",
        f"Define sum_multiples(limit). Return the sum of all positive numbers below limit that are divisible by {multiple}.",
        "A loop can build a result one step at a time. A running total is one of the most useful beginner patterns.",
        starter="def sum_multiples(limit):\n    total = 0\n    # add matching numbers below limit\n    return total\n",
        tests=[
            CodeTest("sum_multiples(1)", 0),
            CodeTest(f"sum_multiples({limit})", sum(x for x in range(limit) if x % multiple == 0)),
            CodeTest(f"sum_multiples({limit + 7})", sum(x for x in range(limit + 7) if x % multiple == 0)),
        ],
    )


@register("functions")
def functions_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    low = rng.randint(-8, -1)
    high = rng.randint(4, 12)
    return _exercise(
        "functions",
        exercise_id,
        "code",
        "Clamp a value",
        f"Define clamp(value, low={low}, high={high}). Return low when value is too small, high when value is too large, otherwise return value.",
        "A function packages a rule. Good function practice means checking the lower edge, the upper edge, and the normal path.",
        starter="def clamp(value, low, high):\n    pass\n",
        tests=[
            CodeTest(f"clamp({low - 5}, {low}, {high})", low),
            CodeTest(f"clamp({high + 5}, {low}, {high})", high),
            CodeTest(f"clamp({(low + high) // 2}, {low}, {high})", (low + high) // 2),
        ],
    )


@register("lists")
def lists_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    values = rng.sample(range(1, 30), 5)
    add = rng.randint(30, 45)
    idx = rng.randint(0, 2)
    mutated = values[:]
    removed = mutated.pop(idx)
    mutated.append(add)
    expected = sorted(mutated)[-2]
    code = f"values = {values!r}\nremoved = values.pop({idx})\nvalues.append({add})\nprint(removed, sorted(values)[-2])"
    return _exercise(
        "lists",
        exercise_id,
        "short",
        "Follow list mutation",
        "What exact two values are printed, separated by one space?",
        "List methods can mutate the original list. pop removes and returns a value; sorted returns a new sorted list.",
        code=code,
        expected=f"{removed} {expected}",
    )


@register("dicts")
def dicts_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    return _exercise(
        "dicts",
        exercise_id,
        "code",
        "Count repeated items",
        "Define count_items(items). Return a dictionary mapping each item to the number of times it appears.",
        "Dictionaries are ideal for frequency tables because keys give direct access to the current count.",
        starter="def count_items(items):\n    counts = {}\n    # fill counts\n    return counts\n",
        tests=[
            CodeTest("count_items([])", {}),
            CodeTest("count_items(['red', 'blue', 'red'])", {"red": 2, "blue": 1}),
            CodeTest("count_items([2, 2, 3, 2, 3])", {2: 3, 3: 2}),
        ],
    )


@register("comprehensions")
def comprehensions_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    divisor = rng.randint(2, 5)
    offset = rng.randint(1, 4)
    return _exercise(
        "comprehensions",
        exercise_id,
        "code",
        "Transform with a comprehension",
        f"Define shifted_squares(values). Return a list of (value + {offset}) squared for values divisible by {divisor}. Preserve the original order.",
        "A comprehension is a compact loop: expression first, source next, optional filter last.",
        starter="def shifted_squares(values):\n    return []\n",
        tests=[
            CodeTest("shifted_squares([])", []),
            CodeTest("shifted_squares([1, 2, 3, 4, 5, 6])", [(x + offset) ** 2 for x in [1, 2, 3, 4, 5, 6] if x % divisor == 0]),
            CodeTest("shifted_squares([10, 11, 12])", [(x + offset) ** 2 for x in [10, 11, 12] if x % divisor == 0]),
        ],
    )


@register("json")
def json_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    min_score = rng.randint(60, 85)
    return _exercise(
        "json",
        exercise_id,
        "code",
        "Parse structured text",
        f"Define passing_names(raw). raw is a JSON string containing a list of objects with name and score. Return the names with score >= {min_score}, sorted alphabetically.",
        "JSON turns text into Python dictionaries and lists. Parsing belongs at the boundary; after that, normal collection tools work.",
        starter="import json\n\n\ndef passing_names(raw):\n    data = json.loads(raw)\n    return []\n",
        tests=[
            CodeTest('passing_names(\'[{"name": "Ada", "score": 90}, {"name": "Lin", "score": 50}]\')', ["Ada"] if 90 >= min_score else []),
            CodeTest(
                'passing_names(\'[{"name": "Tao", "score": 88}, {"name": "Bea", "score": 91}, {"name": "Kai", "score": 40}]\')',
                sorted(name for name, score in [("Tao", 88), ("Bea", 91), ("Kai", 40)] if score >= min_score),
            ),
        ],
    )


@register("exceptions")
def exceptions_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    default = rng.randint(-3, 3)
    return _exercise(
        "exceptions",
        exercise_id,
        "code",
        "Recover from bad input",
        f"Define safe_int(text, default={default}). Return int(text), but return default when conversion fails.",
        "Use exceptions for boundary failures you expect. Keep the try block small so successful code stays readable.",
        starter="def safe_int(text, default):\n    pass\n",
        tests=[
            CodeTest(f"safe_int('42', {default})", 42),
            CodeTest(f"safe_int('-7', {default})", -7),
            CodeTest(f"safe_int('nope', {default})", default),
        ],
    )


@register("testing")
def testing_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    bug = rng.choice(["empty list", "negative values", "duplicate maximum"])
    choices = [
        "A list with one item",
        "An empty list",
        "A list containing negative values",
        "A list that is already sorted",
        "A list with repeated maximum values",
    ]
    expected = {
        "empty list": "An empty list",
        "negative values": "A list containing negative values",
        "duplicate maximum": "A list with repeated maximum values",
    }[bug]
    return _exercise(
        "testing",
        exercise_id,
        "choice",
        "Choose the edge case",
        f"A function returns the largest gap between neighboring numbers after sorting. Which test is most likely to catch a bug around {bug}?",
        "Strong tests pin down edges: empty input, one item, ties, negative numbers, and values exactly on a boundary.",
        choices=choices,
        expected=expected,
    )


@register("classes")
def classes_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    start = rng.randint(0, 5)
    return _exercise(
        "classes",
        exercise_id,
        "code",
        "Keep state in an object",
        f"Create a Counter class. __init__ should accept start={start} by default. inc(amount=1) should add amount and return the new count. reset() should set the count to 0.",
        "Classes bundle state with the operations that change it. self is the object being changed.",
        starter="class Counter:\n    pass\n",
        tests=[
            CodeTest(f"Counter().inc()", start + 1),
            CodeTest(f"Counter().inc(4)", start + 4),
            CodeTest("(lambda c: (c.inc(3), c.reset(), c.inc()))(Counter(10))", (13, None, 1)),
        ],
    )


@register("generators")
def generators_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    stop = rng.randint(4, 9)
    return _exercise(
        "generators",
        exercise_id,
        "code",
        "Yield a lazy sequence",
        f"Define until_stop(values). Yield values one by one, but stop before yielding the first value equal to {stop}.",
        "A generator yields values lazily. return or falling off the function ends the iteration.",
        starter="def until_stop(values):\n    # yield values until the stop marker appears\n    pass\n",
        tests=[
            CodeTest(f"list(until_stop([1, 2, {stop}, 99]))", [1, 2]),
            CodeTest("list(until_stop([]))", []),
            CodeTest(f"list(until_stop([{stop}, 1, 2]))", []),
        ],
    )


@register("decorators")
def decorators_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    return _exercise(
        "decorators",
        exercise_id,
        "code",
        "Wrap a function",
        "Define call_counter(fn). It should return a wrapper that calls fn, returns fn's result, and stores the number of calls on wrapper.calls.",
        "A decorator is a function that receives a function and returns a replacement function. The wrapper can close over shared state.",
        starter="def call_counter(fn):\n    pass\n",
        tests=[
            CodeTest("(lambda f: (f(2), f(5), f.calls))(call_counter(lambda x: x * 3))", (6, 15, 2)),
            CodeTest("(lambda f: (f('a'), f.calls))(call_counter(lambda x: x.upper()))", ("A", 1)),
        ],
    )


@register("context")
def context_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    start = rng.choice(["open", "begin", "start"])
    finish = rng.choice(["closed", "done", "finish"])
    return _exercise(
        "context",
        exercise_id,
        "code",
        "Manage entry and cleanup",
        f"Create ListSession. __init__(log) stores a list. __enter__ appends {start!r} and returns self. add(value) appends value. __exit__ appends {finish!r} and does not suppress exceptions.",
        "A context manager gives code a guaranteed entry and exit path. It is the protocol behind with blocks.",
        starter="class ListSession:\n    pass\n",
        tests=[
            CodeTest(
                "(lambda log: (lambda s: (s.__enter__() is s, s.add('x'), s.__exit__(None, None, None) in (None, False), log))(ListSession(log)))([])",
                (True, None, True, [start, "x", finish]),
            ),
        ],
    )


@register("asyncio")
def asyncio_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    offset = rng.randint(1, 5)
    return _exercise(
        "asyncio",
        exercise_id,
        "code",
        "Await a coroutine",
        f"Define async shifted(values). It should await asyncio.sleep(0), then return a list with {offset} added to each value.",
        "async def creates a coroutine. await pauses until another awaitable has finished.",
        starter="import asyncio\n\n\nasync def shifted(values):\n    await asyncio.sleep(0)\n    return []\n",
        tests=[
            CodeTest("asyncio.run(shifted([]))", []),
            CodeTest("asyncio.run(shifted([1, 2, 3]))", [1 + offset, 2 + offset, 3 + offset]),
        ],
    )


@register("descriptors")
def descriptors_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    return _exercise(
        "descriptors",
        exercise_id,
        "code",
        "Validate attributes with a descriptor",
        "Create PositiveNumber. It should work as a descriptor that stores per-instance values and raises ValueError when assigned a value <= 0.",
        "Descriptors let one object control attribute access on another object. They power properties, methods, and many framework fields.",
        starter="class PositiveNumber:\n    pass\n",
        tests=[
            CodeTest("(lambda p: (setattr(p, 'price', 4), p.price))(type('Product', (), {'price': PositiveNumber()})())", (None, 4)),
            CodeTest("(lambda p: setattr(p, 'price', 0))(type('Product', (), {'price': PositiveNumber()})())", raises="ValueError"),
        ],
    )


@register("algorithms")
def algorithms_exercise(rng: random.Random, exercise_id: str) -> Exercise:
    needle = rng.randint(10, 90)
    values = sorted(set(rng.sample(range(1, 100), 8) + [needle]))
    return _exercise(
        "algorithms",
        exercise_id,
        "code",
        "Binary search",
        "Define binary_search(items, target). items is sorted. Return the index of target, or -1 if target is missing.",
        "Binary search keeps a low/high search window and discards half the remaining candidates each step.",
        starter="def binary_search(items, target):\n    return -1\n",
        tests=[
            CodeTest(f"binary_search({values!r}, {needle})", values.index(needle)),
            CodeTest("binary_search([], 5)", -1),
            CodeTest("binary_search([1, 3, 7, 9], 4)", -1),
            CodeTest("binary_search([1, 3, 7, 9], 9)", 3),
        ],
    )
