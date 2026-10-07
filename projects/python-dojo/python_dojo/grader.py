from __future__ import annotations

from dataclasses import asdict
import ast
import json
import math
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from .models import AttemptResult, Exercise


def check_answer(exercise: Exercise, answer: str) -> AttemptResult:
    if exercise.mode == "code":
        return _check_code_answer(exercise, answer)
    if exercise.mode == "choice":
        ok = _normalize(answer) == _normalize(str(exercise.expected))
        return AttemptResult(ok, "Correct." if ok else f"Not quite. Expected {exercise.expected!r}.")
    if isinstance(exercise.expected, (int, float)) and not isinstance(exercise.expected, bool):
        try:
            actual = float(answer.strip())
        except ValueError:
            return AttemptResult(False, f"Expected a number close to {exercise.expected}.")
        ok = math.isclose(actual, float(exercise.expected), rel_tol=1e-9, abs_tol=1e-9)
        return AttemptResult(ok, "Correct." if ok else f"Not quite. Expected {exercise.expected}.")
    accepted = [_normalize(str(item)) for item in ([exercise.expected] + exercise.accepted)]
    ok = _normalize(answer) in accepted
    return AttemptResult(ok, "Correct." if ok else f"Not quite. Expected {exercise.expected!r}.")


def _normalize(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _check_code_answer(exercise: Exercise, answer: str) -> AttemptResult:
    if not answer.strip():
        return AttemptResult(False, "Add some Python first.")
    lint_error = lint_student_code(answer)
    if lint_error:
        return AttemptResult(False, lint_error)
    result = run_student_code(answer, [asdict(test) for test in exercise.tests])
    if not result.get("ok"):
        return AttemptResult(False, result.get("message", "The checker could not run the code."))
    details = result.get("details", [])
    failed = [row for row in details if not row.get("ok")]
    if failed:
        first = failed[0]
        if first.get("raises"):
            msg = f"{first['expression']} should raise {first['raises']}, but got {first.get('actual', 'no exception')}."
        else:
            msg = f"{first['expression']} returned {first.get('actual')}; expected {first.get('expected')}."
        return AttemptResult(False, msg, details)
    return AttemptResult(True, "All checks passed.", details)


def lint_student_code(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"SyntaxError on line {exc.lineno}: {exc.msg}"
    banned_names = {
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
        "__import__",
    }
    allowed_imports = {
        "asyncio",
        "bisect",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "enum",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "re",
        "statistics",
        "string",
        "typing",
    }
    banned_attr_names = {"system", "popen", "spawn", "fork", "unlink", "rmdir", "chmod"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_imports:
                    return f"Import {alias.name!r} is blocked in this local practice sandbox."
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in allowed_imports:
                return f"Import from {node.module!r} is blocked in this local practice sandbox."
        elif isinstance(node, ast.Name) and node.id in banned_names:
            return f"{node.id} is blocked in this local practice sandbox."
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return "Dunder attribute access is blocked in this local practice sandbox."
            if node.attr in banned_attr_names:
                return f"{node.attr} is blocked in this local practice sandbox."
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if "__" in node.value:
                return "Dunder string access is blocked in this local practice sandbox."
            if any(token in lowered for token in ("subprocess", "socket", "/etc/", "site-packages")):
                return "That string is blocked in this local practice sandbox."
    return None


def run_student_code(source: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    runner = textwrap.dedent(
        """
        from __future__ import annotations

        import asyncio
        import bisect
        import builtins
        import collections
        import copy
        import dataclasses
        import datetime
        import enum
        import functools
        import heapq
        import io
        import itertools
        import json
        import math
        import operator
        import re
        import statistics
        import string
        import sys
        import typing

        SOURCE = __SOURCE__
        TESTS = __TESTS__
        ALLOWED_MODULES = {
            "asyncio": asyncio,
            "bisect": bisect,
            "collections": collections,
            "copy": copy,
            "dataclasses": dataclasses,
            "datetime": datetime,
            "enum": enum,
            "functools": functools,
            "heapq": heapq,
            "itertools": itertools,
            "json": json,
            "math": math,
            "operator": operator,
            "re": re,
            "statistics": statistics,
            "string": string,
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
            "ArithmeticError": ArithmeticError,
            "AttributeError": AttributeError,
            "bool": bool,
            "callable": callable,
            "chr": chr,
            "classmethod": classmethod,
            "dict": dict,
            "divmod": divmod,
            "enumerate": enumerate,
            "Exception": Exception,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "getattr": getattr,
            "hasattr": hasattr,
            "hash": hash,
            "IndexError": IndexError,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "iter": iter,
            "KeyError": KeyError,
            "len": len,
            "list": list,
            "LookupError": LookupError,
            "map": map,
            "max": max,
            "min": min,
            "NameError": NameError,
            "next": next,
            "NotImplementedError": NotImplementedError,
            "object": object,
            "ord": ord,
            "OverflowError": OverflowError,
            "pow": pow,
            "print": print,
            "property": property,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "RuntimeError": RuntimeError,
            "set": set,
            "setattr": setattr,
            "slice": slice,
            "sorted": sorted,
            "staticmethod": staticmethod,
            "StopIteration": StopIteration,
            "str": str,
            "sum": sum,
            "super": super,
            "tuple": tuple,
            "type": type,
            "TypeError": TypeError,
            "ValueError": ValueError,
            "ZeroDivisionError": ZeroDivisionError,
            "zip": zip,
        }
        globals_box = {
            "__builtins__": safe_builtins,
            "__name__": "__python_dojo_student__",
            "asyncio": asyncio,
            "bisect": bisect,
            "collections": collections,
            "copy": copy,
            "dataclass": dataclasses.dataclass,
            "dataclasses": dataclasses,
            "datetime": datetime,
            "enum": enum,
            "functools": functools,
            "heapq": heapq,
            "itertools": itertools,
            "json": json,
            "math": math,
            "operator": operator,
            "re": re,
            "statistics": statistics,
            "string": string,
            "typing": typing,
        }

        def payload(ok, message="", details=None):
            print(json.dumps({"ok": ok, "message": message, "details": details or []}, default=repr))

        captured = io.StringIO()
        original_stdout = sys.stdout
        try:
            sys.stdout = captured
            exec(compile(SOURCE, "<student>", "exec"), globals_box, globals_box)
            sys.stdout = original_stdout
            globals_box["__stdout__"] = captured.getvalue()
            details = []
            for test in TESTS:
                expression = test["expression"]
                expected = test.get("expected")
                raises = test.get("raises")
                try:
                    actual = eval(expression, globals_box, globals_box)
                    if raises:
                        details.append({"expression": expression, "ok": False, "raises": raises, "actual": repr(actual)})
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
            sys.stdout = original_stdout
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
                timeout=5.0,
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
