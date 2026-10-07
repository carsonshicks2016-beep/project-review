from __future__ import annotations

from .models import Lesson, Topic


STAGES: tuple[str, ...] = (
    "Absolute Basics",
    "Beginner Flow",
    "Core Data",
    "Functions in Depth",
    "Working Python",
    "Object-Oriented Python",
    "Advanced Python",
    "Professional Practice",
)


def _topics() -> tuple[Topic, ...]:
    entries: list[tuple[str, str, str, str]] = [
        # Stage 1 - Absolute Basics
        ("print_output", "Printing output", "Absolute Basics", "print, strings, visible output"),
        ("numbers", "Numbers and arithmetic", "Absolute Basics", "operators, precedence, // % **"),
        ("variables", "Variables and reassignment", "Absolute Basics", "names, assignment, updating values"),
        ("types_conversion", "Types and conversion", "Absolute Basics", "int, float, str, bool, casting"),
        ("strings", "Strings and f-strings", "Absolute Basics", "text, indexing, f-strings"),
        # Stage 2 - Beginner Flow
        ("booleans", "Booleans and comparisons", "Beginner Flow", "True, False, ==, <, and/or/not"),
        ("functions", "Functions and returns", "Beginner Flow", "def, parameters, return values"),
        ("conditionals", "Conditional branches", "Beginner Flow", "if, elif, else, decision trees"),
        ("loops", "For loops and range", "Beginner Flow", "for, range, accumulation"),
        ("while_loops", "While loops", "Beginner Flow", "while, counters, stopping conditions"),
        ("nested_logic", "Nested loops and conditions", "Beginner Flow", "loops inside loops, grids, filters"),
        # Stage 3 - Core Data
        ("lists", "Lists", "Core Data", "indexing, append, pop, mutation"),
        ("tuples_unpacking", "Tuples and unpacking", "Core Data", "fixed records, a, b = pair, swap"),
        ("slicing", "Slicing sequences", "Core Data", "start:stop:step, reversal, copies"),
        ("string_methods", "String methods", "Core Data", "split, join, strip, replace, find"),
        ("dicts", "Dictionaries", "Core Data", "lookups, get, counting, iteration"),
        ("sets", "Sets and uniqueness", "Core Data", "membership, union, intersection"),
        ("mutability", "Mutability and copying", "Core Data", "aliasing, copies, shared state"),
        ("enumerate_zip", "enumerate and zip", "Core Data", "index+value pairs, parallel lists"),
        ("comprehensions", "Comprehensions", "Core Data", "list/dict/set comprehensions, filters"),
        ("sorting_lambdas", "Sorting and lambdas", "Core Data", "sorted, key=, reverse, lambda"),
        # Stage 4 - Functions in Depth
        ("default_args", "Default and keyword arguments", "Functions in Depth", "defaults, keyword calls, None pattern"),
        ("args_kwargs", "*args and **kwargs", "Functions in Depth", "variable arguments, forwarding"),
        ("scope_closures", "Scope and closures", "Functions in Depth", "local vs global, enclosing state"),
        ("higher_order", "Higher-order functions", "Functions in Depth", "functions as values, map/filter"),
        ("recursion", "Recursion", "Functions in Depth", "base case, recursive case, trees"),
        # Stage 5 - Working Python
        ("text_files", "File-shaped text", "Working Python", "line parsing, records, cleanup"),
        ("exceptions", "Exceptions and boundaries", "Working Python", "try, except, raise, error types"),
        ("json_data", "JSON and structured data", "Working Python", "json.loads/dumps, nested data"),
        ("modules", "Modules and the standard library", "Working Python", "imports, math, statistics, datetime"),
        ("regex", "Regular expressions", "Working Python", "re.findall, groups, patterns"),
        ("collections_module", "The collections toolbox", "Working Python", "Counter, defaultdict, deque"),
        ("itertools_module", "itertools and functools", "Working Python", "chain, product, reduce, cache"),
        ("match_case", "Structural pattern matching", "Working Python", "match, case, patterns, guards"),
        # Stage 6 - Object-Oriented Python
        ("classes", "Classes and state", "Object-Oriented Python", "objects, methods, self, mutation"),
        ("dunder_methods", "Dunder methods", "Object-Oriented Python", "__repr__, __eq__, __len__, protocols"),
        ("inheritance", "Inheritance and super()", "Object-Oriented Python", "subclasses, overrides, super()"),
        ("dataclasses_topic", "Dataclasses", "Object-Oriented Python", "@dataclass, fields, defaults"),
        ("properties", "Properties and classmethods", "Object-Oriented Python", "@property, @classmethod, validation"),
        # Stage 7 - Advanced Python
        ("iterators", "Iterators", "Advanced Python", "iter, next, StopIteration, protocols"),
        ("generators", "Generators", "Advanced Python", "yield, lazy sequences, pipelines"),
        ("decorators", "Decorators", "Advanced Python", "wrappers, closures, shared state"),
        ("context_managers", "Context managers", "Advanced Python", "with protocol, guaranteed cleanup"),
        ("typing", "Type hints in practice", "Advanced Python", "annotations, Optional, unions"),
        ("async_python", "Async Python", "Advanced Python", "async def, await, gather"),
        # Stage 8 - Professional Practice
        ("testing", "Testing instincts", "Professional Practice", "edge cases, expected behavior"),
        ("debugging", "Debugging by reading code", "Professional Practice", "trace values, locate bugs"),
        ("cli_args", "Command-line thinking", "Professional Practice", "parse arguments represented as lists"),
        ("api_json", "API payload handling", "Professional Practice", "validate JSON-like payloads"),
        ("algorithms", "Algorithms and complexity", "Professional Practice", "search, invariants, efficient loops"),
        ("performance", "Performance-minded Python", "Professional Practice", "avoid repeated work, right structure"),
        ("refactoring", "Refactoring for clarity", "Professional Practice", "decompose tangled logic"),
        ("data_pipeline", "Data pipeline capstone", "Professional Practice", "parse, transform, summarize"),
        ("capstone_cli", "CLI capstone", "Professional Practice", "small production-style command function"),
    ]
    stage_rank = {stage: index + 1 for index, stage in enumerate(STAGES)}
    return tuple(
        Topic(topic_id, label, stage, (index + 1) * 10, summary, stage_rank[stage])
        for index, (topic_id, label, stage, summary) in enumerate(entries)
    )


TOPICS: tuple[Topic, ...] = _topics()
TOPIC_BY_ID = {topic.id: topic for topic in TOPICS}


def lesson_for(topic_id: str) -> Lesson:
    from .lessons import LESSONS

    topic = TOPIC_BY_ID[topic_id]
    return LESSONS.get(
        topic_id,
        Lesson(
            topic.id,
            topic.label,
            f"{topic.label} is about {topic.summary}. Focus on one small rule, solve it, then repeat with new values until the pattern feels natural.",
            "# Read the prompt, write a tiny solution, run the checks.",
        ),
    )


def curriculum_payload() -> dict[str, object]:
    from .gen import archetype_count, total_archetypes

    return {
        "stages": [
            {
                "name": stage,
                "topics": [
                    {
                        "id": topic.id,
                        "label": topic.label,
                        "summary": topic.summary,
                        "order": topic.order,
                        "difficulty": topic.difficulty,
                        "forms": archetype_count(topic.id),
                    }
                    for topic in TOPICS
                    if topic.stage == stage
                ],
            }
            for stage in STAGES
        ],
        "topic_count": len(TOPICS),
        "form_count": total_archetypes(),
        "exercise_modes": ["short", "choice", "code"],
    }
