from __future__ import annotations

from .models import Lesson


# Placeholder during the build-out; every topic receives a full lesson below.
LESSONS: dict[str, Lesson] = {
    "print_output": Lesson(
        "print_output",
        "Printing output",
        "The first Python superpower is making the computer say something. print(...) sends text to the screen. Text goes inside quotes.",
        'print("hello dojo")',
    ),
}
