from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from prompt_toolkit import prompt
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.python import PythonLexer

from .generators import generate_exercise
from .grader import check_answer
from .progress import ProgressStore, topic_label
from .tutor import TutorCoach


console = Console()


def run_terminal() -> int:
    store = ProgressStore()
    console.print("[bold green]Welcome to Python Dojo.[/bold green]")
    while True:
        progress = store.state()
        display_dashboard(progress)
        topic_id = progress.next_topic_id
        coach = TutorCoach(store.settings())
        lesson = coach.lesson(topic_id)
        console.print(Panel(f"{lesson['body']}\n\n[dim]{lesson['example']}[/dim]", title=f"Lesson: {lesson['title']}", border_style="blue"))
        exercise = generate_exercise(topic_id=topic_id)
        console.print(Panel(exercise.prompt, title=f"{exercise.topic_label}: {exercise.title}", border_style="yellow"))
        if exercise.code:
            console.print(Syntax(exercise.code, "python", theme="monokai", line_numbers=False))
        if exercise.mode == "choice":
            for index, choice in enumerate(exercise.choices, start=1):
                console.print(f"[cyan]{index}.[/cyan] {choice}")
        answer = collect_answer(exercise)
        if answer is None:
            break
        result = check_answer(exercise, answer)
        store.record_attempt(exercise.id, exercise.topic_id, exercise.seed, exercise.mode, result.correct)
        console.print(Panel(result.message, title="Success" if result.correct else "Try again", border_style="green" if result.correct else "red"))
        if not result.correct:
            console.print(Panel(coach.hint(exercise, answer, result), title="Hint", border_style="magenta"))
        try:
            choice = prompt("Press Enter for the next challenge, or type q to quit: ")
        except (KeyboardInterrupt, EOFError):
            break
        if choice.strip().lower() == "q":
            break
    console.print("[bold green]Training saved. See you next session.[/bold green]")
    return 0


def display_dashboard(progress) -> None:
    table = Table(title="Python Dojo Progress")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Solved", f"{progress.total_correct}/{progress.total_attempts}")
    table.add_row("Streak", str(progress.streak))
    table.add_row("Next Topic", topic_label(progress.next_topic_id))
    table.add_row("Review Queue", ", ".join(topic_label(topic) for topic in progress.review_queue) or "clear")
    console.print(table)


def collect_answer(exercise) -> str | None:
    try:
        if exercise.mode == "code":
            console.print("[dim]Write your solution. Press Esc then Enter to submit.[/dim]")
            return prompt(">>> ", multiline=True, lexer=PygmentsLexer(PythonLexer), default=exercise.starter)
        if exercise.mode == "choice":
            raw = prompt("Choice: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(exercise.choices):
                return exercise.choices[int(raw) - 1]
            return raw
        return prompt("Answer: ")
    except (KeyboardInterrupt, EOFError):
        return None
