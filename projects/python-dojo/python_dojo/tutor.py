from __future__ import annotations

from .curriculum import lesson_for
from .models import AttemptResult, Exercise, LearnerSettings


class TutorCoach:
    def __init__(self, settings: LearnerSettings) -> None:
        self.settings = settings

    def ollama_available(self) -> bool:
        if not self.settings.use_ollama:
            return False
        try:
            import ollama

            client = ollama.Client(host=self.settings.ollama_host)
            client.show(self.settings.ollama_model)
            return True
        except Exception:
            return False

    def lesson(self, topic_id: str) -> dict[str, str]:
        lesson = lesson_for(topic_id)
        return {
            "topic_id": lesson.topic_id,
            "title": lesson.title,
            "body": lesson.body,
            "example": lesson.example,
        }

    def hint(self, exercise: Exercise, answer: str, result: AttemptResult) -> str:
        if self.settings.use_ollama:
            generated = self._ollama_hint(exercise, answer, result)
            if generated:
                return generated
        return self._fallback_hint(exercise, result)

    def _ollama_hint(self, exercise: Exercise, answer: str, result: AttemptResult) -> str | None:
        try:
            import ollama

            client = ollama.Client(host=self.settings.ollama_host, timeout=8)
            response = client.chat(
                model=self.settings.ollama_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise Python tutor. Give one short conceptual hint. Do not reveal full answer code.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Exercise: {exercise.prompt}\n"
                            f"Concept: {exercise.concept}\n"
                            f"Student answer:\n{answer}\n"
                            f"Checker feedback: {result.message}"
                        ),
                    },
                ],
            )
            content = response["message"]["content"].strip()
            return content.splitlines()[0][:280] if content else None
        except Exception:
            return None

    def _fallback_hint(self, exercise: Exercise, result: AttemptResult) -> str:
        if exercise.mode == "short":
            return "Trace the code one line at a time and write down the value after each assignment."
        if exercise.mode == "choice":
            return "Eliminate the choices that do not match the edge case named in the prompt."
        if "SyntaxError" in result.message:
            return "Check punctuation, indentation, and whether every block ending in ':' has an indented body."
        if "blocked" in result.message:
            return "Stay inside the small sandbox: solve this with core Python rather than files, processes, or introspection."
        if result.details:
            first = result.details[0]
            return f"Focus on the case `{first.get('expression', 'the first check')}`; make that one pass before generalizing."
        return "Compare your function name, return value, and edge cases with the prompt."
