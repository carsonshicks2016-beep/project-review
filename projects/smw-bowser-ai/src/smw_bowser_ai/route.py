from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import default_config_path, load_config
from .protocol import BridgeObservation


class RouteError(RuntimeError):
    """Raised when a route manifest is invalid."""


@dataclass(frozen=True)
class RouteStep:
    id: str
    kind: str
    name: str = ""
    mode: str = ""
    intended_exit: str = ""
    required: bool = True
    tags: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class RouteValidationIssue:
    code: str
    message: str
    frame: int | None = None
    step_id: str | None = None


@dataclass
class RouteValidationResult:
    ok: bool
    issues: list[RouteValidationIssue] = field(default_factory=list)
    required_steps_seen: set[str] = field(default_factory=set)


class RouteManifest:
    def __init__(
        self,
        version: str,
        name: str,
        steps: list[RouteStep],
        forbid_starworld: bool,
        allow_savestates_in_final: bool,
        allow_memory_writes: bool,
        final_goal: str,
    ):
        self.version = version
        self.name = name
        self.steps = steps
        self.forbid_starworld = forbid_starworld
        self.allow_savestates_in_final = allow_savestates_in_final
        self.allow_memory_writes = allow_memory_writes
        self.final_goal = final_goal
        self.by_id = {step.id: step for step in steps}
        self.forbidden_names = {
            "star road",
            "star world",
            "star world 1",
            "star world 2",
            "star world 3",
            "star world 4",
            "star world 5",
        }

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "RouteManifest":
        data = load_config(path or default_config_path("route_no_starworld_safety.yaml"))
        steps: list[RouteStep] = []
        for entry in data.get("steps", []):
            steps.append(
                RouteStep(
                    id=str(entry["id"]),
                    kind=str(entry.get("kind", "")),
                    name=str(entry.get("name", "")),
                    mode=str(entry.get("mode", "")),
                    intended_exit=str(entry.get("intended_exit", "")),
                    required=bool(entry.get("required", True)),
                    tags=tuple(str(tag) for tag in entry.get("tags", [])),
                    notes=str(entry.get("notes", "")),
                )
            )
        if not steps:
            raise RouteError("route manifest must contain at least one step")
        return cls(
            version=str(data.get("version", "unknown")),
            name=str(data.get("name", "Unnamed Route")),
            steps=steps,
            forbid_starworld=bool(data.get("forbid_starworld", True)),
            allow_savestates_in_final=bool(data.get("allow_savestates_in_final", False)),
            allow_memory_writes=bool(data.get("allow_memory_writes", False)),
            final_goal=str(data.get("final_goal", "bowser_defeated")),
        )

    def next_required_after(self, completed: set[str]) -> RouteStep | None:
        for step in self.steps:
            if step.required and step.id not in completed:
                return step
        return None

    def detect_starworld(self, observation: BridgeObservation | dict[str, Any]) -> bool:
        if not self.forbid_starworld:
            return False
        raw = observation.raw if isinstance(observation, BridgeObservation) else observation
        candidates = [
            str(raw.get("level_name", "")),
            str(raw.get("route_step", "")),
            str(raw.get("route_tag", "")),
            str(raw.get("mode", "")),
        ]
        if isinstance(raw.get("tags"), list):
            candidates.extend(str(tag) for tag in raw["tags"])
        haystack = " ".join(candidates).lower()
        return any(name in haystack for name in self.forbidden_names)

    def validate_trace(self, events: Iterable[dict[str, Any]]) -> RouteValidationResult:
        issues: list[RouteValidationIssue] = []
        required_seen: set[str] = set()

        for event in events:
            frame = event.get("frame") if isinstance(event.get("frame"), int) else None
            step_id = event.get("route_step") or event.get("step_id")
            if isinstance(step_id, str) and step_id in self.by_id:
                step = self.by_id[step_id]
                if step.required:
                    required_seen.add(step.id)
                if self.forbid_starworld and "starworld" in step.tags:
                    issues.append(
                        RouteValidationIssue(
                            code="starworld_step",
                            message=f"forbidden Starworld route step {step.id}",
                            frame=frame,
                            step_id=step.id,
                        )
                    )
            if self.detect_starworld(event):
                issues.append(
                    RouteValidationIssue(
                        code="starworld_detected",
                        message="trace appears to enter Starworld/Star Road",
                        frame=frame,
                        step_id=str(step_id) if step_id else None,
                    )
                )
            if event.get("savestate_used") and not self.allow_savestates_in_final:
                issues.append(
                    RouteValidationIssue(
                        code="savestate_used",
                        message="savestate used in final-validation trace",
                        frame=frame,
                        step_id=str(step_id) if step_id else None,
                    )
                )
            if event.get("memory_write_used") and not self.allow_memory_writes:
                issues.append(
                    RouteValidationIssue(
                        code="memory_write_used",
                        message="memory write used in trace",
                        frame=frame,
                        step_id=str(step_id) if step_id else None,
                    )
                )

        missing = [step.id for step in self.steps if step.required and step.id not in required_seen]
        for step_id in missing:
            issues.append(
                RouteValidationIssue(
                    code="missing_required_step",
                    message=f"required step not observed: {step_id}",
                    step_id=step_id,
                )
            )
        return RouteValidationResult(ok=not issues, issues=issues, required_steps_seen=required_seen)

