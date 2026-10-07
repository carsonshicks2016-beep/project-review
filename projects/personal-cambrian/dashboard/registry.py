"""Action registry -- the keystone of the dashboard.

Every controllable function in the engine is registered as an `Action` with a typed
field schema. The HTTP layer auto-generates a form from the schema and the job
orchestrator runs the action's `runner(params, ctx)`. Adding a control = registering
an action; nothing in the engine is touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Field:
    name: str
    type: str                       # int | float | bool | str | enum
    default: Any = None
    label: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list] = None
    help: str = ""

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type, "default": self.default,
             "label": self.label or self.name, "help": self.help}
        for k in ("min", "max", "step", "options"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    def coerce(self, v):
        if v is None:
            return self.default
        try:
            if self.type == "int":
                return int(float(v))
            if self.type == "float":
                return float(v)
            if self.type == "bool":
                return v in (True, "true", "True", 1, "1", "on")
            return str(v)
        except (TypeError, ValueError):
            return self.default


@dataclass
class Action:
    id: str
    label: str
    category: str
    runner: Callable
    kind: str = "value"             # value | stream | render
    fields: list = field(default_factory=list)
    streams: list = field(default_factory=list)   # metric keys worth charting
    description: str = ""
    danger: bool = False            # heavy / sensitive -> UI confirms

    def schema(self) -> dict:
        return {"id": self.id, "label": self.label, "category": self.category,
                "kind": self.kind, "description": self.description,
                "danger": self.danger, "streams": self.streams,
                "fields": [f.to_dict() for f in self.fields]}

    def coerce(self, params: dict) -> dict:
        params = params or {}
        return {f.name: f.coerce(params.get(f.name)) for f in self.fields}


ACTIONS: dict = {}


def action(*, id: str, label: str, category: str, kind: str = "value",
           fields: Optional[list] = None, streams: Optional[list] = None,
           description: str = "", danger: bool = False):
    def deco(fn):
        ACTIONS[id] = Action(id=id, label=label, category=category, kind=kind,
                             fields=fields or [], streams=streams or [],
                             description=description, danger=danger, runner=fn)
        return fn
    return deco


def all_actions() -> list:
    return list(ACTIONS.values())


def get_action(action_id: str) -> Optional[Action]:
    return ACTIONS.get(action_id)


def categories() -> list:
    seen = []
    for a in ACTIONS.values():
        if a.category not in seen:
            seen.append(a.category)
    return seen
