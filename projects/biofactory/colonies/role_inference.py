from __future__ import annotations

from typing import Protocol


class RoleLike(Protocol):
    deliveries: int
    distance_from_nest_max: float
    construction_contributed: float
    building_ticks: int
    defend_ticks: int
    attack_intent_ticks: int
    waste_removed: float
    kills: int
    carrying_ticks: int
    nursing_ticks: int
    larvae_helped: float
    fungus_tended: float
    target_chamber_id: int | None
    age: float


def infer_role(ant: RoleLike) -> str:
    if ant.kills > 0:
        return "Soldier"
    if ant.attack_intent_ticks > max(30, ant.age * 0.12):
        return "Soldier"
    if ant.defend_ticks > max(45, ant.age * 0.18):
        return "Defender"
    if ant.construction_contributed > 5 or ant.building_ticks > max(60, ant.age * 0.20):
        return "Builder"
    if ant.waste_removed > 3:
        return "Recycler"
    if ant.nursing_ticks > 120 or ant.larvae_helped > 1.0:
        return "Nurse"
    if ant.fungus_tended > 1.0:
        return "Farmer"
    if ant.deliveries >= 12 and ant.target_chamber_id is not None and ant.distance_from_nest_max < 35:
        return "Courier"
    if ant.deliveries >= 4 or ant.carrying_ticks > ant.age * 0.25:
        return "Carrier"
    if ant.distance_from_nest_max > 45:
        return "Scout"
    if ant.age > 200 and ant.deliveries == 0:
        return "Idle"
    return "Worker"
