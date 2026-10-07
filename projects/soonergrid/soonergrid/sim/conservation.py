"""Conservative shared receiving allocation and explicit boundary queues.

Flows are proposed simultaneously, then scaled proportionally at merges.
A receiving budget is shared by every upstream edge in a timestep.
"""
import math


def validate_run(dt, duration, interval, timeline):
    if not all(math.isfinite(v) and v > 0 for v in (dt, duration, interval)):
        raise ValueError("Time parameters must be finite and positive")
    if not math.isclose(duration / dt, round(duration / dt), abs_tol=1e-8):
        raise ValueError("Duration must be an integer number of timesteps")
    if not timeline:
        raise ValueError("Demand timeline cannot be empty")
    for row in timeline:
        if any(not math.isfinite(v) or v < 0 for v in row.get("gateway_inflows", {}).values()):
            raise ValueError("Demand must be finite and nonnegative")


def allocate_movements(proposals, receiving):
    """Return (source, target, volume) with aggregate target inflow bounded."""
    totals = {}
    for source, target, volume in proposals:
        totals[target] = totals.get(target, 0.0) + volume
    return [(source, target, volume * min(1.0, receiving[target] / totals[target]))
            for source, target, volume in proposals if volume > 0]


def admit_boundary(vehicles, metadata, queue, arrivals, dt):
    """Queue rejected demand outside the modeled network; never discard it."""
    admitted = 0.0
    for eid, volume in arrivals.items():
        queue[eid] = queue.get(eid, 0.0) + volume
    for eid in list(queue):
        m = metadata[eid]
        amount = min(queue[eid], max(0.0, m["jam_storage_veh"] - vehicles[eid]),
                     m["capacity_vps"] * dt)
        vehicles[eid] += amount
        queue[eid] -= amount
        admitted += amount
    return admitted
