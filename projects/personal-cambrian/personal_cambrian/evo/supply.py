"""SUPPLY / CONNECTIVITY CHECK (ROADMAP Stage 6.4): every muscle must be reachable.

A muscle needs abstract blood + neural supply, which can only reach it through
the body. We model that as graph reachability: a part is supplied iff there is a
path of connection edges from the root to it. A muscle whose origin, insertion,
or any routing site sits on an UNREACHABLE part is an orphan -- it could never be
fed or innervated -- so the body is rejected before it is grown or simulated.

`validate_shape` already guarantees a muscle's anchor parts *exist*; it does NOT
check that they are *connected* to the root (a part can be defined but never
attached by any edge). This check closes that gap. It is pure graph analysis on
the genome -- no morphogenesis, no MuJoCo -- so it is the cheapest gate of all.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..encoding.genome import Genome


def _reachable_parts(genome: Genome) -> set:
    """Parts reachable from the root by following connection edges (parent->child)."""
    children: dict[str, list[str]] = defaultdict(list)
    for e in genome.edges:
        children[e.parent_part].append(e.child_part)
    reachable = set()
    stack = [genome.root]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(children.get(node, ()))
    return reachable


@dataclass
class SupplyReport:
    """Connectivity of every muscle to the root supply."""
    supplied: bool
    reasons: list = field(default_factory=list)
    orphan_muscles: list = field(default_factory=list)   # muscle ids
    unreachable_parts: list = field(default_factory=list)  # part ids (diagnostic)
    n_parts: int = 0
    n_reachable: int = 0

    def __bool__(self) -> bool:
        return self.supplied

    def to_dict(self) -> dict:
        return {
            "supplied": self.supplied, "reasons": list(self.reasons),
            "orphan_muscles": list(self.orphan_muscles),
            "unreachable_parts": list(self.unreachable_parts),
            "n_parts": self.n_parts, "n_reachable": self.n_reachable,
        }


def check_supply(genome: Genome) -> SupplyReport:
    """Reject bodies with orphaned (unreachable) muscles.

    Returns a `SupplyReport`; `bool(report)` is True iff every muscle is supplied.
    """
    reachable = _reachable_parts(genome)
    all_ids = set(genome.part_ids)
    report = SupplyReport(
        supplied=True,
        unreachable_parts=sorted(all_ids - reachable),
        n_parts=len(all_ids),
        n_reachable=len(reachable & all_ids),
    )
    for m in genome.muscles:
        anchors = {m.origin[0], m.insertion[0]}
        anchors.update(r[0] for r in m.route_sites)
        if not anchors <= reachable:        # any anchor part unreachable
            report.orphan_muscles.append(m.id)

    if report.orphan_muscles:
        report.reasons.append("orphan_muscle")
    report.supplied = not report.reasons
    return report
