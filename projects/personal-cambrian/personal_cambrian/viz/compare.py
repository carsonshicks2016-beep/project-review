"""SIDE-BY-SIDE LINEAGE COMPARISON (ROADMAP Stage 10.5).

Arrays several morphologies -- typically an ancestor->descendant LINEAGE -- into one
`ScenePlan`, each creature offset along an axis (with prefixed object names so they
don't collide), so a single render shows the morphological DRIFT across generations.
Reuses `morphology_plan` + the static `blender_build.build_from_plan`; pure and
bpy-free.
"""
from __future__ import annotations

from ..morphogenesis import develop
from .morphology_plan import GeomSpec, MuscleSpec, ScenePlan, build_plan


def _axis_extent(plan, axis):
    xs = [g.pos[axis] for g in plan.geoms] or [0.0]
    return max(xs) - min(xs)


def lineage_plan(genomes, *, spacing: float = None, axis: int = 0,
                 name: str = "lineage") -> ScenePlan:
    """Combine N developed genomes into one ScenePlan, offset by `spacing` along `axis`.
    Names are prefixed `gI_` so bodies/muscles from different creatures stay distinct."""
    plans = [build_plan(develop(g)) for g in genomes]
    if spacing is None:
        spacing = max([_axis_extent(p, axis) for p in plans] + [0.4]) + 0.8

    geoms, muscles = [], []
    for i, p in enumerate(plans):
        off = [0.0, 0.0, 0.0]
        off[axis] = i * spacing
        for g in p.geoms:
            geoms.append(GeomSpec(
                name=f"g{i}_{g.name}", shape=g.shape, dims=dict(g.dims),
                pos=tuple(g.pos[k] + off[k] for k in range(3)), quat=g.quat,
                depth=g.depth, part_id=g.part_id,
                parent=None if g.parent is None else f"g{i}_{g.parent}"))
        for m in p.muscles:
            muscles.append(MuscleSpec(
                name=f"g{i}_{m.name}",
                points=[tuple(pt[k] + off[k] for k in range(3)) for pt in m.points],
                fmax=m.fmax))
    return ScenePlan(name=name, geoms=geoms, muscles=muscles)


def lineage_genomes(phylo, leaf: str = None) -> list:
    """Genomes along a root->leaf lineage of a phylogeny (default: the most-divergent
    leaf), for an ancestor->descendant comparison. Needs the in-memory genome cache."""
    if leaf is None:
        from ..evo.deeptime import most_divergent
        leaf, _ = most_divergent(phylo)
    if leaf is None:
        return []
    out = []
    for nid in phylo.ancestry(leaf):
        g = phylo.genome(nid)
        if g is not None:
            out.append(g)
    return out


def lineage_plan_from_phylogeny(phylo, *, leaf: str = None, name: str = "lineage",
                                **kw) -> ScenePlan:
    return lineage_plan(lineage_genomes(phylo, leaf), name=name, **kw)
