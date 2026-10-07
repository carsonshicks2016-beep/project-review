"""PHYLOGENY TREE RENDER (ROADMAP Stage 10.3).

Renders the anatomical family tree -- the project's headline output -- as a 2D
rectangular cladogram: generation runs left->right (so the x-axis doubles as the
INNOVATION TIMELINE), leaves stack vertically, and a coloured marker is placed on the
branch where each structural innovation (+limb / +muscle / topology_change / ...)
FIRST appears. Reads either a live `Phylogeny` (Stage 4.5) or its exported JSON
(Stage 7.4), and writes a PNG (matplotlib).

The layout/marker logic is pure and unit-testable; only the final draw needs
matplotlib.
"""
from __future__ import annotations

import json
from typing import Optional

# fixed colours for the innovation vocabulary (Stage 4.4 flags)
INNOVATION_COLORS = {
    "+limb": "#2ca02c", "-limb": "#98df8a",
    "+muscle": "#d62728", "-muscle": "#ff9896",
    "+compartment": "#9467bd", "topology_change": "#1f77b4",
    "dof_change": "#ff7f0e",
}
_DEFAULT_COLOR = "#7f7f7f"


# --- normalize a source into (nodes, children, roots) ----------------------
def _normalize(source):
    """Accept a Phylogeny, an exported dict, or a JSON path -> (nodes, children, roots).
    `nodes[id] = {gen, parent, innov, fitness}`."""
    if isinstance(source, str):
        with open(source) as f:
            source = json.load(f)

    if isinstance(source, dict):                         # exported JSON {"nodes": [...]}
        nodes = {}
        for n in source["nodes"]:
            nodes[n["id"]] = {"gen": int(n.get("generation", 0)),
                              "parent": n.get("parent"),
                              "innov": list(n.get("innovations", []) or []),
                              "fitness": (n.get("meta") or {}).get("fitness")}
    else:                                                # a live Phylogeny
        nodes = {nid: {"gen": nd.generation, "parent": nd.parent,
                       "innov": list(nd.innovations or []),
                       "fitness": (nd.meta or {}).get("fitness")}
                 for nid, nd in source.nodes.items()}

    children: dict = {nid: [] for nid in nodes}
    roots = []
    for nid, n in nodes.items():
        p = n["parent"]
        if p is not None and p in children:
            children[p].append(nid)
        else:
            roots.append(nid)
    for k in children:                                   # stable order
        children[k].sort()
    return nodes, children, sorted(roots)


# --- layout (pure) ----------------------------------------------------------
def tree_layout(source) -> dict:
    """(x, y) per node: x = generation, y from an in-order leaf assignment (internal
    nodes centred over their descendants)."""
    nodes, children, roots = _normalize(source)
    pos: dict = {}
    counter = [0]

    def assign(nid):
        kids = children[nid]
        if not kids:
            y = float(counter[0]); counter[0] += 1
        else:
            y = sum(assign(k) for k in kids) / len(kids)
        pos[nid] = (float(nodes[nid]["gen"]), y)
        return y

    for r in roots:
        assign(r)
    return pos


def innovation_markers(source) -> list:
    """[(x, y, flag)] -- one marker per innovation, at the branch (node) where it first
    appears. Together these are the innovation timeline laid over the tree."""
    nodes, _children, _roots = _normalize(source)
    pos = tree_layout(source)
    out = []
    for nid, n in nodes.items():
        x, y = pos[nid]
        for flag in n["innov"]:
            out.append((x, y, flag))
    return out


# --- render (matplotlib) ----------------------------------------------------
def render_phylogeny(source, out: str, *, title: str = "Phylogeny", figsize=None) -> str:
    """Draw the rectangular cladogram with innovation markers and save a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nodes, children, roots = _normalize(source)
    pos = tree_layout(source)
    n_leaves = sum(1 for nid in nodes if not children[nid])
    max_gen = max((p[0] for p in pos.values()), default=1)
    fig, ax = plt.subplots(figsize=figsize or (max(6, max_gen * 0.5 + 2),
                                               max(3, n_leaves * 0.22 + 1)))

    # edges: rectangular cladogram (vertical bar at parent x, horizontals to kids)
    for nid, kids in children.items():
        if not kids:
            continue
        xp, _ = pos[nid]
        ys = [pos[k][1] for k in kids]
        ax.plot([xp, xp], [min(ys), max(ys)], color="#bbbbbb", lw=1.0, zorder=1)
        for k in kids:
            xk, yk = pos[k]
            ax.plot([xp, xk], [yk, yk], color="#bbbbbb", lw=1.0, zorder=1)

    # innovation markers on the introducing branch
    seen = set()
    for x, y, flag in innovation_markers(source):
        c = INNOVATION_COLORS.get(flag, _DEFAULT_COLOR)
        ax.scatter([x], [y], s=42, color=c, edgecolors="black", linewidths=0.4,
                   zorder=3, label=flag if flag not in seen else None)
        seen.add(flag)

    # roots
    for r in roots:
        ax.scatter([pos[r][0]], [pos[r][1]], marker="s", s=36, color="black", zorder=4)

    ax.set_xlabel("generation  (innovation timeline →)")
    ax.set_yticks([])
    ax.set_title(f"{title}   ({len(nodes)} creatures, {n_leaves} leaves)")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=INNOVATION_COLORS[f],
                      markeredgecolor="black", markersize=7, label=f)
               for f in sorted(seen)]
    if handles:
        ax.legend(handles=handles, title="innovations", fontsize=8,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.margins(y=0.05)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def render_from_json(json_path: str, out: str, *, title: Optional[str] = None) -> str:
    return render_phylogeny(json_path, out, title=title or "Phylogeny")
