"""Phylogeny: the lineage tree of evolved body plans (ROADMAP Stage 4.5).

Each node records a creature's genome hash, its parent, the mutations that made it,
and the structural innovations they introduced (auto-classified via Stage 4.4
`diff`). The tree answers ancestry and innovation-timeline queries — e.g. "when did
this lineage first grow a limb?" — and serializes to JSON. (When the SQLite store,
F.5, lands, this becomes the `lineage` table; the API stays the same.)

Genomes are cached in-memory so innovations can be computed at insertion time, but
they are NOT serialized — the persisted tree keeps hashes + flags + mutations, so
ancestry and the innovation timeline remain queryable after load.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from ..encoding.genome import Genome
from ..encoding.innovation import diff


@dataclass
class PhyloNode:
    id: str
    genome_hash: str
    parent: Optional[str] = None
    generation: int = 0
    mutations: list = field(default_factory=list)     # mutation kinds, parent->this
    innovations: list = field(default_factory=list)   # structural-innovation flags
    distance: int = 0                                  # graph-edit distance from parent
    meta: dict = field(default_factory=dict)           # e.g. fitness, niche


def _kinds(mutations) -> list:
    out = []
    for m in mutations or []:
        out.append(m.kind if hasattr(m, "kind") else str(m))
    return out


class Phylogeny:
    def __init__(self):
        self.nodes: dict[str, PhyloNode] = {}
        self._children: dict[str, list[str]] = {}
        self._genomes: dict[str, Genome] = {}          # in-memory only (not serialized)
        self._n = 0

    def _new_id(self) -> str:
        self._n += 1
        return f"n{self._n}"

    # -- building -----------------------------------------------------------
    def add_root(self, genome: Genome, *, generation: int = 0,
                 node_id: Optional[str] = None, **meta) -> str:
        nid = node_id or self._new_id()
        self.nodes[nid] = PhyloNode(nid, genome.hash(), None, generation, meta=dict(meta))
        self._children[nid] = []
        self._genomes[nid] = genome
        return nid

    def add_offspring(self, parent_id: str, child_genome: Genome, mutations=None, *,
                      generation: Optional[int] = None, node_id: Optional[str] = None,
                      **meta) -> str:
        if parent_id not in self.nodes:
            raise KeyError(f"unknown parent '{parent_id}'")
        nid = node_id or self._new_id()
        innov, dist = [], 0
        pg = self._genomes.get(parent_id)
        if pg is not None:                              # classify the structural change
            rep = diff(pg, child_genome)
            innov, dist = sorted(rep.flags), rep.distance
        gen = generation if generation is not None else self.nodes[parent_id].generation + 1
        self.nodes[nid] = PhyloNode(nid, child_genome.hash(), parent_id, gen,
                                    _kinds(mutations), innov, dist, dict(meta))
        self._children[parent_id].append(nid)
        self._children[nid] = []
        self._genomes[nid] = child_genome
        return nid

    # -- queries ------------------------------------------------------------
    def roots(self) -> list:
        return [nid for nid, n in self.nodes.items() if n.parent is None]

    def children(self, nid: str) -> list:
        return list(self._children.get(nid, []))

    def ancestry(self, nid: str) -> list:
        """Node ids from root down to `nid` (inclusive)."""
        chain, cur = [], nid
        while cur is not None:
            chain.append(cur)
            cur = self.nodes[cur].parent
        return list(reversed(chain))

    def descendants(self, nid: str) -> list:
        out, stack = [], list(self._children.get(nid, []))
        while stack:
            c = stack.pop()
            out.append(c)
            stack.extend(self._children.get(c, []))
        return out

    def depth(self, nid: str) -> int:
        return len(self.ancestry(nid)) - 1

    def genome(self, nid: str):
        """The in-memory genome cached for `nid` (None if not retained)."""
        return self._genomes.get(nid)

    def leaves(self) -> list:
        """Node ids with no offspring (the tips of the lineage tree)."""
        return [nid for nid in self.nodes if not self._children.get(nid)]

    def innovation_timeline(self, nid: str) -> list:
        """Along `nid`'s ancestry, the (node, generation, flags) where each
        structural innovation first appeared on this lineage."""
        return [(a, self.nodes[a].generation, self.nodes[a].innovations)
                for a in self.ancestry(nid) if self.nodes[a].innovations]

    def first_appearance(self, flag: str) -> Optional[str]:
        """Earliest node (by generation) anywhere in the tree to introduce `flag`."""
        cands = [n for n in self.nodes.values() if flag in n.innovations]
        return min(cands, key=lambda n: (n.generation, n.id)).id if cands else None

    def _nhx_tag(self, nid: str) -> str:
        """NHX (New Hampshire eXtended) comment block annotating one node with its
        generation, structural innovations, mutations, and any meta (fitness/niche).
        NHX is read by FigTree / Dendroscope / ETE / Archaeopteryx tree viewers."""
        n = self.nodes[nid]
        fields = [f"gen={n.generation}", f"dist={n.distance}"]
        if n.innovations:
            fields.append("innov=" + "|".join(n.innovations))
        if n.mutations:
            fields.append("mut=" + "|".join(n.mutations))
        fit = n.meta.get("fitness")
        if isinstance(fit, (int, float)):
            fields.append(f"fit={fit:.4g}")
        if "niche" in n.meta:
            fields.append(f"niche={n.meta['niche']}")
        fields.append(f"hash={n.genome_hash[:8]}")
        return "[&&NHX:" + ":".join(fields) + "]"

    def to_newick(self, *, branch_lengths: bool = False, nhx: bool = False) -> str:
        """Newick string. Default is topology-only (Stage 4.5, back-compatible).

        `branch_lengths=True` sets each branch length to the graph-edit distance
        from its parent (morphological change along the branch); `nhx=True`
        appends an innovation-annotated NHX comment to every node."""
        def rec(nid):
            kids = self._children.get(nid, [])
            s = (f"({','.join(rec(k) for k in kids)}){nid}" if kids else nid)
            if branch_lengths:
                s += f":{self.nodes[nid].distance}"
            if nhx:
                s += self._nhx_tag(nid)
            return s
        return ";".join(rec(r) for r in self.roots()) + ";"

    def to_nhx(self) -> str:
        """Newick with branch lengths + innovation-annotated NHX on every node."""
        return self.to_newick(branch_lengths=True, nhx=True)

    def export(self, path_stem: str) -> dict:
        """Write the family tree as `<stem>.nwk` (NHX, for tree viewers / Stage-10
        Blender) and `<stem>.json` (full structural record). Returns the paths."""
        nwk_path, json_path = path_stem + ".nwk", path_stem + ".json"
        with open(nwk_path, "w") as f:
            f.write(self.to_nhx() + "\n")
        self.save(json_path)
        return {"newick": nwk_path, "json": json_path}

    # -- persistence (structure only; genomes are not serialized) -----------
    def to_dict(self) -> dict:
        return {"nodes": [asdict(self.nodes[n]) for n in self.nodes]}

    @classmethod
    def from_dict(cls, d: dict) -> "Phylogeny":
        p = cls()
        for nd in d["nodes"]:
            node = PhyloNode(**nd)
            p.nodes[node.id] = node
            p._children.setdefault(node.id, [])
            if node.parent is not None:
                p._children.setdefault(node.parent, []).append(node.id)
            n = int(node.id[1:]) if node.id[1:].isdigit() else 0
            p._n = max(p._n, n)
        return p

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Phylogeny":
        with open(path) as f:
            return cls.from_dict(json.load(f))
