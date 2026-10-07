"""Live evolution session -- the Play / Step / Reset engine behind the Theater.

Holds a persistent MAP-Elites archive + phylogeny that can be STEPPED one generation
at a time (or run continuously in a background thread for Play). Because the genomes
are kept in the phylogeny's in-memory cache, any node in the tree can be fetched and
rendered (3D, anatomy, budgets). The slow RL training happens OUTSIDE the data lock so
the tree stays responsive while a generation evaluates.
"""
from __future__ import annotations

import threading
import time


class EvolutionSession:
    def __init__(self):
        self._step_lock = threading.Lock()      # serialize generations (one at a time)
        self._data_lock = threading.Lock()      # brief: archive/phylogeny reads + writes
        self._play_thread = None
        self._ready = False                     # engine objects built lazily (light startup)
        self._cfg = dict(seed_creature="quadruped", niche="locomotion", bins=10, seed=0)
        self.seed_creature = "quadruped"
        self.niche = "locomotion"
        self.generation = 0
        self.playing = False

    def _ensure(self):
        if not self._ready:
            self._build(**self._cfg)

    # -- lifecycle ----------------------------------------------------------
    def reset(self, seed_creature: str, niche: str, bins: int, seed: int):
        self._cfg = dict(seed_creature=seed_creature, niche=niche, bins=bins, seed=seed)
        self._ready = False
        self._build(seed_creature, niche, bins, seed)

    def _build(self, seed_creature: str, niche: str, bins: int, seed: int):
        self.playing = False
        if self._play_thread:
            self._play_thread.join(timeout=0.1)
        from personal_cambrian.evo import MAPElites
        from personal_cambrian.evo.phylogeny import Phylogeny
        from personal_cambrian.encoding.mutate import MutationSchedule
        from personal_cambrian.seeds import SEEDS
        import numpy as np
        self.seed_creature = seed_creature
        self.niche = niche
        self.archive = MAPElites(["aspect", "limb_count"], bins)
        self.phylo = Phylogeny()
        self.policies = {}
        self.rng = np.random.default_rng(seed)
        # mostly MICRO (refine proportions + control) with occasional macro (new
        # structure) -- so a lineage hill-climbs a working body instead of bloating.
        self.sched = MutationSchedule(macro_rate=0.25, micro_sigma=0.1)
        self.seed0 = SEEDS[seed_creature]()
        self.generation = 0
        self._seeded = False
        # REAL training budgets: enough PPO that the controller actually learns a GAIT
        # (warm-start compounds down a lineage). Bigger = better gaits, slower gens.
        self.ep_steps, self.fine_tune, self.seed_steps = 220, 4500, 28000
        self.n_envs = 4
        # progress tracking (the honest "is it improving?" signal)
        self.best_fitness = -1e9
        self.best_distance = -1e9
        self.best_node = None
        self.history = []                       # [{gen, best_fit, best_dist}]
        self._ready = True

    def _overlap_penalty(self, morph) -> float:
        """Summed inter-penetration depth of NON-adjacent bodies -- discourages the
        self-colliding, overlapping-limb morphologies evolution otherwise drifts into."""
        import numpy as np
        bodies = morph.bodies

        def rad(b):
            d = b.dims
            s = b.shape.value if hasattr(b.shape, "value") else str(b.shape)
            if s == "capsule":
                return float(d["radius"]) + 0.5 * float(d["length"])
            if s in ("box", "ellipsoid"):
                return max(float(d["x"]), float(d["y"]), float(d["z"]))
            return float(d["radius"])

        pos = [np.array(b.world_pos, dtype=float) for b in bodies]
        r = [rad(b) for b in bodies]
        pen = 0.0
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                bi, bj = bodies[i], bodies[j]
                if bj.parent_id == bi.id or bi.parent_id == bj.id:
                    continue                                    # adjacent -> allowed to touch
                overlap = (r[i] + r[j]) * 0.8 - float(np.linalg.norm(pos[i] - pos[j]))
                if overlap > 0:
                    pen += overlap
        return pen

    # -- one generation -----------------------------------------------------
    def _env_fn(self, genome):
        from personal_cambrian.sim import CreatureEnv
        from personal_cambrian.sim.tasks import NICHES
        return lambda s: CreatureEnv(genome, task=NICHES[self.niche](max_steps=self.ep_steps),
                                     obs_mode="structured")

    def _eval_and_insert(self, genome, parent_policy, parent_node, mutations, gen, steps):
        from personal_cambrian.evo.qd import pre_sim_gate, budget_penalty
        from personal_cambrian.control import (PPOConfig, evaluate,
                                               warm_started_make_agent, scratch_make_agent)
        from personal_cambrian.control.ppo import train
        from personal_cambrian.evo.descriptors import morphology_descriptors
        ok, _r = pre_sim_gate(genome)
        if not ok:
            return None
        cme = self._env_fn(genome)
        try:
            env = cme(0)
        except Exception:        # noqa: BLE001
            return None
        ma = (warm_started_make_agent(cme, parent_policy) if parent_policy is not None
              else scratch_make_agent(cme, 32))
        policy, _ = train(cme, PPOConfig(total_timesteps=steps, n_envs=self.n_envs,
                                         n_steps=128, hidden=32, seed=0), make_agent=ma)
        from personal_cambrian.morphogenesis import develop
        perf = evaluate(policy, env, max_steps=self.ep_steps)
        env.close()
        dist = perf["distance"]
        # SELECT ON THE SHAPED LOCOMOTION REWARD, not raw distance: forward velocity −
        # energy + alive bonus, with fall TERMINATION -> a creature that topples forward
        # is cut short (low return), one that stays upright and strides scores high. This
        # is what kills the "grow tall legs, fall forward" exploit.
        overlap = self._overlap_penalty(develop(genome))
        fitness = perf["return"] - budget_penalty(genome, 1.0) - 2.5 * overlap
        desc = morphology_descriptors(genome)
        with self._data_lock:
            if parent_node is None:
                node = self.phylo.add_root(genome, fitness=fitness, generation=gen)
            else:
                node = self.phylo.add_offspring(parent_node, genome, mutations,
                                                fitness=fitness, generation=gen)
            self.policies[node] = policy
            self.archive.add(genome, fitness, desc, node=node)
            if fitness > self.best_fitness:
                self.best_fitness, self.best_distance, self.best_node = fitness, dist, node
            self.history.append({"gen": gen, "best_fit": round(self.best_fitness, 3),
                                 "best_dist": round(self.best_distance, 3)})
            if len(self.history) > 500:
                self.history = self.history[-500:]
        return node

    def ensure_seeded(self):
        if not self._seeded:
            self._eval_and_insert(self.seed0, None, None, None, 0, self.seed_steps)
            self._seeded = True

    def step(self, n: int = 1):
        from personal_cambrian.encoding.mutate import mutate
        self._ensure()
        with self._step_lock:
            self.ensure_seeded()
            for _ in range(n):
                with self._data_lock:
                    # 45% elitism (mutate the current best -> push the frontier up),
                    # else uniform cell sampling (keep exploring the diversity grid).
                    if self.best_node is not None and self.rng.random() < 0.45:
                        elite = self.archive.best()
                    else:
                        elite = self.archive.sample(self.rng)
                if elite is None:
                    break
                pnode = elite.meta["node"]
                child, recs = mutate(elite.genome, self.rng,
                                     generation=self.generation + 1, schedule=self.sched)
                self._eval_and_insert(child, self.policies.get(pnode), pnode, recs,
                                      self.generation + 1, self.fine_tune)
                self.generation += 1

    # -- play (background) --------------------------------------------------
    def play(self):
        if self.playing:
            return
        self.playing = True
        self._play_thread = threading.Thread(target=self._loop, daemon=True)
        self._play_thread.start()

    def pause(self):
        self.playing = False

    def _loop(self):
        while self.playing:
            self.step(1)
            time.sleep(0.03)

    # -- read views ---------------------------------------------------------
    def genome(self, node_id: str):
        return self.phylo.genome(node_id)

    def _trait(self, desc) -> str:
        bc, asp, sym = desc["limb_count"], desc["aspect"], desc["symmetry"]
        if asp > 3.2:
            return "serpentine"
        if sym > 0.45 and bc >= 10:
            return "quadruped"
        if bc <= 8:
            return "compact"
        return "radial"

    def tree(self) -> dict:
        if not self._ready:
            return {"nodes": [], "generation": 0, "max_gen": 0, "playing": False,
                    "n_nodes": 0, "seeded": False, "seed_creature": self.seed_creature,
                    "niche": self.niche}
        from personal_cambrian.viz import tree_layout
        from personal_cambrian.evo.descriptors import morphology_descriptors
        with self._data_lock:
            nodes_copy = dict(self.phylo.nodes)
            pos = tree_layout(self.phylo) if nodes_copy else {}
            genomes = {nid: self.phylo.genome(nid) for nid in nodes_copy}
        nodes = []
        for nid, nd in nodes_copy.items():
            g = genomes.get(nid)
            desc = morphology_descriptors(g) if g is not None else {"limb_count": 0,
                  "muscle_count": 0, "mass": 0, "aspect": 1, "symmetry": 0, "height": 0}
            x, y = pos.get(nid, (nd.generation, 0))
            nodes.append({
                "id": nid, "parent": nd.parent, "gen": nd.generation, "x": x, "y": y,
                "innov": list(nd.innovations or []),
                "fitness": round((nd.meta or {}).get("fitness", 0.0), 3),
                "trait": self._trait(desc),
                "limbs": int(desc["limb_count"]), "muscles": int(desc["muscle_count"]),
                "mass": round(desc["mass"], 1),
            })
        return {"nodes": nodes, "generation": self.generation,
                "max_gen": max((n["gen"] for n in nodes), default=0),
                "playing": self.playing, "n_nodes": len(nodes), "seeded": self._seeded,
                "seed_creature": self.seed_creature, "niche": self.niche,
                "best_node": self.best_node,
                "best_fitness": round(self.best_fitness, 3) if self.best_fitness > -1e8 else None,
                "best_distance": round(self.best_distance, 3) if self.best_distance > -1e8 else None,
                "history": self.history[-160:]}

    def status(self) -> dict:
        return {"generation": self.generation, "playing": self.playing,
                "n_nodes": len(self.phylo.nodes) if self._ready else 0,
                "seeded": getattr(self, "_seeded", False),
                "best_distance": round(self.best_distance, 3) if getattr(self, "best_distance", -1e9) > -1e8 else None,
                "seed_creature": self.seed_creature, "niche": self.niche}


SESSION = EvolutionSession()
