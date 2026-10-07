import random
import numpy as np
from brain import Genome, NodeGene, ConnectionGene


class InnovationTracker:
    def __init__(self, start=0):
        self._map = {}
        self._counter = start

    def get(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self._map:
            self._map[key] = self._counter
            self._counter += 1
        return self._map[key]

    @property
    def count(self):
        return self._counter


class Species:
    def __init__(self, species_id, representative, parent_id=None, birth_gen=0):
        self.id = species_id
        self.representative = representative
        self.members = []
        self.best_fitness = 0.0
        self.stagnation = 0
        self.age = 0
        # Lineage for the phylogenetic tree
        self.parent_id  = parent_id
        self.birth_gen  = birth_gen
        self.death_gen  = None      # set when this species goes extinct

    def add(self, genome):
        self.members.append(genome)
        genome.species_id = self.id


# NEAT hyperparameters
class Cfg:
    weight_mutate_rate    = 0.80
    weight_perturb_rate   = 0.90   # vs random reset
    weight_perturb_power  = 0.40
    add_conn_rate         = 0.06
    add_node_rate         = 0.03

    compat_threshold      = 3.0
    c1 = 1.0   # excess coefficient
    c2 = 1.0   # disjoint coefficient
    c3 = 0.4   # avg weight diff coefficient

    survival_ratio        = 0.20
    elitism               = 2      # top N per species pass unchanged
    max_stagnation        = 15
    interspecies_rate     = 0.001


class NEAT:
    def __init__(self, pop_size, n_inputs, n_outputs):
        self.pop_size  = pop_size
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        self.innov     = InnovationTracker()
        self.species   = []
        self.species_archive = []   # every Species ever seen (for lineage tree)
        self._sid      = 0
        self.generation = 0
        self.best_genome = None
        self.best_fitness_ever = 0.0

        # Seed innovation numbers for input->output connections
        for i in range(n_inputs):
            for j in range(n_outputs):
                self.innov.get(i, n_inputs + j)

        self.population = self._init_population()
        self._speciate()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            g = Genome(self.n_inputs, self.n_outputs)
            # Sparse random initial connections
            for i in range(self.n_inputs):
                for j in range(self.n_outputs):
                    if random.random() < 0.5:
                        inn = self.innov.get(i, self.n_inputs + j)
                        g.connections[inn] = ConnectionGene(
                            i, self.n_inputs + j,
                            random.uniform(-1.0, 1.0), True, inn
                        )
            pop.append(g)
        return pop

    # ------------------------------------------------------------------
    # Speciation
    # ------------------------------------------------------------------

    def _genomic_distance(self, g1, g2):
        s1 = set(g1.connections)
        s2 = set(g2.connections)
        if not s1 and not s2:
            return 0.0

        max1 = max(s1) if s1 else 0
        max2 = max(s2) if s2 else 0
        min_max = min(max1, max2)

        all_keys = s1 | s2
        matching = s1 & s2
        non_match = all_keys - matching

        excess   = sum(1 for k in non_match if k > min_max)
        disjoint = len(non_match) - excess

        avg_w = 0.0
        if matching:
            avg_w = sum(abs(g1.connections[k].weight - g2.connections[k].weight)
                        for k in matching) / len(matching)

        n = max(len(s1), len(s2), 1)
        n = 1 if n < 20 else n

        return (Cfg.c1 * excess / n + Cfg.c2 * disjoint / n + Cfg.c3 * avg_w)

    def _speciate(self):
        for s in self.species:
            s.members.clear()

        for genome in self.population:
            placed = False
            best_match = None
            best_dist  = float('inf')
            for sp in self.species:
                d = self._genomic_distance(genome, sp.representative)
                if d < best_dist:
                    best_dist  = d
                    best_match = sp
                if d < Cfg.compat_threshold:
                    sp.add(genome)
                    placed = True
                    break
            if not placed:
                # New species — record its parent (closest existing species)
                parent_id = best_match.id if best_match else None
                sp = Species(self._sid, genome,
                             parent_id=parent_id,
                             birth_gen=self.generation)
                self._sid += 1
                sp.add(genome)
                self.species.append(sp)
                self.species_archive.append(sp)

        # Mark extinct species
        survivors = [s for s in self.species if s.members]
        for s in self.species:
            if not s.members and s.death_gen is None:
                s.death_gen = self.generation
        self.species = survivors
        for s in self.species:
            s.representative = random.choice(s.members)

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(self):
        self.generation += 1

        # Track global best
        best = max(self.population, key=lambda g: g.fitness)
        if best.fitness > self.best_fitness_ever:
            self.best_fitness_ever = best.fitness
            self.best_genome = best.copy()

        # Adjusted fitness
        for sp in self.species:
            n = len(sp.members) or 1
            for g in sp.members:
                g.adjusted_fitness = g.fitness / n

        # Stagnation
        best_sp = max(self.species, key=lambda s: s.best_fitness) if self.species else None
        for sp in self.species:
            cur_best = max((g.fitness for g in sp.members), default=0.0)
            if cur_best > sp.best_fitness:
                sp.best_fitness = cur_best
                sp.stagnation = 0
            else:
                sp.stagnation += 1
            sp.age += 1

        kept = []
        for s in self.species:
            if s.stagnation < Cfg.max_stagnation or s is best_sp:
                kept.append(s)
            elif s.death_gen is None:
                s.death_gen = self.generation
        self.species = kept

        total_adj = sum(g.adjusted_fitness for g in self.population) or 1.0
        new_pop = []

        for sp in self.species:
            if not sp.members:
                continue

            quota = max(1, round(self.pop_size * sum(g.adjusted_fitness for g in sp.members) / total_adj))
            sp.members.sort(key=lambda g: g.fitness, reverse=True)

            # Elitism
            for g in sp.members[:Cfg.elitism]:
                new_pop.append(g.copy())

            cutoff = max(1, int(len(sp.members) * Cfg.survival_ratio))
            survivors = sp.members[:cutoff]

            for _ in range(max(0, quota - Cfg.elitism)):
                if len(survivors) > 1 and random.random() < 0.75:
                    p1 = random.choice(survivors)
                    # Interspecies mating
                    if random.random() < Cfg.interspecies_rate and len(self.species) > 1:
                        other_sp = random.choice([s for s in self.species if s is not sp])
                        p2 = random.choice(other_sp.members)
                    else:
                        p2 = random.choice(survivors)
                    child = self._crossover(p1, p2)
                else:
                    child = random.choice(survivors).copy()
                self._mutate(child)
                new_pop.append(child)

        # Fill up to pop_size
        while len(new_pop) < self.pop_size:
            parent = random.choice(self.population)
            child = parent.copy()
            self._mutate(child)
            new_pop.append(child)

        self.population = new_pop[:self.pop_size]
        self._speciate()

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover(self, p1, p2):
        if p2.fitness > p1.fitness:
            p1, p2 = p2, p1  # p1 is always the fitter parent

        child = Genome(p1.n_inputs, p1.n_outputs)

        # Inherit nodes from fitter parent
        for nid, n in p1.nodes.items():
            child.nodes[nid] = NodeGene(n.id, n.type, n.activation)

        s1 = set(p1.connections)
        s2 = set(p2.connections)
        matching = s1 & s2

        for inn in s1:
            c1 = p1.connections[inn]
            if inn in matching:
                c2 = p2.connections[inn]
                src = c1 if random.random() < 0.5 else c2
                # If disabled in either parent, 75% chance stays disabled
                enabled = src.enabled
                if not c1.enabled or not c2.enabled:
                    enabled = random.random() > 0.75
                child.connections[inn] = ConnectionGene(
                    src.in_node, src.out_node, src.weight, enabled, inn
                )
            else:
                # Disjoint/excess from fitter parent
                child.connections[inn] = ConnectionGene(
                    c1.in_node, c1.out_node, c1.weight, c1.enabled, inn
                )

            # Ensure referenced nodes exist in child
            for nid in (child.connections[inn].in_node, child.connections[inn].out_node):
                if nid not in child.nodes:
                    if nid in p1.nodes:
                        n = p1.nodes[nid]
                        child.nodes[nid] = NodeGene(n.id, n.type, n.activation)
                    elif nid in p2.nodes:
                        n = p2.nodes[nid]
                        child.nodes[nid] = NodeGene(n.id, n.type, n.activation)

        return child

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, genome):
        if random.random() < Cfg.weight_mutate_rate:
            self._mutate_weights(genome)
        if random.random() < Cfg.add_conn_rate:
            self._mutate_add_conn(genome)
        if random.random() < Cfg.add_node_rate:
            self._mutate_add_node(genome)

    def _mutate_weights(self, genome):
        for conn in genome.connections.values():
            if random.random() < Cfg.weight_perturb_rate:
                conn.weight += random.gauss(0, Cfg.weight_perturb_power)
                conn.weight = float(np.clip(conn.weight, -4.0, 4.0))
            else:
                conn.weight = random.uniform(-2.0, 2.0)

    def _mutate_add_conn(self, genome, tries=30):
        inputs_hidden = [nid for nid, n in genome.nodes.items() if n.type != 'output']
        hidden_outputs = [nid for nid, n in genome.nodes.items() if n.type != 'input']
        if not inputs_hidden or not hidden_outputs:
            return

        for _ in range(tries):
            a = random.choice(inputs_hidden)
            b = random.choice(hidden_outputs)
            if a == b:
                continue

            inn = self.innov.get(a, b)
            if inn in genome.connections:
                if not genome.connections[inn].enabled:
                    genome.connections[inn].enabled = True
                return

            genome.connections[inn] = ConnectionGene(
                a, b, random.uniform(-1.0, 1.0), True, inn
            )
            return

    def _mutate_add_node(self, genome):
        active = [c for c in genome.connections.values() if c.enabled]
        if not active:
            return

        conn = random.choice(active)
        conn.enabled = False

        new_id = max(genome.nodes) + 1
        genome.nodes[new_id] = NodeGene(new_id, 'hidden', 'tanh')

        inn1 = self.innov.get(conn.in_node, new_id)
        inn2 = self.innov.get(new_id, conn.out_node)

        genome.connections[inn1] = ConnectionGene(conn.in_node, new_id, 1.0, True, inn1)
        genome.connections[inn2] = ConnectionGene(new_id, conn.out_node, conn.weight, True, inn2)

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def stats(self):
        fits = [g.fitness for g in self.population]
        return {
            'generation': self.generation,
            'best': max(fits),
            'mean': float(np.mean(fits)),
            'species': len(self.species),
            'nodes_best': len(self.best_genome.nodes) if self.best_genome else 0,
        }
