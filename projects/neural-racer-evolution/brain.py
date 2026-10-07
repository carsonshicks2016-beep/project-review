import numpy as np
import json


class NodeGene:
    def __init__(self, node_id, node_type, activation='tanh'):
        self.id = node_id
        self.type = node_type      # 'input', 'hidden', 'output'
        self.activation = activation


class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation


class Genome:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes = {}        # id -> NodeGene
        self.connections = {}  # innovation -> ConnectionGene
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species_id = -1
        self.last_values = {}   # filled by activate(); used by live network viz

        for i in range(n_inputs):
            self.nodes[i] = NodeGene(i, 'input', 'linear')

        for i in range(n_outputs):
            nid = n_inputs + i
            self.nodes[nid] = NodeGene(nid, 'output', 'tanh')

    def activate(self, inputs):
        # Resilient to input-count drift: truncate or zero-pad as needed
        # so old hall-of-fame brains keep working when N_INPUTS changes.
        if len(inputs) > self.n_inputs:
            inputs = inputs[:self.n_inputs]
        elif len(inputs) < self.n_inputs:
            inputs = list(inputs) + [0.0] * (self.n_inputs - len(inputs))

        values = {}
        for i, v in enumerate(inputs):
            values[i] = float(v)

        hidden_out = sorted(
            [nid for nid, n in self.nodes.items() if n.type in ('hidden', 'output')]
        )

        # Iterative activation handles recurrent topologies
        n_iter = max(len(hidden_out), 1)
        for _ in range(n_iter):
            for nid in hidden_out:
                total = 0.0
                for conn in self.connections.values():
                    if conn.enabled and conn.out_node == nid:
                        total += values.get(conn.in_node, 0.0) * conn.weight
                node = self.nodes[nid]
                if node.activation == 'tanh':
                    values[nid] = float(np.tanh(total))
                elif node.activation == 'relu':
                    values[nid] = float(max(0.0, total))
                elif node.activation == 'sigmoid':
                    values[nid] = float(1.0 / (1.0 + np.exp(-np.clip(total, -30, 30))))
                else:
                    values[nid] = float(total)

        self.last_values = dict(values)
        return [values.get(self.n_inputs + i, 0.0) for i in range(self.n_outputs)]

    def to_dict(self):
        return {
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'fitness': self.fitness,
            'nodes': {
                str(k): {'id': v.id, 'type': v.type, 'activation': v.activation}
                for k, v in self.nodes.items()
            },
            'connections': {
                str(k): {
                    'in_node': v.in_node,
                    'out_node': v.out_node,
                    'weight': v.weight,
                    'enabled': v.enabled,
                    'innovation': v.innovation,
                }
                for k, v in self.connections.items()
            },
        }

    @classmethod
    def from_dict(cls, d):
        g = cls.__new__(cls)
        g.n_inputs = d['n_inputs']
        g.n_outputs = d['n_outputs']
        g.fitness = d.get('fitness', 0.0)
        g.adjusted_fitness = 0.0
        g.species_id = -1
        g.nodes = {
            int(k): NodeGene(v['id'], v['type'], v['activation'])
            for k, v in d['nodes'].items()
        }
        g.connections = {
            int(k): ConnectionGene(
                v['in_node'], v['out_node'], v['weight'], v['enabled'], v['innovation']
            )
            for k, v in d['connections'].items()
        }
        return g

    def copy(self):
        return Genome.from_dict(self.to_dict())
