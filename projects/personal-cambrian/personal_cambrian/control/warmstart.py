"""Parent -> child controller warm-start (ROADMAP Stage 3.4).

When a body mutates, its child inherits its parent's controller as a starting
point instead of learning from scratch. Because ModularPolicy parameters are
shared across nodes (morphology-independent) and the adjacency is kept out of
`state_dict`, warm-starting is just: build a policy for the child's morphology
(its own node count + kinematic adjacency) and load the parent's weights. The
child then needs only a short fine-tune to adapt to its new body.

This is what makes morphology+control co-evolution affordable (Stages 4+): each
mutant starts near-competent rather than random.
"""
from __future__ import annotations

from typing import Callable

from .modular import ModularPolicy


def warm_start(parent: ModularPolicy, child_env) -> ModularPolicy:
    """Build a controller for `child_env`'s morphology, initialized from `parent`."""
    ob = child_env.obs_builder
    child = ModularPolicy(ob.node_dim, ob.n_nodes, child_env.actuator_adjacency,
                          parent.hidden, parent.rounds)
    child.load_state_dict(parent.state_dict())     # shapes match (params node-independent)
    return child


def warm_started_make_agent(child_make_env: Callable[[int], object],
                            parent: ModularPolicy) -> Callable:
    """A `make_agent` for train(): builds the child morphology's policy and loads
    the parent's weights, so fine-tuning continues from the parent."""
    sample = child_make_env(0)
    ob = sample.obs_builder
    node_dim, n_nodes, adj = ob.node_dim, ob.n_nodes, sample.actuator_adjacency
    sample.close()

    def make_agent(obs_dim, act_dim):
        child = ModularPolicy(node_dim, n_nodes, adj, parent.hidden, parent.rounds)
        child.load_state_dict(parent.state_dict())
        return child

    return make_agent
