"""Stage 7.2 acceptance: the morphological distance metric.

Core sanity (the ROADMAP done-when): two human-ish bodies are far CLOSER to each
other than either is to a tailed quadruped. Plus the metric is well-behaved:
symmetric, zero for identical bodies, bounded, and its two components separate
topology change (graph) from proportion change (descriptors).

Runs:  python3 tests/test_distance.py   (requires mujoco for develop)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.encoding.mutate import micro
from personal_cambrian.evo import (
    morphological_distance, descriptor_distance, graph_distance,
    distance_matrix, descriptor_vector,
)


def _micro_twin(genome, seed=0):
    return micro(genome, np.random.default_rng(seed), 0.1)   # topology-preserving


# --- the headline sanity check ---------------------------------------------
def test_human_human_much_closer_than_human_tailed():
    z, q = agent_zero(), quadruped()
    human_human = morphological_distance(z, _micro_twin(z))
    human_tailed = morphological_distance(z, q)
    assert human_human < human_tailed
    assert human_human * 5 < human_tailed          # not just less -- far less


# --- metric properties ------------------------------------------------------
def test_identical_body_has_zero_distance():
    for g in (agent_zero(), quadruped()):
        assert morphological_distance(g, g) == 0.0
        assert descriptor_distance(g, g) == 0.0
        assert graph_distance(g, g) == 0.0


def test_distance_is_symmetric():
    z, q = agent_zero(), quadruped()
    assert morphological_distance(z, q) == morphological_distance(q, z)
    assert descriptor_distance(z, q) == descriptor_distance(q, z)


def test_distance_is_bounded_unit_ish():
    z, q = agent_zero(), quadruped()
    for d in (morphological_distance(z, q), descriptor_distance(z, q),
              graph_distance(z, q)):
        assert 0.0 <= d <= 1.0 + 1e-9


# --- the two components separate topology vs proportion --------------------
def test_graph_distance_ignores_micro_but_catches_topology():
    z = agent_zero()
    assert graph_distance(z, _micro_twin(z)) == 0.0     # continuous tweak -> no edits
    assert graph_distance(z, quadruped()) > 0.0          # different plan -> edits


def test_descriptor_distance_responds_to_micro():
    z = agent_zero()
    # a continuous tweak moves descriptors a little (but less than a new plan)
    micro_d = descriptor_distance(z, _micro_twin(z))
    plan_d = descriptor_distance(z, quadruped())
    assert 0.0 < micro_d < plan_d


def test_weights_select_components():
    z, q = agent_zero(), quadruped()
    only_graph = morphological_distance(z, q, w_graph=1.0, w_desc=0.0)
    only_desc = morphological_distance(z, q, w_graph=0.0, w_desc=1.0)
    assert only_graph == graph_distance(z, q)
    assert only_desc == descriptor_distance(z, q)


# --- matrix helper ----------------------------------------------------------
def test_distance_matrix_shape_and_symmetry():
    z = agent_zero()
    genomes = [z, _micro_twin(z), quadruped()]
    M = distance_matrix(genomes)
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T)                          # symmetric
    assert np.allclose(np.diag(M), 0.0)                 # zero diagonal
    # the two humans (0,1) are closer than human-vs-tailed (0,2)
    assert M[0, 1] < M[0, 2]


def test_descriptor_vector_is_normalized():
    v = descriptor_vector(agent_zero())
    assert v.shape == (6,)
    assert np.all(v >= 0.0) and np.all(v <= 1.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
