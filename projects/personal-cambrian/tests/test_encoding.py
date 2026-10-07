"""Stage 1.1 acceptance: the genome can be constructed in code, and
`validate_shape()` rejects malformed graphs (dangling sites, unknown parts, bad
parameters).

Runs with zero install:  python3 tests/test_encoding.py
Also collectable by pytest once added (ROADMAP F.6) — the test_* functions need
no fixtures; `approx`/`raises` below are local shims so pytest isn't required.
"""
import os
import sys
import json
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.encoding import (
    Shape, JointType, SymmetryKind,
    AttachmentSite, Joint, PartNode, ConnectionEdge, MuscleGene, Genome,
    GenomeValidationError,
)


# --- tiny test shims (so this runs without pytest) ------------------------- #
class _Approx:
    def __init__(self, v, tol=1e-9):
        self.v, self.tol = v, tol

    def __eq__(self, other):
        return abs(float(other) - self.v) <= self.tol


def approx(v, tol=1e-9):
    return _Approx(v, tol)


@contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


def valid_genome() -> Genome:
    """A minimal but well-formed creature: torso -> limb via a hinge, one muscle."""
    torso = PartNode(
        id="torso", shape=Shape.CAPSULE, dims={"radius": 0.08, "length": 0.40},
        sites=[AttachmentSite("hip", pos=(0.0, 0.0, -0.20)),
               AttachmentSite("origin", pos=(0.0, 0.05, -0.10))],
    )
    limb = PartNode(
        id="limb", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.35},
        recursion_limit=3,
        sites=[AttachmentSite("top", pos=(0.0, 0.0, 0.17)),
               AttachmentSite("insert", pos=(0.0, 0.05, 0.10))],
    )
    edge = ConnectionEdge(
        parent_part="torso", child_part="limb", site_idx=0,
        joint=Joint(type=JointType.HINGE, axis=(0, 1, 0), range=(-1.2, 1.2)),
        symmetry=SymmetryKind.BILATERAL,
    )
    muscle = MuscleGene(
        id="hip_flexor", origin=("torso", 1), insertion=("limb", 1),
        pcsa_cm2=12.0, fiber_type=0.6,
    )
    return Genome(root="torso", parts=[torso, limb], edges=[edge], muscles=[muscle])


# --------------------------------------------------------------------------- #
# construction + well-formed                                                   #
# --------------------------------------------------------------------------- #
def test_construct_and_validate():
    g = valid_genome()
    assert g.validate_shape() == []
    assert g.is_valid_shape()
    assert g.assert_valid() is g  # does not raise, returns self


def test_indexing_and_derived():
    g = valid_genome()
    assert g.part_ids == ["torso", "limb"]
    assert g.get_part("limb").recursion_limit == 3
    assert g.get_part("ghost") is None
    # fmax derived from PCSA * specific tension (25 N/cm^2)
    assert g.muscles[0].fmax == approx(12.0 * 25.0)


def test_reachability_diagnostic():
    g = valid_genome()
    assert g.unreachable_parts() == []
    g.parts.append(PartNode(id="orphan", shape=Shape.SPHERE, dims={"radius": 0.03}))
    # orphan is structurally valid but unreachable (diagnostic, not an error)
    assert g.unreachable_parts() == ["orphan"]
    assert g.validate_shape() == []


# --------------------------------------------------------------------------- #
# malformed graphs are rejected                                                #
# --------------------------------------------------------------------------- #
def test_unknown_child_part():
    g = valid_genome()
    g.edges[0].child_part = "nope"
    assert any("unknown child part 'nope'" in p for p in g.validate_shape())


def test_unknown_root():
    g = valid_genome()
    g.root = "missing"
    assert any("root 'missing'" in p for p in g.validate_shape())


def test_dangling_site_in_edge():
    g = valid_genome()
    g.edges[0].site_idx = 99
    assert any("dangling site_idx 99" in p for p in g.validate_shape())


def test_duplicate_part_ids():
    g = valid_genome()
    g.parts.append(PartNode(id="torso", shape=Shape.SPHERE, dims={"radius": 0.05}))
    assert any("duplicate part ids" in p for p in g.validate_shape())


def test_bad_dims_wrong_keys():
    g = valid_genome()
    g.get_part("limb").dims = {"radius": 0.05}  # capsule needs radius+length
    assert any("needs dims" in p for p in g.validate_shape())


def test_bad_dims_nonpositive():
    g = valid_genome()
    g.get_part("torso").dims["radius"] = -0.1
    assert any("must be > 0" in p for p in g.validate_shape())


def test_muscle_unknown_part():
    g = valid_genome()
    g.muscles[0].origin = ("ghost", 0)
    assert any("muscle 'hip_flexor' origin: unknown part 'ghost'" in p
               for p in g.validate_shape())


def test_muscle_dangling_site():
    g = valid_genome()
    g.muscles[0].insertion = ("limb", 7)
    assert any("dangling site_idx 7" in p for p in g.validate_shape())


def test_muscle_origin_equals_insertion():
    g = valid_genome()
    g.muscles[0].insertion = ("torso", 1)  # same as origin
    assert any("origin and insertion are the same" in p for p in g.validate_shape())


def test_fiber_type_out_of_range():
    g = valid_genome()
    g.muscles[0].fiber_type = 1.5
    assert any("fiber_type must be in [0,1]" in p for p in g.validate_shape())


def test_joint_inverted_range():
    g = valid_genome()
    g.edges[0].joint.range = (1.0, -1.0)
    assert any("joint range lo" in p for p in g.validate_shape())


def test_radial_symmetry_needs_count():
    g = valid_genome()
    g.edges[0].symmetry = SymmetryKind.RADIAL
    g.edges[0].symmetry_count = 1
    assert any("RADIAL symmetry needs symmetry_count >= 2" in p for p in g.validate_shape())


def test_bad_quaternion_zero_norm():
    g = valid_genome()
    g.get_part("limb").sites[0].quat = (0.0, 0.0, 0.0, 0.0)
    assert any("zero norm" in p for p in g.validate_shape())


def test_assert_valid_raises():
    g = valid_genome()
    g.edges[0].site_idx = 50
    with raises(GenomeValidationError):
        g.assert_valid()


def test_multiple_problems_collected():
    g = valid_genome()
    g.root = "x"
    g.edges[0].child_part = "y"
    g.muscles[0].fiber_type = 9.0
    assert len(g.validate_shape()) >= 3


def richer_genome() -> Genome:
    """Exercises more fields for serialization: routes, radial symmetry, mixed shapes."""
    hub = PartNode(id="hub", shape=Shape.SPHERE, dims={"radius": 0.10},
                   sites=[AttachmentSite("s0", pos=(0.10, 0, 0)),
                          AttachmentSite("m_o", pos=(0.05, 0.05, 0))])
    arm = PartNode(id="arm", shape=Shape.BOX, dims={"x": 0.04, "y": 0.04, "z": 0.20},
                   density=1100.0, recursion_limit=2,
                   sites=[AttachmentSite("tip", pos=(0, 0, 0.20)),
                          AttachmentSite("mid", pos=(0, 0.04, 0.10)),
                          AttachmentSite("m_i", pos=(0, 0.04, 0.02))])
    e = ConnectionEdge(parent_part="hub", child_part="arm", site_idx=0,
                       pos=(0.02, 0, 0), joint=Joint(JointType.BALL, range=(-2.0, 2.0)),
                       recursion_count=2, symmetry=SymmetryKind.RADIAL, symmetry_count=4)
    m = MuscleGene(id="m0", origin=("hub", 1), insertion=("arm", 2),
                   route_sites=[("arm", 1)], pcsa_cm2=8.0, fiber_type=0.3, pennation=0.2)
    return Genome(root="hub", parts=[hub, arm], edges=[e], muscles=[m], max_depth=6)


# --------------------------------------------------------------------------- #
# Stage 1.2: canonical serialization + hashing                                 #
# --------------------------------------------------------------------------- #
def test_roundtrip_dict_equality():
    for g in (valid_genome(), richer_genome()):
        assert Genome.from_dict(g.to_dict()) == g


def test_roundtrip_json_equality():
    for g in (valid_genome(), richer_genome()):
        assert Genome.from_json(g.to_json()) == g


def test_enums_and_tuples_survive_roundtrip():
    g2 = Genome.from_json(valid_genome().to_json())
    assert g2.parts[0].shape is Shape.CAPSULE
    assert g2.edges[0].symmetry is SymmetryKind.BILATERAL
    assert g2.edges[0].joint.type is JointType.HINGE
    # JSON has no tuples; coercion must restore them so equality holds
    assert isinstance(g2.parts[0].sites[0].pos, tuple)
    assert isinstance(g2.muscles[0].origin, tuple)


def test_hash_deterministic_and_survives_roundtrip():
    g = richer_genome()
    assert g.hash() == g.hash()
    assert Genome.from_json(g.to_json()).hash() == g.hash()


def test_equal_genomes_share_hash():
    # two independently-constructed identical genomes
    assert valid_genome() == valid_genome()
    assert valid_genome().hash() == valid_genome().hash()


def test_hash_changes_on_edit():
    g = valid_genome()
    h0 = g.hash()
    g.muscles[0].pcsa_cm2 += 1.0
    assert g.hash() != h0


def test_hash_independent_of_dims_insertion_order():
    a = valid_genome()
    b = valid_genome()
    # rebuild one part's dims in the opposite key order
    b.parts[0].dims = {"length": 0.40, "radius": 0.08}
    assert a == b                      # dict equality ignores order
    assert a.hash() == b.hash()        # sorted-keys canonical form ignores it too


def test_to_json_is_parseable_and_versioned():
    d = json.loads(valid_genome().to_json())
    assert d["_schema"] == 1
    assert d["root"] == "torso"
    assert isinstance(d["parts"], list) and isinstance(d["edges"], list)


def test_int_float_inputs_hash_identically():
    a = MuscleGene(id="m", origin=("p", 0), insertion=("p", 1), pcsa_cm2=12)    # int
    b = MuscleGene(id="m", origin=("p", 0), insertion=("p", 1), pcsa_cm2=12.0)  # float
    g_a = Genome(root="p", parts=[], muscles=[a])
    g_b = Genome(root="p", parts=[], muscles=[b])
    assert g_a.hash() == g_b.hash()


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
