"""Stage 6.3 acceptance: the structural-failure check.

The seeds are structurally sound (huge safety margin). Deliberately under-built
bodies are flagged BEFORE simulation:
  * bone_overload   -- a thin limb anchoring a strong muscle: the attachment
                       stress exceeds the tissue strength (tears out).
  * tendon_overload -- a muscle too strong for any plausible tendon.

Runs:  python3 tests/test_structure.py   (requires mujoco only via develop import)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.encoding.genome import (
    Shape, JointType, AttachmentSite as S, Joint, PartNode, ConnectionEdge,
    MuscleGene, Genome,
)
from personal_cambrian.evo import check_structure, StructuralReport


def _part(rep, part_id):
    """Look up a part's stress entry by its (un-suffixed) genome part id."""
    return next(v for v in rep.per_part.values() if v["part_id"] == part_id)


def _body(child_radius: float, pcsa: float, parent_radius: float = 0.06) -> Genome:
    """A parent limb anchoring one muscle to a child limb of `child_radius`."""
    p = PartNode(id="p", shape=Shape.CAPSULE,
                 dims={"radius": parent_radius, "length": 0.4},
                 sites=[S("a", pos=(0, 0, -0.2)), S("b", pos=(0.05, 0, -0.1))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE,
                  dims={"radius": child_radius, "length": 0.3},
                  sites=[S("c", pos=(0.0, 0, 0.1)), S("d", pos=(0, 0, -0.14))])
    edges = [ConnectionEdge("p", "ch", site_idx=0,
                            joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1)))]
    mg = MuscleGene(id="m", origin=("p", 1), insertion=("ch", 0), pcsa_cm2=pcsa)
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[mg])


# --- valid bodies are sound -------------------------------------------------
def test_seeds_are_sound():
    for seed in (quadruped(), agent_zero()):
        rep = check_structure(seed)
        assert isinstance(rep, StructuralReport)
        assert rep.sound, rep.failures
        assert bool(rep) is True
        assert rep.failures == []
        assert rep.max_bone_stress_ratio < 1.0
        assert rep.max_tendon_stress_ratio < 1.0


def test_modest_body_is_sound():
    rep = check_structure(_body(child_radius=0.05, pcsa=20.0))
    assert rep.sound
    assert _part(rep, "p") and _part(rep, "ch")
    assert any(v["gene_id"] == "m" for v in rep.per_muscle.values())


# --- under-built bodies are flagged ----------------------------------------
def test_underbuilt_attachment_fails_on_bone():
    # thin limb (3.5 mm radius) anchoring a strong (80 cm^2) muscle, but Fmax low
    # enough that the tendon is fine -> isolates the bone/attachment failure.
    rep = check_structure(_body(child_radius=0.0035, pcsa=80.0))
    assert not rep.sound
    assert "bone_overload" in rep.failures
    assert "tendon_overload" not in rep.failures      # tendon still within capacity
    assert rep.n_bone_overloads >= 1
    assert _part(rep, "ch")["ratio"] > 1.0            # the thin child is what fails
    assert rep.max_bone_stress_ratio > 1.0


def test_overpowered_muscle_fails_on_tendon():
    # huge muscle on a thick limb: bone is fine, but no tendon can carry the force
    rep = check_structure(_body(child_radius=0.08, pcsa=150.0, parent_radius=0.08))
    assert not rep.sound
    assert "tendon_overload" in rep.failures
    assert "bone_overload" not in rep.failures
    assert rep.n_tendon_overloads >= 1
    assert rep.max_tendon_stress_ratio > 1.0


def test_stress_scales_with_thinness():
    # thinner attachment -> higher bone stress (monotonic), all else equal
    thick = _part(check_structure(_body(child_radius=0.05, pcsa=40.0)), "ch")["ratio"]
    thin = _part(check_structure(_body(child_radius=0.02, pcsa=40.0)), "ch")["ratio"]
    assert thin > thick


def test_report_roundtrips_to_dict():
    rep = check_structure(quadruped())
    d = rep.to_dict()
    assert d["sound"] is True and d["failures"] == []
    assert set(d) >= {"n_bone_overloads", "n_tendon_overloads",
                      "max_bone_stress_ratio", "max_tendon_stress_ratio"}


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
