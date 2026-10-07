"""Reproducibility guards.

The North Star requires that seed N produces the same stage on any machine, and
that a checkpoint's numbers can be re-derived from the checkpoint. Both rest on
the same foundation: the same inputs must produce byte-identical output.

This is easy to lose silently. ``numpy`` guarantees the PCG64 *bit stream*
across versions but explicitly does not guarantee that distribution methods like
``normal()`` return identical floats. Float maths can differ in the last bit
between CPUs. Neither shows up as an error — it shows up as a stage that is
subtly not the stage a replay was recorded against.
"""

from __future__ import annotations

import numpy as np
import pytest

from rallyai import contracts
from rallyai.stage.fixtures import ALL, flat_straight, proving_ground

# Golden hashes. If a change moves these, that is not automatically a failure —
# but it IS a deliberate act: every replay recorded against the old geometry is
# now invalid, and the generator version should be bumped. Update these only
# with that understood.
# Updated for AUTHORED_VERSION 2 (eased grade/width, sin^2 crests).
GOLDEN = {
    "proving_ground": "sha256:198980010ba78db38917f35c973a4b7eb1b3faf5123d8dba291875e8bab5ab69",
    "flat_straight": "sha256:969646b01bb2aceab68e063336734ac500a39d4de8e6593a01cdb72f5dcec32f",
}


@pytest.mark.parametrize("name", sorted(ALL))
def test_authored_stage_matches_its_golden_hash(name):
    stage = contracts.stamp(ALL[name]())
    assert stage["content_hash"] == GOLDEN[name], (
        f"{name} geometry changed.\n"
        f"  was {GOLDEN[name]}\n"
        f"  now {stage['content_hash']}\n"
        "If this was intentional, bump the generator version and update GOLDEN — "
        "every replay recorded against the old geometry is now stale."
    )


@pytest.mark.parametrize("name", sorted(ALL))
def test_building_the_same_stage_twice_is_identical(name):
    a = contracts.canonical_bytes(contracts.stamp(ALL[name]()))
    b = contracts.canonical_bytes(contracts.stamp(ALL[name]()))
    assert a == b


def test_write_read_write_is_stable(tmp_path):
    """A stage that round-trips through disk must come back identical. If it
    does not, the file is not a faithful record of the geometry."""
    first = contracts.write_json(proving_ground(), tmp_path / "a.json", kind="stage")
    doc = contracts.read_json(first, kind="stage")
    second = contracts.write_json(doc, tmp_path / "b.json", kind="stage")
    assert first.read_bytes() == second.read_bytes()


def test_quantisation_absorbs_last_bit_differences():
    """The reason floats are quantised before hashing: a difference far below
    anything the physics can resolve must not change a stage's identity."""
    stage = flat_straight(200.0)
    perturbed = contracts.quantize(stage)
    # A nanometre of drift — well under the 1e-6 quantisation step.
    for point in perturbed["centerline"]:
        point["x"] = point["x"] + 1e-11
    assert contracts.content_hash(contracts.quantize(perturbed)) == \
           contracts.content_hash(contracts.quantize(stage))


def test_quantisation_does_not_hide_real_differences():
    """Positive control for the test above. A millimetre is real and must change
    the hash; otherwise quantisation would be laundering actual geometry."""
    a = contracts.quantize(flat_straight(200.0))
    b = contracts.quantize(flat_straight(200.0))
    b["centerline"][5]["x"] += 1e-3
    assert contracts.content_hash(a) != contracts.content_hash(b)


def test_numpy_generator_stream_is_the_documented_one():
    """The generator will be built on default_rng. Pin the assumption that it is
    PCG64, so an environment change that silently swaps the bit generator is
    caught here rather than by stages quietly differing between machines."""
    rng = np.random.default_rng(12345)
    assert type(rng.bit_generator).__name__ == "PCG64"
    # Integers are drawn straight from the bit stream, which numpy does
    # guarantee across versions — unlike the distribution methods.
    assert [int(v) for v in rng.integers(0, 1_000_000, 4)] == [699215, 227336, 788646, 316758]
