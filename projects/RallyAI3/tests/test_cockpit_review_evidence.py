import json
from pathlib import Path
import tempfile
import unittest
from Tools.check_cockpit_review import validate


class CockpitReviewEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.cycles = [{"cockpits": 1, "meshes": 20, "materials": 5,
                        "colliders": 3, "steeringBound": True} for _ in range(8)]
        self.frames = [{"rpm": 4000, "needleDegrees": 90, "gear": 3, "speed": 80,
                        "renderedGear": "3", "renderedSpeed": "080"} for _ in range(40)]

    def write(self):
        for name, rows in (("cockpit-lifecycle", self.cycles), ("cockpit-telemetry", self.frames)):
            (self.folder / (name + ".jsonl")).write_text("\n".join(json.dumps(row) for row in rows))

    def test_complete_evidence(self):
        self.write()
        self.assertEqual(validate(self.folder)["instrument_samples"], 40)

    def test_incomplete_cycles(self):
        self.cycles.pop()
        self.write()
        with self.assertRaisesRegex(AssertionError, "eight"):
            validate(self.folder)

    def test_resource_growth(self):
        self.cycles[-1]["meshes"] += 1
        self.write()
        with self.assertRaisesRegex(AssertionError, "grew"):
            validate(self.folder)

    def test_false_instrument_readout(self):
        self.frames[-1]["renderedGear"] = "4"
        self.write()
        with self.assertRaisesRegex(AssertionError, "gear"):
            validate(self.folder)

    def test_separate_cockpit_motion(self):
        self.frames[-1]["cockpitLocalOffset"] = .05
        self.frames[-1]["cockpitLocalAngle"] = 0
        self.write()
        with self.assertRaisesRegex(AssertionError, "detached"):
            validate(self.folder)
