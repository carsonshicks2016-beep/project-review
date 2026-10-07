import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CarMeshContractTests(unittest.TestCase):
    def test_cockpit_uses_each_surface_slot(self):
        data = json.loads((ROOT / "Assets/Art/Models/Vehicles/GC8_Cockpit.json").read_text())
        self.assertTrue(all(part["indices"] for part in data["submeshes"]))

    def test_authored_mesh_contract(self):
        for name, slots in (("GC8_Body", 5), ("GC8_Wheel", 2), ("GC8_Cockpit", 5)):
            with self.subTest(mesh=name):
                data = json.loads((ROOT / "Assets/Art/Models/Vehicles" / (name + ".json")).read_text())
                count = len(data["positions"])
                self.assertEqual(data["schema"], 1)
                self.assertEqual(len(data["submeshes"]), slots)
                self.assertEqual(count, len(data["normals"]))
                self.assertEqual(count, len(data["uvs"]))
                for field in ("positions", "normals", "uvs"):
                    self.assertTrue(all(math.isfinite(v) for row in data[field] for v in row.values()))
                for normal in data["normals"]:
                    self.assertAlmostEqual(sum(v * v for v in normal.values()), 1, delta=.01)
                for submesh in data["submeshes"]:
                    self.assertEqual(len(submesh["indices"]) % 3, 0)
                    self.assertTrue(all(0 <= index < count for index in submesh["indices"]))
                limits = (.3, .7, .7) if name == "GC8_Wheel" else (2.1, 1.6, 4.6)
                for axis, limit in zip("xyz", limits):
                    values = [point[axis] for point in data["positions"]]
                    self.assertLessEqual(max(values) - min(values), limit)
