import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AuthoredCarTests(unittest.TestCase):
    def load(self, name):
        return json.loads((ROOT / f"Assets/Art/Models/Vehicles/GC8_{name}.json").read_text())

    def test_mesh_contract_and_normals(self):
        for name, slots in [("Body", 5), ("Wheel", 2)]:
            data = self.load(name)
            self.assertEqual(data["schema"], 1)
            self.assertEqual(len(data["submeshes"]), slots)
            count = len(data["positions"])
            self.assertEqual(count, len(data["normals"]))
            self.assertEqual(count, len(data["uvs"]))
            for submesh in data["submeshes"]:
                self.assertEqual(len(submesh["indices"]) % 3, 0)
                self.assertTrue(all(0 <= index < count for index in submesh["indices"]))
            for normal in data["normals"]:
                self.assertAlmostEqual(sum(v*v for v in normal.values()), 1, places=4)
            for position in data["positions"]:
                self.assertTrue(all(math.isfinite(v) for v in position.values()))
            for submesh in data["submeshes"]:
                for i in range(0,len(submesh["indices"]),3):
                    a,b,c=[data["positions"][index] for index in submesh["indices"][i:i+3]]
                    ab=[b[k]-a[k] for k in "xyz"]
                    ac=[c[k]-a[k] for k in "xyz"]
                    cross=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]]
                    self.assertGreater(sum(v*v for v in cross),1e-14)

    def test_suspension_visual_envelope(self):
        wheel = self.load("Wheel")["positions"]
        self.assertAlmostEqual(max(math.hypot(v["y"], v["z"]) for v in wheel), .330, places=5)
        self.assertLessEqual(max(abs(v["x"]) for v in wheel), .13)
        body = self.load("Body")["positions"]
        self.assertLess(max(v["y"] for v in body), 1.40)
        self.assertLess(max(v["z"] for v in body), 2.16)
        self.assertGreater(min(v["z"] for v in body), -2.28)

    def test_extended_cabin_retains_low_roof(self):
        body=self.load("Body")["positions"]
        roof=[v for v in body if v["y"]>1.30]
        self.assertGreaterEqual(max(v["z"] for v in roof),.239)
        self.assertLessEqual(min(v["z"] for v in roof),-.649)


if __name__ == "__main__":
    unittest.main()
