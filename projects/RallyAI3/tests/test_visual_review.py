import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ForestKitSourceTests(unittest.TestCase):
    def test_four_editable_archetypes_and_bounded_review(self):
        kit = json.loads((ROOT / "art-source/forest/tree-kit-v3.json").read_text())
        self.assertEqual(kit["schema"], 1)
        self.assertEqual(kit["reviewSeed"], 41200)
        self.assertGreater(kit["sectionEnd"], kit["sectionStart"])
        self.assertLessEqual(kit["sectionEnd"] - kit["sectionStart"], 300)
        self.assertEqual(len(kit["species"]), 4)
        self.assertTrue(kit["productionCoverage"])
        self.assertEqual({s["atlasTile"] for s in kit["species"]}, set(range(4)))
        for species in kit["species"]:
            self.assertTrue(0 < species["crownStart"] < 1)
            self.assertTrue(0 < species["crownRadius"] < 0.3)
            self.assertTrue(3 <= species["tiers"] <= 16)
            self.assertTrue(3 <= species["boughs"] <= 8)

    def test_baseline_tag_documented_without_promotion_claim(self):
        ledger = (ROOT / "Tools/VISUALS-V3.md").read_text()
        self.assertIn("visuals-v2-refined", ledger)
        self.assertIn("User visual approval required", ledger)
        self.assertIn("-1 mean unavailable", ledger)

    def test_runtime_copy_matches_editable_kit(self):
        authored = json.loads((ROOT / "art-source/forest/tree-kit-v3.json").read_text())
        runtime = json.loads((ROOT / "Assets/Resources/StageDressing/TreeKitV3.json").read_text())
        self.assertEqual(authored, runtime)


if __name__ == "__main__":
    unittest.main()
