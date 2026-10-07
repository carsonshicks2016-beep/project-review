import importlib.util
import unittest
from pathlib import Path

from rallylab import worker

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("circuit_trainability", ROOT / "Tools/circuit_trainability.py")
trainability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainability)


def passing(build="b"):
    results = {}
    for case in trainability.CASES:
        outcome = "HitObstacle" if case == "barrier" else "TimedOut"
        results[case] = ({"missingExpectedRegions": [], "wrongSurfaceSamples": 0,
                          "barrierContactEvidence": case == "barrier", "records": [{"outcome": outcome}]}, build)
    return results


class CircuitTrainabilityTests(unittest.TestCase):
    def test_circuit_training_uses_distributed_starts_and_short_budget(self):
        launch = worker.circuit_training_launch({"definition": {"topology": "circuit"}})
        self.assertEqual(launch, {"spawnProfile": "distributed", "episodeSeconds": 180})

    def test_procedural_and_generalist_training_launch_unchanged(self):
        self.assertEqual(worker.circuit_training_launch({"definition": {"gates": []}}), {})
        self.assertEqual(worker.circuit_training_launch(None), {})

    def test_complete_passing_suite_validates(self):
        self.assertEqual(trainability.evaluate(passing(), "b"), [])

    def test_missing_case_refuses(self):
        results = passing()
        del results["flugplatz-crest"]
        self.assertTrue(any("flugplatz-crest" in f for f in trainability.evaluate(results, "b")))

    def test_fell_off_or_wrong_surface_refuses(self):
        results = passing()
        results["seam"][0]["records"] = [{"outcome": "FellOff"}]
        results["karussell-channel"][0]["wrongSurfaceSamples"] = 3
        failures = trainability.evaluate(results, "b")
        self.assertTrue(any(f.startswith("seam") for f in failures))
        self.assertTrue(any(f.startswith("karussell-channel") for f in failures))

    def test_probes_from_another_build_refuse(self):
        self.assertTrue(trainability.evaluate(passing("old"), "b"))

    def test_barrier_case_requires_contact(self):
        results = passing()
        results["barrier"][0]["records"] = [{"outcome": "TimedOut"}]
        self.assertTrue(any(f.startswith("barrier") for f in trainability.evaluate(results, "b")))


if __name__ == "__main__":
    unittest.main()
