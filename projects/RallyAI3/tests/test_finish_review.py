import json
from pathlib import Path
import tempfile
import unittest
from Tools.check_finish_review import validate


class FinishReviewTests(unittest.TestCase):
    def fixture(self, folder, valid=True):
        samples = [{"viewerPaused": True, "simulationSeconds": 28,
                    "carPosition": {"x": 1, "y": 2, "z": 3},
                    "cameraPosition": {"x": i, "y": 4, "z": 5}} for i in range(6)]
        (folder / "performance.json").write_text(json.dumps({"samples": samples}))
        (folder / "viewer-result.json").write_text(json.dumps({"attempt": 1, "valid": valid, "seconds": 27.35}))
        (folder / "attempts").mkdir()
        (folder / "attempts/episodes-test.jsonl").write_text(json.dumps(
            {"attempt": 1, "valid": valid, "seconds": 27.35,
             "outcome": "Finished" if valid else "FellOff"}) + "\n")

    def test_success_and_failure_stay_distinct(self):
        for valid in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                self.fixture(folder, valid)
                self.assertEqual(validate(folder)["valid"], valid)

    def test_rejects_motion_while_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            data = json.loads((folder / "performance.json").read_text())
            data["samples"][-1]["carPosition"]["x"] = 100
            (folder / "performance.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError, "car moved"):
                validate(folder)

    def test_rejects_wrong_result_time(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            (folder / "viewer-result.json").write_text(json.dumps({"attempt": 1, "valid": True, "seconds": 10}))
            with self.assertRaisesRegex(AssertionError, "time differs"):
                validate(folder)
