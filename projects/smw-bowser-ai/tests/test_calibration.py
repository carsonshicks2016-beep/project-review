import tempfile
import unittest
from pathlib import Path

from smw_bowser_ai.calibration import calibrate_trace
from smw_bowser_ai.memory import MemoryMap
from smw_bowser_ai.telemetry import write_jsonl


class CalibrationTests(unittest.TestCase):
    def test_required_signals_seen(self):
        memory_map = MemoryMap.from_file()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            ram = {name: 1 for name in memory_map.required_signal_names()}
            write_jsonl(path, [{"frame": 1, "ram": ram}, {"frame": 2, "ram": {**ram, "mario_x": 2}}])
            report = calibrate_trace(path, memory_map)
            self.assertTrue(report.ok)
            self.assertTrue(report.signals["mario_x"].changed)


if __name__ == "__main__":
    unittest.main()

