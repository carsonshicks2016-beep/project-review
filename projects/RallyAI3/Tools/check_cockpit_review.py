"""Validate bounded native cockpit lifecycle and instrument evidence."""
import argparse
import json
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(folder):
    cycles = records(folder / "cockpit-lifecycle.jsonl")
    assert len(cycles) == 8, "eight cockpit reconstruction cycles required"
    assert all(c == cycles[0] for c in cycles), "cockpit resources grew or steering binding changed"
    assert cycles[0]["cockpits"] == 1 and cycles[0]["steeringBound"]
    frames = records(folder / "cockpit-telemetry.jsonl")
    assert len(frames) == 40, "bounded instrument sample incomplete"
    for frame in frames:
        if "cockpitLocalOffset" in frame:
            assert frame["cockpitLocalOffset"] < .00001, "cockpit detached from body position"
            assert frame["cockpitLocalAngle"] < .001, "cockpit detached from body rotation"
        expected = "R" if frame["gear"] < 0 else "N" if frame["gear"] == 0 else str(frame["gear"])
        assert frame["renderedGear"] == expected, "gear display differs from source"
        assert abs(int(frame["renderedSpeed"]) - frame["speed"]) <= 2, "speed display differs from source"
        target = (225 - min(8, max(0, frame["rpm"] / 1000)) * 33.75) % 360
        error = abs((frame["needleDegrees"] - target + 180) % 360 - 180)
        assert error < 8, "tachometer differs from source beyond one-frame tolerance"
    return {"cycles": len(cycles), "resources": cycles[0], "instrument_samples": len(frames),
            "coupling_samples": sum("cockpitLocalOffset" in frame for frame in frames),
            "scope": "viewer cockpit reconstruction, not complete course scene reload"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    folder = parser.parse_args().folder
    result = validate(folder)
    (folder / "verified.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
