"""Check native frozen-result presentation against its recorded attempt."""
import argparse
import json
from pathlib import Path


def validate(folder):
    performance = json.loads((folder / "performance.json").read_text())
    samples = [s for s in performance["samples"] if s.get("viewerPaused")]
    assert len(samples) >= 5, "insufficient frozen-presentation samples"
    assert len({s["simulationSeconds"] for s in samples}) == 1, "simulation advanced during orbit"
    assert len({tuple(s["carPosition"].values()) for s in samples}) == 1, "car moved during orbit"
    assert len({tuple(s["cameraPosition"].values()) for s in samples}) > 1, "orbit camera did not move"
    result = json.loads((folder / "viewer-result.json").read_text())
    records = [json.loads(line) for file in (folder / "attempts").glob("episodes*.jsonl")
               for line in file.read_text().splitlines() if line.strip()]
    record = next(r for r in records if r["attempt"] == result["attempt"])
    assert result["valid"] == record["valid"], "result validity differs from record"
    assert abs(result["seconds"] - record["seconds"]) < .001, "result time differs from record"
    assert not record["valid"] or record["outcome"] == "Finished"
    return {"frozen_samples": len(samples), "outcome": record["outcome"],
            "valid": record["valid"], "seconds": record["seconds"],
            "scope": "watch inference, not a ranked evaluation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    folder = parser.parse_args().folder
    result = validate(folder)
    (folder / "result-verified.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
