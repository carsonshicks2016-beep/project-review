"""Validate opt-in standalone audio recordings without making listening-quality claims."""
import argparse
import json
from pathlib import Path
import wave


def inspect(folder):
    audio = json.loads((folder / "audio.json").read_text())
    performance = json.loads((folder / "performance.json").read_text())
    launch = json.loads((folder / "launch.json").read_text())
    build = json.loads((folder / "build.json").read_text())
    frames = [json.loads(line) for line in (folder / "audio-frames.jsonl").read_text().splitlines()]
    with wave.open(str(folder / "listener.wav")) as wav:
        assert wav.getnchannels() == audio["channels"]
        assert wav.getframerate() == audio["sampleRate"]
        assert wav.getnframes() * wav.getnchannels() == audio["samples"]
    assert audio["seconds"] > 5 and audio["rms"] > 0
    assert audio["nonFinite"] == 0 and audio["fullScaleSamples"] == 0 and audio["peak"] < .98
    assert audio["course"] == launch["courseId"] == performance["course"]
    assert audio["checkpoint"] == launch["checkpointId"] == performance["checkpoint"]
    assert frames and all(f["listeners"] == 1 and f["sources"] == 5 for f in frames)
    assert all(f["droppedEngineEvents"] == f["droppedImpactEvents"] == 0 for f in frames)
    assert all(abs(f["pitch"] - 1) < 1e-6 for f in frames if f["cut"] or f["reset"])
    assert all(abs(f["pitch"] - 1) < 1e-6 for f in frames if f["perspective"] in (0, 1, 2))
    # Allow the documented 150 ms mix transition before testing exterior wind suppression.
    settled = []
    last_perspective = None
    since = 0
    for frame in frames:
        if frame["perspective"] != last_perspective:
            since = frame["time"]
            last_perspective = frame["perspective"]
        if frame["time"] - since > 2:
            settled.append(frame)
    assert all(f["windGain"] < 1e-4 for f in settled if f["perspective"] in (3, 4))
    assert all(abs(f["engineCutoff"] - 2200) < 10 for f in settled if f["perspective"] == 2)
    return {
        "folder": str(folder), "camera": audio["camera"], "course": audio["course"],
        "checkpoint": audio["checkpoint"], "build": build["source"]["hash"],
        "sample_rate": audio["sampleRate"], "seconds": audio["seconds"], "rms": audio["rms"],
        "peak": audio["peak"], "full_scale_samples": audio["fullScaleSamples"],
        "dimensions": [performance["width"], performance["height"]],
        "median_ms": performance["medianMs"], "p95_ms": performance["p95Ms"], "p99_ms": performance["p99Ms"],
        "pitch_range": [min(f["pitch"] for f in frames), max(f["pitch"] for f in frames)],
        "perspectives": sorted({f["perspective"] for f in frames}),
        "cuts": sum(f["cut"] for f in frames), "resets": sum(f["reset"] for f in frames),
        "shifts": max(f.get("shiftEvents", 0) for f in frames), "checks_passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [inspect(folder) for folder in args.folders]
    assert len({r["course"] for r in reports}) == len({r["checkpoint"] for r in reports}) == 1
    assert len({r["build"] for r in reports}) == 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"recordings": reports, "listening_approval": "pending"}, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
