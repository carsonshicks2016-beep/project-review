"""Package level-preserving listening references; raw recordings remain in .rally."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "art-source/audio/perspective-review"
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for mode in ("chase", "driver", "trackside"):
        for label, prefix in (("before", "audio-baseline"), ("after", "audio-final")):
            source = ROOT / ".rally/visual-review/v3" / f"{prefix}-{mode}-720"
            audio = json.loads((source / "audio.json").read_text())
            build = json.loads((source / "build.json").read_text())
            launch = json.loads((source / "launch.json").read_text())
            performance = json.loads((source / "performance.json").read_text())
            destination = output / f"{label}-{mode}.mp3"
            subprocess.run(["/opt/homebrew/bin/ffmpeg", "-nostdin", "-v", "error", "-i",
                            str(source / "listener.wav"), "-codec:a", "libmp3lame", "-b:a", "160k",
                            "-map_metadata", "-1", str(destination)], check=True)
            records.append({"file": destination.name, "source": str(source.relative_to(ROOT)),
                            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                            "raw_sha256": hashlib.sha256((source / "listener.wav").read_bytes()).hexdigest(),
                            "build": build["source"]["hash"], "course": launch["courseId"],
                            "checkpoint": launch["checkpointId"], "seed": launch["seed"],
                            "startingGear": launch["startingGear"], "deterministic": launch["deterministic"],
                            "width": performance["width"], "height": performance["height"],
                            "raw_audio": audio})
    assert len({r["course"] for r in records}) == len({r["checkpoint"] for r in records}) == 1
    assert len({(r["seed"], r["startingGear"], r["deterministic"]) for r in records}) == 1
    report = json.loads((ROOT / ".rally/audio-review/20261004/runtime-evidence.json").read_text())
    report.update({"schema": 1, "codec": "MP3 160 kbps; no gain adjustment or normalization",
                   "listening": records,
                   "native_lifetime": json.loads((ROOT / ".rally/visual-review/v3/audio-native-lifetime-720/audio-lifecycle.json").read_text()),
                   "headless": json.loads((ROOT / ".rally/audio-review/20261004/headless/isolation.json").read_text())})
    (output / "evidence.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
