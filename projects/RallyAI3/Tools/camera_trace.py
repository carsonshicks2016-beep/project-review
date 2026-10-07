"""Summarize an opt-in standalone camera trace; mode cuts are not jitter."""
import json
import statistics
import sys


def summarize(rows):
    result = {}
    settled=[]
    previous=None
    start=0
    for row in rows:
        if (previous is None or row["mode"]!=previous["mode"] or
                row.get("shot",row["mode"])!=previous.get("shot",previous["mode"]) or
                row.get("station",-1)!=previous.get("station",-1) or
                row["frame"]!=previous["frame"]+1):
            start=row["time"]
        if row["time"]-start>.5:
            settled.append(row)
        previous=row
    for mode in sorted({row["mode"] for row in rows}):
        frames = [row for row in settled if row["mode"] == mode and row["scale"] > 0 and row["time"] > 1]
        if len(frames) < 3:
            continue
        pairs = [(a,b) for a,b in zip(frames,frames[1:]) if b["frame"] == a["frame"]+1]
        steps = [abs(b["fov"]-a["fov"]) for a,b in pairs]
        dt = [row["dt"] for row in frames]
        result[str(mode)] = {
            "frames": len(frames), "mean_fps": 1/statistics.mean(dt),
            "median_frame_ms": statistics.median(dt)*1000,
            "max_fov_step_degrees": max(steps, default=0),
            "fov_jumps_over_one_degree": sum(step>1 for step in steps),
            "vsync": sorted({row["vsync"] for row in frames}),
            "frame_cap": sorted({row["fpsCap"] for row in frames}),
            "safe_frame_fraction": (sum(row["carInSafeFrame"] for row in frames) / len(frames)
                                    if all("carInSafeFrame" in row for row in frames) else None),
        }
    return result


if __name__ == "__main__":
    with open(sys.argv[1]) as stream:
        print(json.dumps(summarize([json.loads(line) for line in stream]),indent=2))
