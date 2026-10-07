"""Offline, pinned Nordschleife conversion. No dependency on the Supra runtime."""
from __future__ import annotations
import bisect
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "art-source/circuits/nordschleife/source.json"
PROFILES = ROOT / "art-source/circuits/nordschleife/profiles.json"
VERSION = "nordschleife-import-v2"
GENERATOR_VERSION = "circuit-geometry-v7"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(source=SOURCE, profiles=PROFILES):
    raw = json.loads(Path(source).read_text())
    profile = json.loads(Path(profiles).read_text())
    if raw["schema"] != "supra-real-track-v1" or profile["schema"] not in (1, 2):
        raise ValueError("Unsupported circuit source/profile")
    xy, elevations = raw["center"], raw["elevation"]
    if len(xy) != len(elevations) or len(xy) < 4:
        raise ValueError("Circuit arrays do not match")
    if not all(math.isfinite(v) for point in xy for v in point) or not all(map(math.isfinite, elevations)):
        raise ValueError("Non-finite circuit source")
    arc = [0.0]
    for i, a in enumerate(xy):
        b = xy[(i + 1) % len(xy)]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length <= 0 or length > 8:
            raise ValueError("Invalid source segment")
        arc.append(arc[-1] + length)
    total = arc[-1]
    if abs(total - raw["official_length_m"]) > 1:
        raise ValueError("Source length differs from pinned target")
    t13 = next(l for l in raw["landmarks"] if l["name"] == "T13")
    origin_station = t13["start_arc"]

    def sample(station):
        station %= total
        i = min(len(xy) - 1, bisect.bisect_right(arc, station) - 1)
        t = (station - arc[i]) / (arc[i + 1] - arc[i])
        j = (i + 1) % len(xy)
        return [xy[i][0] * (1-t) + xy[j][0] * t,
                elevations[i] * (1-t) + elevations[j] * t,
                xy[i][1] * (1-t) + xy[j][1] * t]

    origin = sample(origin_station)
    # Preserve every original vertex and insert the T13 seam. Subdivide the source
    # polyline, rather than fitting a spline that can silently move sharp corners.
    stations = sorted({0.0, *((a-origin_station) % total for a in arc[:-1]),
                       *(float(i) for i in range(200, math.ceil(total), 200)),
                       *(v for section in profile.get("sections", []) for v in (section["start"], section["end"]) if 0 < v < total),
                       *(v for feature in profile.get("edgeFeatures", []) for v in (feature["start"], feature["start"]+feature["taper"], feature["end"]-feature["taper"], feature["end"]) if 0 < v < total)})
    points, distances = [], []
    for i, start in enumerate(stations):
        end = stations[i+1] if i+1 < len(stations) else total
        steps = max(1, math.ceil((end-start)/1.5))
        for j in range(steps):
            station = start + (end-start)*j/steps
            p = sample(station + origin_station)
            points.append(dict(zip(("x", "y", "z"), (v-o for v,o in zip(p,origin)))))
            distances.append(station)
    points.append(points[0].copy()); distances.append(total)
    landmarks = [{"name": l["name"], "station": (l["start_arc"]-origin_station) % total}
                 for l in raw["landmarks"]]
    landmarks.sort(key=lambda item: item["station"])
    intervals = []
    for interval in profile["intervals"]:
        landmark = next(l for l in landmarks if l["name"] == interval["landmark"])
        intervals.append({**interval, "start": landmark["station"]+interval["startOffset"],
                          "end": landmark["station"]+interval["endOffset"]})
    source_hash, profile_hash = digest(source), digest(profiles)
    revision = hashlib.sha256((VERSION+GENERATOR_VERSION+source_hash+profile_hash).encode()).hexdigest()
    road_profiles = []
    if profile["schema"] == 2:
        validate_profiles(profile, total)
        for station in distances:
            values = dict(profile["defaults"])
            section = next((s for s in profile["sections"] if s["start"] <= station % total < s["end"]), None)
            if section:
                index=profile["sections"].index(section);previous=profile["sections"][index-1]["values"]
                u=max(0,min(1,(station % total-section["start"])/min(40,(section["end"]-section["start"])*.25)))
                u=u*u*(3-2*u)
                for key,value in section["values"].items():
                    values[key]=previous.get(key,values[key])*(1-u)+value*u if isinstance(value,(int,float)) else value
            for feature in profile.get("edgeFeatures", []):
                if feature["start"] <= station % total <= feature["end"]:
                    t = min(1, (station % total-feature["start"])/feature["taper"], (feature["end"]-station % total)/feature["taper"])
                    t = max(0, t); t = t*t*(3-2*t)
                    values["leftKerb" if feature["side"] < 0 else "rightKerb"] = feature["width"]*t
                    side="left" if feature["side"]<0 else "right"
                    values[side+"KerbHeight"] = feature["height"]
                    values[side+"KerbBevel"] = feature.get("bevel",values.get(side+"KerbBevel",values["kerbBevel"]))
            for side in ("left","right"):
                values.setdefault(side+"KerbHeight",values["kerbHeight"]);values.setdefault(side+"KerbBevel",values["kerbBevel"])
            values["independentKerbDimensions"]=True
            road_profiles.append(values)
    return {"schema": 2, "name": "Nordschleife", "revision": revision,
            "sourceHash": source_hash, "profileHash": profile_hash, "importVersion": VERSION,
            "generatorVersion": GENERATOR_VERSION, "patches": [{**item,"start":next(l["station"] for l in landmarks if l["name"]==item["landmark"])+item["startOffset"],"end":next(l["station"] for l in landmarks if l["name"]==item["landmark"])+item["endOffset"]} for item in profile.get("surfaceDetails",[])], "structures": [{**item,"station":next(l["station"] for l in landmarks if l["name"]==item["landmark"])+item["stationOffset"]} for item in profile.get("structures",[])], "roadProfiles": road_profiles, "sections": profile.get("sections", []),
            "length": total, "width": raw["width_m"], "origin": dict(zip(("x","y","z"), origin)),
            "sourceStartStation": origin_station, "points": points, "stations": distances,
            "landmarks": landmarks, "intervals": intervals, "metadata": raw["metadata"],
            "confidence": "documented reconstruction; estimated road/edge profiles",
            "startReference": "T13 OSM landmark start; reconstructed timing gate, not surveyed timing line"}


def validate_profiles(profile, length):
    sections = profile["sections"]
    cursor = 0.0
    numeric = ("left", "right", "crossfall", "leftKerb", "rightKerb", "kerbHeight", "kerbBevel", "leftShoulder", "rightShoulder", "leftBarrier", "rightBarrier", "barrierHeight", "leftMarking", "rightMarking")
    for section in sections:
        if abs(section["start"]-cursor) > .001 or section["end"] <= cursor:
            raise ValueError("Profiles must cover the route exactly once")
        values = {**profile["defaults"], **section["values"]}
        side_dimensions=[values.get(side+suffix,values[legacy]) for side in ("left","right") for suffix,legacy in (("KerbHeight","kerbHeight"),("KerbBevel","kerbBevel"))]
        if any(not math.isfinite(v) or v<0 for v in side_dimensions):raise ValueError("Invalid independent kerb dimensions")
        if any(not math.isfinite(values[k]) for k in numeric): raise ValueError("Non-finite profile")
        if min(values[k] for k in numeric if k != "crossfall") < 0 or min(values["left"],values["right"]) < 2:
            raise ValueError("Invalid profile dimensions")
        if values["ground"] not in ("grass","soil"):raise ValueError("Unsupported ground material")
        if abs(values["crossfall"]) > 15: raise ValueError("Unsupported crossfall")
        if not section["evidence"] or not section["confidence"]: raise ValueError("Missing provenance")
        cursor = section["end"]
    if abs(cursor-length) > .001: raise ValueError("Incomplete profile coverage")
    for f in profile.get("edgeFeatures", []):
        if any(not math.isfinite(f[k]) or f[k]<0 for k in ("start","end","width","height","taper")) or not math.isfinite(f.get("bevel",0)) or f.get("bevel",0)<0:raise ValueError("Invalid kerb feature dimensions")
        if not 0 <= f["start"] < f["end"] <= length or f["side"] not in (-1,1) or f["taper"] <= 0:
            raise ValueError("Invalid edge interval")
    if sections[0]["values"] != sections[-1]["values"]: raise ValueError("Profile seam mismatch")


def variant(data, first=0, count=16):
    if not 0 <= first < 16 or not 1 <= count <= 16 or first+count > 16:
        raise ValueError("Select a contiguous non-wrapping range of 1-16 sectors")
    start, end = first*data["length"]/16, (first+count)*data["length"]/16
    return {"firstSector": first, "sectorCount": count, "startStation": start, "endStation": end,
            "episodeSeconds": 1800 if count == 16 else max(240, math.ceil((end-start)/12+60)),
            "fullLap": count == 16, "revision": data["revision"]}


def write_assets():
    data = convert()
    target = ROOT / "Assets/Resources/Circuits/Nordschleife.json"
    target.write_text(json.dumps(data, separators=(",", ":"))+"\n")
    report = {k: data[k] for k in ("revision", "sourceHash", "profileHash", "importVersion",
                                   "length", "sourceStartStation", "startReference", "confidence", "generatorVersion")}
    report.update(sourcePath="/Users/REVIEW_USER/Desktop/untitled folder/01 Core Work/Python/Supra Ai 2/supra/data/tracks/nordschleife.json",
                  samples=len(data["points"]), interpolation="periodic source polyline, <=1.5 m spacing; preserves source vertices",
                  elevationRange=max(p["y"] for p in data["points"])-min(p["y"] for p in data["points"]),
                  attribution=data["metadata"]["attribution"], sourceMetadata=data["metadata"])
    (SOURCE.parent/"import-report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    print(json.dumps(write_assets(), indent=2))
    from .circuit_terrain import prepare
    print(json.dumps(prepare(convert(), ROOT/"Assets/Resources/Circuits/NordschleifeTerrain.json"), indent=2))
