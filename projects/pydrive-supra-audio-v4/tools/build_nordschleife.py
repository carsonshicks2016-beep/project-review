"""Build the pinned real-scale Nordschleife track asset.

Inputs:
  * OpenStreetMap relation 38566 (Nuerburgring Nordschleife route)
  * Rheinland-Pfalz DGM1 GeoTIFF tiles (ETRS89 / UTM zone 32N)

The generated runtime asset is intentionally small: centreline, elevation,
landmarks, sectors, and source metadata. The raw DEM tiles are cached outside
the repo by default because they are large and reproducible from the metadata.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "supra" / "data" / "tracks"
OSM_RELATION_ID = 38566
OFFICIAL_LENGTH_M = 20_832.0
DEFAULT_SPACING_M = 3.0

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
)
OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"
OFFICIAL_FACT_URL = (
    "https://nuerburgring.de/news/"
    "rekordjagd-auf-der-nordschleife-die-offiziellen-bestzeiten-2025?locale=en"
)
OPENDTM_URL = "https://www.opendem.info/opendtm_de.html"
RLP_DGM1_CONFIG_URL = "https://geoshop.rlp.de/opendata-dgm1.html"
DGM1_META4_URL = "https://geobasis-rlp.de/data/dgm1/current/meta4/dgm1_tif_07.meta4"


# WGS84 -> UTM zone 32N. Kept local so rebuilding does not require pyproj.
_A = 6378137.0
_F = 1.0 / 298.257223563
_K0 = 0.9996
_E = math.sqrt(_F * (2.0 - _F))
_EP2 = _E * _E / (1.0 - _E * _E)


def _utm32(lat: float, lon: float) -> tuple[float, float]:
    latr = math.radians(lat)
    lonr = math.radians(lon)
    lon0 = math.radians(9.0)
    sin_lat = math.sin(latr)
    cos_lat = math.cos(latr)
    n = _A / math.sqrt(1.0 - _E * _E * sin_lat * sin_lat)
    t = math.tan(latr) ** 2
    c = _EP2 * cos_lat * cos_lat
    a = cos_lat * (lonr - lon0)
    m = _A * (
        (1 - _E**2 / 4 - 3 * _E**4 / 64 - 5 * _E**6 / 256) * latr
        - (3 * _E**2 / 8 + 3 * _E**4 / 32 + 45 * _E**6 / 1024) * math.sin(2 * latr)
        + (15 * _E**4 / 256 + 45 * _E**6 / 1024) * math.sin(4 * latr)
        - (35 * _E**6 / 3072) * math.sin(6 * latr)
    )
    x = _K0 * n * (
        a
        + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t * t + 72 * c - 58 * _EP2) * a**5 / 120
    ) + 500000.0
    y = _K0 * (
        m
        + n * math.tan(latr) * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c * c) * a**4 / 24
            + (61 - 58 * t + t * t + 600 * c - 330 * _EP2) * a**6 / 720
        )
    )
    return x, y


def _get(url: str, *, timeout: int = 60) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": "SupraAI/0.1 local"}, timeout=timeout)
    r.raise_for_status()
    return r


def fetch_osm_relation(cache_path: Path, force: bool = False) -> dict:
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())
    query = f"""[out:json][timeout:60];
relation({OSM_RELATION_ID});
(._;>;);
out body;"""
    errors = []
    data = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(
                url,
                data={"data": query},
                headers={"User-Agent": "SupraAI/0.1 local"},
                timeout=90,
            )
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            errors.append(f"{url}: {e}")
    if data is None:
        raise RuntimeError("Could not fetch OSM relation:\n" + "\n".join(errors))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def ordered_relation_points(osm: dict):
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in osm["elements"] if e["type"] == "node"}
    ways = {e["id"]: e for e in osm["elements"] if e["type"] == "way"}
    rel = next(e for e in osm["elements"] if e["type"] == "relation")

    seq: list[int] = []
    intervals = []
    for member in rel["members"]:
        way = ways[member["ref"]]
        ns = list(way["nodes"])
        if not seq:
            use = ns
        elif seq[-1] == ns[0]:
            use = ns[1:]
        elif seq[-1] == ns[-1]:
            use = list(reversed(ns[:-1]))
        else:
            raise RuntimeError(
                f"OSM relation gap before way {member['ref']}: "
                f"current {seq[-1]} vs way ends {ns[0]}/{ns[-1]}"
            )
        start_idx = max(0, len(seq) - 1)
        seq.extend(use)
        end_idx = len(seq) - 1
        name = way.get("tags", {}).get("name")
        if name:
            intervals.append((name, start_idx, end_idx, way["id"], way.get("tags", {})))

    if seq[0] != seq[-1]:
        raise RuntimeError("OSM Nordschleife relation did not close")

    lat = np.asarray([nodes[n][0] for n in seq], dtype=float)
    lon = np.asarray([nodes[n][1] for n in seq], dtype=float)
    xy = np.asarray([_utm32(float(a), float(o)) for a, o in zip(lat, lon)], dtype=float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return rel, lat, lon, xy, arc, intervals


def _interp_closed(values: np.ndarray, src_arc: np.ndarray, dst_arc: np.ndarray) -> np.ndarray:
    return np.interp(dst_arc, src_arc, values)


class DgmSampler:
    def __init__(self, cache_dir: Path, force: bool = False):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = cache_dir / "dgm1_tif_07.meta4"
        if force or not self.meta_path.exists():
            self.meta_path.write_text(_get(DGM1_META4_URL, timeout=90).text)
        self.meta = self.meta_path.read_text()
        self.images = {}

    def _tile_url(self, km_x: int, km_y: int) -> tuple[str, str]:
        pat = (
            rf'<file name="(dgm1_32_{km_x}_{km_y}_1_rp_\d{{4}}\.tif)".*?'
            rf"<url>(https://[^<]+)</url>"
        )
        m = re.search(pat, self.meta, re.S)
        if not m:
            raise RuntimeError(f"No RLP DGM1 tile found for UTM32 km {km_x}/{km_y}")
        return m.group(1), m.group(2)

    def _load_tile(self, km_x: int, km_y: int):
        key = (int(km_x), int(km_y))
        if key in self.images:
            return self.images[key]
        from PIL import Image

        name, url = self._tile_url(*key)
        path = self.cache_dir / name
        if not path.exists():
            path.write_bytes(_get(url, timeout=90).content)
        im = Image.open(path)
        arr = np.asarray(im, dtype=float)
        tie = im.tag_v2[33922]
        pix = im.tag_v2[33550]
        nodata = float(im.tag_v2.get(42113, -9999))
        item = (arr, float(tie[3]), float(tie[4]), float(pix[0]), float(pix[1]), nodata)
        self.images[key] = item
        return item

    def sample(self, x: float, y: float) -> float:
        tile = (int(x // 1000), int(y // 1000))
        arr, ox, oy, px, py, nodata = self._load_tile(*tile)
        col = (x - ox) / px
        row = (oy - y) / py
        c0 = int(math.floor(col))
        r0 = int(math.floor(row))
        if c0 < 0 or r0 < 0 or c0 >= arr.shape[1] - 1 or r0 >= arr.shape[0] - 1:
            return float("nan")
        dc = col - c0
        dr = row - r0
        vals = np.asarray(
            [arr[r0, c0], arr[r0, c0 + 1], arr[r0 + 1, c0], arr[r0 + 1, c0 + 1]],
            dtype=float,
        )
        if np.any(vals == nodata):
            return float("nan")
        return float(
            (1 - dc) * (1 - dr) * vals[0]
            + dc * (1 - dr) * vals[1]
            + (1 - dc) * dr * vals[2]
            + dc * dr * vals[3]
        )


def _fill_nans_loop(z: np.ndarray) -> tuple[np.ndarray, int]:
    bad = np.isnan(z)
    count = int(bad.sum())
    if not count:
        return z, 0
    good = np.where(~bad)[0]
    if good.size < 2:
        raise RuntimeError("DEM sampling produced too many NaNs to repair")
    idx = np.arange(len(z))
    # Periodic interpolation by extending good samples one loop left/right.
    gx = np.concatenate([good - len(z), good, good + len(z)])
    gz = np.concatenate([z[good], z[good], z[good]])
    z[bad] = np.interp(idx[bad], gx, gz)
    return z, count


def _landmarks(intervals, raw_arc: np.ndarray, scale: float) -> list[dict]:
    out = []
    for name, start_idx, end_idx, wid, tags in intervals:
        out.append(
            {
                "name": name,
                "start_arc": float(raw_arc[start_idx] * scale),
                "end_arc": float(raw_arc[end_idx] * scale),
                "osm_way": int(wid),
                "surface": tags.get("surface"),
                "maxspeed": tags.get("maxspeed"),
            }
        )
    return out


def _sectors() -> list[dict]:
    step = OFFICIAL_LENGTH_M / 4.0
    return [
        {"name": "Sector 1", "start_arc": 0.0, "end_arc": step},
        {"name": "Sector 2", "start_arc": step, "end_arc": step * 2},
        {"name": "Sector 3", "start_arc": step * 2, "end_arc": step * 3},
        {"name": "Sector 4", "start_arc": step * 3, "end_arc": OFFICIAL_LENGTH_M},
    ]


def build(args) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "nordschleife_osm_relation_38566.json"
    osm = fetch_osm_relation(raw_path, force=args.force_osm)
    rel, lat, lon, raw_xy, raw_arc, intervals = ordered_relation_points(osm)
    raw_length = float(raw_arc[-1])
    length_scale = OFFICIAL_LENGTH_M / raw_length

    n = int(round(OFFICIAL_LENGTH_M / args.spacing))
    spacing = OFFICIAL_LENGTH_M / n
    sim_arc = np.arange(n, dtype=float) * spacing
    source_arc = sim_arc / length_scale

    raw_x = _interp_closed(raw_xy[:, 0], raw_arc, source_arc)
    raw_y = _interp_closed(raw_xy[:, 1], raw_arc, source_arc)
    out_lat = _interp_closed(lat, raw_arc, source_arc)
    out_lon = _interp_closed(lon, raw_arc, source_arc)
    centroid = np.array([raw_x.mean(), raw_y.mean()])
    center = (np.stack([raw_x, raw_y], axis=1) - centroid) * length_scale
    chord_length = float(np.linalg.norm(np.diff(center, axis=0, append=center[:1]), axis=1).sum())
    final_chord_scale = OFFICIAL_LENGTH_M / chord_length
    center *= final_chord_scale

    sampler = DgmSampler(args.cache_dir, force=args.force_dem_meta)
    z = np.asarray([sampler.sample(float(x), float(y)) for x, y in zip(raw_x, raw_y)], dtype=float)
    z, nan_count = _fill_nans_loop(z)

    elev_range = float(np.max(z) - np.min(z))
    elev_validation = (
        "ok" if elev_range >= 300.0
        else "warning_dem_centerline_below_official_over_300m_claim"
    )

    asset = {
        "schema": "supra-real-track-v1",
        "name": "nordschleife",
        "display_name": "Nuerburgring Nordschleife",
        "official_length_m": OFFICIAL_LENGTH_M,
        "generated_spacing_m": spacing,
        "raw_osm_length_m": raw_length,
        "length_scale": length_scale,
        "final_chord_scale": final_chord_scale,
        "width_m": args.width,
        "width_source": "approximate constant racing width; OSM member widths are incomplete",
        "center": np.round(center, 4).tolist(),
        "elevation": np.round(z, 4).tolist(),
        "latlon": np.round(np.stack([out_lat, out_lon], axis=1), 8).tolist(),
        "landmarks": _landmarks(intervals, raw_arc, length_scale),
        "sectors": _sectors(),
        "metadata": {
            "real_track": True,
            "long_track": True,
            "profile": "nordschleife-full-20.832km",
            "source_confidence": "open-data-calibrated",
            "official_length_target_m": OFFICIAL_LENGTH_M,
            "official_elevation_claim": "over 300 m elevation difference",
            "dem_elevation_range_m": elev_range,
            "dem_elevation_min_m": float(np.min(z)),
            "dem_elevation_max_m": float(np.max(z)),
            "dem_nan_filled_samples": nan_count,
            "elevation_validation": elev_validation,
            "osm_relation_id": OSM_RELATION_ID,
            "osm_relation_name": rel.get("tags", {}).get("name"),
            "osm_relation_timestamp": osm.get("osm3s", {}).get("timestamp_osm_base"),
            "sources": [
                {"label": "Nuerburgring official facts", "url": OFFICIAL_FACT_URL},
                {"label": "OpenStreetMap relation 38566", "url": OSM_COPYRIGHT_URL},
                {"label": "OpenDTM-DE overview", "url": OPENDTM_URL},
                {"label": "RLP DGM1 product page", "url": RLP_DGM1_CONFIG_URL},
                {"label": "RLP DGM1 tile manifest", "url": DGM1_META4_URL},
            ],
            "attribution": (
                "Contains OpenStreetMap data (ODbL) and Rheinland-Pfalz DGM1 "
                "terrain data (dl-de/by-2-0)."
            ),
        },
    }

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asset, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    print(
        f"points={n} length_scale={length_scale:.8f} "
        f"final_chord_scale={final_chord_scale:.8f} raw_osm_length={raw_length:.2f}m"
    )
    print(f"elevation range={elev_range:.2f}m min={np.min(z):.2f} max={np.max(z):.2f} nan_filled={nan_count}")
    if elev_validation != "ok":
        print(f"WARNING: {elev_validation}")
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "nordschleife.json")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "nordschleife_dgm1")
    ap.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M)
    ap.add_argument("--width", type=float, default=11.0)
    ap.add_argument("--force-osm", action="store_true")
    ap.add_argument("--force-dem-meta", action="store_true")
    args = ap.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
