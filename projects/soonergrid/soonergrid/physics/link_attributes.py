"""
Physical and transportation engineering attributes for road network edges.

Implements:
- Bureau of Public Roads (BPR) link performance function
- Lighthill-Whitham-Richards (LWR) jam storage and vehicle density limits
- Functional classification heuristics for speeds, capacities, and lane counts
- Contraflow (reversible lane) twin-edge pairing logic
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import re


# Standard SI and traffic engineering constants
DEFAULT_VEHICLE_JAM_SPACING_M = 7.5  # ~133 vehicles / km / lane
METERS_PER_MILE = 1609.34
MPS_PER_MPH = 0.44704

# Default nominal saturation capacities (vehicles / hour / lane)
NOMINAL_LANE_CAPACITIES_VPH = {
    "motorway": 2000,
    "motorway_link": 1400,
    "trunk": 1700,
    "trunk_link": 1200,
    "primary": 1100,       # Signalized multi-lane arterials (e.g. Lindsey, Classen)
    "primary_link": 900,
    "secondary": 900,       # Main St, Robinson St, 12th Ave
    "secondary_link": 800,
    "tertiary": 700,        # Jenkins Ave, Berry Rd
    "tertiary_link": 600,
    "residential": 500,     # Campus & neighborhood streets
    "unclassified": 600,
    "living_street": 300,
}

# Default speed limits in mph when OSM maxspeed tag is absent
DEFAULT_SPEED_LIMITS_MPH = {
    "motorway": 65,         # I-35 mainline
    "motorway_link": 35,    # Off-ramps / on-ramps
    "trunk": 55,            # SH-9 divided corridor
    "trunk_link": 35,
    "primary": 35,          # Lindsey St, Classen Blvd
    "primary_link": 25,
    "secondary": 30,        # Main St, Robinson St
    "secondary_link": 25,
    "tertiary": 25,         # Jenkins Ave, Chautauqua Ave
    "tertiary_link": 20,
    "residential": 20,      # Campus streets (Asp, Elm, Brooks)
    "unclassified": 25,
    "living_street": 15,
}

# Corridors explicitly evaluated for dynamic contraflow (reversible lane) operations
CONTRAFLOW_CANDIDATE_KEYWORDS = [
    "lindsey",
    "jenkins",
    "brooks",
    "chautauqua",
    "asp",
    "elm",
    "timberdell",
]


@dataclass
class EdgeAttributes:
    """Research-grade transportation edge container."""
    edge_id: str
    u: int
    v: int
    name: str
    highway_type: str
    length_m: float
    lanes: int
    free_speed_mps: float
    free_speed_mph: float
    free_flow_time_s: float
    capacity_vph: float
    jam_storage_veh: int
    is_oneway: bool
    is_reversible: bool
    twin_edge_id: Optional[str] = None
    geometry: List[Tuple[float, float]] = field(default_factory=list)  # Local metric (x, y) meters
    lat_lon_points: List[Tuple[float, float]] = field(default_factory=list)  # (lat, lon) coordinates

    def bpr_travel_time(self, volume_vph: float, alpha: float = 0.15, beta: float = 4.0) -> float:
        """
        Calculates travel time using the Bureau of Public Roads (BPR) function:
        t(V) = t_0 * (1 + alpha * (V / C)^beta)
        """
        if self.capacity_vph <= 0:
            return self.free_flow_time_s * 10.0
        vc_ratio = max(0.0, volume_vph / self.capacity_vph)
        return self.free_flow_time_s * (1.0 + alpha * (vc_ratio ** beta))

    def density_to_velocity_lwr(self, vehicles_on_edge: float) -> float:
        """
        Greenshields continuous LWR velocity function:
        v(k) = v_max * max(0, 1 - (k / k_jam))
        """
        if self.jam_storage_veh <= 0:
            return 0.0
        ratio = min(1.0, max(0.0, vehicles_on_edge / self.jam_storage_veh))
        return self.free_speed_mps * (1.0 - ratio)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "u": self.u,
            "v": self.v,
            "name": self.name,
            "highway_type": self.highway_type,
            "length_m": round(self.length_m, 2),
            "lanes": self.lanes,
            "free_speed_mph": round(self.free_speed_mph, 1),
            "free_speed_mps": round(self.free_speed_mps, 2),
            "free_flow_time_s": round(self.free_flow_time_s, 2),
            "capacity_vph": round(self.capacity_vph, 1),
            "jam_storage_veh": self.jam_storage_veh,
            "is_oneway": self.is_oneway,
            "is_reversible": self.is_reversible,
            "twin_edge_id": self.twin_edge_id,
            "geometry": [(round(x, 2), round(y, 2)) for x, y in self.geometry],
            "lat_lon_points": [(round(lat, 6), round(lon, 6)) for lat, lon in self.lat_lon_points],
        }


def parse_speed_mph(maxspeed_str: Optional[str], highway_type: str) -> float:
    """Parses OSM maxspeed tag or falls back to road type heuristics."""
    if maxspeed_str:
        # Match digits
        m = re.search(r"(\d+)", str(maxspeed_str))
        if m:
            val = float(m.group(1))
            # If specified in knots or km/h (rare in US OSM, but standard precaution)
            if "km/h" in str(maxspeed_str).lower():
                val = val * 0.621371
            if 10.0 <= val <= 85.0:
                return val
    return float(DEFAULT_SPEED_LIMITS_MPH.get(highway_type, 25))


def parse_lanes_count(lanes_str: Optional[str], highway_type: str, is_oneway: bool) -> int:
    """Parses OSM lanes tag or applies functional classification defaults."""
    if lanes_str:
        m = re.search(r"(\d+)", str(lanes_str))
        if m:
            val = int(m.group(1))
            if 1 <= val <= 8:
                # If bidirectional street with total lanes, one-way directional graph edge gets half
                if not is_oneway and val >= 2:
                    return max(1, val // 2)
                return val
    # Defaults
    if highway_type in ["motorway"]:
        return 3
    elif highway_type in ["trunk", "primary"]:
        return 2 if is_oneway else 1
    return 1


def check_contraflow_candidate(name: str) -> bool:
    """Identifies key game-day arterials that can be reconfigured dynamically."""
    if not name:
        return False
    lower = name.lower()
    return any(keyword in lower for keyword in CONTRAFLOW_CANDIDATE_KEYWORDS)


def compute_edge_attributes(
    edge_id: str,
    u: int,
    v: int,
    osm_tags: Dict[str, Any],
    length_m: float,
    is_oneway: bool,
    geometry: List[Tuple[float, float]],
    lat_lon_points: List[Tuple[float, float]]
) -> EdgeAttributes:
    """Factory function converting raw OSM data into calibrated EdgeAttributes."""
    name = str(osm_tags.get("name", "Unnamed Road"))
    hwy = str(osm_tags.get("highway", "residential")).lower()
    if ";" in hwy:
        hwy = hwy.split(";")[0].strip()

    lanes = parse_lanes_count(osm_tags.get("lanes"), hwy, is_oneway)
    speed_mph = parse_speed_mph(osm_tags.get("maxspeed"), hwy)
    speed_mps = speed_mph * MPS_PER_MPH

    # Free-flow traversal time (seconds)
    eff_length = max(1.0, length_m)
    fft_s = eff_length / max(1.0, speed_mps)

    # Saturation capacity (vph)
    nominal_lane_cap = NOMINAL_LANE_CAPACITIES_VPH.get(hwy, 600)
    capacity_vph = float(lanes * nominal_lane_cap)

    # Jam storage (vehicles)
    jam_density_per_m = lanes / DEFAULT_VEHICLE_JAM_SPACING_M
    jam_storage = max(1, int(round(eff_length * jam_density_per_m)))

    is_reversible = check_contraflow_candidate(name)

    return EdgeAttributes(
        edge_id=edge_id,
        u=u,
        v=v,
        name=name,
        highway_type=hwy,
        length_m=eff_length,
        lanes=lanes,
        free_speed_mps=speed_mps,
        free_speed_mph=speed_mph,
        free_flow_time_s=fft_s,
        capacity_vph=capacity_vph,
        jam_storage_veh=jam_storage,
        is_oneway=is_oneway,
        is_reversible=is_reversible,
        geometry=geometry,
        lat_lon_points=lat_lon_points
    )
