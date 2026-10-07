"""
Rule-based coordinated signal controller.
Historical MARL names are retained for API compatibility; no policy is learned.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Target arterial progression speed along Lindsey St
PROGRESSION_SPEED_MPS = 13.41  # 30 mph in m/s


@dataclass
class MARLSignalNode:
    id: str
    name: str
    cycle_length_s: float
    base_arterial_green_s: float
    min_green_s: float
    max_green_s: float
    dist_from_spui_m: float      # Distance along Lindsey St from I-35 interchange
    arterial_keywords: List[str]
    cross_keywords: List[str]
    upstream_id: Optional[str] = None
    downstream_id: Optional[str] = None


MARL_SIGNAL_NODES: List[MARLSignalNode] = [
    MARLSignalNode(
        id="SIG_I35_LINDSEY_SPUI",
        name="I-35 & W Lindsey St SPUI",
        cycle_length_s=90.0,
        base_arterial_green_s=45.0,
        min_green_s=20.0,
        max_green_s=65.0,
        dist_from_spui_m=0.0,
        arterial_keywords=["lindsey"],
        cross_keywords=["i 35", "interstate 35", "i-35"],
        downstream_id="SIG_LINDSEY_MCGEE"
    ),
    MARLSignalNode(
        id="SIG_LINDSEY_MCGEE",
        name="W Lindsey St & McGee Dr",
        cycle_length_s=90.0,
        base_arterial_green_s=45.0,
        min_green_s=20.0,
        max_green_s=65.0,
        dist_from_spui_m=650.0,
        arterial_keywords=["lindsey"],
        cross_keywords=["mcgee"],
        upstream_id="SIG_I35_LINDSEY_SPUI",
        downstream_id="SIG_LINDSEY_BERRY"
    ),
    MARLSignalNode(
        id="SIG_LINDSEY_BERRY",
        name="W Lindsey St & Berry Rd (Major Chokepoint)",
        cycle_length_s=90.0,
        base_arterial_green_s=45.0,
        min_green_s=20.0,
        max_green_s=65.0,
        dist_from_spui_m=1450.0,
        arterial_keywords=["lindsey"],
        cross_keywords=["berry"],
        upstream_id="SIG_LINDSEY_MCGEE",
        downstream_id="SIG_LINDSEY_CHAUTAUQUA"
    ),
    MARLSignalNode(
        id="SIG_LINDSEY_CHAUTAUQUA",
        name="W Lindsey St & Chautauqua Ave",
        cycle_length_s=90.0,
        base_arterial_green_s=45.0,
        min_green_s=20.0,
        max_green_s=65.0,
        dist_from_spui_m=2250.0,
        arterial_keywords=["lindsey"],
        cross_keywords=["chautauqua"],
        upstream_id="SIG_LINDSEY_BERRY",
        downstream_id="SIG_LINDSEY_JENKINS"
    ),
    MARLSignalNode(
        id="SIG_LINDSEY_JENKINS",
        name="W Lindsey St & S Jenkins Ave (Stadium SE Corner)",
        cycle_length_s=90.0,
        base_arterial_green_s=45.0,
        min_green_s=20.0,
        max_green_s=65.0,
        dist_from_spui_m=3050.0,
        arterial_keywords=["lindsey"],
        cross_keywords=["jenkins"],
        upstream_id="SIG_LINDSEY_CHAUTAUQUA"
    ),
    MARLSignalNode(
        id="SIG_CLASSEN_LINDSEY",
        name="Classen Blvd & E Lindsey St",
        cycle_length_s=90.0,
        base_arterial_green_s=45.0,
        min_green_s=20.0,
        max_green_s=65.0,
        dist_from_spui_m=3800.0,
        arterial_keywords=["classen"],
        cross_keywords=["lindsey"],
    ),
    MARLSignalNode(
        id="SIG_SH9_JENKINS",
        name="State Highway 9 & S Jenkins Ave (LNC Entrance)",
        cycle_length_s=90.0,
        base_arterial_green_s=55.0,
        min_green_s=25.0,
        max_green_s=70.0,
        dist_from_spui_m=0.0,
        arterial_keywords=["state highway 9", "sh 9", "highway 9"],
        cross_keywords=["jenkins"],
    ),
]


class MARLSignalController:
    """
    Rule-based signal controller (historical API name).
    Dynamically aligns phase offsets to create moving Green Waves along Lindsey St.
    """

    def __init__(self, progression_speed_mps: float = PROGRESSION_SPEED_MPS):
        self.v_prog = progression_speed_mps
        self.nodes = MARL_SIGNAL_NODES
        self.nodes_by_id = {n.id: n for n in self.nodes}

        # Dynamic green splits per node: node_id -> active_arterial_green_s
        self.active_greens: Dict[str, float] = {n.id: n.base_arterial_green_s for n in self.nodes}
        # Dynamic phase offsets per node: node_id -> offset_s
        self.active_offsets: Dict[str, float] = {n.id: 0.0 for n in self.nodes}

        # Fast lookup mapping: edge_id -> (node_id, is_arterial)
        self.edge_to_signal_map: Dict[str, Tuple[str, bool]] = {}

    def register_signal_edges(self, edge_meta: Dict[str, Dict[str, Any]]):
        """Pre-maps road edge IDs to signal nodes for O(1) evaluation."""
        for eid, meta in edge_meta.items():
            lower = meta["name"].lower()
            for node in self.nodes:
                is_art = any(k in lower for k in node.arterial_keywords)
                is_cross = any(k in lower for k in node.cross_keywords)
                if is_art or is_cross:
                    self.edge_to_signal_map[eid] = (node.id, is_art)
                    break

        print(f"[MARLSignals] Pre-mapped {len(self.edge_to_signal_map)} links to {len(self.nodes)} MARL signal agents.")

    def synchronize_green_waves(self, is_egress: bool = False):
        """
        Coordinates phase offsets between intersections along Lindsey St to produce
        continuous moving green waves at 30 mph:
        - Ingress (Eastbound): Offset increases with distance from I-35
        - Egress (Westbound): Offset increases with distance from Stadium towards I-35
        """
        for node in self.nodes:
            if "LINDSEY" not in node.id:
                self.active_offsets[node.id] = 0.0
                continue

            dist = node.dist_from_spui_m
            if not is_egress:
                # Eastbound progression (toward stadium)
                # Offset = (distance / speed) % cycle
                ideal_offset = (dist / self.v_prog) % node.cycle_length_s
            else:
                # Westbound progression (toward I-35)
                # Max distance is at Jenkins (~3050m)
                dist_from_stadium = max(0.0, 3050.0 - dist)
                ideal_offset = (dist_from_stadium / self.v_prog) % node.cycle_length_s

            self.active_offsets[node.id] = round(ideal_offset, 1)

    def adapt_green_splits(self, link_states: Dict[str, Any], is_peak_surge: bool = True):
        """
        Adjusts green splits based on real-time arterial queue pressure.
        During peak surge, arterial green expands up to max_green (e.g. 65s of 90s).
        """
        for node in self.nodes:
            if is_peak_surge:
                # Allocate 65s of 90s (72% green ratio) to mainline arterial
                self.active_greens[node.id] = min(node.max_green_s, node.base_arterial_green_s + 15.0)
            else:
                self.active_greens[node.id] = node.base_arterial_green_s

    def get_signal_green_fraction(self, edge_id: str, t_sim_s: float) -> float:
        """
        Returns instantaneous effective green fraction [0.0, 1.0] for link edge_id in O(1).
        """
        mapping = self.edge_to_signal_map.get(edge_id)
        if not mapping:
            return 1.0  # Uncontrolled free flow

        node_id, is_arterial = mapping
        node = self.nodes_by_id[node_id]
        cycle = node.cycle_length_s
        offset = self.active_offsets.get(node_id, 0.0)
        art_green = self.active_greens.get(node_id, node.base_arterial_green_s)

        t_rel = (t_sim_s - offset) % cycle

        if is_arterial:
            # Mainline arterial green phase
            return 1.0 if t_rel < art_green else 0.05
        else:
            # Cross street green phase
            start_cross = art_green + 6.0
            end_cross = start_cross + (cycle - art_green - 12.0)
            return 1.0 if start_cross <= t_rel < end_cross else 0.05
