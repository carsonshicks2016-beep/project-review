"""
Dynamic route assignment and behavioral driver choice (Greedy GPS vs. Local Route-Informed).
"""

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from soonergrid.physics.link_attributes import EdgeAttributes


class RouteAssignmentEngine:
    """
    Computes and manages diverse candidate routes between Gateway Ingresses
    and Parking Sinks, modeling the split between greedy navigation app users and local drivers.
    """

    def __init__(self, graph: nx.MultiDiGraph, greedy_ratio: float = 0.75):
        self.graph = graph
        self.greedy_ratio = greedy_ratio
        # Cache of pre-computed K-shortest paths: (gw_node, sink_node) -> List[Tuple[path_nodes, edge_ids, dist_m, time_s]]
        self.route_cache: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    def _edge_weight_fft(self, u: int, v: int, d: Any) -> float:
        if "attr" in d:
            return d["attr"].free_flow_time_s
        weights = [val["attr"].free_flow_time_s for val in d.values() if "attr" in val]
        return min(weights) if weights else 1.0

    def get_k_shortest_paths(self, source: int, target: int, k: int = 3) -> List[Dict[str, Any]]:
        """
        Computes up to k diverse shortest paths using Yen's algorithm variation (edge-penalty heuristic).
        """
        cache_key = (source, target)
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]

        paths: List[Dict[str, Any]] = []
        
        # Path 1: Pure free-flow shortest path
        try:
            p1_nodes = nx.shortest_path(self.graph, source=source, target=target, weight=self._edge_weight_fft)
            p1_meta = self._compile_path_metadata(p1_nodes, "Primary Shortest Arterial")
            paths.append(p1_meta)
        except nx.NetworkXNoPath:
            return []

        # Path 2 & 3: Penalize edges of primary path to find genuine diverse alternatives
        penalized_edges = set()
        for i in range(len(p1_nodes) - 1):
            penalized_edges.add((p1_nodes[i], p1_nodes[i+1]))

        def penalized_weight(u: int, v: int, d: Any) -> float:
            base_w = self._edge_weight_fft(u, v, d)
            if (u, v) in penalized_edges:
                return base_w * 3.5  # Heavy penalty forces detour search
            return base_w

        try:
            p2_nodes = nx.shortest_path(self.graph, source=source, target=target, weight=penalized_weight)
            if p2_nodes != p1_nodes:
                p2_meta = self._compile_path_metadata(p2_nodes, "Secondary Bypass Route")
                paths.append(p2_meta)
        except nx.NetworkXNoPath:
            pass

        self.route_cache[cache_key] = paths
        return paths

    def _compile_path_metadata(self, node_path: List[int], label: str) -> Dict[str, Any]:
        """Extracts lengths, travel times, and edge IDs for a node sequence."""
        total_len_m = 0.0
        total_time_s = 0.0
        edge_ids = []
        street_names = []

        for i in range(len(node_path) - 1):
            u = node_path[i]
            v = node_path[i + 1]
            best_key = min(
                self.graph[u][v].keys(),
                key=lambda k: self.graph[u][v][k]["attr"].free_flow_time_s
            )
            attr: EdgeAttributes = self.graph[u][v][best_key]["attr"]
            total_len_m += attr.length_m
            total_time_s += attr.free_flow_time_s
            edge_ids.append(attr.edge_id)
            if attr.name not in street_names:
                street_names.append(attr.name)

        return {
            "label": label,
            "node_path": node_path,
            "edge_ids": edge_ids,
            "distance_km": round(total_len_m / 1000.0, 2),
            "free_flow_time_min": round(total_time_s / 60.0, 2),
            "corridors": street_names[:5],
        }

    def assign_route(self, source: int, target: int, is_local_driver: bool = False) -> Optional[Dict[str, Any]]:
        """
        Assigns a vehicle to a path based on driver knowledge profile:
        - Greedy GPS user: picks shortest path (Path 1).
        - Local informed driver: picks perimeter bypass if available (Path 2).
        """
        paths = self.get_k_shortest_paths(source, target, k=2)
        if not paths:
            return None

        if is_local_driver and len(paths) > 1:
            return paths[1]  # Local bypass route
        return paths[0]      # Standard primary route
