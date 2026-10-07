#!/usr/bin/env python3
"""
Master Execution Pipeline for SoonerGrid Step 1:
Norman Road Network Ingestion, Attribution, and Topological Validation.
"""

import sys
import os
import time

# Ensure package directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soonergrid.data.osm_extractor import OSMExtractor, DEFAULT_BBOX
from soonergrid.data.network_builder import NetworkBuilder, ORIGIN_LAT, ORIGIN_LON
from soonergrid.data.game_day_zones import NORMAN_GATEWAYS, NORMAN_PARKING_SINKS, bind_zones_to_graph
from soonergrid.visualizer.network_qc import NetworkQC


def run_pipeline(force_refresh: bool = False):
    print("=" * 70)
    print("🏈 SOONERGRID STEP 1: RESEARCH-GRADE NORMAN ROAD NETWORK PIPELINE")
    print("=" * 70)
    start_time = time.time()

    # 1. Ingest OSM data
    extractor = OSMExtractor(bbox=DEFAULT_BBOX, cache_dir="data/cache")
    osm_raw = extractor.fetch(force_refresh=force_refresh)

    # 2. Build and simplify metric topological graph
    builder = NetworkBuilder(origin_lat=ORIGIN_LAT, origin_lon=ORIGIN_LON)
    graph = builder.build_graph_from_osm(osm_raw, simplify=True)

    # 3. Map nodes dictionary for gateway & sink binding
    nodes_metric = {
        nid: (data["x"], data["y"])
        for nid, data in graph.nodes(data=True)
    }

    # 4. Bind Gateways and Parking Sinks to road graph
    gateways, sinks = bind_zones_to_graph(nodes_metric, ORIGIN_LAT, ORIGIN_LON)
    print(f"\n[GameDayZones] Bound {len(gateways)} Regional Gateways:")
    for gw in gateways:
        print(f"  • {gw.name} -> Node {gw.nearest_node} (snap dist: {gw.dist_to_node_m:.1f} m)")

    print(f"\n[GameDayZones] Bound {len(sinks)} Calibrated Parking Sinks:")
    for sink in sinks:
        print(f"  • {sink.name} ({sink.capacity_stalls:,} stalls) -> Node {sink.nearest_node} (snap dist: {sink.dist_to_node_m:.1f} m)")

    # 5. Run Quality Control, Dijkstra pathfinding, and dataset export
    qc = NetworkQC(graph, gateways, sinks)
    stats = qc.compute_summary_statistics()

    print("\n" + "=" * 50)
    print("📊 TOPOLOGICAL & PHYSICAL ATTRIBUTION SUMMARY")
    print("=" * 50)
    print(f"  Total Network Nodes:               {stats['node_count']:,}")
    print(f"  Total Directed Links:              {stats['edge_count']:,}")
    print(f"  Centerline Roadway Length:         {stats['centerline_km']:,.2f} km")
    print(f"  Total Directional Lane Length:     {stats['lane_km']:,.2f} km")
    print(f"  Total Physical Storage Capacity:   {stats['network_storage_capacity_veh']:,} vehicles")
    print(f"  Reversible (Contraflow) Links:     {stats['reversible_edge_count']:,}")
    print("  Road Functional Breakdown:")
    for hwy, count in sorted(stats['highway_type_breakdown'].items(), key=lambda x: x[1], reverse=True):
        print(f"    - {hwy:<16}: {count:>4} links")

    # 6. Export Dataset and Visualizer
    data_out = "data/norman_network_processed.json"
    html_out = "visualizer/network_viewer.html"

    payload = qc.export_dataset(data_out)
    qc.generate_html_viewer(payload, html_out)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ PIPELINE COMPLETE in {elapsed:.2f} seconds!")
    print(f"📦 Graph Data: {os.path.abspath(data_out)}")
    print(f"🌐 Interactive Visualizer: {os.path.abspath(html_out)}")
    print("=" * 70)

    return graph, stats, payload


if __name__ == "__main__":
    force = "--refresh" in sys.argv
    run_pipeline(force_refresh=force)
