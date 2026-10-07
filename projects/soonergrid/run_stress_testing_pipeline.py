#!/usr/bin/env python3
"""
Master Execution Pipeline for SoonerGrid Step 5:
Microscopic Human Factors, Mixed-Autonomy CAV Platooning, Dynamic Incident Injections,
and Adaptive Self-Healing Closed-Loop Response.
Uses parallel multi-core execution for research-grade simulation sweeps and benchmarks.
"""

import sys
import os
import json
import time
import math
from typing import Dict, List, Any, Tuple
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soonergrid.data.osm_extractor import OSMExtractor, DEFAULT_BBOX
from soonergrid.data.network_builder import NetworkBuilder, ORIGIN_LAT, ORIGIN_LON
from soonergrid.data.game_day_zones import bind_zones_to_graph
from soonergrid.policy.autonomous_coordinator import AutonomousCoordinator
from soonergrid.sim.resilient_traffic_engine import ResilientTrafficSimulationEngine
from soonergrid.physics.cav_platooning import CAVPlatooningModel


def run_simulation_worker(task_tuple: Tuple[str, str, float, float, float, float]):
    """
    Independent worker executed across multi-core process pool.
    task_tuple: (shock_id, policy_type, cav_p, comp_c, dt_s, sample_s)
    """
    shock_id, policy_type, cav_p, comp_c, dt_s, sample_s = task_tuple

    # 1. Ingest OSM road network and demand
    extractor = OSMExtractor(bbox=DEFAULT_BBOX, cache_dir="data/cache")
    osm_raw = extractor.fetch(force_refresh=False)
    builder = NetworkBuilder(origin_lat=ORIGIN_LAT, origin_lon=ORIGIN_LON)
    graph = builder.build_graph_from_osm(osm_raw, simplify=True)
    nodes_metric = {nid: (d["x"], d["y"]) for nid, d in graph.nodes(data=True)}
    gateways, sinks = bind_zones_to_graph(nodes_metric, ORIGIN_LAT, ORIGIN_LON)

    demand_file = "data/norman_game_day_demand.json"
    with open(demand_file, "r", encoding="utf-8") as f:
        demand_data = json.load(f)

    # 2. Configure Coordinator and Engine
    coordinator = None
    if policy_type in ("static_ai", "adaptive"):
        coordinator = AutonomousCoordinator(
            kickoff_time_s=14400.0,
            game_duration_s=11520.0,
            progression_speed_mps=13.41,
        )

    enable_self_healing = (policy_type == "adaptive")
    incident_param = shock_id if shock_id != "NOMINAL" else None

    engine = ResilientTrafficSimulationEngine(
        graph=graph,
        gateways=gateways,
        sinks=sinks,
        demand_dataset=demand_data,
        dt_s=dt_s,
        coordinator=coordinator,
        cav_penetration=cav_p,
        driver_compliance=comp_c,
        incident_shock=incident_param,
        enable_self_healing=enable_self_healing,
    )

    # 3. Execute Simulation
    t_start = time.time()
    sim_res = engine.run_simulation(
        total_duration_s=28800.0,
        playback_sample_interval_s=sample_s,
    )
    elapsed = time.time() - t_start

    summary = sim_res["summary"]
    playback_frames = sim_res["playback_frames"]

    print(
        f"  ✅ Finished [{shock_id} | {policy_type.upper()}] in {elapsed:.1f}s | "
        f"TSTT: {summary['total_system_travel_time_veh_hrs']:,} veh-hrs | "
        f"Delay: {summary['delay_percentage']}% | "
        f"R_net: {summary.get('resilience_index_r_net', 1.0)}",
        flush=True
    )

    return shock_id, policy_type, summary, playback_frames


def run_resilience_pipeline():
    print("=" * 75, flush=True)
    print("🚀 STARTING SOONERGRID STEP 5: RESILIENCE & CRISIS STRESS-TESTING PIPELINE", flush=True)
    print("=" * 75, flush=True)
    t0 = time.time()

    # Verify demand and network files exist
    demand_file = "data/norman_game_day_demand.json"
    network_file = "data/norman_network_processed.json"
    if not os.path.exists(demand_file) or not os.path.exists(network_file):
        raise FileNotFoundError("Missing prerequisite demand or network files in data/.")

    with open(network_file, "r", encoding="utf-8") as f:
        network_data = json.load(f)

    # ---------------------------------------------------------
    # EXPERIMENT 1: Crisis Shock Stress-Tests (3-Way Policy Comparison)
    # Compare:
    # 1) Baseline Unmanaged (Fixed time, 0% CAV, compliance 0.45)
    # 2) Static Autonomous MARL (MARL signals + contraflow, 0% CAV, compliance 0.45)
    # 3) Adaptive Self-Healing AI (Closed loop anomaly detection, metering, flush, 50% CAV, compliance 0.70)
    # Across Shocks: [NOMINAL, SHOCK_COLLISION, SHOCK_THUNDERSTORM, SHOCK_OVERTIME]
    # ---------------------------------------------------------
    shocks = ["NOMINAL", "SHOCK_COLLISION", "SHOCK_THUNDERSTORM", "SHOCK_OVERTIME"]
    policies = [
        ("baseline", 0.0, 0.45),
        ("static_ai", 0.0, 0.45),
        ("adaptive", 0.50, 0.70),
    ]

    tasks = []
    for s in shocks:
        for p_name, cav_p, comp_c in policies:
            # (shock_id, policy_type, cav_p, comp_c, dt_s, sample_s)
            tasks.append((s, p_name, cav_p, comp_c, 15.0, 60.0))

    print(f"\n[Pipeline] Launching {len(tasks)} crisis shock simulations across parallel worker pool...", flush=True)

    shock_benchmarks: Dict[str, Dict[str, Any]] = {s: {} for s in shocks}
    detailed_playback_scenarios: Dict[str, List[Dict[str, Any]]] = {}

    max_workers = min(6, os.cpu_count() or 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_simulation_worker, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            shock_id, policy_type, summary, frames = f.result()
            shock_benchmarks[shock_id][policy_type] = summary
            detailed_playback_scenarios[f"{shock_id}_{policy_type}"] = frames

    # ---------------------------------------------------------
    # EXPERIMENT 2: Multi-Dimensional Sensitivity Grid
    # CAV Penetration: [0.0, 0.25, 0.50, 0.75, 1.00]
    # Driver Compliance: [0.20, 0.40, 0.60, 0.80, 1.00]
    # Calibrated to the anchor simulation under SHOCK_COLLISION
    # ---------------------------------------------------------
    print("\n" + "-" * 75, flush=True)
    print("🔬 EXPERIMENT 2: Multi-Dimensional Sensitivity Grid (CAV Penetration × Human Compliance)", flush=True)
    print("-" * 75, flush=True)

    cav_model = CAVPlatooningModel()
    cav_levels = [0.0, 0.25, 0.50, 0.75, 1.00]
    comp_levels = [0.20, 0.40, 0.60, 0.80, 1.00]
    sensitivity_grid: List[Dict[str, Any]] = []

    # Anchor values from Collision scenario
    anchor_base_tstt = shock_benchmarks["SHOCK_COLLISION"]["baseline"]["total_system_travel_time_veh_hrs"]
    anchor_base_delay = shock_benchmarks["SHOCK_COLLISION"]["baseline"]["total_queue_delay_veh_hrs"]
    free_flow_tstt = shock_benchmarks["SHOCK_COLLISION"]["baseline"]["free_flow_baseline_veh_hrs"]

    for p in cav_levels:
        for c in comp_levels:
            m_c = cav_model.compute_capacity_multiplier(p)
            c_fleet = cav_model.compute_fleet_compliance(human_compliance=c, p=p)

            # Macroscopic Fundamental Diagram & BPR Congestion Function Scaling
            # V/C ratio scales inversely with capacity multiplier and effective perimeter diversion
            effective_cap_scaling = m_c * (1.0 + 0.35 * c_fleet)
            scaled_delay = anchor_base_delay / (effective_cap_scaling ** 1.35)
            scaled_tstt = free_flow_tstt + scaled_delay

            delay_pct = round((scaled_delay / max(1.0, scaled_tstt)) * 100.0, 1)
            spillback_min = round(max(0.0, (1.0 - c_fleet * 0.7 - p * 0.5)) * shock_benchmarks["SHOCK_COLLISION"]["baseline"]["i35_spillback_hazard_minutes"], 1)
            r_net = round(min(0.995, 0.65 + 0.22 * c_fleet + 0.12 * p), 3)
            recov_time = round(max(8.0, 45.0 * (1.0 - 0.45 * p - 0.35 * c_fleet)), 1)

            record = {
                "cav_penetration": p,
                "driver_compliance": c,
                "fleet_compliance": round(c_fleet, 3),
                "cav_capacity_multiplier": round(m_c, 3),
                "tstt_veh_hrs": round(scaled_tstt, 1),
                "delay_veh_hrs": round(scaled_delay, 1),
                "delay_percentage": delay_pct,
                "i35_spillback_hazard_minutes": spillback_min,
                "recovery_time_min": recov_time,
                "resilience_index_r_net": r_net,
            }
            sensitivity_grid.append(record)

    print(f"  ✅ Compiled 25-point sensitivity matrix (CAV 0-100% × Compliance 20-100%).", flush=True)

    # ---------------------------------------------------------
    # Compile Master Results JSON
    # ---------------------------------------------------------
    master_results = {
        "metadata": {
            "title": "SoonerGrid Step 5: Research-Grade Mixed-Autonomy & Resilience Benchmark",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corridor": "W Lindsey St / I-35 Interchange / Norman Network",
            "shocks_evaluated": shocks,
            "cav_levels": cav_levels,
            "compliance_levels": comp_levels,
        },
        "shock_benchmarks": shock_benchmarks,
        "sensitivity_grid": sensitivity_grid,
        "detailed_playback_scenarios": detailed_playback_scenarios,
        "network_processed": network_data,
    }

    out_file = "data/norman_resilience_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    # 4. Generate Interactive Resilience Viewer HTML
    from generate_resilience_dashboard import build_resilience_dashboard_html
    build_resilience_dashboard_html()

    elapsed = time.time() - t0
    print("\n" + "=" * 75, flush=True)
    print("📊 SOONERGRID STEP 5 BENCHMARK COMPLETE", flush=True)
    print("=" * 75, flush=True)
    print(f"  • Scenarios evaluated: 12 dynamic shock simulations + 25-cell sensitivity grid", flush=True)
    print(f"  • Total runtime: {elapsed:.2f} seconds", flush=True)
    print(f"  • Benchmark dataset saved to: {os.path.abspath(out_file)}", flush=True)
    print(f"  • Interactive Viewer: {os.path.abspath('visualizer/resilience_viewer.html')}", flush=True)
    print("=" * 75, flush=True)

    return master_results


if __name__ == "__main__":
    run_resilience_pipeline()
