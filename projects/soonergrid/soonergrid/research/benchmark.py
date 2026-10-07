"""Matched-condition experiments using the processed network, without re-fetching OSM."""
from pathlib import Path
import dataclasses
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
import networkx as nx
from soonergrid.physics.link_attributes import EdgeAttributes
from soonergrid.data.game_day_zones import Gateway, ParkingSink
from soonergrid.sim.resilient_traffic_engine import ResilientTrafficSimulationEngine
from soonergrid.policy.autonomous_coordinator import AutonomousCoordinator

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest():
    h = hashlib.sha256()
    for p in sorted((ROOT / 'soonergrid').rglob('*.py')):
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def load_inputs():
    network = json.loads((ROOT / 'data/norman_network_processed.json').read_text())
    demand = json.loads((ROOT / 'data/norman_game_day_demand.json').read_text())
    graph = nx.MultiDiGraph()
    for node in network['nodes']:
        graph.add_node(node['id'], **node)
    fields = {f.name for f in dataclasses.fields(EdgeAttributes)}
    for edge in network['edges']:
        graph.add_edge(edge['u'], edge['v'], key=edge['edge_id'],
                       attr=EdgeAttributes(**{k: v for k,v in edge.items() if k in fields}))
    def records(cls, rows):
        fields = {f.name for f in dataclasses.fields(cls)}
        aliases = {'weight':'hourly_inflow_weight', 'capacity':'capacity_stalls', 'dist_m':'dist_to_node_m'}
        return [cls(**{aliases.get(k,k):v for k,v in row.items() if aliases.get(k,k) in fields}) for row in rows]
    return network, demand, graph, records(Gateway, network['gateways']), records(ParkingSink, network['sinks'])


def run_case(task):
    policy, shock, duration, dt, scale = task
    network, demand, graph, gateways, sinks = load_inputs()
    for row in demand['timeline']:
        row['gateway_inflows'] = {k:v * scale for k,v in row['gateway_inflows'].items()}
        
    started = time.monotonic()
    
    if policy == 'rl_marl':
        from soonergrid.gym.env import NormanTrafficEnv
        from stable_baselines3 import PPO
        import numpy as np
        
        env = NormanTrafficEnv(graph=graph, gateways=gateways, sinks=sinks, demand_dataset=demand,
                               dt_s=dt, episode_duration_s=duration, incident_shock=None if shock == 'NOMINAL' else shock,
                               enable_self_healing=True, auto_load_network=False)
        
        model_path = ROOT / "data" / "models" / "marl_ppo_final.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained RL model not found at {model_path}. Run train_marl.py first.")
        
        model = PPO.load(str(model_path))
        
        obs, _ = env.reset()
        playback_frames = []
        
        total_steps = round(duration / dt)
        sample_steps = max(1, int(60.0 / dt))
        
        # We need to manually build playback frames since env doesn't do it
        # Actually, env._engine builds playback frames logic is in run_simulation.
        # But we can extract it cleanly:
        for step in range(total_steps):
            # Predict actions
            actions = {}
            for agent in env.agents:
                # PPO expects (1, obs_dim) for single predict
                action, _ = model.predict(np.array(obs[agent]), deterministic=True)
                actions[agent] = action
                
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            # Record playback frame every 60s
            if step % sample_steps == 0:
                # Reconstruct frame data similar to resilient_traffic_engine.py
                t_sim_s = step * dt
                eng = env._engine
                
                total_lindsey_queue_m = sum(eng._link_states[eid].queue_length_m for eid in eng._lindsey_edge_ids if eid in eng._link_states)
                r_net = eng._self_healing.compute_resilience_index(eng._exited, max(1.0, eng._generated)) if eng._self_healing else 0.85
                
                # Fetch metrics from tracker
                if eng._metrics_tracker and eng._metrics_tracker.time_series:
                    metrics = eng._metrics_tracker.time_series[-1]
                    active_veh = metrics.active_vehicles_on_network
                    lindsey_speed = metrics.lindsey_corridor_speed_mph
                    ramp_queue = metrics.i35_ramp_queue_m
                    is_spill = metrics.is_i35_spillback_hazard
                else:
                    active_veh = lindsey_speed = ramp_queue = 0.0
                    is_spill = False
                    
                frame_data = {
                    "time_hr": round(t_sim_s / 3600.0, 2),
                    "phase": infos[env.possible_agents[0]].get("phase", "Game Day"),
                    "active_veh": active_veh,
                    "lindsey_speed_mph": lindsey_speed,
                    "i35_ramp_queue_m": ramp_queue,
                    "is_spillback": is_spill,
                    "incident_active": eng._self_healing.is_incident_active if eng._self_healing else (shock != 'NOMINAL'),
                    "resilience_index_r_net": r_net,
                    "lindsey_total_queue_m": round(total_lindsey_queue_m, 1),
                    "fleet_compliance_pct": round(eng._fleet_compliance * 100.0, 1),
                    "cav_penetration_pct": round(eng.cav_penetration * 100.0, 1),
                    "corridor_states": {
                        eid: {
                            "speed_mph": round(eng._link_states[eid].speed_mph, 1),
                            "queue_m": round(eng._link_states[eid].queue_length_m, 1),
                            "density_pct": round(min(100.0, (eng._vehicles[eid] / max(1, eng._edge_meta[eid]["jam_storage_veh"])) * 100.0), 1),
                        }
                        for eid in eng._edge_meta # For simplicity, output all congested
                        if eid in eng._link_states and (
                            eng._vehicles[eid] > 0.05
                            or eng._link_states[eid].queue_length_m > 0.1
                            or eng._link_states[eid].speed_mph < eng._edge_meta[eid]["free_speed_mph"] * 0.98
                        )
                    }
                }
                playback_frames.append(frame_data)
                
            if any(terms.values()) or any(truncs.values()):
                break
                
        summary = env.get_episode_summary()
        summary["total_system_time_including_boundary_veh_hrs"] = sum(total_sys)
        result = {
            "summary": summary,
            "playback_frames": playback_frames,
        }
        
    else:
        coordinator = AutonomousCoordinator() if policy == 'coordinated' else None
        engine = ResilientTrafficSimulationEngine(graph, gateways, sinks, demand, dt_s=dt,
            coordinator=coordinator, cav_penetration=0, driver_compliance=.6,
            incident_shock=None if shock == 'NOMINAL' else shock,
            enable_self_healing=policy == 'coordinated')
        result = engine.run_simulation(duration, 60)
        
    return {'id': f'{shock}_{policy}_{scale:g}', 'policy': policy, 'shock': shock,
            'config': {'duration_s':duration, 'dt_s':dt, 'demand_scale':scale,
                       'cav_penetration':0, 'driver_compliance':.6},
            'elapsed_s':time.monotonic()-started, **result}


def benchmark(duration=28800, dt=5, workers=2, scales=(1.0,), shocks=('NOMINAL','SHOCK_COLLISION')):
    from concurrent.futures import ProcessPoolExecutor
    inputs = {p.name: sha(p) for p in [ROOT/'data/norman_network_processed.json', ROOT/'data/norman_game_day_demand.json']}
    digest = source_digest()
    print("Starting benchmark run...", flush=True)
    tasks = [(p,s,duration,dt,scale) for scale in scales for s in shocks for p in ['baseline','coordinated','rl_marl']]
    runs=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(run_case, tasks):
            runs.append(result)
            print(f"Completed {result['id']} in {result['elapsed_s']:.1f}s", flush=True)
    if source_digest() != digest:
        raise RuntimeError('Simulation source changed during experiment; rerun before publishing')
    payload={'schema_version':1, 'created_utc':datetime.now(timezone.utc).isoformat(),
             'evidence_kind':'uncalibrated_network_simulation', 'source_sha256':digest,
             'input_sha256':inputs, 'environment':{'python':platform.python_version(),'networkx':nx.__version__},
             'design':'Matched demand, CAV fraction and compliance. Deterministic runs; no sampling confidence intervals.',
             'limitations':['No observed traffic calibration or held-out field validation.',
                'Receiving-weighted turning replaces destination-specific route assignment.',
                'Parking capacity and explicit parked-vehicle egress are not modeled.',
                'Whole-link flow approximation; timestep convergence is not established.',
                'Street-name signal mapping is approximate, not an intersection movement model.',
                'Speed-based delay and fuel are proxies; boundary waiting is reported separately.'],
             'runs':runs}
    out=ROOT/'data/research_benchmark.json'
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(payload, allow_nan=False));temp.replace(out)
    return out

if __name__ == '__main__':
    benchmark()
