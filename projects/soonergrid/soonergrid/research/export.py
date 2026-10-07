"""Build the compact, offline dashboard bundle and immutable input inventory."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def export():
    data = ROOT / 'data'
    network = json.loads((data / 'norman_network_processed.json').read_text())
    legacy = json.loads((data / 'norman_resilience_benchmark_results.json').read_text())
    research_path = data / 'research_benchmark.json'
    research = json.loads(research_path.read_text()) if research_path.exists() else None
    inventory = []
    for path in sorted(data.glob('*.json')):
        if path.name.startswith('workspace_'):
            continue
        kind = (
            'uncalibrated simulation' if path.name == 'research_benchmark.json'
            else 'prescribed analytical illustration' if 'resilience' in path.name
            else 'legacy simulation — provenance unverified' if 'simulation_results' in path.name
            else 'assumed demand model' if 'demand' in path.name
            else 'OSM-derived geometry / heuristic attributes'
        )
        inventory.append({
            'name': path.name,
            'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'kind': kind,
        })

    compact_research = None
    if research:
        compact_research = {k: v for k, v in research.items() if k != 'runs'}
        edge_speeds = {e['edge_id']: e['free_speed_mph'] for e in network['edges']}
        compact_runs = []
        for r in research.get('runs', []):
            c_run = {k: v for k, v in r.items() if k != 'playback_frames'}
            c_frames = []
            for f in r.get('playback_frames', []):
                c_frame = {k: v for k, v in f.items() if k != 'corridor_states'}
                c_states = {}
                for eid, st in f.get('corridor_states', {}).items():
                    free_spd = edge_speeds.get(eid, 35.0)
                    if (st.get('queue_m', 0) > 2.0 or
                            st.get('speed_mph', 35.0) < free_spd * 0.85 or
                            st.get('density_pct', 0) > 18.0 or
                            'lindsey' in eid.lower()):
                        c_states[eid] = st
                c_frame['corridor_states'] = c_states
                c_frames.append(c_frame)
            c_run['playback_frames'] = c_frames
            compact_runs.append(c_run)
        compact_research['runs'] = compact_runs

    bundle = {
        'built_utc': datetime.now(timezone.utc).isoformat(),
        'inventory': inventory,
        'network': {
            'statistics': network['statistics'],
            'gateways': network['gateways'],
            'sinks': network['sinks'],
            'intersections': [
                {'id': 'INT_LINDSEY_I35', 'name': 'Lindsey St & I-35 Ramp', 'lat': 35.205864, 'lon': -97.476662, 'node_id': 8205332751},
                {'id': 'INT_LINDSEY_24TH', 'name': 'Lindsey St & 24th Ave SW', 'lat': 35.205821, 'lon': -97.467796, 'node_id': 7229412580},
                {'id': 'INT_LINDSEY_MCGEE', 'name': 'Lindsey St & McGee Dr', 'lat': 35.205881, 'lon': -97.457904, 'node_id': 1055332506},
                {'id': 'INT_LINDSEY_BERRY', 'name': 'Lindsey St & Berry Rd', 'lat': 35.206017, 'lon': -97.450139, 'node_id': 8012629250},
                {'id': 'INT_LINDSEY_PICKARD', 'name': 'Lindsey St & Pickard Ave', 'lat': 35.205739, 'lon': -97.444273, 'node_id': 4912651859},
                {'id': 'INT_LINDSEY_JENKINS', 'name': 'Lindsey St & Jenkins Ave', 'lat': 35.207491, 'lon': -97.438990, 'node_id': 1556137735},
                {'id': 'INT_LINDSEY_CLASSEN', 'name': 'Lindsey St & Classen Blvd', 'lat': 35.206006, 'lon': -97.429748, 'node_id': 145171599},
                {'id': 'INT_BOYD_JENKINS', 'name': 'Boyd St & Jenkins Ave (Campus Corner)', 'lat': 35.212129, 'lon': -97.438884, 'node_id': 3828661038},
                {'id': 'INT_SH9_JENKINS', 'name': 'SH-9 & Jenkins Ave (Lloyd Noble)', 'lat': 35.185141, 'lon': -97.439647, 'node_id': 1665582710},
                {'id': 'INT_MAIN_I35', 'name': 'Main St & I-35', 'lat': 35.221026, 'lon': -97.476655, 'node_id': 4003554949}
            ],
            'edges': [
                {
                    'id': e['edge_id'],
                    'name': e['name'],
                    'type': e['highway_type'],
                    'points': e['geometry'],
                    'speed': e['free_speed_mph'],
                }
                for e in network['edges']
            ],
        },
        'research': compact_research,
        'illustration': {
            'summaries': legacy['shock_benchmarks'],
            'sensitivity': legacy['sensitivity_grid'],
        },
    }
    out = ROOT / 'visualizer/workspace-data.js'
    out.write_text('window.WORKSPACE = ' + json.dumps(bundle, separators=(',', ':'), allow_nan=False) + ';\n')
    (data / 'workspace_manifest.json').write_text(json.dumps({'built_utc': bundle['built_utc'], 'files': inventory}, indent=2))
    
    # Older entry points retain their content and receive a persistent evidence label.
    for path in (ROOT / 'visualizer').glob('*.html'):
        if path.name == 'index.html':
            continue
        text = path.read_text()
        if 'legacy-evidence-banner' in text:
            continue
        label = 'Legacy viewer · unverified historical results'
        if path.name in ('research_dashboard.html', 'resilience_viewer.html'):
            label = 'Analytical illustration · prescribed curves, not measured policy performance'
        banner = (
            f'<div id="legacy-evidence-banner" style="position:fixed;bottom:0;left:0;right:0;z-index:999999;'
            f'background:#f5e8cb;color:#352f24;padding:10px 20px;font:13px system-ui;text-align:center">'
            f'{label} · <a href="index.html" style="color:#174c48">Open research workspace →</a></div>'
        )
        text = text.replace('</body>', banner + '\n</body>')
        path.write_text(text)
    return out


if __name__ == '__main__':
    print(export())
