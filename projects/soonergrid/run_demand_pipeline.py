#!/usr/bin/env python3
"""
Master Execution Pipeline for SoonerGrid Step 2:
Dynamic Game Day Origin-Destination Matrix & Multi-Modal Surge Demand Engine.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soonergrid.data.game_day_zones import NORMAN_GATEWAYS, NORMAN_PARKING_SINKS, bind_zones_to_graph
from soonergrid.data.network_builder import NetworkBuilder, ORIGIN_LAT, ORIGIN_LON
from soonergrid.data.osm_extractor import OSMExtractor, DEFAULT_BBOX
from soonergrid.demand.od_matrix_generator import GameDayDemandEngine


def generate_demand_html(demand_data: dict, html_filepath: str):
    """Generates an interactive game-day surge timeline dashboard."""
    os.makedirs(os.path.dirname(html_filepath), exist_ok=True)
    json_blob = json.dumps(demand_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SoonerGrid — Game Day Demand & Parking Surge Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #090b10;
      --card-bg: rgba(16, 20, 30, 0.85);
      --border: rgba(255, 255, 255, 0.1);
      --crimson: #a8192d;
      --crimson-glow: #e0243d;
      --cyan: #00f0ff;
      --amber: #ffb703;
      --emerald: #06d6a0;
      --text: #f0f4fc;
      --text-muted: #8b9bb4;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Outfit', sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      min-height: 100vh;
      overflow-x: hidden;
    }}
    .container {{
      max-width: 1300px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      padding: 16px 24px;
      border-radius: 14px;
    }}
    .logo-box {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .logo-icon {{ font-size: 28px; }}
    .logo-text {{ font-size: 18px; font-weight: 900; letter-spacing: 1.5px; text-transform: uppercase; }}
    .logo-text span {{ color: var(--crimson-glow); }}
    .badge {{
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      background: rgba(0, 240, 255, 0.12);
      border: 1px solid rgba(0, 240, 255, 0.4);
      color: var(--cyan);
      padding: 4px 10px;
      border-radius: 6px;
    }}

    /* Summary Bar */
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }}
    .metric-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .metric-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }}
    .metric-val {{ font-family: 'Space Mono', monospace; font-size: 24px; font-weight: 700; color: #fff; }}

    /* Timeline Scrubber Card */
    .timeline-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .time-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .phase-badge {{
      font-family: 'Space Mono', monospace;
      font-size: 13px;
      color: var(--amber);
      background: rgba(255, 183, 3, 0.12);
      border: 1px solid rgba(255, 183, 3, 0.3);
      padding: 6px 14px;
      border-radius: 8px;
    }}
    input[type=range] {{
      width: 100%;
      height: 8px;
      accent-color: var(--crimson-glow);
      cursor: pointer;
    }}

    /* Main Grid */
    .main-grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
    }}
    .panel {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .panel-title {{
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }}

    /* Parking Gauges */
    .gauge-list {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 6px;
    }}
    .gauge-item {{
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .gauge-top {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
    }}
    .gauge-bar-bg {{
      width: 100%;
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 4px;
      overflow: hidden;
    }}
    .gauge-fill {{
      height: 100%;
      width: 0%;
      border-radius: 4px;
      transition: width 0.15s ease, background 0.15s ease;
    }}

    /* Gateway Inflow Bars */
    .gw-list {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .gw-item {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-box">
        <span class="logo-icon">🏈</span>
        <div class="logo-text">SOONER<span>GRID</span> &bull; DEMAND ENGINE</div>
      </div>
      <div class="badge">GAME DAY SURGE TIMELINE</div>
    </header>

    <div class="summary-grid">
      <div class="metric-card">
        <div class="metric-label">Total Game-Day Fleet</div>
        <div class="metric-val" id="val-fleet">--</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">I-35 North Share</div>
        <div class="metric-val" id="val-i35" style="color:var(--cyan)">--</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Shuttle Passengers</div>
        <div class="metric-val" id="val-shuttle" style="color:var(--emerald)">--</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Private Car Trips Saved</div>
        <div class="metric-val" id="val-saved" style="color:var(--amber)">--</div>
      </div>
    </div>

    <div class="timeline-card">
      <div class="time-header">
        <div>
          <span style="font-size:13px; color:var(--text-muted); text-transform:uppercase;">Simulation Time:</span>
          <span id="label-time" style="font-family:'Space Mono'; font-size:22px; font-weight:700; color:#fff; margin-left:8px;">T - 4.0h</span>
        </div>
        <div class="phase-badge" id="label-phase">Phase I: Early Tailgate Arrival</div>
      </div>
      <input type="range" id="time-slider" min="0" max="31" value="0" step="1">
      <div style="display:flex; justify-content:space-between; font-size:11px; font-family:'Space Mono'; color:var(--text-muted);">
        <span>T - 4.0h (08:00 AM)</span>
        <span>T - 2.0h (Ingress Peak)</span>
        <span>T = 0 (12:00 PM Kickoff)</span>
        <span>T + 3.2h (Final Whistle)</span>
        <span>T + 4.0h (Post-Game Egress)</span>
      </div>
    </div>

    <div class="main-grid">
      <!-- Left Panel: Flow Rates & Gateways -->
      <div class="panel">
        <div class="panel-title">
          <span>Regional Gateway Influx (Vehicles / Hour)</span>
          <span id="label-flow" style="color:var(--cyan); font-family:'Space Mono';">-- veh/hr</span>
        </div>
        <div class="gw-list" id="gw-container">
          <!-- Populated dynamically -->
        </div>

        <div class="panel-title" style="margin-top:10px;">
          <span>Pedestrian Conflict & Crosswalk Capacity Derating</span>
        </div>
        <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:10px; padding:12px; font-size:12px; display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div style="font-weight:600;">W Lindsey St & S Jenkins Ave (Stadium SE Corner)</div>
            <div style="color:var(--text-muted); font-size:11px;">Crosswalk foot traffic choking vehicular turning movements</div>
          </div>
          <div id="label-ped" style="font-family:'Space Mono'; font-weight:700; font-size:16px; color:var(--emerald);">100% Flow</div>
        </div>
      </div>

      <!-- Right Panel: Parking Lot Occupancy Gauges -->
      <div class="panel">
        <div class="panel-title">
          <span>Parking Sink Occupancy (MNL Choice Model)</span>
          <span style="color:var(--emerald)">13 Sinks</span>
        </div>
        <div class="gauge-list" id="sink-container">
          <!-- Populated dynamically -->
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {json_blob};
    const timeline = DATA.timeline;

    // Fill Summary Cards
    document.getElementById('val-fleet').textContent = DATA.summary.total_vehicles_generated.toLocaleString() + ' veh';
    document.getElementById('val-i35').textContent = DATA.summary.gateway_breakdown['GW_I35_NORTH'].pct_share + '%';
    document.getElementById('val-shuttle').textContent = DATA.summary.shuttle_impact.total_passengers_carried.toLocaleString() + ' pax';
    document.getElementById('val-saved').textContent = DATA.summary.shuttle_impact.private_car_trips_avoided.toLocaleString() + ' cars';

    const slider = document.getElementById('time-slider');
    slider.max = timeline.length - 1;

    const gwContainer = document.getElementById('gw-container');
    const sinkContainer = document.getElementById('sink-container');

    function updateUI(stepIdx) {{
      const slice = timeline[stepIdx];
      const relHours = (slice.time_hr - 4.0).toFixed(1);
      const sign = relHours >= 0 ? '+' : '';
      document.getElementById('label-time').textContent = `T ${{sign}}${{relHours}}h (${{slice.time_hr.toFixed(2)}} hrs)`;
      document.getElementById('label-phase').textContent = slice.phase;
      document.getElementById('label-flow').textContent = slice.ingress_rate_vph.toLocaleString() + ' veh/hr';

      // Pedestrian derating
      const pedCap = Math.round(slice.pedestrian_capacity_derating * 100);
      const pedEl = document.getElementById('label-ped');
      pedEl.textContent = pedCap + '% Flow';
      pedEl.style.color = pedCap < 50 ? 'var(--crimson-glow)' : (pedCap < 80 ? 'var(--amber)' : 'var(--emerald)');

      // Gateways
      gwContainer.innerHTML = '';
      for (let [gwId, count] of Object.entries(slice.gateway_inflows || {{}})) {{
        const gwMeta = DATA.summary.gateway_breakdown[gwId] || {{ name: gwId }};
        const rateVph = count * 4; // 15-min slice -> vph
        const pct = Math.min(100, Math.round((rateVph / 7000) * 100));

        const item = document.createElement('div');
        item.className = 'gw-item';
        item.innerHTML = `
          <div style="display:flex; justify-content:space-between; font-size:12px;">
            <span>${{gwMeta.name.split('(')[0]}}</span>
            <span style="font-family:'Space Mono'; color:var(--cyan)">${{rateVph.toLocaleString()}} vph</span>
          </div>
          <div class="gauge-bar-bg">
            <div class="gauge-fill" style="width:${{pct}}%; background:var(--cyan);"></div>
          </div>
        `;
        gwContainer.appendChild(item);
      }}

      // Sinks
      sinkContainer.innerHTML = '';
      for (let [sinkId, pct] of Object.entries(slice.occupancy_percentages || {{}})) {{
        const meta = DATA.summary.parking_sink_stats[sinkId] || {{ name: sinkId, capacity: 1000 }};
        const occupied = Math.round((pct / 100) * meta.capacity);
        let color = 'var(--emerald)';
        if (pct >= 85) color = 'var(--crimson-glow)';
        else if (pct >= 60) color = 'var(--amber)';

        const item = document.createElement('div');
        item.className = 'gauge-item';
        item.innerHTML = `
          <div class="gauge-top">
            <span style="font-weight:600;">${{meta.name.split('(')[0]}}</span>
            <span style="font-family:'Space Mono'; color:${{color}}">${{occupied.toLocaleString()}} / ${{meta.capacity.toLocaleString()}} (${{pct}}%)</span>
          </div>
          <div class="gauge-bar-bg">
            <div class="gauge-fill" style="width:${{pct}}%; background:${{color}};"></div>
          </div>
        `;
        sinkContainer.appendChild(item);
      }}
    }}

    slider.addEventListener('input', (e) => {{
      updateUI(parseInt(e.target.value));
    }});

    // Initialize at peak ingress slice (slice 11 = T - 1.25h)
    slider.value = 11;
    updateUI(11);
  </script>
</body>
</html>
"""
    with open(html_filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[DemandPipeline] Interactive dashboard exported to: {html_filepath}")


def run_pipeline(demand_scale: float = 1.0):
    print("=" * 70)
    print("🏈 SOONERGRID STEP 2: GAME DAY DEMAND & OD SURGE PIPELINE")
    print("=" * 70)
    t0 = time.time()

    # 1. Ingest cached network
    extractor = OSMExtractor(bbox=DEFAULT_BBOX, cache_dir="data/cache")
    osm_raw = extractor.fetch(force_refresh=False)

    builder = NetworkBuilder(origin_lat=ORIGIN_LAT, origin_lon=ORIGIN_LON)
    graph = builder.build_graph_from_osm(osm_raw, simplify=True)

    nodes_metric = {nid: (d["x"], d["y"]) for nid, d in graph.nodes(data=True)}
    gateways, sinks = bind_zones_to_graph(nodes_metric, ORIGIN_LAT, ORIGIN_LON)

    # 2. Instantiate and run demand orchestrator
    demand_engine = GameDayDemandEngine(
        graph=graph,
        gateways=gateways,
        sinks=sinks,
        total_vehicles=34500,
        demand_scale=demand_scale,
        greedy_gps_ratio=0.75,
    )

    data_out = "data/norman_game_day_demand.json"
    html_out = "visualizer/demand_viewer.html"

    demand_data = demand_engine.export_demand_dataset(data_out, time_step_min=15.0)
    generate_demand_html(demand_data, html_out)

    elapsed = time.time() - t0
    summary = demand_data["summary"]

    print("\n" + "=" * 50)
    print("📊 GAME DAY DEMAND SUMMARY")
    print("=" * 50)
    print(f"  Total Ingress Vehicles Spawned:    {summary['total_vehicles_generated']:,}")
    print(f"  Total Post-Game Egress Discharged: {summary['total_vehicles_discharged']:,}")
    print(f"  Shuttle Bus Passengers Carried:    {summary['shuttle_impact']['total_passengers_carried']:,} passengers")
    print(f"  Private Vehicle Trips Avoided:     {summary['shuttle_impact']['private_car_trips_avoided']:,} vehicles")
    print("\n  Ingress Gateway Distribution:")
    for gw_id, gw_info in summary["gateway_breakdown"].items():
        print(f"    - {gw_info['name']:<42}: {gw_info['total_trips']:>6,} veh ({gw_info['pct_share']}%)")

    print("\n  Peak Parking Sink Saturation:")
    for sink_id, s_info in sorted(summary["parking_sink_stats"].items(), key=lambda x: x[1]['peak_occupancy_pct'], reverse=True)[:6]:
        print(f"    - {s_info['name'][:38]:<40}: {int(s_info['peak_occupied']):>5,} / {s_info['capacity']:>5,} ({s_info['peak_occupancy_pct']}%)")

    print("\n" + "=" * 70)
    print(f"✅ STEP 2 PIPELINE COMPLETE in {elapsed:.2f} seconds!")
    print(f"📦 Demand Dataset: {os.path.abspath(data_out)}")
    print(f"🌐 Demand Dashboard: {os.path.abspath(html_out)}")
    print("=" * 70)

    return demand_data


if __name__ == "__main__":
    scale = 1.0
    for arg in sys.argv[1:]:
        if arg.startswith("--scale="):
            scale = float(arg.split("=")[1])
    run_pipeline(demand_scale=scale)
