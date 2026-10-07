#!/usr/bin/env python3
"""
Master Execution Pipeline for SoonerGrid Step 3:
Baseline Simulation Engine (Demonstrating Unmanaged Gridlock on Lindsey St & I-35).
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soonergrid.data.osm_extractor import OSMExtractor, DEFAULT_BBOX
from soonergrid.data.network_builder import NetworkBuilder, ORIGIN_LAT, ORIGIN_LON
from soonergrid.data.game_day_zones import bind_zones_to_graph
from soonergrid.sim.traffic_engine import TrafficSimulationEngine


def generate_simulation_html(sim_results: dict, network_data: dict, html_filepath: str):
    """Generates an interactive simulation playback dashboard."""
    os.makedirs(os.path.dirname(html_filepath), exist_ok=True)
    sim_blob = json.dumps(sim_results)
    network_blob = json.dumps(network_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SoonerGrid — Game Day Baseline Simulation (Unmanaged Gridlock)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #090b10;
      --panel-bg: rgba(16, 20, 30, 0.88);
      --border: rgba(255, 255, 255, 0.12);
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
      overflow: hidden;
      height: 100vh;
      display: flex;
    }}
    #canvas-container {{
      flex: 1;
      position: relative;
      height: 100vh;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
    }}
    canvas:active {{ cursor: grabbing; }}

    /* Top HUD */
    #top-bar {{
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 16px;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      padding: 10px 20px;
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .brand-icon {{ font-size: 24px; }}
    .brand-title {{ font-size: 16px; font-weight: 900; letter-spacing: 1.5px; text-transform: uppercase; }}
    .brand-title span {{ color: var(--crimson-glow); }}
    .badge {{
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      background: rgba(224, 36, 61, 0.15);
      border: 1px solid var(--crimson-glow);
      color: #fff;
      padding: 4px 10px;
      border-radius: 6px;
    }}

    /* Hazard Banner */
    #hazard-banner {{
      position: absolute;
      top: 16px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10;
      background: rgba(224, 36, 61, 0.9);
      border: 1px solid #fff;
      padding: 8px 18px;
      border-radius: 20px;
      font-family: 'Space Mono', monospace;
      font-size: 12px;
      font-weight: 700;
      color: #fff;
      display: none;
      box-shadow: 0 0 24px var(--crimson-glow);
      animation: pulse 1s infinite alternate;
    }}
    @keyframes pulse {{
      from {{ transform: translateX(-50%) scale(1); }}
      to {{ transform: translateX(-50%) scale(1.05); }}
    }}

    /* Sidebar Telemetry */
    #sidebar {{
      position: absolute;
      top: 16px;
      right: 16px;
      bottom: 16px;
      width: 360px;
      z-index: 10;
      background: var(--panel-bg);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border);
      border-radius: 16px;
      display: flex;
      flex-direction: column;
      padding: 18px;
      gap: 14px;
      box-shadow: 0 12px 48px rgba(0,0,0,0.6);
      overflow-y: auto;
    }}
    .section-title {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .stat-card {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.07);
      padding: 10px;
      border-radius: 8px;
    }}
    .stat-label {{ font-size: 11px; color: var(--text-muted); }}
    .stat-val {{ font-family: 'Space Mono', monospace; font-size: 16px; font-weight: 700; color: #fff; margin-top: 3px; }}

    /* Playback Deck */
    .playback-card {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .btn-row {{
      display: flex;
      gap: 8px;
    }}
    .play-btn {{
      flex: 1;
      background: var(--crimson);
      color: #fff;
      border: none;
      padding: 8px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .play-btn:hover {{ background: var(--crimson-glow); }}
    input[type=range] {{
      width: 100%;
      height: 6px;
      accent-color: var(--crimson-glow);
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <div id="canvas-container">
    <canvas id="simCanvas"></canvas>

    <div id="top-bar">
      <div class="brand">
        <span class="brand-icon">🏈</span>
        <div class="brand-title">SOONER<span>GRID</span></div>
      </div>
      <div class="badge">STEP 3: UNMANAGED BASELINE</div>
    </div>

    <div id="hazard-banner">
      ⚠️ I-35 MAINLINE SPILLBACK HAZARD DETECTED
    </div>
  </div>

  <div id="sidebar">
    <div class="playback-card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:12px; color:var(--text-muted);">TIME</span>
        <span id="label-time" style="font-family:'Space Mono'; font-size:16px; font-weight:700; color:var(--amber);">T - 4.0h</span>
      </div>
      <input type="range" id="time-slider" min="0" max="100" value="0">
      <div class="btn-row">
        <button class="play-btn" id="btn-play">▶ PLAY</button>
        <button class="play-btn" id="btn-speed" style="background:rgba(255,255,255,0.08); flex:0.6;">5x</button>
      </div>
      <div id="label-phase" style="font-size:11px; color:var(--text-muted); text-align:center;">Phase I: Early Tailgate Arrival</div>
    </div>

    <div class="section-title">
      <span>Simulation Benchmark Metrics</span>
    </div>
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">Total System Travel Time</div>
        <div class="stat-val" id="val-tstt" style="color:var(--cyan)">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Queue Delay</div>
        <div class="stat-val" id="val-delay" style="color:var(--crimson-glow)">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Lindsey Corridor Speed</div>
        <div class="stat-val" id="val-speed" style="color:var(--amber)">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">I-35 Ramp Queue</div>
        <div class="stat-val" id="val-queue" style="color:#fff">--</div>
      </div>
    </div>

    <div class="section-title">
      <span>Baseline Gridlock Diagnostics</span>
    </div>
    <div style="font-size:11px; display:flex; flex-direction:column; gap:8px; color:var(--text-muted);">
      <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); padding:10px; border-radius:8px;">
        <div style="color:#fff; font-weight:600; margin-bottom:4px;">1. Fixed Signal Gating</div>
        <div>90s uncoordinated signals at Berry & Jenkins create continuous shockwaves stopping arrival platoons.</div>
      </div>
      <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); padding:10px; border-radius:8px;">
        <div style="color:#fff; font-weight:600; margin-bottom:4px;">2. Pedestrian Crosswalk Freeze</div>
        <div>Foot crowds crossing Lindsey reduce right-turn vehicle discharge by 82% at peak.</div>
      </div>
      <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); padding:10px; border-radius:8px;">
        <div style="color:#fff; font-weight:600; margin-bottom:4px;">3. I-35 Freeway Spillback</div>
        <div>Ramp queue exceeds 420m, spilling 5 mph queues onto the 70 mph interstate.</div>
      </div>
    </div>

    <div class="section-title">
      <span>Velocity Legend</span>
    </div>
    <div style="font-size:11px; display:flex; flex-direction:column; gap:6px; color:var(--text-muted);">
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="width:16px; height:3px; background:#06d6a0; display:inline-block;"></span>
        <span>Free-Flow Speed (30 - 45 mph)</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="width:16px; height:3px; background:#ffb703; display:inline-block;"></span>
        <span>Moderate Congestion (12 - 25 mph)</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="width:16px; height:3px; background:#e0243d; display:inline-block;"></span>
        <span>Severe Gridlock Shockwave (0 - 10 mph)</span>
      </div>
    </div>
  </div>

  <script>
    const SIM = {sim_blob};
    const NET = {network_blob};

    const canvas = document.getElementById('simCanvas');
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {{
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      render();
    }});

    // Camera view
    let zoom = 0.085;
    let panX = width / 2;
    let panY = height / 2;
    let isDragging = false;
    let startX = 0, startY = 0;

    // Simulation playback state
    const frames = SIM.playback_frames;
    let currentFrameIdx = 0;
    let isPlaying = false;
    let playSpeed = 5;
    let animTimer = null;

    const slider = document.getElementById('time-slider');
    slider.max = frames.length - 1;

    // Coordinate conversion
    function worldToScreen(x, y) {{
      return {{ x: panX + x * zoom, y: panY - y * zoom }};
    }}
    function screenToWorld(sx, sy) {{
      return {{ x: (sx - panX) / zoom, y: (panY - sy) / zoom }};
    }}

    // Controls
    canvas.addEventListener('mousedown', (e) => {{
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    }});
    window.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('mousemove', (e) => {{
      if (isDragging) {{
        panX = e.clientX - startX;
        panY = e.clientY - startY;
        render();
      }}
    }});
    canvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.85;
      const mw = screenToWorld(e.clientX, e.clientY);
      zoom *= factor;
      panX = e.clientX - mw.x * zoom;
      panY = e.clientY + mw.y * zoom;
      render();
    }});

    // Playback loop
    function setFrame(idx) {{
      currentFrameIdx = Math.max(0, Math.min(frames.length - 1, idx));
      slider.value = currentFrameIdx;
      const frame = frames[currentFrameIdx];

      const relH = (frame.time_hr - 4.0).toFixed(1);
      const sign = relH >= 0 ? '+' : '';
      document.getElementById('label-time').textContent = `T ${{sign}}${{relH}}h (${{frame.time_hr.toFixed(2)}}h)`;
      document.getElementById('label-phase').textContent = frame.phase;
      document.getElementById('val-speed').textContent = frame.lindsey_speed_mph + ' mph';
      document.getElementById('val-queue').textContent = frame.i35_ramp_queue_m + ' m';

      const hazardEl = document.getElementById('hazard-banner');
      if (frame.is_spillback) {{
        hazardEl.style.display = 'block';
      }} else {{
        hazardEl.style.display = 'none';
      }}

      render();
    }}

    slider.addEventListener('input', (e) => {{
      setFrame(parseInt(e.target.value));
    }});

    const btnPlay = document.getElementById('btn-play');
    btnPlay.addEventListener('click', () => {{
      isPlaying = !isPlaying;
      btnPlay.textContent = isPlaying ? '⏸ PAUSE' : '▶ PLAY';
      if (isPlaying) runAnimation();
    }});

    const btnSpeed = document.getElementById('btn-speed');
    btnSpeed.addEventListener('click', () => {{
      if (playSpeed === 1) playSpeed = 5;
      else if (playSpeed === 5) playSpeed = 20;
      else playSpeed = 1;
      btnSpeed.textContent = playSpeed + 'x';
    }});

    function runAnimation() {{
      if (!isPlaying) return;
      if (currentFrameIdx >= frames.length - 1) {{
        currentFrameIdx = 0;
      }} else {{
        currentFrameIdx += 1;
      }}
      setFrame(currentFrameIdx);
      setTimeout(runAnimation, 200 / playSpeed);
    }}

    // Populate initial stats
    document.getElementById('val-tstt').textContent = SIM.summary.total_system_travel_time_veh_hrs.toLocaleString() + ' hrs';
    document.getElementById('val-delay').textContent = SIM.summary.total_queue_delay_veh_hrs.toLocaleString() + ' hrs';

    function render() {{
      ctx.fillStyle = '#090b10';
      ctx.fillRect(0, 0, width, height);

      const frame = frames[currentFrameIdx] || frames[0];
      const corridorStates = frame.corridor_states || {{}};

      // Draw all edges
      NET.edges.forEach(edge => {{
        if (!edge.geometry || edge.geometry.length < 2) return;

        const p0 = worldToScreen(edge.geometry[0][0], edge.geometry[0][1]);
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        for (let i = 1; i < edge.geometry.length; i++) {{
          const pt = worldToScreen(edge.geometry[i][0], edge.geometry[i][1]);
          ctx.lineTo(pt.x, pt.y);
        }}

        const state = corridorStates[edge.edge_id];
        if (state) {{
          // Dynamic velocity color coding
          const spd = state.speed_mph;
          if (spd < 10.0) {{
            ctx.strokeStyle = '#e0243d'; // Severe gridlock
            ctx.lineWidth = Math.max(3.0, 4.5 * (edge.lanes || 1) * zoom);
            ctx.shadowColor = '#e0243d';
            ctx.shadowBlur = 8;
          }} else if (spd < 25.0) {{
            ctx.strokeStyle = '#ffb703'; // Moderate congestion
            ctx.lineWidth = Math.max(2.2, 3.5 * (edge.lanes || 1) * zoom);
            ctx.shadowBlur = 0;
          }} else {{
            ctx.strokeStyle = '#06d6a0'; // Free flow
            ctx.lineWidth = Math.max(1.8, 2.5 * (edge.lanes || 1) * zoom);
            ctx.shadowBlur = 0;
          }}
        }} else if (edge.highway_type.includes('motorway')) {{
          ctx.strokeStyle = '#00f0ff';
          ctx.lineWidth = Math.max(2.0, 3.0 * zoom);
          ctx.shadowBlur = 0;
        }} else {{
          ctx.strokeStyle = 'rgba(139, 155, 180, 0.3)';
          ctx.lineWidth = Math.max(0.7, 1.0 * zoom);
          ctx.shadowBlur = 0;
        }}
        ctx.stroke();
      }});
      ctx.shadowBlur = 0;

      // Draw Sinks
      NET.sinks.forEach(s => {{
        const pt = worldToScreen(
          (s.lon - -97.4423) * 111320 * Math.cos(35.2059 * Math.PI / 180),
          (s.lat - 35.2059) * 110950
        );
        ctx.fillStyle = '#06d6a0';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }});
    }}

    setFrame(0);
  </script>
</body>
</html>
"""
    with open(html_filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SimulationPipeline] Interactive dashboard exported to: {html_filepath}")


def run_pipeline():
    print("=" * 70)
    print("🏈 SOONERGRID STEP 3: BASELINE UNMANAGED SIMULATION PIPELINE")
    print("=" * 70)
    t0 = time.time()

    # 1. Load Road Network
    print("[Pipeline] Ingesting road network...")
    extractor = OSMExtractor(bbox=DEFAULT_BBOX, cache_dir="data/cache")
    osm_raw = extractor.fetch(force_refresh=False)

    builder = NetworkBuilder(origin_lat=ORIGIN_LAT, origin_lon=ORIGIN_LON)
    graph = builder.build_graph_from_osm(osm_raw, simplify=True)

    nodes_metric = {nid: (d["x"], d["y"]) for nid, d in graph.nodes(data=True)}
    gateways, sinks = bind_zones_to_graph(nodes_metric, ORIGIN_LAT, ORIGIN_LON)

    # 2. Load Game Day Demand Matrix from Step 2
    demand_file = "data/norman_game_day_demand.json"
    if not os.path.exists(demand_file):
        print(f"[Pipeline] Running demand generator to create {demand_file}...")
        from soonergrid.demand.od_matrix_generator import GameDayDemandEngine
        demand_engine = GameDayDemandEngine(graph, gateways, sinks, total_vehicles=34500)
        demand_data = demand_engine.export_demand_dataset(demand_file)
    else:
        print(f"[Pipeline] Loading cached game-day demand from {demand_file}...")
        with open(demand_file, "r", encoding="utf-8") as f:
            demand_data = json.load(f)

    # 3. Load Processed Network for Visualizer Overlay
    network_file = "data/norman_network_processed.json"
    with open(network_file, "r", encoding="utf-8") as f:
        network_data = json.load(f)

    # 4. Instantiate and execute dynamic simulation
    sim_engine = TrafficSimulationEngine(
        graph=graph,
        gateways=gateways,
        sinks=sinks,
        demand_dataset=demand_data,
        dt_s=5.0,  # 5-second macroscopic kinematic wave timestep
        backward_wave_speed_mps=5.0,
    )

    sim_results = sim_engine.run_simulation(
        total_duration_s=28800.0,  # Full 8 hours
        playback_sample_interval_s=60.0,  # 1-minute visualizer snapshots
    )

    data_out = "data/norman_baseline_simulation_results.json"
    html_out = "visualizer/simulation_viewer.html"

    with open(data_out, "w", encoding="utf-8") as f:
        json.dump(sim_results, f, indent=2)
    print(f"[Pipeline] Baseline simulation results saved to: {data_out}")

    generate_simulation_html(sim_results, network_data, html_out)

    elapsed = time.time() - t0
    summary = sim_results["summary"]

    print("\n" + "=" * 50)
    print("📊 BASELINE UNMANAGED SIMULATION REPORT")
    print("=" * 50)
    print(f"  Total System Travel Time (TSTT):   {summary['total_system_travel_time_veh_hrs']:,} veh-hrs")
    print(f"  Free-Flow Travel Time Baseline:    {summary['free_flow_baseline_veh_hrs']:,} veh-hrs")
    print(f"  Total Gridlock Queue Delay:        {summary['total_queue_delay_veh_hrs']:,} veh-hrs ({summary['delay_percentage']}% delay)")
    print(f"  I-35 Spillback Hazard Duration:    {summary['i35_spillback_hazard_minutes']} minutes on interstate mainline")
    print(f"  Peak I-35 Off-Ramp Queue:          {summary['i35_peak_ramp_queue_meters']} meters")
    print(f"  Excess Fuel Wasted in Gridlock:    {summary['excess_fuel_wasted_gallons']:,} gallons")

    print("\n" + "=" * 70)
    print(f"✅ STEP 3 PIPELINE COMPLETE in {elapsed:.2f} seconds!")
    print(f"📦 Simulation Results: {os.path.abspath(data_out)}")
    print(f"🌐 Playback Dashboard: {os.path.abspath(html_out)}")
    print("=" * 70)

    return sim_results


if __name__ == "__main__":
    run_pipeline()
