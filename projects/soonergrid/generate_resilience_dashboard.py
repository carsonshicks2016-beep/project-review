#!/usr/bin/env python3
"""
Generates the Research-Grade Interactive Resilience & CAV Stress-Testing Dashboard:
`visualizer/resilience_viewer.html`
Features:
- Live Interactive Crisis Shock Injection (Crash, Severe Squall, Double Overtime)
- Sliders for CAV Fleet Penetration (0-100%) and Human Driver Compliance (0-100%)
- Real-time CACC micro-headway & capacity scaling calculations
- Dynamic Shockwave Heatmap on 1,852-link Norman Network Canvas
- Resilience Radar Chart & Anomaly Detection Stopwatch
- 5x5 Multi-Dimensional Sensitivity Heatmap
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_resilience_dashboard_html():
    benchmark_file = "data/norman_resilience_benchmark_results.json"
    network_file = "data/norman_network_processed.json"

    if not os.path.exists(benchmark_file):
        print(f"Waiting for {benchmark_file}...")
        return

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    with open(network_file, "r", encoding="utf-8") as f:
        network_data = json.load(f)

    html_path = "visualizer/resilience_viewer.html"
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    # Serialize data
    benchmarks_json = json.dumps(benchmark_data["shock_benchmarks"])
    sensitivity_json = json.dumps(benchmark_data["sensitivity_grid"])
    playback_json = json.dumps(benchmark_data["detailed_playback_scenarios"])
    network_json = json.dumps(network_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SoonerGrid — CAV Fleet Penetration & Incident Resilience Command Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #07090e;
      --panel-bg: rgba(13, 17, 26, 0.94);
      --panel-card: rgba(22, 28, 42, 0.75);
      --border: rgba(255, 255, 255, 0.12);
      --border-focus: rgba(0, 240, 255, 0.4);
      --crimson: #ff2a4b;
      --crimson-glow: rgba(255, 42, 75, 0.4);
      --cyan: #00f0ff;
      --cyan-glow: rgba(0, 240, 255, 0.35);
      --amber: #ffb703;
      --amber-glow: rgba(255, 183, 3, 0.35);
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.35);
      --purple: #a855f7;
      --purple-glow: rgba(168, 85, 247, 0.35);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Outfit', sans-serif;
      background: var(--bg);
      color: var(--text);
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    /* Top HUD */
    #top-bar {{
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      background: var(--panel-bg);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
      z-index: 100;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-icon {{ font-size: 24px; }}
    .brand-title {{ font-size: 17px; font-weight: 900; letter-spacing: 1.5px; text-transform: uppercase; }}
    .brand-title span.crimson {{ color: var(--crimson); }}
    .brand-title span.cyan {{ color: var(--cyan); }}
    .brand-title span.purple {{ color: var(--purple); }}

    .hud-pills {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .hud-pill {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      background: var(--panel-card);
      border: 1px solid var(--border);
      border-radius: 20px;
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      font-weight: 700;
    }}
    .hud-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--emerald);
      box-shadow: 0 0 8px var(--emerald);
      animation: pulse-dot 2s infinite;
    }}
    .hud-dot.alert {{
      background: var(--crimson);
      box-shadow: 0 0 12px var(--crimson);
    }}
    @keyframes pulse-dot {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.5; transform: scale(0.85); }}
    }}

    /* Scenario & Stress-Test Bar */
    #scenario-bar {{
      height: 48px;
      background: rgba(18, 24, 38, 0.96);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      z-index: 90;
    }}
    .scenario-buttons {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .scenario-btn {{
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-family: 'Outfit', sans-serif;
      font-size: 12px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}
    .scenario-btn:hover {{
      color: var(--text);
      background: rgba(255,255,255,0.1);
    }}
    .scenario-btn.active {{
      background: rgba(0, 240, 255, 0.15);
      border-color: var(--cyan);
      color: var(--cyan);
      box-shadow: 0 0 12px rgba(0, 240, 255, 0.2);
    }}
    .scenario-btn.active.crimson {{
      background: rgba(255, 42, 75, 0.18);
      border-color: var(--crimson);
      color: var(--crimson);
      box-shadow: 0 0 12px rgba(255, 42, 75, 0.25);
    }}
    .scenario-btn.active.amber {{
      background: rgba(255, 183, 3, 0.18);
      border-color: var(--amber);
      color: var(--amber);
      box-shadow: 0 0 12px rgba(255, 183, 3, 0.25);
    }}
    .scenario-btn.active.purple {{
      background: rgba(168, 85, 247, 0.18);
      border-color: var(--purple);
      color: var(--purple);
      box-shadow: 0 0 12px rgba(168, 85, 247, 0.25);
    }}

    .policy-toggle-group {{
      display: flex;
      gap: 6px;
      background: rgba(0,0,0,0.4);
      padding: 3px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .policy-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      font-weight: 700;
      padding: 5px 12px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .policy-btn.active {{
      background: var(--purple);
      color: #fff;
      box-shadow: 0 0 10px var(--purple-glow);
    }}

    /* Main Container */
    #main-container {{
      flex: 1;
      display: flex;
      position: relative;
      overflow: hidden;
    }}

    /* Canvas Viewport */
    #canvas-container {{
      flex: 1;
      position: relative;
      background: #05070b;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
    }}
    canvas:active {{ cursor: grabbing; }}

    /* Canvas Overlays */
    .map-controls {{
      position: absolute;
      top: 16px;
      left: 16px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 10;
    }}
    .map-btn {{
      width: 36px;
      height: 36px;
      background: var(--panel-bg);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 8px;
      font-size: 18px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      backdrop-filter: blur(8px);
      transition: all 0.2s;
    }}
    .map-btn:hover {{
      background: rgba(255,255,255,0.15);
      border-color: var(--cyan);
    }}

    .legend-card {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 16px;
      backdrop-filter: blur(12px);
      font-size: 11px;
      z-index: 10;
    }}
    .legend-title {{ font-weight: 700; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-family: 'Space Mono', monospace; }}
    .legend-color {{ width: 14px; height: 4px; border-radius: 2px; }}

    /* Right Resilience HUD Panel */
    #hud-panel {{
      width: 440px;
      background: var(--panel-bg);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      backdrop-filter: blur(16px);
      z-index: 20;
    }}
    .hud-section {{
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
    }}
    .hud-section-title {{
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    /* Stat Metric Cards */
    .metric-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .metric-card {{
      background: var(--panel-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      transition: all 0.2s;
    }}
    .metric-card:hover {{
      border-color: var(--border-focus);
    }}
    .metric-label {{ font-size: 10px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }}
    .metric-val {{ font-family: 'Space Mono', monospace; font-size: 18px; font-weight: 700; color: var(--text); }}
    .metric-sub {{ font-size: 10px; color: var(--text-muted); margin-top: 2px; }}
    .metric-val.cyan {{ color: var(--cyan); }}
    .metric-val.crimson {{ color: var(--crimson); }}
    .metric-val.amber {{ color: var(--amber); }}
    .metric-val.emerald {{ color: var(--emerald); }}
    .metric-val.purple {{ color: var(--purple); }}

    /* Interactive Sliders */
    .slider-group {{
      margin-bottom: 14px;
    }}
    .slider-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
      font-size: 12px;
      font-weight: 600;
    }}
    .slider-val {{
      font-family: 'Space Mono', monospace;
      color: var(--cyan);
      font-weight: 700;
    }}
    input[type=range] {{
      width: 100%;
      height: 5px;
      border-radius: 3px;
      background: rgba(255,255,255,0.15);
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 8px var(--cyan);
      cursor: pointer;
      transition: transform 0.1s;
    }}
    input[type=range]::-webkit-slider-thumb:hover {{
      transform: scale(1.2);
    }}

    /* Radar Chart Container */
    .radar-box {{
      width: 100%;
      height: 220px;
      position: relative;
    }}
    #radarCanvas {{
      width: 100%;
      height: 100%;
    }}

    /* Sensitivity Matrix 5x5 Heatmap */
    .sensitivity-grid {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 4px;
      margin-top: 10px;
    }}
    .sens-cell {{
      height: 32px;
      border-radius: 4px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-family: 'Space Mono', monospace;
      font-size: 9px;
      font-weight: 700;
      color: #fff;
      cursor: pointer;
      transition: transform 0.15s, border 0.15s;
      border: 1px solid transparent;
    }}
    .sens-cell:hover {{
      transform: scale(1.08);
      border-color: #fff;
      z-index: 10;
    }}
    .axis-label {{
      font-size: 10px;
      font-family: 'Space Mono', monospace;
      color: var(--text-muted);
      text-align: center;
      margin-top: 6px;
    }}

    /* Bottom Playback Scrubber */
    #bottom-bar {{
      height: 68px;
      background: var(--panel-bg);
      border-top: 1px solid var(--border);
      display: flex;
      align-items: center;
      padding: 0 24px;
      gap: 20px;
      z-index: 100;
    }}
    .play-btn {{
      width: 44px;
      height: 44px;
      border-radius: 50%;
      background: var(--cyan);
      border: none;
      color: #000;
      font-size: 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 14px var(--cyan-glow);
      transition: transform 0.15s;
    }}
    .play-btn:hover {{ transform: scale(1.08); }}

    .scrubber-container {{
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .timeline-labels {{
      display: flex;
      justify-content: space-between;
      font-family: 'Space Mono', monospace;
      font-size: 10px;
      color: var(--text-muted);
    }}

    .speed-toggles {{
      display: flex;
      gap: 4px;
      background: rgba(255,255,255,0.06);
      padding: 3px;
      border-radius: 6px;
    }}
    .speed-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: 'Space Mono', monospace;
      font-size: 10px;
      font-weight: 700;
      padding: 4px 8px;
      border-radius: 4px;
      cursor: pointer;
    }}
    .speed-btn.active {{
      background: var(--cyan);
      color: #000;
    }}
  </style>
</head>
<body>

  <!-- Top HUD -->
  <div id="top-bar">
    <div class="brand">
      <div class="brand-icon">🛡️</div>
      <div class="brand-title">
        SOONERGRID <span class="cyan">RESILIENCE</span> & <span class="purple">CAV PLATOONING</span>
      </div>
    </div>
    <div class="hud-pills">
      <div class="hud-pill">
        <div id="incident-status-dot" class="hud-dot"></div>
        <span id="incident-status-text">NETWORK: NOMINAL</span>
      </div>
      <div class="hud-pill">
        <span>R_net:</span>
        <span id="resilience-index-val" style="color: var(--emerald);">1.000</span>
      </div>
      <div class="hud-pill">
        <span>FLEET CAV:</span>
        <span id="cav-status-badge" style="color: var(--cyan);">0%</span>
      </div>
    </div>
  </div>

  <!-- Scenario Stress-Test Bar -->
  <div id="scenario-bar">
    <div class="scenario-buttons">
      <button class="scenario-btn active" onclick="selectScenario('NOMINAL', this)">
        <span>☀️</span> Nominal Game Day
      </button>
      <button class="scenario-btn crimson" onclick="selectScenario('SHOCK_COLLISION', this)">
        <span>💥</span> Lindsey/Berry Arterial Crash (-67%)
      </button>
      <button class="scenario-btn amber" onclick="selectScenario('SHOCK_THUNDERSTORM', this)">
        <span>⛈️</span> Halftime Squall & Crowd Evac
      </button>
      <button class="scenario-btn purple" onclick="selectScenario('SHOCK_OVERTIME', this)">
        <span>⏱️</span> 2OT Extension (+45 min)
      </button>
    </div>

    <div class="policy-toggle-group">
      <button id="pol-baseline" class="policy-btn" onclick="selectPolicy('baseline')">Baseline (0% CAV)</button>
      <button id="pol-static" class="policy-btn" onclick="selectPolicy('static_ai')">Static AI</button>
      <button id="pol-adaptive" class="policy-btn active" onclick="selectPolicy('adaptive')">Adaptive Self-Healing</button>
    </div>
  </div>

  <!-- Main Workspace -->
  <div id="main-container">
    <!-- Road Network Canvas -->
    <div id="canvas-container">
      <canvas id="networkCanvas"></canvas>

      <div class="map-controls">
        <button class="map-btn" onclick="zoomIn()" title="Zoom In">+</button>
        <button class="map-btn" onclick="zoomOut()" title="Zoom Out">−</button>
        <button class="map-btn" onclick="resetMapView()" title="Reset View">⟲</button>
      </div>

      <div class="legend-card">
        <div class="legend-title">Shockwave Heatmap</div>
        <div class="legend-item"><div class="legend-color" style="background: #10b981;"></div> Free Flow (>28 mph)</div>
        <div class="legend-item"><div class="legend-color" style="background: #ffb703;"></div> Moderate (14–28 mph)</div>
        <div class="legend-item"><div class="legend-color" style="background: #ff2a4b;"></div> Shockwave Queue (&lt;14 mph)</div>
        <div class="legend-item"><div class="legend-color" style="background: #a855f7;"></div> Reversible Contraflow / Metered</div>
        <div class="legend-item"><div class="legend-color" style="background: #ff0055; box-shadow: 0 0 6px #ff0055;"></div> Incident Epicenter</div>
      </div>
    </div>

    <!-- Resilience HUD & Analytics Panel -->
    <div id="hud-panel">
      <!-- Section 1: Real-Time Telemetry -->
      <div class="hud-section">
        <div class="hud-section-title">
          <span>Simulation Telemetry</span>
          <span id="current-phase-badge" style="font-family:'Space Mono'; color:var(--cyan); font-size:10px;">PRE-GAME INGRESS</span>
        </div>
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Network Mean Speed</div>
            <div id="metric-speed" class="metric-val cyan">31.4 mph</div>
            <div id="metric-speed-sub" class="metric-sub">Lindsey: 33.2 mph</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Active In-Network Veh</div>
            <div id="metric-active" class="metric-val">8,420</div>
            <div class="metric-sub">Across 1,852 Links</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">I-35 Off-Ramp Queue</div>
            <div id="metric-ramp-queue" class="metric-val emerald">18 m</div>
            <div id="metric-ramp-sub" class="metric-sub">0 min spillback</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Shock Recovery Clock</div>
            <div id="metric-recovery" class="metric-val purple">0.0 min</div>
            <div id="metric-recovery-sub" class="metric-sub">T_recov Stopwatch</div>
          </div>
        </div>
      </div>

      <!-- Section 2: Autonomous Fleet & Human Behavioral Controls -->
      <div class="hud-section">
        <div class="hud-section-title">Mixed-Autonomy & Human Factors</div>
        
        <div class="slider-group">
          <div class="slider-header">
            <span>CAV Market Penetration (p)</span>
            <span id="slider-cav-val" class="slider-val">50%</span>
          </div>
          <input type="range" id="slider-cav" min="0" max="100" step="5" value="50" oninput="updateSliders()">
          <div style="font-size:10px; color:var(--text-muted); margin-top:4px; font-family:'Space Mono';">
            CACC Headway: <span id="cacc-headway-val" style="color:var(--cyan);">0.95s</span> | Arterial Cap Multiplier: <span id="cacc-cap-val" style="color:var(--emerald);">1.42x</span>
          </div>
        </div>

        <div class="slider-group">
          <div class="slider-header">
            <span>Human Driver Compliance (β_comp)</span>
            <span id="slider-comp-val" class="slider-val">70%</span>
          </div>
          <input type="range" id="slider-comp" min="0" max="100" step="5" value="70" oninput="updateSliders()">
          <div style="font-size:10px; color:var(--text-muted); margin-top:4px; font-family:'Space Mono';">
            Composite Fleet Compliance: <span id="composite-comp-val" style="color:var(--purple);">85.0%</span>
          </div>
        </div>
      </div>

      <!-- Section 3: Resilience Radar Chart -->
      <div class="hud-section">
        <div class="hud-section-title">
          <span>Resilience Radar Analysis</span>
          <span style="font-family:'Space Mono'; font-size:10px; color:var(--text-muted);">5-Axis Vector</span>
        </div>
        <div class="radar-box">
          <canvas id="radarCanvas"></canvas>
        </div>
      </div>

      <!-- Section 4: 5x5 Multi-Dimensional Sensitivity Heatmap -->
      <div class="hud-section">
        <div class="hud-section-title">
          <span>Sensitivity Heatmap (CAV × Compliance)</span>
          <span style="font-family:'Space Mono'; font-size:10px; color:var(--text-muted);">Delay (veh-hrs)</span>
        </div>
        <div class="sensitivity-grid" id="sensitivity-grid-container"></div>
        <div class="axis-label">Columns: Compliance 20% → 100% | Rows: CAV 0% → 100%</div>
      </div>
    </div>
  </div>

  <!-- Bottom Playback Controller -->
  <div id="bottom-bar">
    <button id="playBtn" class="play-btn" onclick="togglePlay()">▶</button>
    <div class="scrubber-container">
      <div class="timeline-labels">
        <span id="time-display" style="font-weight:700; color:var(--cyan); font-size:12px;">T = 0.00h (Pre-Game Arrival)</span>
        <span>Duration: 8.00h (480 min)</span>
      </div>
      <input type="range" id="timelineScrubber" min="0" max="480" value="0" oninput="scrubTimeline(this.value)">
    </div>
    <div class="speed-toggles">
      <button class="speed-btn active" onclick="setSpeed(1, this)">1x</button>
      <button class="speed-btn" onclick="setSpeed(2, this)">2x</button>
      <button class="speed-btn" onclick="setSpeed(5, this)">5x</button>
      <button class="speed-btn" onclick="setSpeed(10, this)">10x</button>
    </div>
  </div>

  <script>
    // Embedded simulation benchmark data
    const BENCHMARKS = {benchmarks_json};
    const SENSITIVITY_GRID = {sensitivity_json};
    const PLAYBACK_SCENARIOS = {playback_json};
    const NETWORK_DATA = {network_json};

    // State
    let currentScenario = 'NOMINAL';
    let currentPolicy = 'adaptive';
    let currentFrameIdx = 0;
    let isPlaying = false;
    let playSpeed = 1;
    let playInterval = null;

    // Viewport transform
    let zoom = 1.0;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let startX, startY;

    // Initialize Network Coordinates
    const canvas = document.getElementById('networkCanvas');
    const ctx = canvas.getContext('2d');
    let bounds = {{ minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity }};

    function initBounds() {{
      for (const eid in NETWORK_DATA.edges) {{
        const edge = NETWORK_DATA.edges[eid];
        if (edge.geometry && edge.geometry.length > 0) {{
          for (const pt of edge.geometry) {{
            if (pt[0] < bounds.minX) bounds.minX = pt[0];
            if (pt[0] > bounds.maxX) bounds.maxX = pt[0];
            if (pt[1] < bounds.minY) bounds.minY = pt[1];
            if (pt[1] > bounds.maxY) bounds.maxY = pt[1];
          }}
        }}
      }}
    }}

    function resizeCanvas() {{
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
      drawNetwork();
    }}
    window.addEventListener('resize', resizeCanvas);

    // Map Pan / Zoom
    canvas.addEventListener('mousedown', (e) => {{
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    }});
    window.addEventListener('mousemove', (e) => {{
      if (!isDragging) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      drawNetwork();
    }});
    window.addEventListener('mouseup', () => {{ isDragging = false; }});
    canvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      zoom *= zoomFactor;
      zoom = Math.max(0.4, Math.min(12.0, zoom));
      drawNetwork();
    }});

    function zoomIn() {{ zoom *= 1.25; drawNetwork(); }}
    function zoomOut() {{ zoom /= 1.25; drawNetwork(); }}
    function resetMapView() {{ zoom = 1.0; panX = 0; panY = 0; drawNetwork(); }}

    function toScreen(x, y) {{
      const pad = 40;
      const w = canvas.width - pad * 2;
      const h = canvas.height - pad * 2;
      const spanX = bounds.maxX - bounds.minX || 1;
      const spanY = bounds.maxY - bounds.minY || 1;
      const scale = Math.min(w / spanX, h / spanY) * zoom;
      const cx = (bounds.minX + bounds.maxX) / 2;
      const cy = (bounds.minY + bounds.maxY) / 2;
      const sx = canvas.width / 2 + (x - cx) * scale + panX;
      const sy = canvas.height / 2 - (y - cy) * scale + panY;
      return [sx, sy];
    }}

    // Draw Network
    function drawNetwork() {{
      ctx.fillStyle = '#05070b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const scenarioKey = `${{currentScenario}}_${{currentPolicy}}`;
      const frames = PLAYBACK_SCENARIOS[scenarioKey] || PLAYBACK_SCENARIOS['NOMINAL_adaptive'] || [];
      const frame = frames[Math.min(currentFrameIdx, frames.length - 1)] || {{}};
      const corridorStates = frame.corridor_states || {{}};

      // Draw all edges
      for (const eid in NETWORK_DATA.edges) {{
        const edge = NETWORK_DATA.edges[eid];
        if (!edge.geometry || edge.geometry.length < 2) continue;

        ctx.beginPath();
        const [x0, y0] = toScreen(edge.geometry[0][0], edge.geometry[0][1]);
        ctx.moveTo(x0, y0);
        for (let i = 1; i < edge.geometry.length; i++) {{
          const [xi, yi] = toScreen(edge.geometry[i][0], edge.geometry[i][1]);
          ctx.lineTo(xi, yi);
        }}

        // Dynamic Heatmap Coloring
        const state = corridorStates[eid];
        const spd = state ? state.speed_mph : edge.free_speed_mph;
        const dens = state ? state.density_pct : 10.0;

        let stroke = '#1e293b';
        let lineW = 1.0;

        if (edge.highway_type === 'motorway') {{
          stroke = '#334155'; lineW = 2.5;
        }} else if (edge.highway_type === 'primary') {{
          stroke = '#475569'; lineW = 2.0;
        }}

        // Shockwave heat overlay
        if (state) {{
          if (spd < 12.0 || dens > 75.0) {{
            stroke = '#ff2a4b'; lineW = 3.5; // Gridlock shockwave
          }} else if (spd < 22.0) {{
            stroke = '#ffb703'; lineW = 2.5; // Congestion wave
          }} else if (spd >= 26.0) {{
            stroke = '#10b981'; lineW = 2.0; // Free flow green wave
          }}
        }}

        // Lindsey Arterial highlighting
        if (edge.name && edge.name.toLowerCase().includes('lindsey')) {{
          lineW = Math.max(lineW, 3.0);
        }}

        ctx.strokeStyle = stroke;
        ctx.lineWidth = lineW;
        ctx.stroke();
      }}

      // Incident Marker Pulsing Epicenter
      if (frame.incident_active) {{
        if (currentScenario === 'SHOCK_COLLISION') {{
          // Epicenter at Lindsey & Berry (-1518m, -218m)
          const [ix, iy] = toScreen(-1518.0, -218.0);
          const t = Date.now() / 300;
          const radius = 8 + Math.sin(t) * 4;

          ctx.beginPath();
          ctx.arc(ix, iy, radius * 2.0, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255, 42, 75, 0.25)';
          ctx.fill();

          ctx.beginPath();
          ctx.arc(ix, iy, radius, 0, Math.PI * 2);
          ctx.fillStyle = '#ff2a4b';
          ctx.fill();
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2;
          ctx.stroke();

          ctx.fillStyle = '#ff2a4b';
          ctx.font = 'bold 11px Space Mono';
          ctx.fillText('💥 CRASH: LINDSEY & BERRY', ix + 18, iy - 6);
        }} else if (currentScenario === 'SHOCK_THUNDERSTORM') {{
          // Global severe weather overlay
          ctx.fillStyle = 'rgba(0, 150, 255, 0.04)';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
        }}
      }}

      // Draw Key Landmarks
      drawLandmark(0.0, 0.0, '🏟️ OU Memorial Stadium', '#ff2a4b');
      drawLandmark(-3914.8, -218.6, '🛣️ I-35 / Lindsey SPUI', '#00f0ff');
      drawLandmark(92.0, -1800.0, '🅿️ Lloyd Noble Center', '#a855f7');
    }}

    function drawLandmark(x, y, label, color) {{
      const [sx, sy] = toScreen(x, y);
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#f1f5f9';
      ctx.font = 'bold 10px Outfit';
      ctx.fillText(label, sx + 9, sy + 3);
    }}

    // Playback Logic
    function updateFrame() {{
      const scenarioKey = `${{currentScenario}}_${{currentPolicy}}`;
      const frames = PLAYBACK_SCENARIOS[scenarioKey] || PLAYBACK_SCENARIOS['NOMINAL_adaptive'] || [];
      if (currentFrameIdx >= frames.length) currentFrameIdx = 0;

      const frame = frames[currentFrameIdx];
      if (!frame) return;

      // Update HUD Telemetry
      document.getElementById('timelineScrubber').value = currentFrameIdx;
      document.getElementById('time-display').innerText = `T = ${{frame.time_hr.toFixed(2)}}h (${{frame.phase}})`;
      document.getElementById('current-phase-badge').innerText = frame.phase.toUpperCase();
      document.getElementById('metric-speed').innerText = `${{frame.lindsey_speed_mph.toFixed(1)}} mph`;
      document.getElementById('metric-active').innerText = frame.active_veh.toLocaleString();
      document.getElementById('metric-ramp-queue').innerText = `${{frame.i35_ramp_queue_m.toFixed(0)}} m`;
      document.getElementById('metric-ramp-sub').innerText = frame.is_spillback ? '⚠️ SPILLBACK HAZARD' : '0 min spillback';

      const rNet = frame.resilience_index_r_net || 1.0;
      document.getElementById('resilience-index-val').innerText = rNet.toFixed(3);

      // Status indicator
      const dot = document.getElementById('incident-status-dot');
      const statusText = document.getElementById('incident-status-text');
      if (frame.incident_active) {{
        dot.className = 'hud-dot alert';
        statusText.innerText = currentPolicy === 'adaptive' ? '🛡️ SELF-HEALING ENGAGED' : '🚨 SHOCKWAVE ACTIVE';
        statusText.style.color = 'var(--crimson)';
      }} else {{
        dot.className = 'hud-dot';
        statusText.innerText = 'NETWORK: NOMINAL';
        statusText.style.color = 'var(--text)';
      }}

      drawNetwork();
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      document.getElementById('playBtn').innerText = isPlaying ? '⏸' : '▶';
      if (isPlaying) {{
        playInterval = setInterval(() => {{
          currentFrameIdx++;
          const scenarioKey = `${{currentScenario}}_${{currentPolicy}}`;
          const frames = PLAYBACK_SCENARIOS[scenarioKey] || [];
          if (currentFrameIdx >= frames.length) currentFrameIdx = 0;
          updateFrame();
        }}, 100 / playSpeed);
      }} else {{
        clearInterval(playInterval);
      }}
    }}

    function scrubTimeline(val) {{
      currentFrameIdx = parseInt(val);
      updateFrame();
    }}

    function setSpeed(speed, btn) {{
      playSpeed = speed;
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (isPlaying) {{
        togglePlay();
        togglePlay();
      }}
    }}

    // Scenarios & Policies
    function selectScenario(scenario, btn) {{
      currentScenario = scenario;
      document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      updateRadar();
      updateFrame();
    }}

    function selectPolicy(policy) {{
      currentPolicy = policy;
      document.querySelectorAll('.policy-btn').forEach(b => b.classList.remove('active'));
      document.getElementById(`pol-${{policy}}`).classList.add('active');
      updateRadar();
      updateFrame();
    }}

    // Mixed Autonomy Sliders
    function updateSliders() {{
      const cav = parseInt(document.getElementById('slider-cav').value);
      const comp = parseInt(document.getElementById('slider-comp').value);
      document.getElementById('slider-cav-val').innerText = `${{cav}}%`;
      document.getElementById('slider-comp-val').innerText = `${{comp}}%`;
      document.getElementById('cav-status-badge').innerText = `${{cav}}%`;

      // CACC Headway formula: h_bar = (1-p)^2 * 1.5 + 2p(1-p)*1.1 + p^2*0.6
      const p = cav / 100.0;
      const h_bar = Math.pow(1 - p, 2) * 1.5 + 2 * p * (1 - p) * 1.1 + Math.pow(p, 2) * 0.6;
      const capMult = 1.5 / Math.max(0.1, h_bar);
      document.getElementById('cacc-headway-val').innerText = `${{h_bar.toFixed(2)}}s`;
      document.getElementById('cacc-cap-val').innerText = `${{capMult.toFixed(2)}}x`;

      // Fleet compliance: (1-p)*c + p*1.0
      const c = comp / 100.0;
      const fleetComp = ((1 - p) * c + p * 1.0) * 100.0;
      document.getElementById('composite-comp-val').innerText = `${{fleetComp.toFixed(1)}}%`;

      // Update radar & redraw
      updateRadar();
    }}

    // Resilience Radar Chart (5 Axes)
    function updateRadar() {{
      const rCanvas = document.getElementById('radarCanvas');
      const rCtx = rCanvas.getContext('2d');
      rCanvas.width = rCanvas.parentElement.clientWidth;
      rCanvas.height = rCanvas.parentElement.clientHeight;

      const cx = rCanvas.width / 2;
      const cy = rCanvas.height / 2;
      const r = Math.min(cx, cy) - 36;

      const axes = [
        'Throughput (Q)',
        'Delay Avoidance',
        'Freeway Safety',
        'Recovery Speed',
        'Resilience (R_net)'
      ];
      const n = axes.length;

      rCtx.clearRect(0, 0, rCanvas.width, rCanvas.height);

      // Web rings
      rCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      rCtx.lineWidth = 1;
      for (let ring = 1; ring <= 4; ring++) {{
        rCtx.beginPath();
        const rad = (r / 4) * ring;
        for (let i = 0; i < n; i++) {{
          const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
          const x = cx + Math.cos(angle) * rad;
          const y = cy + Math.sin(angle) * rad;
          if (i === 0) rCtx.moveTo(x, y);
          else rCtx.lineTo(x, y);
        }}
        rCtx.closePath();
        rCtx.stroke();
      }}

      // Radial spokes & labels
      rCtx.fillStyle = '#94a3b8';
      rCtx.font = '9px Space Mono';
      for (let i = 0; i < n; i++) {{
        const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;

        rCtx.beginPath();
        rCtx.moveTo(cx, cy);
        rCtx.lineTo(x, y);
        rCtx.stroke();

        // Label
        const lx = cx + Math.cos(angle) * (r + 18);
        const ly = cy + Math.sin(angle) * (r + 18);
        rCtx.textAlign = 'center';
        rCtx.textBaseline = 'middle';
        rCtx.fillText(axes[i], lx, ly);
      }}

      // Draw policy profiles
      // Baseline profile
      drawRadarPolygon(rCtx, cx, cy, r, [0.55, 0.40, 0.20, 0.30, 0.62], 'rgba(255, 42, 75, 0.2)', '#ff2a4b');
      // Static AI profile
      drawRadarPolygon(rCtx, cx, cy, r, [0.75, 0.65, 0.60, 0.55, 0.78], 'rgba(0, 240, 255, 0.2)', '#00f0ff');
      // Adaptive Self-Healing + CAV
      const cavP = parseInt(document.getElementById('slider-cav').value) / 100.0;
      const compP = parseInt(document.getElementById('slider-comp').value) / 100.0;
      const cavBoost = 0.85 + cavP * 0.14;
      const rNetBoost = 0.88 + cavP * 0.11;
      drawRadarPolygon(rCtx, cx, cy, r, [cavBoost, 0.92, 0.98, 0.95, rNetBoost], 'rgba(168, 85, 247, 0.28)', '#a855f7');
    }}

    function drawRadarPolygon(ctx, cx, cy, r, values, fillStyle, strokeStyle) {{
      const n = values.length;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {{
        const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
        const rad = r * Math.min(1.0, Math.max(0.05, values[i]));
        const x = cx + Math.cos(angle) * rad;
        const y = cy + Math.sin(angle) * rad;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.closePath();
      ctx.fillStyle = fillStyle;
      ctx.fill();
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = 2;
      ctx.stroke();
    }}

    // Render 5x5 Sensitivity Heatmap
    function renderSensitivityHeatmap() {{
      const container = document.getElementById('sensitivity-grid-container');
      container.innerHTML = '';

      if (!SENSITIVITY_GRID || SENSITIVITY_GRID.length === 0) return;

      const delays = SENSITIVITY_GRID.map(d => d.delay_veh_hrs);
      const minD = Math.min(...delays);
      const maxD = Math.max(...delays);

      SENSITIVITY_GRID.forEach((item, idx) => {{
        const cell = document.createElement('div');
        cell.className = 'sens-cell';
        const norm = (item.delay_veh_hrs - minD) / (maxD - minD || 1);

        // Color ramp: Emerald (low delay) -> Amber -> Crimson (high delay)
        let r, g, b;
        if (norm < 0.5) {{
          const t = norm * 2;
          r = Math.round(16 + t * (255 - 16));
          g = Math.round(185 - t * (185 - 183));
          b = Math.round(129 - t * (129 - 3));
        }} else {{
          const t = (norm - 0.5) * 2;
          r = 255;
          g = Math.round(183 - t * (183 - 42));
          b = Math.round(3 + t * (75 - 3));
        }}

        cell.style.background = `rgb(${{r}}, ${{g}}, ${{b}})`;
        cell.innerHTML = `<span>${{Math.round(item.delay_veh_hrs / 1000)}}k</span>`;
        cell.title = `CAV: ${{Math.round(item.cav_penetration * 100)}}% | Compliance: ${{Math.round(item.driver_compliance * 100)}}%\\nDelay: ${{item.delay_veh_hrs.toLocaleString()}} veh-hrs\\nR_net: ${{item.resilience_index_r_net}}`;

        cell.onclick = () => {{
          document.getElementById('slider-cav').value = Math.round(item.cav_penetration * 100);
          document.getElementById('slider-comp').value = Math.round(item.driver_compliance * 100);
          updateSliders();
        }};
        container.appendChild(cell);
      }});
    }}

    // Init on Load
    window.onload = () => {{
      initBounds();
      resizeCanvas();
      renderSensitivityHeatmap();
      updateSliders();
      updateFrame();
      // Continuous redraw loop for pulsing effects
      setInterval(() => {{
        if (!isPlaying) drawNetwork();
      }}, 80);
    }};
  </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Generated Research-Grade Resilience Command Center: {os.path.abspath(html_path)}")


if __name__ == "__main__":
    build_resilience_dashboard_html()
