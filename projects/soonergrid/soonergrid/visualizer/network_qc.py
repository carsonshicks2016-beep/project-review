"""
Network Quality Control, Shortest Path Analysis, and Interactive Web Visualizer Generator.
"""

import os
import json
import math
from typing import Dict, List, Tuple, Any
import networkx as nx

from soonergrid.physics.link_attributes import EdgeAttributes
from soonergrid.data.game_day_zones import Gateway, ParkingSink


class NetworkQC:
    def __init__(self, graph: nx.MultiDiGraph, gateways: List[Gateway], sinks: List[ParkingSink]):
        self.graph = graph
        self.gateways = gateways
        self.sinks = sinks

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """Calculates topological and physical network metrics."""
        total_length_m = 0.0
        total_lane_length_m = 0.0
        total_storage_veh = 0
        reversible_edges = 0
        hwy_breakdown = {}

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            attr: EdgeAttributes = data["attr"]
            total_length_m += attr.length_m
            total_lane_length_m += attr.length_m * attr.lanes
            total_storage_veh += attr.jam_storage_veh
            if attr.is_reversible:
                reversible_edges += 1
            hwy = attr.highway_type
            hwy_breakdown[hwy] = hwy_breakdown.get(hwy, 0) + 1

        stats = {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "centerline_km": round(total_length_m / 1000.0, 2),
            "lane_km": round(total_lane_length_m / 1000.0, 2),
            "network_storage_capacity_veh": total_storage_veh,
            "reversible_edge_count": reversible_edges,
            "highway_type_breakdown": hwy_breakdown,
        }
        return stats

    def compute_gateway_sink_routes(self) -> List[Dict[str, Any]]:
        """Computes free-flow shortest paths from each gateway to stadium & Lloyd Noble Center."""
        routes = []

        # Find target sinks
        target_sinks = {
            "Stadium": next((s for s in self.sinks if "STADIUM" in s.id), self.sinks[0]),
            "Lloyd_Noble": next((s for s in self.sinks if "LLOYD_NOBLE" in s.id), self.sinks[0]),
        }

        for gw in self.gateways:
            if gw.nearest_node is None:
                continue

            for target_name, sink in target_sinks.items():
                if sink.nearest_node is None or gw.nearest_node == sink.nearest_node:
                    continue

                try:
                    # Dijkstra shortest path by free-flow travel time
                    def edge_weight_fn(u, v, d):
                        if "attr" in d:
                            return d["attr"].free_flow_time_s
                        weights = [val["attr"].free_flow_time_s for val in d.values() if "attr" in val]
                        return min(weights) if weights else 1.0

                    node_path = nx.shortest_path(
                        self.graph,
                        source=gw.nearest_node,
                        target=sink.nearest_node,
                        weight=edge_weight_fn
                    )

                    path_length_m = 0.0
                    path_time_s = 0.0
                    edge_ids = []

                    for i in range(len(node_path) - 1):
                        u_curr = node_path[i]
                        v_next = node_path[i + 1]
                        # Pick edge with minimal travel time
                        best_edge_key = min(
                            self.graph[u_curr][v_next].keys(),
                            key=lambda k: self.graph[u_curr][v_next][k]["attr"].free_flow_time_s
                        )
                        attr: EdgeAttributes = self.graph[u_curr][v_next][best_edge_key]["attr"]
                        path_length_m += attr.length_m
                        path_time_s += attr.free_flow_time_s
                        edge_ids.append(attr.edge_id)

                    routes.append({
                        "gateway_id": gw.id,
                        "gateway_name": gw.name,
                        "target_name": target_name,
                        "sink_id": sink.id,
                        "sink_name": sink.name,
                        "distance_km": round(path_length_m / 1000.0, 2),
                        "free_flow_time_min": round(path_time_s / 60.0, 2),
                        "node_path": node_path,
                        "edge_ids": edge_ids,
                    })
                except nx.NetworkXNoPath:
                    print(f"[NetworkQC] Warning: No path between {gw.name} and {sink.name}")

        return routes

    def export_dataset(self, output_filepath: str) -> Dict[str, Any]:
        """Serializes processed road network, gateways, sinks, and routes to JSON."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        stats = self.compute_summary_statistics()
        routes = self.compute_gateway_sink_routes()

        nodes_list = []
        for n, data in self.graph.nodes(data=True):
            nodes_list.append({
                "id": n,
                "lat": round(data["lat"], 6),
                "lon": round(data["lon"], 6),
                "x": round(data["x"], 2),
                "y": round(data["y"], 2),
            })

        edges_list = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            attr: EdgeAttributes = data["attr"]
            edges_list.append(attr.to_dict())

        gateways_list = [
            {
                "id": g.id,
                "name": g.name,
                "lat": g.lat,
                "lon": g.lon,
                "corridor": g.corridor,
                "primary_origin": g.primary_origin,
                "weight": g.hourly_inflow_weight,
                "nearest_node": g.nearest_node,
                "dist_m": round(g.dist_to_node_m, 1),
            }
            for g in self.gateways
        ]

        sinks_list = [
            {
                "id": s.id,
                "name": s.name,
                "lat": s.lat,
                "lon": s.lon,
                "capacity": s.capacity_stalls,
                "zone_type": s.zone_type,
                "shuttle_served": s.shuttle_served,
                "description": s.description,
                "nearest_node": s.nearest_node,
                "dist_m": round(s.dist_to_node_m, 1),
            }
            for s in self.sinks
        ]

        payload = {
            "statistics": stats,
            "gateways": gateways_list,
            "sinks": sinks_list,
            "routes": routes,
            "nodes": nodes_list,
            "edges": edges_list,
        }

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[NetworkQC] Dataset successfully serialized to: {output_filepath}")
        return payload

    def generate_html_viewer(self, json_data: Dict[str, Any], html_filepath: str):
        """Generates a standalone, interactive HTML5 Canvas visualizer."""
        os.makedirs(os.path.dirname(html_filepath), exist_ok=True)
        json_blob = json.dumps(json_data)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SoonerGrid — Norman Game Day Road Network & Contraflow Explorer</title>
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
      font-family: 'Outfit', -apple-system, sans-serif;
      background-color: var(--bg);
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
    .brand-icon {{
      font-size: 24px;
      filter: drop-shadow(0 0 8px var(--crimson-glow));
    }}
    .brand-title {{
      font-size: 16px;
      font-weight: 900;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--text);
    }}
    .brand-title span {{ color: var(--crimson-glow); }}
    .badge {{
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      background: rgba(0, 240, 255, 0.12);
      border: 1px solid rgba(0, 240, 255, 0.4);
      color: var(--cyan);
      padding: 3px 8px;
      border-radius: 6px;
    }}

    /* Control & Telemetry Sidebar */
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
      gap: 16px;
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
    .stat-label {{
      font-size: 11px;
      color: var(--text-muted);
    }}
    .stat-val {{
      font-family: 'Space Mono', monospace;
      font-size: 15px;
      font-weight: 700;
      color: var(--cyan);
      margin-top: 3px;
    }}
    
    /* Interactive Routes */
    .route-btn {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: inherit;
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 12px;
      text-align: left;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.2s;
    }}
    .route-btn:hover {{
      background: rgba(0, 240, 255, 0.12);
      border-color: var(--cyan);
      transform: translateX(3px);
    }}
    .route-btn.active {{
      background: rgba(224, 36, 61, 0.2);
      border-color: var(--crimson-glow);
      color: #fff;
    }}

    /* Contraflow Toggle Banner */
    .contraflow-card {{
      background: linear-gradient(135deg, rgba(224, 36, 61, 0.15), rgba(255, 183, 3, 0.15));
      border: 1px solid rgba(224, 36, 61, 0.4);
      border-radius: 12px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .contraflow-title {{
      font-size: 13px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .toggle-btn {{
      background: var(--crimson);
      color: #fff;
      border: none;
      padding: 8px 12px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .toggle-btn:hover {{
      background: var(--crimson-glow);
      box-shadow: 0 0 12px var(--crimson-glow);
    }}

    /* Edge Telemetry Modal */
    #telemetry-box {{
      position: absolute;
      bottom: 24px;
      left: 24px;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 18px;
      width: 320px;
      display: none;
      z-index: 10;
      font-family: 'Space Mono', monospace;
      font-size: 11px;
    }}
    .tele-title {{
      font-family: 'Outfit', sans-serif;
      font-size: 14px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 8px;
    }}
  </style>
</head>
<body>
  <div id="canvas-container">
    <canvas id="mapCanvas"></canvas>

    <div id="top-bar">
      <div class="brand">
        <span class="brand-icon">🏈</span>
        <div class="brand-title">SOONER<span>GRID</span></div>
      </div>
      <div class="badge">NORMAN GAME DAY NETWORK</div>
    </div>

    <div id="telemetry-box">
      <div class="tele-title" id="tele-name">Road Telemetry</div>
      <div id="tele-content"></div>
    </div>
  </div>

  <div id="sidebar">
    <div class="section-title">
      <span>Corridor Statistics</span>
      <span style="color:var(--cyan)">LIVE</span>
    </div>
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">Road Length</div>
        <div class="stat-val" id="stat-len">-- km</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Lane Length</div>
        <div class="stat-val" id="stat-lanes">-- km</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Vehicle Storage</div>
        <div class="stat-val" id="stat-storage">-- veh</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Reversible Edges</div>
        <div class="stat-val" id="stat-rev">--</div>
      </div>
    </div>

    <div class="contraflow-card">
      <div class="contraflow-title">
        <span>⚡</span> Dynamic Contraflow Mode
      </div>
      <div style="font-size:11px; color:var(--text-muted);">
        Flip Lindsey St & Jenkins Ave from balanced 2:2 into 4:0 / 3:1 Game Day evacuation pipelines.
      </div>
      <button class="toggle-btn" id="contraflow-toggle">Toggle Lindsey Contraflow</button>
    </div>

    <div class="section-title">
      <span>Game Day Ingress Corridors</span>
    </div>
    <div id="routes-container" style="display:flex; flex-direction:column; gap:6px;">
      <!-- Populated dynamically -->
    </div>

    <div class="section-title">
      <span>Legend</span>
    </div>
    <div style="font-size:11px; display:flex; flex-direction:column; gap:6px; color:var(--text-muted);">
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="width:16px; height:3px; background:#00f0ff; display:inline-block;"></span>
        <span>I-35 Freeway / Arterial Expressway</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="width:16px; height:3px; background:#ffb703; display:inline-block;"></span>
        <span>Lindsey St / Classen / Major Arterial</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="width:16px; height:2px; background:#8b9bb4; display:inline-block;"></span>
        <span>Campus & Residential Streets</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="width:10px; height:10px; border-radius:50%; background:#00f0ff; box-shadow:0 0 8px #00f0ff; display:inline-block;"></span>
        <span>External Ingress Gateways (OKC, Moore, Noble)</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="width:10px; height:10px; border-radius:50%; background:#06d6a0; box-shadow:0 0 8px #06d6a0; display:inline-block;"></span>
        <span>Stadium & Lloyd Noble Parking Sinks</span>
      </div>
    </div>
  </div>

  <script>
    const DATA = {json_blob};

    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {{
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      render();
    }});

    // Population of sidebar stats
    document.getElementById('stat-len').textContent = DATA.statistics.centerline_km + ' km';
    document.getElementById('stat-lanes').textContent = DATA.statistics.lane_km + ' km';
    document.getElementById('stat-storage').textContent = DATA.statistics.network_storage_capacity_veh.toLocaleString() + ' veh';
    document.getElementById('stat-rev').textContent = DATA.statistics.reversible_edge_count;

    // Camera view state
    let zoom = 0.085;
    let panX = width / 2;
    let panY = height / 2;
    let isDragging = false;
    let startX = 0, startY = 0;
    let activeRouteEdges = new Set();
    let contraflowActive = false;

    // Ingress route buttons
    const routesDiv = document.getElementById('routes-container');
    DATA.routes.forEach((r, idx) => {{
      const btn = document.createElement('button');
      btn.className = 'route-btn';
      btn.innerHTML = `
        <div>
          <div style="font-weight:600;">${{r.gateway_name.split('(')[0]}}</div>
          <div style="font-size:10px; color:var(--text-muted); font-family:'Space Mono';">to ${{r.target_name}} (${{r.distance_km}} km)</div>
        </div>
        <div style="font-family:'Space Mono'; font-weight:700; color:var(--amber);">${{r.free_flow_time_min}}m</div>
      `;
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.route-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeRouteEdges = new Set(r.edge_ids);
        render();
      }});
      routesDiv.appendChild(btn);
    }});

    // Contraflow button
    document.getElementById('contraflow-toggle').addEventListener('click', () => {{
      contraflowActive = !contraflowActive;
      const btn = document.getElementById('contraflow-toggle');
      btn.textContent = contraflowActive ? 'Contraflow ACTIVE (4:0 Outbound)' : 'Toggle Lindsey Contraflow';
      btn.style.background = contraflowActive ? '#06d6a0' : '#a8192d';
      render();
    }});

    // Canvas coordinate transforms
    function worldToScreen(x, y) {{
      return {{
        x: panX + x * zoom,
        y: panY - y * zoom // Flip y because screen y goes downward
      }};
    }}

    function screenToWorld(sx, sy) {{
      return {{
        x: (sx - panX) / zoom,
        y: (panY - sy) / zoom
      }};
    }}

    // Pan & Zoom controls
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
      checkHover(e.clientX, e.clientY);
    }});

    canvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      const mouseWorld = screenToWorld(e.clientX, e.clientY);
      zoom *= zoomFactor;
      panX = e.clientX - mouseWorld.x * zoom;
      panY = e.clientY + mouseWorld.y * zoom;
      render();
    }});

    function checkHover(mouseX, mouseY) {{
      const mWorld = screenToWorld(mouseX, mouseY);
      let closestEdge = null;
      let minDist = 30.0 / zoom; // 30 pixels threshold

      for (let edge of DATA.edges) {{
        if (!edge.geometry || edge.geometry.length < 2) continue;
        for (let i = 0; i < edge.geometry.length - 1; i++) {{
          const p1 = edge.geometry[i];
          const p2 = edge.geometry[i + 1];
          const d = distToSegment(mWorld.x, mWorld.y, p1[0], p1[1], p2[0], p2[1]);
          if (d < minDist) {{
            minDist = d;
            closestEdge = edge;
          }}
        }}
      }}

      const box = document.getElementById('telemetry-box');
      if (closestEdge) {{
        box.style.display = 'block';
        document.getElementById('tele-name').textContent = closestEdge.name;
        document.getElementById('tele-content').innerHTML = `
          <div><b style="color:var(--text-muted)">Type:</b> ${{closestEdge.highway_type}}</div>
          <div><b style="color:var(--text-muted)">Lanes:</b> ${{closestEdge.lanes}}</div>
          <div><b style="color:var(--text-muted)">Speed:</b> ${{closestEdge.free_speed_mph}} mph (${{closestEdge.free_speed_mps}} m/s)</div>
          <div><b style="color:var(--text-muted)">Length:</b> ${{closestEdge.length_m}} m</div>
          <div><b style="color:var(--text-muted)">Capacity:</b> ${{closestEdge.capacity_vph}} veh/hr</div>
          <div><b style="color:var(--text-muted)">Jam Storage:</b> ${{closestEdge.jam_storage_veh}} vehicles</div>
          <div><b style="color:var(--text-muted)">Contraflow:</b> <span style="color:${{closestEdge.is_reversible ? '#06d6a0' : '#8b9bb4'}}">${{closestEdge.is_reversible ? 'ELIGIBLE' : 'NO'}}</span></div>
        `;
      }} else {{
        box.style.display = 'none';
      }}
    }}

    function distToSegment(px, py, x1, y1, x2, y2) {{
      const l2 = (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1);
      if (l2 === 0) return Math.hypot(px - x1, py - y1);
      let t = ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / l2;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(px - (x1 + t*(x2 - x1)), py - (y1 + t*(y2 - y1)));
    }}

    function render() {{
      ctx.fillStyle = '#090b10';
      ctx.fillRect(0, 0, width, height);

      // 1. Draw Roads
      DATA.edges.forEach(edge => {{
        if (!edge.geometry || edge.geometry.length < 2) return;

        const isHighlighted = activeRouteEdges.has(edge.edge_id);
        const isContraflowCorr = contraflowActive && edge.is_reversible;

        ctx.beginPath();
        const p0 = worldToScreen(edge.geometry[0][0], edge.geometry[0][1]);
        ctx.moveTo(p0.x, p0.y);
        for (let i = 1; i < edge.geometry.length; i++) {{
          const pt = worldToScreen(edge.geometry[i][0], edge.geometry[i][1]);
          ctx.lineTo(pt.x, pt.y);
        }}

        if (isHighlighted) {{
          ctx.strokeStyle = '#e0243d';
          ctx.lineWidth = Math.max(3.5, 4.5 * (edge.lanes || 1) * zoom);
          ctx.shadowColor = '#e0243d';
          ctx.shadowBlur = 10;
        }} else if (isContraflowCorr) {{
          ctx.strokeStyle = '#06d6a0';
          ctx.lineWidth = Math.max(3, 4 * (edge.lanes || 1) * zoom);
          ctx.shadowColor = '#06d6a0';
          ctx.shadowBlur = 8;
        }} else if (edge.highway_type.includes('motorway')) {{
          ctx.strokeStyle = '#00f0ff';
          ctx.lineWidth = Math.max(2.2, 3.2 * zoom);
          ctx.shadowBlur = 0;
        }} else if (edge.highway_type.includes('trunk') || edge.highway_type.includes('primary')) {{
          ctx.strokeStyle = '#ffb703';
          ctx.lineWidth = Math.max(1.6, 2.4 * zoom);
          ctx.shadowBlur = 0;
        }} else {{
          ctx.strokeStyle = 'rgba(139, 155, 180, 0.45)';
          ctx.lineWidth = Math.max(0.8, 1.2 * zoom);
          ctx.shadowBlur = 0;
        }}
        ctx.stroke();
      }});
      ctx.shadowBlur = 0;

      // 2. Draw Gateways
      DATA.gateways.forEach(g => {{
        const pt = worldToScreen(
          (g.lon - -97.4423) * 111320 * Math.cos(35.2059 * Math.PI / 180),
          (g.lat - 35.2059) * 110950
        );

        ctx.fillStyle = '#00f0ff';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = '11px Space Mono';
        ctx.fillStyle = '#00f0ff';
        ctx.fillText(g.name.split('(')[0], pt.x + 12, pt.y + 4);
      }});

      // 3. Draw Sinks
      DATA.sinks.forEach(s => {{
        const pt = worldToScreen(
          (s.lon - -97.4423) * 111320 * Math.cos(35.2059 * Math.PI / 180),
          (s.lat - 35.2059) * 110950
        );

        ctx.fillStyle = '#06d6a0';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = '10px Outfit';
        ctx.fillStyle = '#f0f4fc';
        ctx.fillText(s.name.split('(')[0], pt.x + 10, pt.y + 3);
      }});
    }}

    render();
  </script>
</body>
</html>
"""
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[NetworkQC] Interactive visualizer written to: {html_filepath}")
