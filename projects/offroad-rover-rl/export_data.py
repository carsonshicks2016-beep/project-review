"""
Exports terrain and checkpoint trajectory data into a single client-side JS bundle
(dashboard/static/rover_data.js) so the 3D WebGL viewer works 100% standalone
without requiring any local HTTP server or network sockets.
"""
import os
import json

def build_data_bundle():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(project_dir, "dashboard", "static")
    checkpoints_dir = os.path.join(project_dir, "checkpoints")

    terrain_path = os.path.join(static_dir, "terrain.json")
    with open(terrain_path, "r") as f:
        terrain_data = json.load(f)

    generations = {}
    for f in sorted(os.listdir(checkpoints_dir)):
        if f.startswith("gen_") and f.endswith(".json"):
            gen_key = f.replace(".json", "")
            with open(os.path.join(checkpoints_dir, f), "r") as cf:
                generations[gen_key] = json.load(cf)

    history_path = os.path.join(checkpoints_dir, "history.json")
    history_data = []
    if os.path.exists(history_path):
        with open(history_path, "r") as hf:
            history_data = json.load(hf)

    js_content = f"""// Standalone Telemetry & Terrain Bundle for Off-Road Rover
window.ROVER_TERRAIN_DATA = {json.dumps(terrain_data)};
window.ROVER_GENERATIONS = {json.dumps(generations)};
window.ROVER_HISTORY = {json.dumps(history_data)};
console.log("Loaded embedded simulation data: terrain + " + Object.keys(window.ROVER_GENERATIONS).length + " generations.");
"""

    out_path = os.path.join(static_dir, "rover_data.js")
    with open(out_path, "w") as out_f:
        out_f.write(js_content)

    print(f"[SUCCESS] Exported standalone simulation bundle to {out_path} ({os.path.getsize(out_path) // 1024} KB)")

if __name__ == "__main__":
    build_data_bundle()
