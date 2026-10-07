"""
Research-Grade Off-Road Rover Telemetry & Neuroevolution Dashboard Server.
Provides complete REST APIs for real-time 3D simulation telemetry, continuous background
training orchestration, hyperparameter tuning, population genome inspection, and checkpoint logging.
Default port: 8780.
"""
import os
import sys
import json
import time
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Set project paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
STATIC_DIR = os.path.join(CURRENT_DIR, "static")
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, "checkpoints")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from terrain import ProceduralTerrain
from vehicle import RoverVehicle
from sensors import RoverSensorSuite
from env import RoverTerrainEnv
from neuroevolution import NeuroevolutionEngine, RoverPolicy

# Global Orchestration State
GLOBAL_STATE = {
    "env": None,
    "manual_vehicle": None,
    "manual_sensors": None,
    "engine": None,
    "training_active": False,
    "stop_requested": False,
    "train_thread": None,
    "hyperparams": {
        "pop_size": 40,
        "mutation_rate": 0.12,
        "mutation_strength": 0.25,
        "elitism_count": 6,
        "tournament_size": 4,
        "seed": 42
    },
    "latest_population": []
}

def init_global_state():
    seed = GLOBAL_STATE["hyperparams"]["seed"]
    terrain = ProceduralTerrain(seed=seed)
    GLOBAL_STATE["manual_vehicle"] = RoverVehicle(terrain, initial_pos=[2.0, 0.0])
    GLOBAL_STATE["manual_sensors"] = RoverSensorSuite(terrain)
    GLOBAL_STATE["env"] = RoverTerrainEnv(seed=seed)
    GLOBAL_STATE["engine"] = NeuroevolutionEngine(
        pop_size=GLOBAL_STATE["hyperparams"]["pop_size"],
        seed=seed
    )

init_global_state()


class RoverDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        # Suppress routine GET logging in terminal
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            engine = GLOBAL_STATE["engine"]
            self.send_json({
                "status": "online",
                "training": GLOBAL_STATE["training_active"],
                "generation": engine.generation if engine else 0,
                "hyperparams": GLOBAL_STATE["hyperparams"],
                "timestamp": time.time()
            })

        elif path == "/api/terrain":
            terrain_file = os.path.join(STATIC_DIR, "terrain.json")
            if os.path.exists(terrain_file):
                with open(terrain_file, "r") as f:
                    self.send_json_raw(f.read())
            else:
                terrain = ProceduralTerrain(seed=GLOBAL_STATE["hyperparams"]["seed"])
                t_data = terrain.export_terrain_data(nx=120, ny=40)
                self.send_json(t_data)

        elif path == "/api/history":
            hist_file = os.path.join(CHECKPOINTS_DIR, "history.json")
            if os.path.exists(hist_file):
                with open(hist_file, "r") as f:
                    self.send_json_raw(f.read())
            else:
                engine = GLOBAL_STATE["engine"]
                self.send_json(engine.history if engine else [])

        elif path == "/api/generations":
            ckpts = []
            if os.path.exists(CHECKPOINTS_DIR):
                for f in sorted(os.listdir(CHECKPOINTS_DIR)):
                    if f.startswith("gen_") and f.endswith(".json"):
                        p = os.path.join(CHECKPOINTS_DIR, f)
                        try:
                            with open(p, "r") as cf:
                                data = json.load(cf)
                                ckpts.append({
                                    "filename": f,
                                    "stats": data.get("stats", {})
                                })
                        except Exception:
                            pass
            self.send_json(ckpts)

        elif path.startswith("/api/generation/"):
            gen_id = path.split("/")[-1]
            if not gen_id.endswith(".json"):
                try:
                    gen_num = int(gen_id)
                    gen_id = f"gen_{gen_num:03d}.json"
                except ValueError:
                    gen_id = f"gen_{gen_id}.json"
            ckpt_path = os.path.join(CHECKPOINTS_DIR, gen_id)
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    self.send_json_raw(f.read())
            else:
                self.send_error(404, f"Checkpoint {gen_id} not found")

        elif path == "/api/population":
            self.send_json(GLOBAL_STATE["latest_population"])

        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            params = json.loads(body)
        except Exception:
            params = {}

        if path == "/api/train_start":
            if GLOBAL_STATE["training_active"]:
                self.send_json({"status": "already_running"})
                return

            GLOBAL_STATE["training_active"] = True
            GLOBAL_STATE["stop_requested"] = False

            def continuous_trainer():
                print("[TRAINING] Continuous background neuroevolution loop started.")
                while GLOBAL_STATE["training_active"] and not GLOBAL_STATE["stop_requested"]:
                    try:
                        stats, champ = GLOBAL_STATE["engine"].evolve_generation(
                            GLOBAL_STATE["env"], save_dir=CHECKPOINTS_DIR
                        )
                        # Re-export data bundle if export_data module exists
                        try:
                            import export_data
                            export_data.build_data_bundle()
                        except Exception:
                            pass
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"[ERROR] Training exception: {e}")
                        break
                GLOBAL_STATE["training_active"] = False
                print("[TRAINING] Continuous background loop stopped.")

            t = threading.Thread(target=continuous_trainer, daemon=True)
            GLOBAL_STATE["train_thread"] = t
            t.start()
            self.send_json({"status": "started"})

        elif path == "/api/train_pause":
            GLOBAL_STATE["training_active"] = False
            GLOBAL_STATE["stop_requested"] = True
            self.send_json({"status": "paused"})

        elif path == "/api/train_step":
            if GLOBAL_STATE["training_active"]:
                self.send_json({"status": "busy", "message": "Continuous training currently running."})
                return

            try:
                stats, champ = GLOBAL_STATE["engine"].evolve_generation(
                    GLOBAL_STATE["env"], save_dir=CHECKPOINTS_DIR
                )
                try:
                    import export_data
                    export_data.build_data_bundle()
                except Exception:
                    pass
                self.send_json({"status": "success", "stats": stats})
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)})

        elif path == "/api/hyperparams":
            # Update hyperparameter configuration
            hp = GLOBAL_STATE["hyperparams"]
            for k in ["mutation_rate", "mutation_strength", "elitism_count", "tournament_size"]:
                if k in params:
                    hp[k] = float(params[k])
            self.send_json({"status": "updated", "hyperparams": hp})

        elif path == "/api/manual_reset":
            v = GLOBAL_STATE["manual_vehicle"]
            v.reset(initial_pos=[2.0, 0.0])
            telem = v.get_telemetry()
            s = GLOBAL_STATE["manual_sensors"]
            dists, endpoints, hits, normals = s.scan(v)
            telem["rays"] = {
                "distances": [round(float(d), 3) for d in dists],
                "endpoints": [[round(float(c), 2) for c in pt] for pt in endpoints],
                "hits": [bool(h) for h in hits]
            }
            self.send_json(telem)

        elif path == "/api/manual_step":
            steer = float(params.get("steer", 0.0))
            throttle = float(params.get("throttle", 0.0))
            v = GLOBAL_STATE["manual_vehicle"]
            for _ in range(5):
                v.step(steer, throttle, dt=0.01)

            telem = v.get_telemetry()
            s = GLOBAL_STATE["manual_sensors"]
            dists, endpoints, hits, normals = s.scan(v)
            telem["rays"] = {
                "distances": [round(float(d), 3) for d in dists],
                "endpoints": [[round(float(c), 2) for c in pt] for pt in endpoints],
                "hits": [bool(h) for h in hits]
            }
            self.send_json(telem)

        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, data):
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json_raw(self, json_str):
        content = json_str.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(port=8780):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, RoverDashboardHandler)
    print(f"==================================================================")
    print(f"  OFF-ROAD ROVER RESEARCH COMMAND CENTER RUNNING                 ")
    print(f"  URL: http://127.0.0.1:{port}/index.html                         ")
    print(f"==================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    port = 8780
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
