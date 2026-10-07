"""
Flapping Flight Command Center - Backend Server.
Zero-dependency HTTP server using standard library (http.server, threading, json).
Serves the web dashboard, real-time aerodynamic telemetry, and training orchestration APIs.
Port: 8775
"""
import os
import sys
import json
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add parent directory to path so we can import simulation modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import numpy as np

try:
    from config import InsectSpec, SimConfig
    from kinematics import KinematicEngine
    from aerodynamics import AerodynamicSolver
    from body import InsectBody
    from env import InsectFlightEnv
except ImportError as e:
    print(f"Warning importing simulation modules: {e}")

# Global state
SIM_STATE = {
    "running": True,
    "training": False,
    "train_iter": 0,
    "train_max_iters": 50,
    "train_loss": 0.0,
    "train_reward": 0.0,
    "manual_mode": False,
    "manual_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "telemetry": {}
}

spec = InsectSpec()
kinematics = KinematicEngine(spec)
aero = AerodynamicSolver(spec)
body = InsectBody(spec)

def simulation_background_loop():
    """Continuously runs the 250 Hz simulation loop and updates live telemetry."""
    sim_time = 0.0
    dt = 1.0 / 2500.0 # 2500 Hz internal physics
    steps_per_tick = 10 # 250 Hz output

    while SIM_STATE["running"]:
        t0 = time.time()
        cycle_lift = 0.0
        cycle_drag = 0.0
        cycle_rot = 0.0
        cycle_mass = 0.0
        cycle_thrust = 0.0

        action = SIM_STATE["manual_action"]
        pitch_L = action[0] * np.radians(25.0)
        pitch_R = action[1] * np.radians(25.0)
        stroke_L = action[2] * np.radians(15.0)
        stroke_R = action[3] * np.radians(15.0)
        elev_L = action[4] * np.radians(10.0)
        elev_R = action[5] * np.radians(10.0)

        for _ in range(steps_per_tick):
            sim_time += dt

            left_wing, right_wing = kinematics.compute_wing_angles(
                t=sim_time,
                bias_phi_L=stroke_L, bias_phi_R=stroke_R,
                pitch_offset_L=pitch_L, pitch_offset_R=pitch_R,
                elevation_offset_L=elev_L, elevation_offset_R=elev_R
            )

            fling_active = (kinematics.fling_boost_timer > 0)
            if fling_active:
                kinematics.fling_boost_timer -= dt

            F_L, M_L, tel_L = aero.compute_wing_wrench(left_wing, is_left=True, fling_active=fling_active)
            F_R, M_R, tel_R = aero.compute_wing_wrench(right_wing, is_left=False, fling_active=fling_active)

            cycle_lift += F_L[2] + F_R[2]
            cycle_drag += tel_L["drag_n"] + tel_R["drag_n"]
            cycle_rot += tel_L["rotational_n"] + tel_R["rotational_n"]
            cycle_mass += tel_L["added_mass_n"] + tel_R["added_mass_n"]
            cycle_thrust += F_L[0] + F_R[0]

            body.step(F_L, M_L, F_R, M_R, dt)

        # Update telemetry packet
        mean_lift = cycle_lift / steps_per_tick
        weight = spec.weight

        R = body.rotation_matrix()
        # Compute Euler angles (pitch, roll, yaw)
        pitch_deg = np.degrees(np.arctan2(R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
        roll_deg = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

        SIM_STATE["telemetry"] = {
            "time_s": round(sim_time, 4),
            "altitude_m": round(float(body.pos[2]), 4),
            "pos_x_m": round(float(body.pos[0]), 4),
            "pos_y_m": round(float(body.pos[1]), 4),
            "vel_z_mps": round(float(body.vel[2]), 4),
            "pitch_deg": round(float(pitch_deg), 2),
            "roll_deg": round(float(roll_deg), 2),
            "yaw_deg": round(float(yaw_deg), 2),
            "phi_L_deg": round(float(np.degrees(left_wing.phi)), 2),
            "phi_R_deg": round(float(np.degrees(right_wing.phi)), 2),
            "psi_L_deg": round(float(np.degrees(left_wing.psi)), 2),
            "psi_R_deg": round(float(np.degrees(right_wing.psi)), 2),
            "theta_L_deg": round(float(np.degrees(left_wing.theta)), 2),
            "theta_R_deg": round(float(np.degrees(right_wing.theta)), 2),
            "lift_mn": round(float(mean_lift * 1e3), 2),
            "weight_mn": round(float(weight * 1e3), 2),
            "lift_ratio": round(float(mean_lift / max(1e-6, weight)), 2),
            "thrust_mn": round(float((cycle_thrust / steps_per_tick) * 1e3), 2),
            "drag_mn": round(float((cycle_drag / steps_per_tick) * 1e3), 2),
            "rotational_mn": round(float((cycle_rot / steps_per_tick) * 1e3), 2),
            "added_mass_mn": round(float((cycle_mass / steps_per_tick) * 1e3), 2),
            "clap_and_fling": bool(fling_active),
            "frequency_hz": spec.nominal_frequency,
            "control_hz": 250.0
        }

        # Sleep to keep ~250 Hz loop pace (4 ms)
        elapsed = time.time() - t0
        time.sleep(max(0.001, (1.0 / 250.0) - elapsed))

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        static_dir = os.path.join(CURRENT_DIR, "static")
        super().__init__(*args, directory=static_dir, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self.send_json({
                "status": "online",
                "training": SIM_STATE["training"],
                "train_iter": SIM_STATE["train_iter"],
                "train_max_iters": SIM_STATE["train_max_iters"],
                "train_loss": SIM_STATE["train_loss"],
                "train_reward": SIM_STATE["train_reward"],
                "spec": {
                    "name": spec.name,
                    "mass_mg": spec.body_mass * 1e6,
                    "span_mm": spec.wing_length * 1e3,
                    "chord_mm": spec.mean_chord * 1e3,
                    "frequency_hz": spec.nominal_frequency,
                    "cl_max": spec.c_l_max,
                    "cd_max": spec.c_d_max,
                    "c_rot": spec.c_rot
                }
            })
        elif parsed.path == "/api/telemetry":
            self.send_json(SIM_STATE["telemetry"])
        elif parsed.path == "/api/checkpoints":
            ck_dir = os.path.join(PARENT_DIR, "checkpoints")
            ck_files = os.listdir(ck_dir) if os.path.exists(ck_dir) else []
            self.send_json({"checkpoints": ck_files})
        elif parsed.path == "/api/reset":
            body.reset()
            self.send_json({"status": "reset_success"})
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception:
            payload = {}

        if parsed.path == "/api/control":
            # Updates manual action offsets [-1, 1]^6
            act = payload.get("action", [0.0]*6)
            SIM_STATE["manual_action"] = [float(x) for x in act]
            self.send_json({"status": "action_updated"})
        elif parsed.path == "/api/train/start":
            iters = payload.get("iterations", 30)
            SIM_STATE["train_max_iters"] = iters
            SIM_STATE["training"] = True
            self.send_json({"status": "training_started", "iterations": iters})
        elif parsed.path == "/api/train/stop":
            SIM_STATE["training"] = False
            self.send_json({"status": "training_stopped"})
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

def run_server(port: int = 8775):
    # Launch simulation loop in background thread
    sim_thread = threading.Thread(target=simulation_background_loop, daemon=True)
    sim_thread.start()

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print("=" * 65)
    print(f"🪲 FLAPPING FLIGHT RL COMMAND CENTER RUNNING")
    print(f"URL: http://localhost:{port}")
    print(f"Control Rate: 250 Hz | Internal Physics: 2500 Hz")
    print("=" * 65)
    server.serve_forever()

if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8775
    run_server(p)
