import os
import time
import threading
import numpy as np
import cv2
import mujoco
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

from olympus_mini.envs import make_env
from olympus_mini.policy import PPO
from olympus_mini.viewer import EpisodeRecorder

app = Flask(__name__, template_folder="templates", static_folder="static")

class LabSimManager:
    """Thread-safe simulator engine powering the live laboratory dashboard."""
    def __init__(self, task="sprint"):
        self.lock = threading.Lock()
        self.task = task
        self.env = make_env(task=self.task, max_steps=1000)
        self.obs, self.info = self.env.reset()
        
        self.is_paused = False
        self.control_mode = "auto"  # 'auto' or 'manual'
        self.camera_preset = "side"
        
        # Manual control registers
        self.manual_lean = 0.25      # Forward torso lean [-0.5, 0.8]
        self.manual_cadence = 1.6   # Stride frequency (Hz) [0.5, 3.5]
        self.manual_thrust = 0.5    # Leg push effort [0.0, 1.0]
        self.manual_jump = 0.0      # Jump impulse
        
        # PPO policy for auto mode
        self.ppo = PPO(obs_dim=self.env.observation_space.shape[0], act_dim=self.env.action_space.shape[0])
        # Try loading champion if available
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"champion_{self.task}.pt")
        if os.path.exists(ckpt_path):
            try:
                self.ppo.load_checkpoint(ckpt_path)
                print(f"[Lab] Loaded champion checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"[Lab] Could not load checkpoint: {e}")
                
        self.latest_frame_jpeg = None
        self.step_count = 0
        self.fps = 0.0
        qpos = self.env.data.qpos
        qvel = self.env.data.qvel
        self.latest_telemetry = {
            "task": self.task,
            "control_mode": self.control_mode,
            "is_paused": self.is_paused,
            "step": 0,
            "fps": 0.0,
            "vx": float(qvel[0]),
            "vy": float(qvel[1]),
            "dist": float(qpos[0]),
            "height": float(qpos[2]),
            "pitch": float(qpos[4]),
            "roll": float(qpos[5]),
            "phase": 0.0,
            "knee_angle": float(qpos[10]),
            "knee_vel": float(qvel[9]),
            "hip_angle": float(qpos[9]),
            "hip_vel": float(qvel[8]),
            "foot_contacts": [0.0] * 8,
            "hurdles_cleared": 0,
            "crossbar_cleared": False,
            "peak_z": float(qpos[2]),
            "manual_lean": self.manual_lean,
            "manual_cadence": self.manual_cadence,
            "manual_thrust": self.manual_thrust
        }
        
        # Clip recording buffer
        self.is_recording = False
        self.recorded_frames = []
        self.record_target_frames = 90
        
        # Camera angles
        self.cam_configs = {
            "side": {"distance": 3.8, "elevation": -10.0, "azimuth": 90.0},
            "orbit": {"distance": 4.5, "elevation": -18.0, "azimuth": 45.0},
            "chase": {"distance": 3.4, "elevation": 5.0, "azimuth": 180.0},
            "front": {"distance": 3.6, "elevation": -8.0, "azimuth": 0.0}
        }
        
        self.running = True
        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.sim_thread.start()

    def set_task(self, new_task):
        with self.lock:
            if new_task != self.task:
                self.task = new_task
                self.env.close()
                self.env = make_env(task=self.task, max_steps=1000)
                self.obs, self.info = self.env.reset()
                self.step_count = 0
                ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"champion_{self.task}.pt")
                if os.path.exists(ckpt_path):
                    try:
                        self.ppo.load_checkpoint(ckpt_path)
                    except:
                        pass

    def reset_episode(self):
        with self.lock:
            self.obs, self.info = self.env.reset()
            self.step_count = 0

    def _sim_loop(self):
        last_time = time.time()
        frame_counter = 0
        fps_timer = time.time()
        
        while self.running:
            start_step = time.time()
            with self.lock:
                if not self.is_paused:
                    # Decide action
                    if self.control_mode == "manual":
                        act = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
                        phase_sin = np.sin(2 * np.pi * self.env.phase)
                        # Modulate torso lean
                        act[0] = self.manual_lean
                        # Arm counter-swing
                        act[2] = 0.5 * phase_sin
                        act[3] = -0.5 * phase_sin
                        # Hip flexion modulated by user thrust & cadence
                        act[6] = self.manual_thrust * phase_sin
                        act[12] = -self.manual_thrust * phase_sin
                        # Knee actuation
                        act[7] = -0.3 + 0.3 * max(0, -phase_sin) + self.manual_jump
                        act[13] = -0.3 + 0.3 * max(0, phase_sin) + self.manual_jump
                        action = act
                    else:
                        action, _, _ = self.ppo.select_action(self.obs, deterministic=True)
                        
                    self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                    self.step_count += 1
                    
                    if terminated or truncated:
                        self.obs, self.info = self.env.reset()
                        self.step_count = 0
                        
                # Render frame
                frame_rgb = self._render_current_view()
                
                # Check clip recording
                if self.is_recording:
                    self.recorded_frames.append(frame_rgb)
                    if len(self.recorded_frames) >= self.record_target_frames:
                        self.is_recording = False
                        
                # Encode to JPEG for MJPEG stream
                ret, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    self.latest_frame_jpeg = jpeg.tobytes()
                    
                # Update telemetry
                qpos = self.env.data.qpos
                qvel = self.env.data.qvel
                contacts = self.env._get_foot_contacts()
                
                self.latest_telemetry = {
                    "task": self.task,
                    "control_mode": self.control_mode,
                    "is_paused": self.is_paused,
                    "step": self.step_count,
                    "fps": round(self.fps, 1),
                    "vx": float(qvel[0]),
                    "vy": float(qvel[1]),
                    "dist": float(qpos[0]),
                    "height": float(qpos[2]),
                    "pitch": float(qpos[4]),
                    "roll": float(qpos[5]),
                    "phase": float(self.env.phase),
                    "knee_angle": float(qpos[10]),
                    "knee_vel": float(qvel[9]),
                    "hip_angle": float(qpos[9]),
                    "hip_vel": float(qvel[8]),
                    "foot_contacts": contacts.tolist(),
                    "hurdles_cleared": int(self.info.get("hurdles_cleared", 0)),
                    "crossbar_cleared": bool(self.info.get("crossbar_cleared", False)),
                    "peak_z": float(self.info.get("peak_z", qpos[2])),
                    "manual_lean": self.manual_lean,
                    "manual_cadence": self.manual_cadence,
                    "manual_thrust": self.manual_thrust
                }
                
            frame_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.fps = frame_counter / (time.time() - fps_timer)
                frame_counter = 0
                fps_timer = time.time()
                
            elapsed = time.time() - start_step
            # Target ~35-40 FPS playback loop
            sleep_time = max(0.001, (1.0 / 35.0) - elapsed)
            time.sleep(sleep_time)

    def _render_current_view(self):
        if self.env.renderer is None:
            self.env.renderer = mujoco.Renderer(self.env.model, height=480, width=640)
            
        cfg = self.cam_configs.get(self.camera_preset, self.cam_configs["side"])
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.env.pelvis_body_id
        cam.distance = cfg["distance"]
        cam.elevation = cfg["elevation"]
        cam.azimuth = cfg["azimuth"]
        
        self.env.renderer.update_scene(self.env.data, camera=cam)
        return self.env.renderer.render()


class LabTrainingManager:
    """Manages background PPO training triggered from the lab UI."""
    def __init__(self, sim_mgr):
        self.sim_mgr = sim_mgr
        self.is_training = False
        self.thread = None
        self.stats = {
            "is_training": False,
            "iteration": 0,
            "step": 0,
            "total_steps": 20000,
            "mean_reward": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "history": []
        }

    def start_training(self, task="sprint", total_steps=20000, lr=3e-4, epochs=8):
        if self.is_training:
            return False
        self.is_training = True
        self.stats["is_training"] = True
        self.stats["total_steps"] = total_steps
        self.stats["history"] = []
        self.thread = threading.Thread(target=self._train_worker, args=(task, total_steps, lr, epochs), daemon=True)
        self.thread.start()
        return True

    def stop_training(self):
        self.is_training = False
        self.stats["is_training"] = False

    def _train_worker(self, task, total_steps, lr, epochs):
        print(f"[Lab Trainer] Starting background PPO for task '{task}', {total_steps} steps...")
        train_env = make_env(task=task)
        obs_dim = train_env.observation_space.shape[0]
        act_dim = train_env.action_space.shape[0]
        device = "mps" if np.bool_(os.uname().sysname == "Darwin") else "cpu"
        trainer_ppo = PPO(obs_dim=obs_dim, act_dim=act_dim, lr=lr, device=device)
        
        step_count = 0
        iteration = 0
        rollout_steps = 1024
        
        while self.is_training and step_count < total_steps:
            iteration += 1
            from olympus_mini.train import collect_rollouts
            rollouts, roll_stats = collect_rollouts(train_env, trainer_ppo, num_steps=rollout_steps)
            step_count += rollout_steps
            
            update_info = trainer_ppo.update(rollouts, epochs=epochs, batch_size=64)
            
            m_rew = float(roll_stats["mean_ep_reward"])
            p_loss = float(update_info["policy_loss"])
            v_loss = float(update_info["value_loss"])
            
            self.stats["iteration"] = iteration
            self.stats["step"] = step_count
            self.stats["mean_reward"] = round(m_rew, 2)
            self.stats["policy_loss"] = round(p_loss, 4)
            self.stats["value_loss"] = round(v_loss, 4)
            self.stats["history"].append({
                "step": step_count,
                "reward": round(m_rew, 2),
                "policy_loss": round(p_loss, 4),
                "value_loss": round(v_loss, 4)
            })
            
            # Auto checkpoint champion
            ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"champion_{task}.pt")
            trainer_ppo.save_checkpoint(ckpt_path)
            
            # Hot-reload into live viewer if matching active task
            with self.sim_mgr.lock:
                if self.sim_mgr.task == task:
                    self.sim_mgr.ppo.load_checkpoint(ckpt_path)
                    
        self.is_training = False
        self.stats["is_training"] = False
        train_env.close()
        print(f"[Lab Trainer] Background training finished.")

# Instantiate global managers
sim_manager = LabSimManager(task="sprint")
train_manager = LabTrainingManager(sim_manager)

@app.route("/")
def index():
    return render_template("index.html")

def generate_mjpeg():
    while True:
        if sim_manager.latest_frame_jpeg:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + sim_manager.latest_frame_jpeg + b'\r\n')
        time.sleep(0.025)

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/telemetry")
def get_telemetry():
    data = sim_manager.latest_telemetry.copy()
    data["training"] = train_manager.stats
    return jsonify(data)

@app.route("/api/control", methods=["POST"])
def post_control():
    body = request.json or {}
    cmd = body.get("command")
    
    if cmd == "pause":
        sim_manager.is_paused = True
    elif cmd == "play":
        sim_manager.is_paused = False
    elif cmd == "toggle_pause":
        sim_manager.is_paused = not sim_manager.is_paused
    elif cmd == "reset":
        sim_manager.reset_episode()
    elif cmd == "set_task":
        task = body.get("task", "sprint")
        sim_manager.set_task(task)
    elif cmd == "set_camera":
        cam = body.get("camera", "side")
        sim_manager.camera_preset = cam
    elif cmd == "set_mode":
        mode = body.get("mode", "auto")
        sim_manager.control_mode = mode
        
    return jsonify({"status": "ok", "command": cmd})

@app.route("/api/manual_action", methods=["POST"])
def post_manual_action():
    body = request.json or {}
    sim_manager.control_mode = "manual"
    
    if "lean" in body:
        sim_manager.manual_lean = float(np.clip(body["lean"], -0.4, 0.8))
    if "cadence" in body:
        sim_manager.manual_cadence = float(np.clip(body["cadence"], 0.5, 4.0))
    if "thrust" in body:
        sim_manager.manual_thrust = float(np.clip(body["thrust"], 0.1, 1.0))
    if "jump" in body:
        sim_manager.manual_jump = float(np.clip(body["jump"], 0.0, 0.6))
        
    return jsonify({"status": "ok"})

@app.route("/api/train/start", methods=["POST"])
def train_start():
    body = request.json or {}
    task = body.get("task", sim_manager.task)
    steps = int(body.get("steps", 20000))
    lr = float(body.get("lr", 3e-4))
    epochs = int(body.get("epochs", 8))
    success = train_manager.start_training(task=task, total_steps=steps, lr=lr, epochs=epochs)
    return jsonify({"started": success})

@app.route("/api/train/stop", methods=["POST"])
def train_stop():
    train_manager.stop_training()
    return jsonify({"status": "stopped"})

@app.route("/api/record_clip", methods=["POST"])
def record_clip():
    with sim_manager.lock:
        sim_manager.recorded_frames = []
        sim_manager.is_recording = True
        
    # Wait for recording to complete in background
    def save_worker():
        while sim_manager.is_recording:
            time.sleep(0.1)
        if sim_manager.recorded_frames:
            import imageio
            out_dir = os.path.join(os.path.dirname(__file__), "recordings")
            os.makedirs(out_dir, exist_ok=True)
            filename = f"clip_{sim_manager.task}_{int(time.time())}.gif"
            filepath = os.path.join(out_dir, filename)
            imageio.mimsave(filepath, sim_manager.recorded_frames, fps=30, loop=0)
            print(f"[Lab] Highlight clip saved: {filepath}")
            
    threading.Thread(target=save_worker, daemon=True).start()
    return jsonify({"status": "recording_started", "duration_frames": sim_manager.record_target_frames})

@app.route("/recordings/<path:filename>")
def get_recording(filename):
    rec_dir = os.path.join(os.path.dirname(__file__), "recordings")
    return send_from_directory(rec_dir, filename)

def run_server(host="127.0.0.1", port=5050):
    print(f"==================================================")
    print(f"  OLYMPUS MINI LABORATORY DASHBOARD")
    print(f"  Live at: http://{host}:{port}")
    print(f"==================================================")
    app.run(host=host, port=port, threaded=True, debug=False)

if __name__ == "__main__":
    run_server()
