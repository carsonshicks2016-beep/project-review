"""
Drift simulation script with logging of steps and screenshot capture.
"""

import numpy as np
from drift_rl.visualizer import DriftVisualizer

def run_controlled_drift():
    vis = DriftVisualizer(track_name="gymkhana", headless=True) # Open arena, plenty of room to slide!
    obs, info = vis.env.reset(options={"spawn_idx": 0})
    vis.camera_pos = np.array([vis.env.state.x, vis.env.state.y])

    # 1. Accelerate forward (25 frames = 0.5s)
    for _ in range(25):
        obs, reward, term, trunc, info = vis.env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        vis.camera_pos += (np.array([vis.env.state.x, vis.env.state.y]) - vis.camera_pos) * 0.3
        vis._update_effects(0.02)

    # 2. Turn in hard with handbrake (35 frames = 0.7s)
    for _ in range(35):
        obs, reward, term, trunc, info = vis.env.step(np.array([1.0, 0.5, 1.0], dtype=np.float32))
        vis.camera_pos += (np.array([vis.env.state.x, vis.env.state.y]) - vis.camera_pos) * 0.3
        vis._update_effects(0.02)

    # 3. Countersteer and power throttle through slide (30 frames = 0.6s)
    for i in range(30):
        obs, reward, term, trunc, info = vis.env.step(np.array([-0.8, 1.0, 0.0], dtype=np.float32))
        vis.camera_pos += (np.array([vis.env.state.x, vis.env.state.y]) - vis.camera_pos) * 0.3
        vis._update_effects(0.02)
        if i % 10 == 0:
            print(f"Step {i} | Slip: {vis.env.state.slip_angle_deg:.1f}° | Speed: {vis.env.state.speed*3.6:.1f} km/h | Score: {info['score']:.0f} | Mult: x{info['multiplier']:.1f} | Smoke: {len(vis.smoke_particles)}")

    vis._render()
    vis.save_frame("gymkhana_drift_screenshot.png")
    print(f"Gymkhana drift screenshot saved! Final Score: {info['score']:.0f}, Multiplier: x{info['multiplier']:.1f}")

if __name__ == "__main__":
    run_controlled_drift()
