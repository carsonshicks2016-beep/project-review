"""
Simulate an active drift maneuver and capture a screenshot with skid marks,
tire smoke, and combo multiplier HUD.
"""

import numpy as np
from drift_rl.visualizer import DriftVisualizer

def generate_drift_screenshot():
    vis = DriftVisualizer(track_name="touge", headless=True)

    # 1. Drive forward with throttle to build speed (30 frames)
    for _ in range(30):
        vis.env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        vis.camera_pos = np.array([vis.env.state.x, vis.env.state.y])
        vis._update_effects(0.02)

    # 2. Initiate sharp flick and handbrake (15 frames)
    for _ in range(15):
        vis.env.step(np.array([0.8, 0.5, 1.0], dtype=np.float32))
        vis.camera_pos = np.array([vis.env.state.x, vis.env.state.y])
        vis._update_effects(0.02)

    # 3. Countersteer and power oversteer through corner with high throttle (40 frames)
    for _ in range(40):
        vis.env.step(np.array([-0.6, 1.0, 0.0], dtype=np.float32))
        vis.camera_pos = np.array([vis.env.state.x, vis.env.state.y])
        vis._update_effects(0.02)

    # Render and save frame
    vis._render()
    vis.save_frame("drift_action_screenshot.png")
    print("Action screenshot saved to drift_action_screenshot.png")

if __name__ == "__main__":
    generate_drift_screenshot()
