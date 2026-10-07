"""MJCF model for BLADE — two sword-and-shield humanoids on uneven terrain.

Each fighter is a ~18-DOF articulated humanoid (pelvis, spine, head, two arms,
two legs) holding a steel sword (welded to the right hand) and a wooden shield
(welded to the left forearm).  Weapons get realistic mass/inertia from geom
density, so swinging the sword carries momentum and the shield is a heavy guard.

The model is built programmatically so joints/actuators stay in sync, and the
terrain heightfield is filled with smooth low-amplitude noise at load time.
"""
from __future__ import annotations

import numpy as np
import mujoco

from . import weapons

# Per-fighter joints: (name, axis, range_deg, kp).  Order defines the action layout.
JOINTS = [
    ("abdomen_z", "0 0 1", (-45, 45), 120),
    ("abdomen_y", "0 1 0", (-30, 45), 120),
    ("r_hip_x",   "1 0 0", (-110, 45), 240),
    ("r_hip_z",   "0 0 1", (-45, 45), 140),
    ("r_hip_y",   "0 1 0", (-45, 20), 140),
    ("r_knee",    "0 -1 0", (-160, 2), 280),   # axis -y: negative q = human flexion (shin swings back)
    ("r_ankle",   "0 1 0", (-45, 45), 180),
    ("l_hip_x",   "1 0 0", (-110, 45), 240),
    ("l_hip_z",   "0 0 1", (-45, 45), 140),
    ("l_hip_y",   "0 1 0", (-20, 45), 140),
    ("l_knee",    "0 -1 0", (-160, 2), 280),
    ("l_ankle",   "0 1 0", (-45, 45), 180),
    ("r_shoulder1", "0 1 0", (-120, 120), 80),
    ("r_shoulder2", "1 0 0", (-120, 90), 80),
    ("r_elbow",   "0 1 0", (-150, 5), 70),
    ("l_shoulder1", "0 1 0", (-120, 120), 80),
    ("l_shoulder2", "1 0 0", (-90, 120), 80),
    ("l_elbow",   "0 1 0", (-150, 5), 70),
]
JOINT_NAMES = [j[0] for j in JOINTS]
N_JOINT = len(JOINTS)


def _fighter(p, x, y, yaw, loadout):
    """MJCF body tree for one fighter with name prefix `p` at (x,y) facing `yaw` deg."""
    def jt(name):
        axis, (lo, hi), _ = next((a, r, k) for n, a, r, k in JOINTS if n == name)
        return f'<joint name="{p}{name}" axis="{axis}" range="{lo} {hi}"/>'
    rh = weapons.right_hand_xml(p, loadout)
    sh = weapons.shield_xml(p, loadout)
    return f'''
    <body name="{p}pelvis" pos="{x} {y} 1.05" euler="0 0 {yaw}">
      <freejoint name="{p}root"/>
      <geom name="{p}pelvis" type="capsule" fromto="0 -0.08 0 0 0.08 0" size="0.085"/>
      <body name="{p}torso" pos="0 0 0.14">
        {jt("abdomen_z")}{jt("abdomen_y")}
        <geom name="{p}torso" type="capsule" fromto="0 -0.09 0 0 0.09 0" size="0.10"/>
        <geom name="{p}chest" type="capsule" fromto="0 -0.07 0.16 0 0.07 0.16" size="0.095"/>
        <body name="{p}head" pos="0 0 0.35">
          <geom name="{p}head" type="sphere" size="0.10"/>
        </body>
        <body name="{p}r_uarm" pos="0 -0.18 0.22">
          {jt("r_shoulder1")}{jt("r_shoulder2")}
          <geom name="{p}r_uarm" type="capsule" fromto="0 0 0 0 -0.02 -0.27" size="0.045"/>
          <body name="{p}r_larm" pos="0 -0.02 -0.27">
            {jt("r_elbow")}
            <geom name="{p}r_larm" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.038"/>
            <body name="{p}r_hand" pos="0 0 -0.27">
              <geom name="{p}r_hand" type="sphere" size="0.05"/>
              {rh}
            </body>
          </body>
        </body>
        <body name="{p}l_uarm" pos="0 0.18 0.22">
          {jt("l_shoulder1")}{jt("l_shoulder2")}
          <geom name="{p}l_uarm" type="capsule" fromto="0 0 0 0 0.02 -0.27" size="0.045"/>
          <body name="{p}l_larm" pos="0 0.02 -0.27">
            {jt("l_elbow")}
            <geom name="{p}l_larm" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.038"/>
            <body name="{p}l_hand" pos="0 0 -0.27">
              <geom name="{p}l_hand" type="sphere" size="0.05"/>
              {sh}
            </body>
          </body>
        </body>
      </body>
      <body name="{p}r_thigh" pos="0 -0.10 -0.04">
        {jt("r_hip_x")}{jt("r_hip_z")}{jt("r_hip_y")}
        <geom name="{p}r_thigh" type="capsule" fromto="0 0 0 0 0.01 -0.40" size="0.06"/>
        <body name="{p}r_shin" pos="0 0.01 -0.40">
          {jt("r_knee")}
          <geom name="{p}r_shin" type="capsule" fromto="0 0 0 0 0 -0.40" size="0.05"/>
          <body name="{p}r_foot" pos="0 0 -0.40">
            {jt("r_ankle")}
            <geom name="{p}r_foot" type="box" pos="0.035 0 -0.02" size="0.15 0.075 0.028"/>
          </body>
        </body>
      </body>
      <body name="{p}l_thigh" pos="0 0.10 -0.04">
        {jt("l_hip_x")}{jt("l_hip_z")}{jt("l_hip_y")}
        <geom name="{p}l_thigh" type="capsule" fromto="0 0 0 0 -0.01 -0.40" size="0.06"/>
        <body name="{p}l_shin" pos="0 -0.01 -0.40">
          {jt("l_knee")}
          <geom name="{p}l_shin" type="capsule" fromto="0 0 0 0 0 -0.40" size="0.05"/>
          <body name="{p}l_foot" pos="0 0 -0.40">
            {jt("l_ankle")}
            <geom name="{p}l_foot" type="box" pos="0.035 0 -0.02" size="0.15 0.075 0.028"/>
          </body>
        </body>
      </body>
    </body>'''


def _actuators(p):
    out = []
    for name, _axis, (lo, hi), kp in JOINTS:
        out.append(f'<position name="{p}{name}" joint="{p}{name}" kp="{kp}" '
                   f'ctrlrange="{np.deg2rad(lo):.3f} {np.deg2rad(hi):.3f}"/>')
    return "\n    ".join(out)


def build_xml(loadout="sword_shield", terrain=True, loadout_b=None):
    loadout_b = loadout_b or loadout
    floor = ('<geom name="floor" type="hfield" hfield="terrain" material="grid"/>' if terrain
             else '<geom name="floor" type="plane" size="0 0 .1" material="grid"/>')
    return f'''<mujoco model="blade">
  <option timestep="0.005" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.32 0.32 0.32" specular="0.1 0.1 0.1"/>
    <map znear="0.05"/>
  </visual>
  <default>
    <joint type="hinge" limited="true" damping="4" armature="0.02" stiffness="2"/>
    <geom condim="3" friction="1.0 0.05 0.01" density="1000" rgba="0.75 0.78 0.85 1"/>
    <position forcerange="-320 320"/>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.5 0.65 0.85" rgb2="0.08 0.10 0.18" width="64" height="64"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.22 0.30" rgb2="0.12 0.15 0.22" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <hfield name="terrain" nrow="80" ncol="80" size="6 6 0.18 0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="3 -3 5" dir="-0.5 0.5 -1" diffuse="0.4 0.4 0.5"/>
    <camera name="side" pos="0 -6 2.4" xyaxes="1 0 0 0 0.38 1"/>
    <camera name="hero" pos="4.5 -4.5 2.6" xyaxes="0.7 0.7 0 -0.28 0.28 1"/>
    {floor}
    {_fighter("a_", -1.6, 0, 0, loadout)}
    {_fighter("b_", 1.6, 0, 180, loadout_b)}
  </worldbody>
  <actuator>
    {_actuators("a_")}
    {_actuators("b_")}
  </actuator>
</mujoco>'''


def make_model(loadout="sword_shield", terrain=True, seed=0, bump=1.0, loadout_b=None):
    """Build the model and fill the heightfield with smooth low-amplitude noise.

    bump scales terrain roughness (0 = flat); terrain=False uses a clean plane.
    loadout_b lets fighter B carry a different armament (cross-loadout duels).
    """
    model = mujoco.MjModel.from_xml_string(build_xml(loadout, terrain, loadout_b))
    if terrain:
        rng = np.random.default_rng(seed)
        nr, nc = int(model.hfield_nrow[0]), int(model.hfield_ncol[0])
        h = rng.standard_normal((nr, nc))
        # smooth it into gentle rolling bumps
        for _ in range(6):
            h = (h + np.roll(h, 1, 0) + np.roll(h, -1, 0) +
                 np.roll(h, 1, 1) + np.roll(h, -1, 1)) / 5.0
        h -= h.min()
        h /= max(h.max(), 1e-6)
        # flatten a spawn strip so fighters start on near-level ground
        h *= bump
        model.hfield_data[:] = h.ravel()
    return model


if __name__ == "__main__":
    for lo in weapons.LOADOUT_NAMES:
        m = make_model(loadout=lo, terrain=False)
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_pelvis")
        wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_weapon")
        wmass = m.body_subtreemass[wid] if wid >= 0 else 0.0
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_shield")
        smass = m.body_subtreemass[sid] if sid >= 0 else 0.0
        print(f"{lo:13s} nu={m.nu:3d}  weapon={wmass:4.2f}kg  shield={smass:4.2f}kg  "
              f"fighter={m.body_subtreemass[pid]:.1f}kg")
