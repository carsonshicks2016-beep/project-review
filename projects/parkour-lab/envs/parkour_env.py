"""Custom parkour humanoid: procedural terrain, body-relative height sensing, obstacle clearance bonuses.

Integrates the stock MuJoCo Humanoid body with procedurally generated terrain courses.
Features 18-dimensional downward raycast height scanning appended to the standard 348-dim
proprioceptive state vector, forming a 366-dim observation space.
"""
import os
import tempfile
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from typing import Optional

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

try:
    from . import terrain
except ImportError:
    from envs import terrain

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.5,
    "lookat": np.array((0.0, 0.0, 1.5)),
    "elevation": -18.0,
}

# Height-scan grid: 6 forward distances × 3 lateral offsets = 18 rays
SCAN_FORWARD = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
SCAN_LATERAL = np.array([-0.3, 0.0, 0.3])
N_HEIGHT_SAMPLES = len(SCAN_FORWARD) * len(SCAN_LATERAL)  # 18
ENV_VERSION = "parkour-v2"

# Geom group mask for raycasting: group 2 only (terrain + death plane)
TERRAIN_GEOM_GROUP = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)


def _mass_center(model, data):
    """Weighted center of mass (x, y only)."""
    num = np.einsum("b,bj->j", model.body_mass, data.xipos)
    denom = model.body_mass.sum()
    return (num / denom)[0:2].copy()


def _build_mjcf_template(terrain_xml: str) -> str:
    """Builds complete MJCF XML string with the humanoid body and terrain geoms."""
    return f"""\
<mujoco model="parkour_humanoid">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="1" condim="3" friction="1.2 .1 .1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <option integrator="RK4" timestep="0.003"/>
    <visual>
        <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
        <map force="0.1" znear="0.05" zfar="100.0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <quality shadowsize="2048"/>
        <global offwidth="640" offheight="480"/>
    </visual>
    <asset>
        <texture builtin="gradient" height="100" rgb1=".55 .70 .85" rgb2=".25 .40 .55" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0.2 0.25 0.3" rgb2="0.3 0.35 0.4" type="2d" width="100"/>
        <texture name="texterrain" builtin="flat" rgb1="0.85 0.85 0.88" rgb2="0.75 0.75 0.78" type="cube" width="127" height="127"/>
        <material name="MatPlane" reflectance="0.2" shininess="0.5" specular="0.5" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
        <material name="terrain_mat" texture="texterrain" texuniform="true" rgba="0.85 0.85 0.88 1"/>
    </asset>
    <worldbody>
        <light directional="true" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3" pos="0 0 8" dir="0 0 -1"/>
        <light directional="true" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" pos="10 -10 10" dir="-0.3 0.5 -1"/>
        <!-- Death plane: gap / fall detection (in group 2 for raycast) -->
        <geom name="death_plane" type="plane" pos="0 0 -2" size="200 200 0.1" group="2" rgba="0.3 0.35 0.4 1" material="MatPlane"/>
        <!-- Procedural terrain geoms directly in worldbody (group 2) -->
{terrain_xml}
        <!-- Humanoid body hierarchy (13 bodies, identical to Humanoid-v5) -->
        <body name="torso" pos="2.5 0 1.35">
            <camera name="track" mode="trackcom" pos="0 -4.5 0" xyaxes="1 0 0 0 0 1"/>
            <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>
            <geom fromto="0 -.07 0 0 .07 0" name="torso1" size="0.07" type="capsule"/>
            <geom name="head" pos="0 0 .19" size=".09" type="sphere" user="258"/>
            <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist" size="0.06" type="capsule"/>
            <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
                <geom fromto="0 -.06 0 0 .06 0" name="lwaist" size="0.06" type="capsule"/>
                <joint armature="0.02" axis="0 0 1" damping="5" name="abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
                <joint armature="0.02" axis="0 1 0" damping="5" name="abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
                <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
                    <joint armature="0.02" axis="1 0 0" damping="5" name="abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
                    <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt" size="0.09" type="capsule"/>
                    <body name="right_thigh" pos="0 -0.1 -0.04">
                        <joint armature="0.01" axis="1 0 0" damping="5" name="right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 1" damping="5" name="right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.0080" axis="0 1 0" damping="5" name="right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1" size="0.06" type="capsule"/>
                        <body name="right_shin" pos="0 0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="right_shin1" size="0.049" type="capsule"/>
                            <body name="right_foot" pos="0 0 -0.45">
                                <geom name="right_foot" pos="0 0 0.1" size="0.075" type="sphere" user="0"/>
                            </body>
                        </body>
                    </body>
                    <body name="left_thigh" pos="0 0.1 -0.04">
                        <joint armature="0.01" axis="-1 0 0" damping="5" name="left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 -1" damping="5" name="left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 1 0" damping="5" name="left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size="0.06" type="capsule"/>
                        <body name="left_shin" pos="0 -0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size="0.049" type="capsule"/>
                            <body name="left_foot" pos="0 0 -0.45">
                                <geom name="left_foot" type="sphere" size="0.075" pos="0 0 0.1" user="0"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="right_upper_arm" pos="0 -0.17 0.06">
                <joint armature="0.0068" axis="2 1 1" name="right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 -1 1" name="right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1" size="0.04 0.16" type="capsule"/>
                <body name="right_lower_arm" pos=".18 -.18 -.18">
                    <joint armature="0.0028" axis="0 -1 1" name="right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="right_larm" size="0.031" type="capsule"/>
                    <geom name="right_hand" pos=".18 .18 .18" size="0.04" type="sphere"/>
                    <camera pos="0 0 0"/>
                </body>
            </body>
            <body name="left_upper_arm" pos="0 0.17 0.06">
                <joint armature="0.0068" axis="2 -1 1" name="left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 1 1" name="left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1" size="0.04 0.16" type="capsule"/>
                <body name="left_lower_arm" pos=".18 .18 -.18">
                    <joint armature="0.0028" axis="0 -1 -1" name="left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="left_larm" size="0.031" type="capsule"/>
                    <geom name="left_hand" pos=".18 -.18 .18" size="0.04" type="sphere"/>
                </body>
            </body>
        </body>
    </worldbody>
    <tendon>
        <fixed name="left_hipknee">
            <joint coef="-1" joint="left_hip_y"/>
            <joint coef="1" joint="left_knee"/>
        </fixed>
        <fixed name="right_hipknee">
            <joint coef="-1" joint="right_hip_y"/>
            <joint coef="1" joint="right_knee"/>
        </fixed>
    </tendon>
    <actuator>
        <motor gear="100" joint="abdomen_y" name="abdomen_y"/>
        <motor gear="100" joint="abdomen_z" name="abdomen_z"/>
        <motor gear="100" joint="abdomen_x" name="abdomen_x"/>
        <motor gear="100" joint="right_hip_x" name="right_hip_x"/>
        <motor gear="100" joint="right_hip_z" name="right_hip_z"/>
        <motor gear="300" joint="right_hip_y" name="right_hip_y"/>
        <motor gear="200" joint="right_knee" name="right_knee"/>
        <motor gear="100" joint="left_hip_x" name="left_hip_x"/>
        <motor gear="100" joint="left_hip_z" name="left_hip_z"/>
        <motor gear="300" joint="left_hip_y" name="left_hip_y"/>
        <motor gear="200" joint="left_knee" name="left_knee"/>
        <motor gear="25" joint="right_shoulder1" name="right_shoulder1"/>
        <motor gear="25" joint="right_shoulder2" name="right_shoulder2"/>
        <motor gear="25" joint="right_elbow" name="right_elbow"/>
        <motor gear="25" joint="left_shoulder1" name="left_shoulder1"/>
        <motor gear="25" joint="left_shoulder2" name="left_shoulder2"/>
        <motor gear="25" joint="left_elbow" name="left_elbow"/>
    </actuator>
</mujoco>
"""


def _segments_to_geom_xml(segments) -> str:
    """Converts a list of Segments into MuJoCo geom tags with group=2 and clear obstacle colors."""
    lines = []
    for i, s in enumerate(segments):
        for j, g in enumerate(s.geom_specs):
            pos_str = " ".join(f"{x:.4f}" for x in g["pos"])
            size_str = " ".join(f"{x:.4f}" for x in g["size"])
            if s.kind == "low_wall":
                rgba = "0.95 0.45 0.15 1" if j > 0 else "0.85 0.85 0.88 1"
            elif s.kind == "boxes":
                rgba = "0.20 0.70 0.95 1" if j > 0 else "0.85 0.85 0.88 1"
            elif "stairs" in s.kind:
                rgba = "0.88 0.80 0.68 1"
            else:
                rgba = "0.85 0.85 0.88 1"
            lines.append(f'        <geom name="t_{i}_{j}" type="box" pos="{pos_str}" size="{size_str}" group="2" rgba="{rgba}"/>')
    return "\n".join(lines)


class ParkourHumanoidEnv(MujocoEnv, utils.EzPickle):
    """Humanoid obstacle course navigation with procedural terrain and height scanning.

    Observation Space (366 dims):
      - 348 standard proprioceptive features (identical to Humanoid-v5)
      - 18 body-relative height-above-ground raycast samples
    Action Space (17 dims):
      - Joint torques in [-0.4, 0.4]
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-7,
        contact_cost_range: tuple = (-np.inf, 10.0),
        healthy_reward: float = 5.0,
        upright_reward_weight: float = 0.5,
        facing_reward_weight: float = 2.0,
        obstacle_bonus: float = 50.0,
        terminate_when_unhealthy: bool = True,
        terminate_when_backward: bool = False,
        healthy_z_range: tuple = (0.75, 2.0),
        reset_noise_scale: float = 5e-3,
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        difficulty: float = 1.0,
        replay_prob: float = 0.0,
        max_episode_steps: int = 1000,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, render_mode, width, height, forward_reward_weight,
            ctrl_cost_weight, contact_cost_weight, contact_cost_range,
            healthy_reward, upright_reward_weight, facing_reward_weight,
            obstacle_bonus, terminate_when_unhealthy, terminate_when_backward,
            healthy_z_range, reset_noise_scale,
            frame_skip, default_camera_config, difficulty, replay_prob,
            max_episode_steps, **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._upright_reward_weight = upright_reward_weight
        self._facing_reward_weight = facing_reward_weight
        self._obstacle_bonus = obstacle_bonus
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._terminate_when_backward = terminate_when_backward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self.difficulty = float(difficulty)
        self.replay_prob = float(replay_prob)
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0
        self.effective_difficulty = self.difficulty
        self._episode_difficulty = self.difficulty

        # Generate initial course and write temporary XML file
        init_rng = np.random.default_rng(42)
        self._segments, self._course_length = terrain.generate_course(
            init_rng, difficulty=self.difficulty, replay_prob=self.replay_prob
        )
        terrain_xml = _segments_to_geom_xml(self._segments)
        xml_content = _build_mjcf_template(terrain_xml)
        self._current_xml = xml_content

        self._tmpdir = tempfile.mkdtemp(prefix="parkour_")
        self._xml_path = os.path.join(self._tmpdir, "parkour_humanoid.xml")
        with open(self._xml_path, "w") as f:
            f.write(xml_content)

        MujocoEnv.__init__(
            self,
            self._xml_path,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            render_mode=render_mode,
            width=width,
            height=height,
            **kwargs,
        )

        self.metadata = {**self.metadata, "render_fps": int(np.round(1.0 / self.dt))}

        # 13 humanoid bodies (world is index 0)
        n_humanoid_bodies = self.model.nbody - 1
        obs_size = (
            self.data.qpos.size - 2
            + self.data.qvel.size
            + n_humanoid_bodies * 10
            + n_humanoid_bodies * 6
            + (self.data.qvel.size - 6)
            + n_humanoid_bodies * 6
            + N_HEIGHT_SAMPLES
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self._cache_body_ids()
        self._cleared_segments = set()
        self._course_completed = False

    def _cache_body_ids(self):
        """Caches critical body IDs from the compiled model."""
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._feet_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_foot", "right_foot")
        ]
        self._death_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "death_plane")

    def set_difficulty(self, difficulty: float):
        """Set the next reset's level; an in-flight course keeps its original label."""
        terrain.sample_difficulty(np.random.default_rng(0), float(difficulty), 0.0)
        self.difficulty = float(difficulty)

    def get_contract(self):
        """Serializable physical/observation contract saved alongside each policy."""
        return {
            "version": ENV_VERSION,
            "terrain_version": terrain.TERRAIN_VERSION,
            "observation_layout": "humanoid-v5-348+heading-height-rays-18-v1",
            "observation_shape": list(self.observation_space.shape),
            "action_shape": list(self.action_space.shape),
            "action_layout": "humanoid-v5-motor-order-17",
            "action_bounds": [-0.4, 0.4],
            "control_dt": self.dt,
            "max_episode_steps": self.max_episode_steps,
            "success": "healthy, both feet past finish platform start, terrain-supported landing",
            "clearance": "non-flat segment traversed with both feet past its end and supported landing; boxes may be bypassed",
            "reward_weights": {
                "forward": self._forward_reward_weight, "control": self._ctrl_cost_weight,
                "contact": self._contact_cost_weight, "healthy": self._healthy_reward,
                "upright": self._upright_reward_weight, "facing": self._facing_reward_weight,
                "obstacle": self._obstacle_bonus,
            },
        }

    def export_scene(self, path):
        """Save the exact current MJCF and a versioned layout manifest for inspection."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._current_xml)
        manifest = {
            "env_version": ENV_VERSION, "terrain_version": terrain.TERRAIN_VERSION,
            "xml_sha256": hashlib.sha256(self._current_xml.encode()).hexdigest(),
            "seed": int(self.np_random_seed), "curriculum_level": self._episode_difficulty,
            "effective_difficulty": self.effective_difficulty,
            "course_length": self._course_length,
            "segments": [asdict(segment) for segment in self._segments],
        }
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
        return manifest

    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, "_tmpdir") and os.path.exists(self._tmpdir):
            for f in os.listdir(self._tmpdir):
                try:
                    os.remove(os.path.join(self._tmpdir, f))
                except OSError:
                    pass
            try:
                os.rmdir(self._tmpdir)
            except OSError:
                pass

    def _sample_heights(self) -> np.ndarray:
        """Casts 18 rays downward to sample terrain height ahead/below the torso."""
        torso_pos = self.data.xpos[self._torso_id].copy()
        torso_mat = self.data.xmat[self._torso_id].reshape(3, 3)

        forward_xy = torso_mat[:, 0].copy()
        forward_xy[2] = 0.0
        norm = np.linalg.norm(forward_xy)
        forward_xy = forward_xy / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

        lateral = np.array([-forward_xy[1], forward_xy[0], 0.0])

        heights = np.empty(N_HEIGHT_SAMPLES, dtype=np.float64)
        ray_dir = np.array([0.0, 0.0, -1.0])
        geomid_out = np.zeros(1, dtype=np.int32)
        idx = 0

        for fwd_dist in SCAN_FORWARD:
            for lat_dist in SCAN_LATERAL:
                pnt = torso_pos + fwd_dist * forward_xy + lat_dist * lateral
                pnt[2] = torso_pos[2]

                dist = mujoco.mj_ray(
                    self.model, self.data,
                    pnt, ray_dir,
                    TERRAIN_GEOM_GROUP,
                    1,
                    -1,
                    geomid_out,
                )

                if dist >= 0:
                    heights[idx] = -dist
                else:
                    heights[idx] = -3.5  # void/chasm
                idx += 1

        return heights

    @property
    def is_healthy(self) -> bool:
        """Determines whether the humanoid remains in an upright, stable state."""
        torso_z = self.data.qpos[2]
        torso_y = self.data.qpos[1]

        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return False
        # The death plane is a physical catch surface, never a landing platform.
        if torso_z < -1.5:
            return False

        # Fallen off the side of the 3m-wide course
        if abs(torso_y) > 2.0:
            return False

        # Check local ground height directly below torso
        geomid_out = np.zeros(1, dtype=np.int32)
        ray_dir = np.array([0.0, 0.0, -1.0])
        dist = mujoco.mj_ray(
            self.model, self.data,
            self.data.xpos[self._torso_id], ray_dir,
            TERRAIN_GEOM_GROUP,
            1,
            -1,
            geomid_out
        )
        if dist >= 0:
            height_above_ground = dist
            if int(geomid_out[0]) == self._death_geom_id:
                # Flight over a gap must not instantly fail because the death
                # plane is >2m below the torso. Use the gap's takeoff elevation.
                segment = terrain.segment_at_x(self._segments, float(self.data.qpos[0]))
                if segment is None or segment.kind != "gap":
                    return False
                height_above_ground = torso_z - (segment.height + 0.05)
            min_z, max_z = self._healthy_z_range
            if not (min_z < height_above_ground < max_z):
                return False

        if self._terminate_when_backward:
            torso_fwd_x = float(self.data.xmat[self._torso_id].reshape(3, 3)[0, 0])
            if torso_fwd_x < -0.5:
                return False

        return dist >= 0

    def _supported_feet(self):
        """Foot bodies in contact with course terrain, excluding the death plane."""
        supported = set()
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            for foot_geom, ground_geom in ((a, b), (b, a)):
                body = int(self.model.geom_bodyid[foot_geom])
                if (body in self._feet_ids and ground_geom != self._death_geom_id
                        and self.model.geom_group[ground_geom] == 2
                        and self.model.geom_bodyid[ground_geom] == 0):
                    supported.add(body)
        return supported

    def _course_progress(self, healthy):
        """Award traversal only after a healthy supported landing beyond the obstacle."""
        pelvis_x = float(self.data.qpos[0])
        rear_foot_x = float(np.min(self.data.xpos[self._feet_ids, 0]))
        landed = healthy and bool(self._supported_feet())
        bonus = 0.0
        obstacle_indices = {i for i, s in enumerate(self._segments) if s.kind != "flat"}
        if landed:
            for i in obstacle_indices - self._cleared_segments:
                if min(pelvis_x, rear_foot_x) > self._segments[i].x_end:
                    self._cleared_segments.add(i)
                    bonus += self._obstacle_bonus
        finish = self._segments[-1].x_start + 1.0
        completed = bool(landed and pelvis_x >= finish
                         and rear_foot_x > self._segments[-1].x_start)
        if completed and not self._course_completed:
            bonus += 200.0
            self._course_completed = True
        cleared = len(self._cleared_segments & obstacle_indices)
        clearance = cleared / len(obstacle_indices) if obstacle_indices else float(completed)
        progress = float(np.clip((pelvis_x - 2.5) / max(1.0, finish - 2.5), 0, 1))
        return bonus, completed, cleared, len(obstacle_indices), clearance, progress

    def _get_obs(self) -> np.ndarray:
        """Assembles the 366-dimensional observation vector."""
        position = self.data.qpos[2:].flatten()
        velocity = self.data.qvel.flatten()
        com_inertia = self.data.cinert[1:].flatten()
        com_velocity = self.data.cvel[1:].flatten()
        actuator_forces = self.data.qfrc_actuator[6:].flatten()
        external_contact_forces = self.data.cfrc_ext[1:].flatten()
        heights = self._sample_heights()

        return np.concatenate((
            position, velocity, com_inertia, com_velocity,
            actuator_forces, external_contact_forces, heights,
        ))

    def step(self, action):
        xy_before = _mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        self._elapsed_steps += 1
        xy_after = _mass_center(self.model, self.data)

        xy_velocity = (xy_after - xy_before) / self.dt
        x_velocity = xy_velocity[0]

        # Orientation: torso local X (forward) and local Z (upright) projected into world
        torso_xmat = self.data.xmat[self._torso_id].reshape(3, 3)
        torso_fwd_x = float(torso_xmat[0, 0])
        torso_zz = float(torso_xmat[2, 2])

        # Rewards
        # Gate forward velocity reward by positive facing alignment (facing +x) so backpedaling earns no forward credit
        forward_reward = self._forward_reward_weight * x_velocity * max(0.0, torso_fwd_x)
        healthy = self.is_healthy
        healthy_reward = float(healthy) * self._healthy_reward
        upright_reward = self._upright_reward_weight * max(0.0, torso_zz)
        facing_reward = self._facing_reward_weight * torso_fwd_x

        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))

        contact_forces = self.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        contact_cost = np.clip(contact_cost, *self._contact_cost_range)

        obstacle_bonus, completed, n_cleared, n_obstacles, clearance_rate, progress = self._course_progress(healthy)

        reward = forward_reward + healthy_reward + upright_reward + facing_reward - ctrl_cost - contact_cost + obstacle_bonus
        failed = bool((not healthy) and self._terminate_when_unhealthy)
        terminated = bool(failed or completed)
        truncated = bool(self._elapsed_steps >= self.max_episode_steps and not terminated)

        observation = self._get_obs()
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "x_velocity": x_velocity,
            "facing_x": torso_fwd_x,
            "curriculum_level": self._episode_difficulty,
            "effective_difficulty": self.effective_difficulty,
            "env_version": ENV_VERSION,
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_upright": upright_reward,
            "reward_facing": facing_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_obstacle": obstacle_bonus,
            "segments_cleared": n_cleared,
            "obstacles_total": n_obstacles,
            "course_length": self._course_length,
            "clearance_rate": clearance_rate,
            "completion_rate": float(completed),
            "progress_fraction": progress,
            "success": completed,
            "failed": failed,
            "is_healthy": healthy,
            "termination_reason": "success" if completed else "unhealthy" if failed else "time_limit" if truncated else None,
            "completed": completed,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        """Generates a new course and positions the humanoid on the start platform."""
        self._episode_difficulty = self.difficulty
        self.effective_difficulty = terrain.sample_difficulty(self.np_random, self.difficulty, self.replay_prob)
        self._segments, self._course_length = terrain.generate_course(
            self.np_random, difficulty=self.effective_difficulty, replay_prob=0.0
        )
        terrain_xml = _segments_to_geom_xml(self._segments)
        xml_content = _build_mjcf_template(terrain_xml)
        self._current_xml = xml_content

        # Recompile model with freshly built course BVH
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)

        if hasattr(self, "mujoco_renderer") and self.mujoco_renderer is not None:
            # A viewer owns model-sized scene buffers and retains model/data
            # pointers. Recreate it when course compilation changes the model.
            self.mujoco_renderer.close()
            self.mujoco_renderer._viewers.clear()
            self.mujoco_renderer.viewer = None
            self.mujoco_renderer.model = self.model
            self.mujoco_renderer.data = self.data

        self._cache_body_ids()
        self._cleared_segments = set()
        self._course_completed = False
        self._elapsed_steps = 0

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Add small initial noise to avoid identical trajectories
        qpos[7:] += self.np_random.uniform(-self._reset_noise_scale, self._reset_noise_scale, size=self.model.nq - 7)
        qvel += self.np_random.uniform(-self._reset_noise_scale, self._reset_noise_scale, size=self.model.nv)

        # Initial root pose: centered on start runway, standing on platform
        qpos[0] = 2.5
        qpos[1] = 0.0
        qpos[2] = 1.35
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # unit quaternion

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        torso_fwd_x = float(self.data.xmat[self._torso_id].reshape(3, 3)[0, 0])
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "facing_x": torso_fwd_x,
            "curriculum_level": self._episode_difficulty,
            "effective_difficulty": self.effective_difficulty,
            "env_version": ENV_VERSION,
            "course_length": self._course_length,
            "n_segments": len(self._segments),
        }
