"""Versioned physical world stepping without changing the frozen body candidate."""
import numpy as np
import mujoco as mj
from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action
from .synchronized import SynchronizedBody
from .body import Body


class LiveBody(SynchronizedBody):
    version = 'copy-safe-world-body-v1'

    def __init__(self, *args, **kwargs):
        # Retain the frozen candidate's constructor exactly. Only subsequent
        # world changes use the copy-safe path.
        self._initial_visual = True
        super().__init__(*args, **kwargs)
        self._initial_visual = False

    def update_visual_world(self, foods, predator):
        if self._initial_visual:
            return Body.update_visual_world(self, foods, predator)
        # Updating a visual-only mocap must not call forward on integration data:
        # that would change solver warm starts and the supplied gait's inputs.
        data, model = self.sim.mj_data, self.sim.mj_model
        data.mocap_pos[0] = ([predator.get('x', 0), predator.get('y', 0), 1.2]
                             if predator.get('enabled') else [0, 0, -100])
        heading = predator.get('heading', 0)
        data.mocap_quat[0] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
        for food in foods:
            index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"food_{food['id']}")
            if index >= 0:
                model.geom_rgba[index, 3] = float(food['units'] > 0)

    def advance(self, dt, motor, capture=None, world_step=None):
        ticks, stride = round(dt / self.dt), round(self.control_dt / self.dt)
        if ticks <= 0 or ticks % stride or abs(ticks * self.dt - dt) > 1e-9:
            raise ValueError('Window must match physics and gait clocks')
        motor = np.asarray(motor, dtype=float)
        if motor.shape != (2,) or not np.isfinite(motor).all():
            raise ValueError('Two finite motor commands required')
        motor = np.clip(motor, 0, 1.2)
        next_sample = (int(np.floor(self.steps * self.dt * 30 + 1e-9)) + 1) / 30
        for _ in range(ticks // stride):
            obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
            action = self.controller.step(motor, obs)
            apply_locomotion_action(self.sim, self.fly.name, action)
            mj.mj_step(self.sim.mj_model, self.sim.mj_data, nstep=stride)
            self.steps += stride
            if world_step is not None:
                # Kinematics on a separate data object; no invented contact input.
                world_step(self.steps * self.dt, self.recording_pose()['body'])
            if capture is not None and self.steps * self.dt + 1e-10 >= next_sample:
                capture(self.steps * self.dt, self.recording_pose())
                next_sample += 1 / 30
        if not all(np.isfinite(x).all() for x in
                   (self.sim.mj_data.qpos, self.sim.mj_data.qvel, self.sim.mj_data.qacc)):
            raise RuntimeError('Non-finite body state')
        return self.observation()
