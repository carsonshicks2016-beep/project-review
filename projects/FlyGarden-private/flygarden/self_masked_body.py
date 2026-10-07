"""Eye RGB plus an engineered own-body visibility mask, with copied physics."""
import numpy as np
import mujoco as mj
from .live_body import LiveBody


class SelfMaskedBody(LiveBody):
    version = 'copy-safe-self-masked-body-v3'

    def eye_frames(self):
        with self.observed_state():
            rgb = self.sim.get_raw_vision(self.fly.name)
            renderer = self.sim.eye_renderer
            own_bodies = self.sim._internal_bodyids_by_fly[self.fly.name]
            own_geoms = np.flatnonzero(np.isin(self.sim.mj_model.geom_bodyid, own_bodies))
            masks = []
            renderer.enable_segmentation_rendering()
            try:
                for camera in self.sim._intern_eye_camera_ids_by_fly[self.fly.name]:
                    renderer.update_scene(self.sim.mj_data, camera,
                                          scene_option=self.sim.eye_renderer_scene_option)
                    segmentation = renderer.render()
                    self_pixel = (segmentation[..., 1] == mj.mjtObj.mjOBJ_GEOM) & np.isin(segmentation[..., 0], own_geoms)
                    raw = np.repeat((~self_pixel)[..., None], 3, axis=2).astype(np.uint8)*255
                    # The same nearest-pixel fisheye mapping as RGB; outside
                    # the camera's image circle remains invalid.
                    masks.append(self.sim.retina.correct_fisheye(raw)[..., 0] > 0)
            finally:
                renderer.disable_segmentation_rendering()
            return {'rgb': rgb, 'valid': np.asarray(masks),
                    'mask_identity': 'Own-body geoms only; no external object identities or coordinates'}
