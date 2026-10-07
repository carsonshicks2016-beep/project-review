"""Native rectilinear eye views plus independently verified fisheye views."""
import numpy as np
import mujoco as mj
from .self_masked_body import SelfMaskedBody


class RectilinearBody(SelfMaskedBody):
    version='copy-safe-rectilinear-eye-body-v5'

    def eye_frames(self):
        with self.observed_state():
            if self.sim.eye_renderer is None:self.sim.get_raw_vision(self.fly.name)
            renderer=self.sim.eye_renderer
            own_geoms=np.flatnonzero(np.isin(self.sim.mj_model.geom_bodyid,
                                    self.sim._internal_bodyids_by_fly[self.fly.name]))
            images=[];masks=[];fisheye=[]
            try:
                for camera in self.sim._intern_eye_camera_ids_by_fly[self.fly.name]:
                    renderer.disable_segmentation_rendering()
                    renderer.update_scene(self.sim.mj_data,camera,scene_option=self.sim.eye_renderer_scene_option)
                    raw=renderer.render().copy();images.append(raw)
                    fisheye.append(self.sim.retina.correct_fisheye(raw))
                    renderer.enable_segmentation_rendering()
                    segmentation=renderer.render()
                    self_pixel=(segmentation[...,1]==mj.mjtObj.mjOBJ_GEOM)&np.isin(segmentation[...,0],own_geoms)
                    masks.append(~self_pixel)
            finally:renderer.disable_segmentation_rendering()
            return {'rgb':np.asarray(images),'valid':np.asarray(masks),'fisheye_rgb':np.asarray(fisheye),
                'projection':'Native rectilinear eye cameras,not physiological retina',
                'mask_identity':'Visible own-body only; external identities unavailable'}
