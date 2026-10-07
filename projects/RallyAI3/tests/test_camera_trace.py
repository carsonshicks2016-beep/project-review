import unittest
from pathlib import Path
from Tools.camera_trace import summarize


class CameraTraceTests(unittest.TestCase):
    def frame(self, index, fov, mode=0):
        return dict(frame=index,mode=mode,scale=1,time=2+index/30,dt=1/30,
                    fov=fov,vsync=0,fpsCap=30)

    def test_repeated_zoom_resets_are_detected(self):
        rows=[self.frame(i,58 if i%15==0 else 65) for i in range(60)]
        self.assertGreater(summarize(rows)["0"]["fov_jumps_over_one_degree"],0)

    def test_smooth_fov_and_mode_cuts_are_not_jitter(self):
        rows=[self.frame(i,56+i*.01) for i in range(30)]
        rows += [self.frame(i,78,1) for i in range(30,60)]
        summary=summarize(rows)
        self.assertEqual(summary["0"]["fov_jumps_over_one_degree"],0)
        self.assertEqual(summary["1"]["fov_jumps_over_one_degree"],0)
        self.assertAlmostEqual(summary["0"]["mean_fps"],30)

    def test_cinematic_shot_cut_is_excluded(self):
        rows=[{**self.frame(i,56 if i<30 else 50,5),"shot":0 if i<30 else 3} for i in range(60)]
        self.assertEqual(summarize(rows)["5"]["fov_jumps_over_one_degree"],0)

    def test_presentation_poll_does_not_overwrite_animated_fov(self):
        source=(Path(__file__).resolve().parents[1]/"Assets/Core/Presentation/RallyVisualRenovation.cs").read_text()
        method=source.split("void ApplyCamera(",1)[1].split("void CreateDust(",1)[0]
        self.assertNotIn(".fieldOfView =",method)

    def test_station_cuts_and_framing_are_reported_separately(self):
        rows=[{**self.frame(i,48 if i<30 else 52,2),"station":0 if i<30 else 1,
               "carInSafeFrame":True} for i in range(90)]
        report=summarize(rows)["2"]
        self.assertEqual(report["fov_jumps_over_one_degree"],0)
        self.assertEqual(report["safe_frame_fraction"],1)

    def test_missing_framing_telemetry_is_not_zero(self):
        self.assertIsNone(summarize([self.frame(i,48) for i in range(60)])["0"]["safe_frame_fraction"])
