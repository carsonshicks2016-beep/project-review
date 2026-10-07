#!/usr/bin/env python3
"""Silent real-device subprocess lifecycle smoke for Audio V4."""
from __future__ import annotations
import argparse, os, time
from types import SimpleNamespace
from supra.audio_v4 import RATIOS_787B, SpatialAudioMixerV4

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",type=int); p.add_argument("--seconds",type=float,default=2.0); a=p.parse_args()
    if a.device is not None:
        import sounddevice as sd
        os.environ["SUPRA_AUDIO_DEVICE"]=sd.query_devices(a.device)["name"]
    spec=SimpleNamespace(name="mazda787b",redline_rpm=9000,gear_ratios=RATIOS_787B)
    v=SimpleNamespace(spec=spec,x=0.,y=0.,yaw=0.,speed=0.,rpm=900.,gear=1,boost=0.,slip_angle=0.,wheel_sr=[0.,0.,0.,0.])
    mixer=SpatialAudioMixerV4(master=0.0).start([v]); time.sleep(a.seconds); mixer.reset([v]); time.sleep(.2)
    metrics=mixer.metrics; alive=bool(mixer.process and mixer.process.is_alive()); mixer.stop()
    print({"alive":alive,"reset_ack":metrics.get("reset_ack"),"failed":metrics.get("failed"),"metrics":metrics})
    return 0 if alive and not metrics.get("failed") and metrics.get("reset_ack",0)>=1 else 1
if __name__ == "__main__": raise SystemExit(main())
