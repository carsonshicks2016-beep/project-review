"""Explicit supplied comparison policy; only egocentric sensory signals."""
import numpy as np

def motor_command(sensory,time):
 left=float(np.sum(sensory['odor_left']));right=float(np.sum(sensory['odor_right']))
 total=left+right
 gradient=(left-right)/max(total,.01)
 exploration=np.sin(time*.7)*.15 if total<.02 else 0.
 turn=float(np.clip(gradient*40+exploration,-.65,.65))
 motor=np.array([.85-turn,.85+turn])
 if sensory['threat']>.25:
  turn=.6 if sensory['threat_bearing']>0 else -.6
  motor=np.array([1.+turn,1.-turn])
 ranges=sensory.get('obstacle_ranges')
 if ranges and (ranges[1]<6. or min(ranges)<1.5):
  # Turn toward the clearer local ray; exact ties choose left deterministically.
  motor=np.array([.12,1.05]) if ranges[0]>=ranges[2] else np.array([1.05,.12])
 return np.clip(motor,0,1.2)
