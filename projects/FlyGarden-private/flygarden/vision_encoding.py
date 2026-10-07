"""Experimental image-only expansion proxy; bypasses retina/T4/T5 physiology.

No coordinates, predator state, camera heading or requested motor command enter
this encoder. Uniformly drive annotated ipsilateral LPLC2; receptive fields are
not known and no per-cell retinotopy is invented.
"""
import numpy as np
from scipy.ndimage import label
class EyeExpansion:
 version='eye-dark-expansion-v1'
 def __init__(self,baseline):
  self.baseline=self.luminance(baseline);self.previous_area=np.zeros(2);self.previous_centroid=[None,None]
 @staticmethod
 def luminance(frames):
  f=np.asarray(frames)
  if f.ndim!=4 or f.shape[0]!=2 or f.shape[-1]!=3 or not np.isfinite(f).all() or np.any(f<0) or np.any(f>255):raise ValueError('Two finite RGB eye images in [0,255] required')
  return np.tensordot(f.astype(float)/255,[.2126,.7152,.0722],axes=([-1],[0]))
 def advance(self,frames,dt):
  if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite image interval required')
  lum=self.luminance(frames)
  if lum.shape!=self.baseline.shape:raise ValueError('Eye image dimensions changed')
  features=[]
  for side in range(2):
   mask=self.baseline[side]-lum[side]>.12;components,n=label(mask);sizes=np.bincount(components.ravel());sizes[0]=0
   if n and sizes.max()>3:
    component=components==sizes.argmax();area=float(component.mean());centroid=np.argwhere(component).mean(axis=0)/np.array(component.shape)
   else:area=0.;centroid=None
   prev=self.previous_centroid[side];stable=prev is None or centroid is None or np.linalg.norm(centroid-prev)<.08
   growth=max(0,(area-self.previous_area[side])/dt) if stable and area<.5 else 0.
   # Fixed engineered gain chosen before full-brain tests; no fitted biology.
   rate=float(np.clip(growth*1000,0,100))
   features.append({'brightness':float(lum[side].mean()),'motion':float(abs(lum[side]-getattr(self,'previous_lum',self.baseline)[side]).mean()/dt),'dark_area_fraction':area,'expansion_per_second':growth,'lplc2_hz':rate})
   self.previous_area[side]=area;self.previous_centroid[side]=centroid
  self.previous_lum=lum.copy();return features
