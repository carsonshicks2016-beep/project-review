"""Predeclared measurements for the experimental recorded-eye causal test."""
import numpy as np

def trajectory_metrics(initial,frames,rows,cue):
 if cue not in ('loom_left','loom_right') or not frames or not rows:raise ValueError('A complete sided response trajectory is required')
 times=np.array([f['time'] for f in frames]);positions=np.array([initial['position']]+[f['body']['position'] for f in frames]);headings=np.unwrap([initial['heading']]+[f['body']['heading'] for f in frames]);motors=np.array([r['applied_motor'] for r in rows])
 if not np.isfinite(positions).all() or not np.isfinite(headings).all() or not np.isfinite(motors).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid trajectory')
 sign=-1 if cue=='loom_left' else 1
 distance=np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1)
 active=float(np.sqrt(np.mean(motors**2)))>1e-4
 late=positions[1:][times>=1.5,:2]
 late_distance=float(np.linalg.norm(np.diff(late,axis=0),axis=1).sum())
 return {'away_heading_radians':float(sign*(headings[-1]-headings[0])),'heading_change_radians':float(headings[-1]-headings[0]),'horizontal_travel_mm':float(distance.sum()),'net_horizontal_displacement_mm':float(np.linalg.norm(positions[-1,:2]-positions[0,:2])),'late_horizontal_travel_mm':late_distance,'motor_rms':float(np.sqrt(np.mean(motors**2))),'motor_active':active,'stalled':bool(active and late_distance<.1),'flipped_frames':int(sum(f['body']['flipped'] for f in frames))}

def paired_bootstrap(values,sides,seed=8200,repetitions=10000):
 values=np.asarray(values,dtype=float);sides=np.asarray(sides)
 if values.ndim!=1 or len(values)!=len(sides) or not np.isfinite(values).all():raise ValueError('Finite matched effects required')
 groups=[np.flatnonzero(sides==side) for side in ('loom_left','loom_right')]
 if any(len(g)==0 for g in groups):raise ValueError('Both directions required')
 rng=np.random.default_rng(seed);samples=np.concatenate([rng.choice(g,size=(repetitions,len(g)),replace=True) for g in groups],axis=1);means=values[samples].mean(axis=1)
 # Two paired comparisons per family: 97.5% intervals, Bonferroni simultaneous95%.
 return {'mean':float(values.mean()),'ci_97_5':[float(v) for v in np.quantile(means,[.0125,.9875])],'repetitions':repetitions,'seed':seed,'stratified_by_cue':True,'positive':bool(np.quantile(means,.0125)>0)}
