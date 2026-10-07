"""Validate committed full-brain recording and timestamp markers without recomputation."""
import sys,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
from flygarden.recorded_events import recorded_markers
from flygarden.recording import run_folder,atomic_json
report=root/'reports/recording-delivery';result=json.loads((report/'obstacle-trial.json').read_text());folder=run_folder(result['run']);meta=json.loads((folder/'recording.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
frames=[]
with (folder/'frames.jsonl').open() as f:
 for i,line in enumerate(f):
  if i>=meta['frames']:break
  frames.append(json.loads(line))
clock=np.array([f['world']['time'] for f in frames]);positions=np.array([f['body']['position'] for f in frames]);counts=np.zeros(138639,dtype=np.int64);last=-np.inf;events=0
for chunk in meta['chunks']:
 p=folder/chunk['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==chunk['sha256']
 with np.load(p,allow_pickle=False) as a:
  indices=a['indices'];times=a['times'];events+=len(indices)
  assert len(indices)==chunk['events']
  if len(indices):
   assert indices.min()>=0 and indices.max()<138639 and times.min()>=last-1e-9 and np.all(np.diff(times)>=0) and times.max()<=chunk['end']+1e-9
   np.add.at(counts,indices,1);last=times[-1]
with np.load(folder/'activity-totals.npz',allow_pickle=False) as a:assert np.array_equal(a['counts'],counts)
assert len(frames)==901 and np.all(np.diff(clock)>0) and abs(clock[-1]-30)<1e-8 and np.max(np.diff(clock))<=.0335+1e-8
assert np.isfinite(positions).all() and all(np.isfinite(np.array(g['p']+g['r'])).all() for f in frames for g in f['visuals'])
assert events==result['monitor_spikes']==meta['spike_count']
markers=recorded_markers(result['run']);atomic_json(report/'obstacle-events.json',markers)
# New recordings carry the actual sensory sampling clock, preceding physical poses.
for marker in markers['events']:
 if marker['kind'] in ('odor','odor_end','obstacle','threat'):
  assert any(abs(f.get('sensory_time',-1)-marker['time'])<1e-8 for f in frames)
# Cross-check a real captured diagnostic: displayed capture is its original saved event clock.
capture=recorded_markers('ed20d7d471344674a330d216d6aad0dd')
assert [e['time'] for e in capture['events'] if e['kind']=='capture']==[.1]
summary=dict(status='passed',run=result['run'],frames=len(frames),spikes=events,sha256_chunks=len(meta['chunks']),frame_interval_max=float(np.diff(clock).max()),physical_pose_clock_start=float(clock[0]),physical_pose_clock_end=float(clock[-1]),marker_count=len(markers['events']),real_capture_marker_verified=.1,checks=['all chunk hashes','all spike ordering/ranges','all per-neuron totals','monitor event count','901 finite body poses','finite recorded geometry','strict increasing pose clock','sensory-marker sample timestamps','real capture timestamp'],scope='Recording integrity and timestamp verification, not behavioral learning')
atomic_json(report/'obstacle-verification.json',summary);print(json.dumps(summary,indent=2))
