import json,struct
from pathlib import Path
import numpy as np
import pytest
from flygarden import recording,morphology

def frame(t):return dict(world=dict(time=t,status='running'),body=dict(position=[0,0,0]))

def test_atomic_chunks_ordering_and_interruption(tmp_path,monkeypatch):
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path));r=recording.Recorder(dict(mode='full'),{},[9007199254740993,9007199254740995]);r.append_window([frame(0),frame(.1)],[0,1,0],[.01,.02,.09]);r.append_window([frame(.133333),frame(.2)],[1],[.1001]);r.finish(dict(status='captured'),enqueue=False)
 m=recording.metadata(r.id);assert m['recording']['status']=='complete';assert m['recording']['spike_count']==4;assert m['recording']['chunks'][1]['start']==.1001
 assert recording.spike_chunk(r.id,0)==dict(indices=[0,1,0],times=[.01,.02,.09]);assert not list(r.folder.glob('*.tmp'))
 with np.load(r.folder/'activity-totals.npz') as a:assert a['counts'].tolist()==[2,2]
 with (r.folder/'frames.jsonl').open('a') as f:f.write('{"interrupted":')
 from flygarden.video import read_frames
 assert len(read_frames(r.folder))==4
 (r.folder/'spikes-000000.npz').write_bytes(b'broken')
 with pytest.raises(ValueError,match='integrity'):recording.spike_chunk(r.id,0)

def test_low_space_preserves_files(tmp_path,monkeypatch):
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path));r=recording.Recorder({},{});original=(r.folder/'recording.json').read_bytes();monkeypatch.setattr(recording.shutil,'disk_usage',lambda path:type('Usage',(),dict(free=recording.RESERVE-1))())
 with pytest.raises(OSError,match='reserve'):r.append_window([frame(.1)])
 assert (r.folder/'recording.json').read_bytes()==original;assert not (r.folder/'frames.jsonl').exists()

def test_morphology_binary_units_and_invalid_geometry():
 vertices=np.array([[1000,2000,3000],[4000,5000,6000]],dtype='<f4');edges=np.array([[0,1]],dtype='<u4');raw=struct.pack('<II',2,1)+vertices.tobytes()+edges.tobytes()+np.array([1,1],dtype='<f4').tobytes();v,e=morphology.parse_skeleton(raw)
 assert np.array_equal(v,vertices);assert np.array_equal(e,edges)
 with pytest.raises(ValueError):morphology.parse_skeleton(raw[:10])
 with pytest.raises(ValueError):morphology.parse_skeleton(struct.pack('<II',2,1)+vertices.tobytes()+np.array([[0,3]],dtype='<u4').tobytes())

def test_recording_cannot_traverse():
 with pytest.raises(ValueError):recording.run_folder('../other')

def test_sampling_does_not_change_body_state():
 from flygarden.body import Body
 body=Body(seed=431);initial=body.snapshot();expected=body.advance(.1,[.7,.9]);expected_state=body.snapshot();body.restore(initial);samples=[];actual=body.advance(.1,[.7,.9],capture=lambda t,p:samples.append((t,p)));actual_state=body.snapshot();body.close()
 assert len(samples)==3;assert np.allclose([x[0] for x in samples],[.0335,.067,.1],atol=1e-10)
 assert np.array_equal(actual_state['physics'],expected_state['physics']);assert np.array_equal(actual['position'],expected['position'])
 assert len(samples[0][1]['visuals'])>0

def test_brian_monitor_drain_preserves_cumulative_counts():
 import brian2 as b
 from flygarden.brain import FullBrain
 b.prefs.codegen.target='numpy';clock=b.Clock(dt=.1*b.ms);g=b.NeuronGroup(2,'dv/dt = 0*Hz : 1',threshold='v>1',reset='v=0',clock=clock);monitor=b.SpikeMonitor(g,record=True);net=b.Network(g,monitor)
 g.v=[2,2];net.run(.001*b.second);assert len(monitor.i)==2
 adapter=type('Adapter',(),dict(record_spikes=True,monitor=monitor))();FullBrain.clear_recorded_spikes(adapter);assert len(monitor.i)==0;assert np.asarray(monitor.count[:]).tolist()==[1,1]
 g.v=[2,0];net.run(.001*b.second);assert np.asarray(monitor.count[:]).tolist()==[2,1];assert np.asarray(monitor.i[:]).tolist()==[0]

def test_committed_frames_exclude_uncommitted_complete_tail(tmp_path,monkeypatch):
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path));r=recording.Recorder({},{});r.append_window([frame(0),frame(.1)]);(r.folder/'frames.jsonl').open('a').write(json.dumps(frame(.2))+'\n')
 from flygarden.video import read_frames
 assert len(read_frames(r.folder))==2

def test_export_retry_and_restart_keep_recording(tmp_path,monkeypatch):
 from flygarden import video_jobs
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path));monkeypatch.setattr(video_jobs,'ensure_worker',lambda:None);r=recording.Recorder({},{});r.append_window([frame(0),frame(.1)]);r.finish(enqueue=False)
 j=video_jobs.enqueue_export(r.id);assert j['status']=='queued';assert video_jobs.enqueue_export(r.id)['id']==j['id']
 path=video_jobs.jobs_root()/j['id']/'job.json';j.update(status='rendering',pid=99999999,process_created=0);recording.atomic_json(path,j)
 interrupted=video_jobs.statuses()[0];assert interrupted['status']=='interrupted';assert r.folder.exists();assert video_jobs.retry(j['id'])['status']=='queued'
 with pytest.raises(ValueError):video_jobs.retry('../invalid')
 with pytest.raises(ValueError):video_jobs.enqueue_export(r.id,dict(camera='shell-command'))

def test_recording_endpoints_reject_invalid_ids_and_negative_chunks(monkeypatch,tmp_path):
 from fastapi.testclient import TestClient
 from flygarden.server import app
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path));client=TestClient(app)
 assert client.get('/api/recordings/not-a-run').status_code==400
 assert client.get('/api/recordings/'+'a'*32+'/spikes/-1').status_code==400
 assert client.get('/api/morphology/123').status_code==400
 assert client.get('/api/exports/not-a-job/video').status_code==400
 assert client.get('/api/recordings/'+'a'*32+'/neurons?limit=100000').status_code==400

def test_coarse_shapes_preserve_branch_end_nodes_and_cycles():
 v=np.arange(33,dtype=np.float32).reshape(11,3);e=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[4,9]],dtype=np.uint32)
 coarse,edges=morphology.coarse_geometry(v,e);assert set(map(tuple,v[[0,4,8,9]]))<=set(map(tuple,coarse));assert len(coarse)<len(v);assert edges.max()<len(coarse)
 cycle=np.array([[0,1],[1,2],[2,0]],dtype=np.uint32);cv,ce=morphology.coarse_geometry(v[:3],cycle);assert len(ce)==3;assert len(cv)==3

def test_neural_recording_without_body_retains_exact_events(tmp_path,monkeypatch):
 import brian2 as b
 from flygarden.trial_recording import NeuralRecording
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path))
 class Brain:
  ids=np.array([9007199254740993],dtype=np.int64)
  plasticity=type('Plasticity',(),dict(enabled=False))()
  network=type('Network',(),dict(t=0*b.second))()
  last_rates={}
  def advance(self,dt,**kwargs):
   self.last_spikes=(np.array([0]),np.array([float(self.network.t/b.second)+.001]));self.network.t+=dt*b.second;return np.array([0,0])
 r=NeuralRecording(Brain(),11,'frozen');r.advance(.2,odor=(1,0));r.advance(.2,odor=(0,1));r.recorder.finish(enqueue=False)
 assert r.recorder.meta['frames']==13;assert r.recorder.meta['duration']==.4;assert r.recorder.manifest['body_recorded'] is False
 assert recording.spike_chunk(r.recorder.id,1)['times']==[.201]

def test_dead_writer_recovery_queues_video_but_preserves_live_recording(tmp_path,monkeypatch):
 from flygarden import video_jobs
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path));monkeypatch.setattr(video_jobs,'ensure_worker',lambda:None);monkeypatch.setattr(recording,'ROOT',tmp_path/'project');monkeypatch.setattr(video_jobs,'ROOT',tmp_path/'project')
 live=recording.Recorder({},{});live.append_window([frame(0)])
 dead=recording.Recorder({},{});dead.append_window([frame(0),frame(.1)]);dead.manifest['writer']=dict(pid=99999999,process_created=0);recording.atomic_json(dead.folder/'manifest.json',dead.manifest)
 before=(dead.folder/'frames.jsonl').read_bytes();assert recording.recover_interrupted()==[dead.id]
 assert recording.metadata(dead.id)['recording']['status']=='interrupted';assert recording.metadata(live.id)['recording']['status']=='recording'
 assert (dead.folder/'frames.jsonl').read_bytes()==before;assert video_jobs.statuses()[0]['status']=='queued';assert recording.recover_interrupted()==[]

def test_full_neuron_history_preserves_ordering_and_bins(tmp_path,monkeypatch):
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path));r=recording.Recorder({}, {}, [9007199254740993,9007199254740995]);r.append_window([frame(0),frame(.1)],[0,1,0],[.01,.05,.09]);r.append_window([frame(.1335),frame(.2)],[0,1],[.11,.15]);r.finish(enqueue=False)
 history=recording.neuron_history(r.id,0);assert history['id']=='9007199254740993';assert history['spike_count']==3;assert history['rates_hz']==[20,10]
 with pytest.raises(ValueError):recording.neuron_history(r.id,-1)


def test_free_export_preserves_view_and_rejects_invalid_coordinates(tmp_path,monkeypatch):
 from flygarden import video_jobs
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path));monkeypatch.setattr(video_jobs,'ensure_worker',lambda:None)
 r=recording.Recorder({},{});r.append_window([frame(0),frame(.1)]);r.finish(enqueue=False)
 view=dict(position=[-8,-11,7],target=[0,0,1])
 job=video_jobs.enqueue_export(r.id,dict(camera='free',camera_view=view),automatic=False)
 assert job['settings']['camera_view']==view
 assert json.loads((video_jobs.jobs_root()/job['id']/'job.json').read_text())['settings']['camera_view']==view
 for invalid in (None,{},dict(position=[float('nan'),0,0],target=[0,0,0]),dict(position=[0,0],target=[0,0,0]),dict(position=[10001,0,0],target=[0,0,0])):
  with pytest.raises(ValueError):video_jobs.enqueue_export(r.id,dict(camera='free',camera_view=invalid),automatic=False)
