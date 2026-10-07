from flygarden.recorded_events import markers_from_frames

def frame(t,odor=(0,0),events=(),sample=None):
 f=dict(world=dict(time=t),sensory=dict(odor=list(odor)),events=list(events))
 if sample is not None:f['sensory_time']=sample
 return f

def test_markers_keep_true_contact_times_and_deduplicate_pose_copies():
 contact=dict(time=.1,kind='food',message='Food contact')
 frames=[frame(0),frame(.0335,(.2,0),sample=0),frame(.1,(.2,0),[contact],sample=0),frame(.1335,(.2,0),[contact],sample=.1),frame(.2,(.05,0),[contact],sample=.1)]
 events=markers_from_frames(frames,[dict(time=.1,kind='reinforcement',message='Positive reinforcement',value=1)])
 assert [(e['time'],e['kind']) for e in events if e['kind']=='food']==[(.1,'food')]
 assert [(e['time'],e['kind']) for e in events if e['kind']=='odor']==[(0.,'odor')]
 assert [(e['time'],e['kind']) for e in events if e['kind']=='odor_end']==[(.1,'odor_end')]
 assert [e['time'] for e in events if e['kind']=='reinforcement']==[.1]

def test_hysteresis_nonzero_origin_and_future_events():
 frames=[frame(10,(.16,0)),frame(10.1,(.14,0)),frame(10.2,(.11,0)),frame(10.3,(.09,0))]
 events=markers_from_frames(frames,[dict(time=11,kind='food')],status='interrupted')
 assert len([e for e in events if e['kind']=='odor'])==1
 assert not any(e['kind']=='food' for e in events)
 assert events[-1]['kind']=='termination'
 assert events[-1]['time']==10.3

def test_recording_endpoint_ignores_uncommitted_frame_tail(tmp_path,monkeypatch):
 from flygarden import recording
 from flygarden.recorded_events import recorded_markers
 monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path))
 r=recording.Recorder({},{});r.append_window([frame(0),frame(.1)]);r.finish(enqueue=False)
 with (r.folder/'frames.jsonl').open('a') as f:f.write('{"world":{"time":50},"events":[{"time":50,"kind":"capture"}]}\n')
 events=recorded_markers(r.id)['events'];assert max(e['time'] for e in events)==.1
 assert not any(e['kind']=='capture' for e in events)
