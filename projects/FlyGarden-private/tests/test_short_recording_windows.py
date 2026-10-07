import numpy as np
from flygarden.recording import Recorder,spike_chunk

def test_neural_windows_without_pose_are_committed(tmp_path,monkeypatch):
    monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path))
    recorder=Recorder({}, {},[101,102])
    recorder.append_window([], [0],[.01],time_bounds=(0,.025))
    recorder.append_window([{'world':{'time':.0335}}],[1],[.03],time_bounds=(.025,.05))
    recorder.append_window([],[],[],time_bounds=(.05,.075))
    recorder.finish(enqueue=False)
    assert len(recorder.meta['chunks'])==3 and recorder.meta['spike_count']==2
    assert recorder.meta['duration']==.075 and recorder.meta['frames']==1
    assert spike_chunk(recorder.id,0)['times']==[.01]
    assert spike_chunk(recorder.id,1)['times']==[.03]
    with np.load(recorder.folder/'activity-totals.npz') as z:assert z['counts'].tolist()==[1,1]
