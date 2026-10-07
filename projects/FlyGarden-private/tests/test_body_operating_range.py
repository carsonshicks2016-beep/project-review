import gzip,json
import numpy as np
from scripts import measure_body_operating_range as runner

class BodyFixture:
    def __init__(self,seed):self.steps=0;self.x=0.
    def observation(self):return {'position':[self.x,0.,0.],'heading':0.,'contacts':[[0.,0.,0.]]*18,'flipped':False}
    def advance(self,dt,motor,capture):
        self.x+=float(np.mean(motor))*dt;self.steps+=1
        capture(self.steps*dt,{'body':self.observation(),'visuals':[]})
        return self.observation()
    def snapshot(self):return {'steps':self.steps,'x':self.x}
    def restore(self,state):self.steps=state['steps'];self.x=state['x']
    def geometry(self):return {}
    def close(self):pass

def test_ac_pause_and_continuation_preserve_commands_and_chunks(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'SynchronizedBody',BodyFixture)
    monkeypatch.setattr(runner,'OUT',tmp_path/'paused')
    power=iter([True,False]);monkeypatch.setattr(runner,'on_ac_power',lambda:next(power))
    case={'rates':{'DNp09_left':65,'DNp09_right':65,'DNa02_left':50,'DNa02_right':0}}
    assert not runner.trial(1,'fixture',case,3.)
    folder=runner.OUT/'trials/1-fixture';initial=json.loads((folder/'manifest.json').read_text())
    assert initial['completed_windows']==40
    original=(folder/initial['chunks'][0]['file']).read_bytes()
    monkeypatch.setattr(runner,'on_ac_power',lambda:True)
    assert runner.trial(1,'fixture',case,3.)
    resumed=json.loads((folder/'manifest.json').read_text())
    assert (folder/initial['chunks'][0]['file']).read_bytes()==original
    def rows(base,meta):
        result=[]
        for c in meta['chunks']:
            with gzip.open(base/c['file'],'rt') as f:result.extend(json.load(f)['rows'])
        return result
    monkeypatch.setattr(runner,'OUT',tmp_path/'fresh')
    assert runner.trial(1,'fixture',case,3.)
    fresh=runner.OUT/'trials/1-fixture';meta=json.loads((fresh/'manifest.json').read_text())
    assert rows(folder,resumed)==rows(fresh,meta)
    assert resumed['metrics']==meta['metrics']

def test_power_loss_before_first_window_can_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'SynchronizedBody',BodyFixture)
    monkeypatch.setattr(runner,'OUT',tmp_path)
    monkeypatch.setattr(runner,'on_ac_power',lambda:False)
    assert not runner.trial(1,'start',{'motor':[.4,.4]},3.)
    monkeypatch.setattr(runner,'on_ac_power',lambda:True)
    assert runner.trial(1,'start',{'motor':[.4,.4]},3.)
