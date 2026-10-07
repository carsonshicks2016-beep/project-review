"""Numerical and end-to-end checks using temporary runs; no user runs are modified."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import numpy as np
import torch
from hybrid_lab.config import Config
from hybrid_lab.environment import Environment
from hybrid_lab.learning import Policy, advantages, load_checkpoint


class Numerics(unittest.TestCase):
    def test_gae_bootstraps_time_limit_without_cross_episode_leak(self):
        rewards = np.array([[1.], [2.]], dtype=np.float32)
        values = np.zeros_like(rewards)
        next_values = np.array([[10.], [20.]], dtype=np.float32)
        terminated = np.array([[0.], [1.]], dtype=np.float32)
        done = np.ones_like(rewards)
        adv, ret = advantages(rewards, values, next_values, terminated, done, .9, .95)
        np.testing.assert_allclose(adv[:, 0], [10., 2.])
        np.testing.assert_allclose(ret, adv)

    def test_environment_reproducibility_and_real_motion(self):
        a, b = Environment(), Environment()
        np.testing.assert_array_equal(a.track.z, b.track.z)
        self.assertGreater(np.ptp(a.track.z), .1)
        for _ in range(180):
            action=a.baseline_action()
            oa, ra, ta, ca, ia = a.step(action)
            ob, rb, tb, cb, ib = b.step(action)
        np.testing.assert_array_equal(oa, ob)
        self.assertEqual(ra, rb)
        self.assertEqual(oa.shape, (60,))
        self.assertTrue(np.isfinite(oa).all())
        self.assertGreater(a.car.speed, 3.)
        self.assertGreater(a.progress, .01)
        with self.assertRaises(ValueError): a.step([float('nan'), 0, 0])

    def test_config_rejects_invalid_or_unbounded_inputs(self):
        for data in ({'environments':0}, {'updates':True}, {'learning_rate':float('nan')},
                     {'car':'unknown'}, {'surprise':1}, {'hills':'false'}):
            with self.assertRaises(ValueError): Config.parse(data)

    def test_tanh_policy_density_ratio_is_consistent(self):
        torch.manual_seed(10)
        p=Policy()
        obs=torch.zeros(16,60)
        with torch.no_grad(): raw, old, value=p.sample(obs)
        new, entropy, value=p.evaluate(obs,raw)
        torch.testing.assert_close((new-old).exp(),torch.ones(16))
        self.assertTrue((raw.tanh().abs() <= 1).all())
        self.assertTrue(torch.isfinite(entropy).all())


class DashboardIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='hybrid-lab-tests-')
        cls.root=Path(cls.tmp.name)
        cls.proc=subprocess.Popen([sys.executable,'-m','hybrid_lab','--port','0','--runs',str(cls.root)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True)
        line=cls.proc.stdout.readline().strip()
        if not line.startswith('Hybrid Lab running at '): raise RuntimeError(line)
        cls.url=line.split(' at ')[1]

    @classmethod
    def tearDownClass(cls):
        import signal
        cls.proc.send_signal(signal.SIGINT)
        cls.proc.communicate(timeout=25)
        cls.tmp.cleanup()

    def tearDown(self):
        # One server is shared by the whole class, so a run left alive by one
        # test makes the next test's start() fail with 400 ("Stop the current
        # run before starting another") — correct product behaviour, wrong test
        # isolation. Leave the server idle after every test.
        try:
            state = self.request('state')
        except Exception:
            return
        if state.get('training', {}).get('status') in ('training', 'paused', 'evaluating', 'starting'):
            try:
                self.request('command', dict(op='resume'))
                self.request('command', dict(op='stop'))
                self.wait_for(lambda s: s['training']['status'] in ('stopped', 'failed', 'completed'))
            except Exception:
                pass

    def request(self, route, data=None, headers=None):
        req=urllib.request.Request(self.url+'/api/'+route,
             data=json.dumps(data).encode() if data else None,
             headers=headers or ({'Content-Type':'application/json'} if data else {}))
        with urllib.request.urlopen(req,timeout=10) as response:return json.load(response)

    def wait_for(self, predicate, seconds=25):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            state=self.request('state')
            if predicate(state):return state
            time.sleep(.08)
        self.fail('Timed out. Last state: '+json.dumps(state.get('training')))

    def test_full_train_save_evaluate_watch_resume_lifecycle(self):
        config=dict(name='Integration smoke test',updates=2,environments=1,rollout=16,
                    epochs=1,batch_size=16,episode_seconds=5,eval_every=2,save_every=1,seed=17)
        first=self.request('command',dict(op='start',config=config))['run']
        state=self.wait_for(lambda s:s['training']['status'] in ('completed','failed'))
        self.assertEqual(state['training']['status'],'completed',state['training'])
        self.assertEqual(state['training']['steps'],32)
        latest=self.root/first/'latest.pt'
        policy,payload=load_checkpoint(latest)
        self.assertEqual(payload['update'],2)
        torch.manual_seed(17)
        original=Policy()
        self.assertTrue(any(not torch.equal(policy.state_dict()[k],v) for k,v in original.state_dict().items()))
        rows=self.request('metrics?run='+first)
        self.assertEqual(len(rows),2)
        self.assertTrue(np.isfinite(rows[-1]['value_loss']))
        self.assertTrue((self.root/first/'best.pt').is_file())
        self.request('command',dict(op='watch',run=first))
        state=self.wait_for(lambda s:s.get('watching',{}).get('source')=='Saved PPO')
        self.assertEqual(state['watching']['car'],'supra')
        self.assertTrue(self.request('track')['center'])
        self.request('command',dict(op='viewer',action='pause',paused=True))
        self.wait_for(lambda s:s['watching']['paused'])
        frame=self.request('state')['watching']['frame']
        time.sleep(.15)
        self.assertEqual(frame['elapsed'],self.request('state')['watching']['frame']['elapsed'])
        self.request('command',dict(op='evaluate',run=first))
        end=time.monotonic()+15
        while time.monotonic()<end:
            runs=self.request('runs')
            ev=next(r for r in runs if r['id']==first)['evaluation']
            if ev.get('status')!='evaluating':break
            time.sleep(.1)
        self.assertEqual(ev['status'],'completed',ev)
        before=hashlib.sha256(latest.read_bytes()).hexdigest()
        child=self.request('command',dict(op='start',resume_run=first,additional_updates=1,config={'updates':1}))['run']
        self.assertNotEqual(first,child)
        state=self.wait_for(lambda s:s['training']['status'] in ('completed','failed'))
        self.assertEqual(state['training']['status'],'completed',state['training'])
        self.assertEqual(state['training']['update'],3)
        self.assertEqual(hashlib.sha256(latest.read_bytes()).hexdigest(),before)
        self.assertEqual(json.loads((self.root/child/'lineage.json').read_text())['parent_run'],first)

    def test_pause_save_stop_and_reject_duplicate_training(self):
        run=self.request('command',dict(op='start',config=dict(updates=1000,rollout=64,environments=1,eval_every=1000)))['run']
        self.wait_for(lambda s:s['training']['status']=='training')
        with self.assertRaises(urllib.error.HTTPError):
            self.request('command',dict(op='start',config={}))
        self.request('command',dict(op='pause'))
        self.wait_for(lambda s:s['training']['status']=='paused')
        self.request('command',dict(op='save'))
        self.wait_for(lambda s:bool(s['training'].get('saved_at')))
        self.assertTrue((self.root/run/'latest.pt').is_file())
        self.request('command',dict(op='resume'))
        self.wait_for(lambda s:s['training']['status']=='training')
        self.request('command',dict(op='stop'))
        result=self.wait_for(lambda s:s['training']['status'] in ('stopped','failed'))
        self.assertEqual(result['training']['status'],'stopped',result['training'])
        load_checkpoint(self.root/run/'latest.pt')

    def test_static_assets_and_api_boundaries(self):
        for path in ('/','/style.css','/app.js','/viewer.js','/vendor/three.module.js','/vendor/three.core.js'):
            with urllib.request.urlopen(self.url+path) as r:
                self.assertEqual(r.status,200)
                self.assertGreater(len(r.read()),20)
        for route in ('metrics?run=../../supra','download?run=../../supra&file=config.py'):
            with self.assertRaises(urllib.error.HTTPError) as ctx:self.request(route)
            self.assertEqual(ctx.exception.code,400)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.request('command',dict(op='start'),{'Content-Type':'application/json','Origin':'https://example.com'})
        self.assertEqual(ctx.exception.code,403)


if __name__=='__main__': unittest.main(verbosity=2)
