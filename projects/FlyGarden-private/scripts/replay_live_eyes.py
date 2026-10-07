"""Reconstruct exact body/world replay and retain native eye images."""
import fcntl,json,pickle,sys,time,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.live_body import LiveBody
from flygarden.live_coupling import LiveCoupling
from flygarden.local_senses_v2 import LocalSensesV2
from flygarden.causal_stimulus_world import CausalStimulusWorld
from flygarden.world import default_arena
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json,space_check
from scripts.check_candidate_full_state import digest
SOURCE=ROOT/'reports/brain-integration/recovery/live-causal-v2/screen'


class MotorReplay:
    def __init__(self,rows): self.time=0.;self.rows=rows;self.index=0
    def advance(self,dt,*args):
        command=self.rows[self.index]['next_motor'];self.index+=1;self.time=self.index*dt
        return command


class EyeRecorder(LocalSensesV2):
    def sample(self,world,observation,images,*args,**kwargs):
        self.images=np.asarray(images).copy()
        return super().sample(world,observation,images,*args,**kwargs)


def replay(seed,output):
    source=SOURCE/str(seed)/'intact';manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['status']=='complete'
    rows=json.loads((source/'rows.json').read_text());original_frames=json.loads((source/'frames.json').read_text())
    folder=output/str(seed);folder.mkdir();arena=default_arena();arena['spawn']=[0.,0.]
    world=CausalStimulusWorld(seed=seed,arena=arena);world.configure(manifest['cue'])
    body=LiveBody(seed=seed,blocks=arena['blocks'],spawn=arena['spawn'])
    senses=EyeRecorder.from_body(body);loop=LiveCoupling(MotorReplay(rows),body,world,senses=senses)
    frames=[];images=[];features=[];started=time.monotonic()
    try:
        for index,row in enumerate(rows):
            transition=loop.advance();frames.extend(transition.pop('frames'))
            assert transition['sensory']==row['sensory'],'Eye feature replay mismatch'
            assert transition['body']==row['body'],'Body observation replay mismatch'
            chunk=manifest['chunks'][index];state=source/chunk['state_file']
            assert file_sha(state)==chunk['state_sha256']
            with state.open('rb') as stream:original=pickle.load(stream)
            assert digest({'body':body.snapshot(),'world':world.snapshot()})==digest(original)
            images.append(senses.images);features.append(transition['sensory'])
        assert frames==original_frames,'Recorded pose replay mismatch'
        space_check(folder,200*1024**2)
        with (folder/'eyes.npz').open('wb') as stream:
            np.savez_compressed(stream,frames=np.asarray(images),baseline=senses.eye.baseline,
                                times=np.array([r['start'] for r in rows]))
        atomic_json(folder/'features.json',features)
        # Scientific contact sheet of raw views and difference masks.
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(4,4,figsize=(12,10))
        for k,index in enumerate((20,30,40,50)):
            for side in range(2):
                image=images[index][side];lum=senses.eye.luminance(images[index])[side]
                axes[k,side*2].imshow(image);axes[k,side*2+1].imshow(senses.eye.baseline[side]-lum>.12,cmap='gray')
                for ax in axes[k,side*2:side*2+2]:ax.axis('off')
                axes[k,side*2].set_title(f'{index*.025:.2f}s {("left","right")[side]}')
        fig.tight_layout();fig.savefig(folder/'eye-contact-sheet.png');plt.close(fig)
        atomic_json(folder/'results.json',{'status':'passed','seed':seed,'windows':len(rows),
            'complete_body_world_states_exact':True,'poses_exact':True,'eye_features_exact':True,
            'brain_recomputed':False,'eyes_sha256':file_sha(folder/'eyes.npz'),
            'source_manifest_sha256':file_sha(source/'manifest.json'),'wall_seconds':time.monotonic()-started})
        print(f'Exact eye replay passed {seed}',flush=True)
    finally:body.close()


def main():
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=ROOT/'reports/brain-integration/recovery'/f'live-eye-replay-{time.time_ns()}';output.mkdir()
        files=('scripts/replay_live_eyes.py','flygarden/live_body.py','flygarden/live_coupling.py',
            'flygarden/local_senses_v2.py','flygarden/vision_continuity.py','flygarden/causal_stimulus_world.py')
        atomic_json(output/'protocol.json',{'sources':{n:file_sha(ROOT/n) for n in files},'seeds':[9751,9752],
            'scope':'Exact motor replay with fresh native images; neural records reused,not recomputed'})
        for name in files:
            target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        print(output,flush=True)
        for seed in (9751,9752):replay(seed,output)


if __name__=='__main__':main()
