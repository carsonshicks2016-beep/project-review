"""Exact native body/world replay of the predeclared odor motor timelines."""
import argparse,fcntl,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.live_body import LiveBody
from flygarden.live_coupling import LiveCoupling
from flygarden.odor_only_senses import OdorOnlySenses
from flygarden.timed_odor_world import TimedOdorWorld
from flygarden.world import default_arena
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json
from scripts.check_candidate_full_state import digest
SOURCE=ROOT/'reports/brain-integration/recovery/odor-causal-v2'


class RecordedMotor:
    def __init__(self,rows):self.rows=rows;self.index=0;self.time=0.
    def advance(self,dt,*args):
        command=self.rows[self.index]['next_motor'];self.index+=1;self.time=self.index*dt
        return command


def replay(seed,screen=False):
    source=SOURCE/('screen' if screen else 'trials')/str(seed)/'intact'
    manifest=json.loads((source/'manifest.json').read_text());assert manifest['status']=='complete'
    rows=json.loads((source/'rows.json').read_text());original_frames=json.loads((source/'frames.json').read_text())
    folder=SOURCE/('screen-replay' if screen else 'motor-replay')/str(seed);folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();arena=default_arena();arena['spawn']=[0.,0.]
    world=TimedOdorWorld(seed=seed,arena=arena);world.configure(manifest['cue'])
    body=LiveBody(seed=seed,blocks=arena['blocks'],spawn=arena['spawn'],foods=world.arena['foods'])
    senses=OdorOnlySenses.from_body(body);loop=LiveCoupling(RecordedMotor(rows),body,world,senses=senses)
    frames=[];boundaries=[]
    try:
        for k,row in enumerate(rows):
            result=loop.advance();frames.extend(result.pop('frames'))
            assert result['body']==row['body'] and result['sensory']==row['sensory']
            assert result['applied_motor']==row['applied_motor'] and result['next_motor']==row['next_motor']
            chunk=manifest['chunks'][k];state_path=source/chunk['state_file']
            assert file_sha(state_path)==chunk['state_sha256']
            with state_path.open('rb') as stream:original=pickle.load(stream)
            actual={'body':body.snapshot(),'world':world.snapshot()}
            assert digest(actual)==digest(original),f'Physical replay mismatch at{row["end"]}'
            boundaries.append({'time':row['end'],'state_sha256':digest(actual),'source_state_sha256':chunk['state_sha256']})
            atomic_json(folder/'boundaries.json',boundaries)
        assert frames==original_frames
        result={'status':'passed','seed':seed,'boundaries':len(boundaries),'body_world_and_gait_exact':True,
            'poses_and_senses_exact':True,'brain_recomputed':False,'wall_seconds':time.monotonic()-started,
            'source_manifest_sha256':file_sha(source/'manifest.json'),'checker_sha256':file_sha(Path(__file__))}
        atomic_json(folder/'results.json',result);print(f'Exact odor motor replay passed {seed}',flush=True)
        return result
    finally:body.close()


def main(screen):
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=json.loads((SOURCE/'protocol.json').read_text())
        for name,sha in protocol['sources'].items():assert file_sha(ROOT/name)==sha
        results=[]
        for seed in protocol['screening_seeds'] if screen else protocol['seeds']:
            path=SOURCE/('screen-replay' if screen else 'motor-replay')/str(seed)/'results.json'
            if path.exists():
                result=json.loads(path.read_text());assert result['status']=='passed' and result['checker_sha256']==file_sha(Path(__file__));results.append(result)
            else:results.append(replay(seed,screen))
        atomic_json(SOURCE/('screen-replay-results.json' if screen else 'motor-replay-results.json'),
                    {'status':'passed','records':results,'source_protocol_sha256':file_sha(SOURCE/'protocol.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--screen',action='store_true');args=parser.parse_args();main(args.screen)
