"""Source-pinned live causal evaluation; no fitting to held-out outcomes."""
import argparse, fcntl, json, pickle, subprocess, sys, time, resource
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json, space_check
from flygarden.power import on_ac_power
OUT = ROOT/'reports/brain-integration/recovery/odor-causal-v2'
SOURCES = ('scripts/evaluate_odor_candidate.py', 'flygarden/timed_odor_world.py', 'flygarden/odor_only_senses.py', 'flygarden/odor_causal_metrics.py',
           'flygarden/continuous_candidate.py', 'flygarden/candidate_inputs.py', 'flygarden/brain.py',
           'flygarden/descending.py', 'flygarden/body.py', 'flygarden/synchronized.py',
           'flygarden/live_body.py', 'flygarden/live_coupling.py', 'flygarden/local_senses.py',
           'flygarden/local_senses_v2.py', 'flygarden/vision_continuity.py', 'flygarden/world.py',
           'flygarden/causal_metrics.py', 'data/annotations.tsv',
           'vendor/fly-brain/data/2025_Completeness_783.csv', 'vendor/fly-brain/data/2025_Connectivity_783.parquet')
ARMS = ('intact', 'sensory_off', 'steering_cut')


def sources(): return {n: file_sha(ROOT/n) for n in SOURCES}
def cue_for(seed): return 'odor_left' if seed % 2 else 'odor_right'


def prepare():
    import shutil
    OUT.mkdir(exist_ok=True)
    protocol = {'version': 1, 'sources': sources(), 'seeds': list(range(9601, 9621)),
        'screening_seeds': [9761, 9762], 'neural_arms': ARMS, 'duration': 3., 'interval': .025,
        'support_hz': 65, 'cue_assignment': {str(s): cue_for(s) for s in range(9601,9621)},
        'support_only_control': 'sensory_off is exactly support-only: all odor/visual rates zero, unchanged65Hz DNp09 support and supplied gait. One shared arm, not duplicate independent evidence.',
        'steering_intervention': 'Zero every incoming anatomical weight onto both exact-root DNa02 outputs; no cell or edge omitted. Original signs/weights and root IDs retained.',
        'gates': {'causal_motor': 'Positive stratified paired bootstrap effects vs sensory_off and steering_cut',
            'direction': 'Positive paired toward-odor-heading effects vs both controls',
            'useful': 'At least16/20 gain0.05rad toward heading over sensory_off;16/20 depart0.1mm from its final xy;18/20 have no flips. Same numerical movement targets as prior loom test; toward-odor direction declared prospectively for this different task.',
            'replay': 'All20 held-command body/scene replays exactly match saved physics/gait/world state and poses',
            'statistics': '10000 paired stratified resamples, seed8200;97.5% intervals for simultaneous95% coverage of each two-comparison family'},
        'screen_criterion': 'Both non-evaluation seeds deliver positive physical antenna odor exposure and complete without non-finite values; no gain or motor fitting',
        'scope': 'Live physical antenna odor feedback to full-network engineered hybrid; fixed descending decoder and leg controller, no escape override, no learning. Vision input is disabled; finite odor source switches on at0.5s and off at1.5s. No position, route or cue side enters the brain. Fixed starting weights across independent stimulus/gait RNG seeds.'}
    path = OUT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else:
        atomic_json(path, protocol)
        for name in SOURCES:
            if name.startswith('vendor/') or name == 'data/annotations.tsv': continue
            dst = OUT/'source'/name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dst)
    return protocol


def worker(seed, arm):
    import brian2 as b
    from flygarden.continuous_candidate import ContinuousCandidate
    from flygarden.candidate_inputs import CandidateInputs
    from flygarden.live_body import LiveBody
    from flygarden.live_coupling import LiveCoupling
    from flygarden.odor_only_senses import OdorOnlySenses as LocalSensesV2
    from flygarden.timed_odor_world import TimedOdorWorld as CausalStimulusWorld
    from flygarden.world import default_arena
    from flygarden.odor_causal_metrics import odor_metrics as trajectory_metrics
    from scripts.diagnose_recovery_interfaces import mappings
    protocol = prepare()
    folder = OUT/('screen' if seed in protocol['screening_seeds'] else 'trials')/str(seed)/arm
    folder.mkdir(parents=True,exist_ok=False)
    started = time.monotonic(); _, mapping = mappings()
    candidate = ContinuousCandidate(mapping, seed, CandidateInputs(65))
    baseline_weights = np.asarray(candidate.brain.synapses.w[:]/b.mV).copy()
    original_sha = __import__('hashlib').sha256(baseline_weights.tobytes()).hexdigest()
    cut = np.array([],dtype=np.int32)
    if arm == 'steering_cut':
        targets = np.concatenate([candidate.populations[k] for k in ('DNa02_left','DNa02_right')])
        posts = np.asarray(candidate.brain.synapses.j[:],dtype=np.int32)
        cut = np.flatnonzero(np.isin(posts,targets))
        pres = np.asarray(candidate.brain.synapses.i[:],dtype=np.int32)
        np.savez_compressed(folder/'intervention.npz', indices=cut, pre=pres[cut], post=posts[cut],
            original_w_mV=baseline_weights[cut], pre_roots=candidate.brain.ids[pres[cut]], post_roots=candidate.brain.ids[posts[cut]])
        candidate.brain.synapses.w[cut] = 0*b.mV
        baseline_weights[cut] = 0
    expected_sha = __import__('hashlib').sha256(baseline_weights.tobytes()).hexdigest()
    del baseline_weights
    arena = default_arena(); arena['spawn'] = [0.,0.]
    world = CausalStimulusWorld(seed=seed,arena=arena); world.configure(cue_for(seed))
    body = LiveBody(seed=seed, blocks=arena['blocks'],spawn=arena['spawn'],foods=world.arena['foods'])
    loop = LiveCoupling(candidate,body,world,senses=LocalSensesV2.from_body(body))
    initial = body.observation(); frames = []; rows = []
    controller = candidate.manifest(); controller['signed_connectome_weights_modified'] = bool(len(cut))
    manifest = {'status':'running','seed':seed,'arm':arm,'cue':cue_for(seed),'sources':protocol['sources'],
        'controller':controller,'sensory_adapter':loop.senses.version,'initial':initial,
        'baseline_weight_sha256':original_sha,'expected_weight_sha256':expected_sha,
        'interrupted_records':len(cut),'chunks':[],'learning':False,'support_hz':65,
        'support_only_alias':arm=='sensory_off','visual_fixture':world.version}
    atomic_json(folder/'manifest.json',manifest); atomic_json(folder/'geometry.json',body.geometry())
    try:
        for tick in range(120):
            if world.status != 'running': break
            transition = loop.advance(sensory_enabled=arm!='sensory_off')
            frames.extend(transition.pop('frames'))
            row = {**transition,'population_hz':candidate.last['population_hz'],
                   'requested_hz':candidate.last['requested_hz']}
            rows.append(row)
            spike_i,spike_t = candidate.last_spikes
            assert len(spike_i) == candidate.last['spike_counts'].sum()
            space_check(folder,16*1024**2)
            file = folder/f'window-{tick:04d}.npz'
            with file.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, spike_i=spike_i,spike_t=spike_t,
                    counts=candidate.last['spike_counts'], external_i=candidate.last['external_indices'],
                    external_t=candidate.last['external_times'])
            file.with_suffix('.tmp').replace(file)
            state_file = folder/f'scene-{tick:04d}.pkl'
            with state_file.open('wb') as stream:
                # Exact motor replay needs no neural recomputation.
                pickle.dump({'body':body.snapshot(),'world':world.snapshot()},stream,protocol=5)
            manifest['chunks'].append({'file':file.name,'sha256':file_sha(file),
                'state_file':state_file.name,'state_sha256':file_sha(state_file),'end':loop.clock.time})
            atomic_json(folder/'frames.json',frames); atomic_json(folder/'rows.json',rows)
            atomic_json(folder/'manifest.json',manifest)
        assert __import__('hashlib').sha256(np.asarray(candidate.brain.synapses.w[:]/b.mV).tobytes()).hexdigest()==expected_sha
        side = int(cue_for(seed).endswith('right'))
        manifest.update(status='complete',duration=loop.clock.time,outcome=world.status,
            metrics=trajectory_metrics(initial,frames,rows,cue_for(seed)),
            antenna_odor_peak=max(max(v[0] for v in r['sensory']['antenna_odors']) for r in rows),
            wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(folder/'manifest.json',manifest)
        print(json.dumps({k:manifest[k] for k in ('seed','arm','duration','antenna_odor_peak','wall_seconds')}),flush=True)
    except BaseException as exc:
        manifest.update(status='interrupted',error=str(exc));atomic_json(folder/'manifest.json',manifest);raise
    finally: body.close()


def run(screen=False):
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert on_ac_power(), 'Causal batch requires AC power'
        protocol=prepare(); seeds=protocol['screening_seeds'] if screen else protocol['seeds']
        if not screen:
            for seed in protocol['screening_seeds']:
                m=json.loads((OUT/'screen'/str(seed)/'intact/manifest.json').read_text())
                assert m['status']=='complete' and m['antenna_odor_peak']>0 and m['duration']==3.
        for seed in seeds:
            for arm in ('intact',) if screen else ARMS:
                folder=OUT/('screen' if screen else 'trials')/str(seed)/arm
                if folder.exists():
                    m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and m['sources']==protocol['sources'];continue
                assert on_ac_power(),'AC disconnected; completed arms preserved'
                subprocess.run([sys.executable,__file__,'--worker',str(seed),'--arm',arm],cwd=ROOT,check=True)
        print('Screen completed' if screen else 'Neural causal batch completed; independent audit and motor replay still required',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--screen',action='store_true');parser.add_argument('--worker',type=int);parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    if args.worker: worker(args.worker,args.arm)
    else: run(args.screen)
