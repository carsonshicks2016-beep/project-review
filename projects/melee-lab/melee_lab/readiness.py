"""Report implementation and measured evidence separately. Never infer human skill from CPU wins."""
import shutil
from .state import OBS_SIZE
from .actions import ACTIONS
from .execution import VERSION as EXECUTION_VERSION

def report(manager, checkpoints, ready, issue):
    evidence={e['id']:e for c in checkpoints for e in c.get('evaluations',[])}
    benchmarks=[e for e in evidence.values() if e.get('benchmark')]
    challenges=list(manager.runs.glob('*/challenge.json'))
    best=max(benchmarks,key=lambda e:e.get('win_rate',0),default=None)
    demonstrations=[d for d in manager.datasets() if d.get('samples')]
    samples=sum(d['samples'] for d in demonstrations)
    comparisons=[r for r in (manager.cached_json(p) for p in manager.runs.glob('*/results.json')) if r.get('gate')]
    promotable=sum(1 for r in comparisons for c in r['gate']['candidates'] if c['eligible'])
    unverified=[e for e in evidence.values() if not e.get('execution_verified')]
    return dict(label='Superhuman performance unproven',
        explanation='CPU wins measure a baseline. Strong human opponents, held-out matchups, and repeated trials are still required.',
        architecture=dict(policy='PPO · 256 × 256 MLP',observation_size=OBS_SIZE,history_decisions=2,actions=len(ACTIONS),
                          stages=['FINAL_DESTINATION'],stocks=3,recurrent=False,
                          replay_imitation=bool(demonstrations),demonstration_samples=samples,
                          execution_rules=EXECUTION_VERSION,observation_delay_frames=0),
        evidence=dict(evaluation_trials=len(evidence),evaluation_games=sum(e.get('episodes',0) for e in evidence.values()),
                      benchmark_trials=len(benchmarks),human_sessions=len(challenges),best_benchmark=best),
        checks=[
            dict(name='Game connection',status='ready' if ready else 'blocked',detail=issue or 'Dolphin and game image paths verified; live startup is checked per run.'),
            dict(name='Saved opponent',status='ready' if checkpoints else 'needed',detail=f'{len(checkpoints)} checkpoint artifacts available.'),
            dict(name='Human challenge',status='implemented',detail='Real-time P2 keyboard or standard gamepad, disconnect release, automatic match setup.'),
            dict(name='Unattended recovery',status='implemented',detail='Three startup attempts per emulator, up to 12 local recoveries, then four bounded worker restarts from checkpoints.'),
            dict(name='Frame-level inputs',status='implemented',detail='Optional expanded controller with one-frame decisions; old policy heads require explicit transfer and fresh evaluation.'),
            dict(name='Frozen-opponent training',status='implemented',detail='Train against an immutable policy snapshot with independent P2 observations.'),
            dict(name='Roster benchmark',status='measured' if benchmarks else 'needed',detail=f'{len(benchmarks)} completed CPU-9 benchmark trials; six matchups per trial.'),
            dict(name='Human performance',status='unproven',detail=f'{len(challenges)} recorded local challenge sessions. No verified expert-human rating.'),
            dict(name='Expert replay learning',status='implemented' if demonstrations else 'missing',
                 detail=f'{len(demonstrations)} parsed tournament datasets, {samples:,} demonstration samples, usable as a cloning anchor during PPO. Cloning alone does not reach CPU-level play; it is a starting point, not evidence.'),
            dict(name='Symmetric controller rules',status='implemented',
                 detail=f'Both policies run {EXECUTION_VERSION}: identical recovery assistance with independent state, and every report carries its mode and its count of altered inputs. '
                        + (f'{len(unverified)} older evaluations predate it and cannot say how they were played.' if unverified else 'Every retained evaluation is labelled.')),
            dict(name='Historical opponent league',status='implemented',
                 detail='Training can sample a frozen panel of champion, historical policies, and varied CPU 7-9 characters, hashed so the panel cannot change under a run. Whether it beats a single frozen opponent is not yet measured.'),
            dict(name='Promotion gauntlet',status='measured' if comparisons else 'needed',
                 detail=f'{len(comparisons)} checkpoint comparisons under matched conditions, both ports, with self mirrors; {promotable} candidate(s) have cleared the gate. Promotion is never automatic.'),
            dict(name='Recurrent policy & stage coverage',status='missing',detail='Two observations, fixed macro actions, Final Destination only. Existing policies keep their compatible contract.')],
        storage=dict(free_gb=round(shutil.disk_usage(manager.root).free/1024**3,1)))
