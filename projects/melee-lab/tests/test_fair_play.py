"""Both policies play by the same rules, and a comparison is evidence or nothing.

These cover the pieces that decide whether a measured difference between two
checkpoints means anything: identical controller processing for either port, an
opponent panel that cannot be edited underneath a run, and a comparison suite that
refuses a job whose report does not match what was asked for.
"""
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
import melee
import numpy as np
import pytest
from melee_lab.catalog import Catalog
from melee_lab.config import Config
from melee_lab.execution import VERSION, ControllerRules, perspective
from melee_lab.gauntlet import make_jobs, prepare, promotion_gate, run_suite, summarize_job
from melee_lab.league import League, snapshot_league
from melee_lab.storage import action_contract, write_json
from test_dashboard import lab


def player(x, y, *, off_stage=False, on_ground=False, jumps=1, character='FOX', action='FALLING'):
    p = melee.PlayerState()
    p.position.x, p.position.y = x, y
    p.off_stage, p.on_ground, p.jumps_left = off_stage, on_ground, jumps
    p.character = melee.Character[character]
    p.action = melee.Action[action]
    p.stock, p.percent = 3, 0
    return p


def state(first, second):
    g = melee.GameState()
    g.players[1], g.players[2] = first, second
    return g


OFFSTAGE = dict(off_stage=True, on_ground=False, jumps=0)


def test_both_ports_are_processed_by_the_same_rules():
    """The learner used to get recovery help its frozen opponent did not, so the two
    networks were not playing the same game and a win rate between them meant nothing."""
    action = np.array([10, 3, 1, 1, 0, 0, 2], dtype=np.int64)
    as_p1 = ControllerRules().apply(action, state(player(-90, -20, **OFFSTAGE), player(0, 0, on_ground=True, jumps=2)), 1)
    as_p2 = ControllerRules().apply(action, state(player(0, 0, on_ground=True, jumps=2), player(-90, -20, **OFFSTAGE)), 2)
    np.testing.assert_array_equal(as_p1, as_p2)
    assert not np.array_equal(as_p1, action)


def test_each_player_keeps_its_own_recovery_state():
    charging = state(player(-95, -30, **OFFSTAGE, action='SWORD_DANCE_3_MID_AIR'),
                     player(95, -30, **OFFSTAGE, action='SWORD_DANCE_3_MID_AIR'))
    one, two = ControllerRules(), ControllerRules()
    one.apply(np.array([48, 0, 0, 1, 0, 0, 0]), charging, 1)
    two.apply(np.array([48, 0, 0, 1, 0, 0, 0]), charging, 2)
    # Both are mid-Firefox on opposite sides, so a shared latch would send one of them
    # out to sea. The angles must differ and must each point back at the stage.
    assert one.firefox_latch != two.firefox_latch
    one.reset()
    assert one.firefox_latch is None and two.firefox_latch is not None


def test_raw_mode_hands_the_network_output_straight_to_the_pad():
    action = np.array([10, 3, 1, 1, 0, 0, 2], dtype=np.int64)
    dying = state(player(-90, -20, **OFFSTAGE), player(0, 0, on_ground=True, jumps=2))
    raw = ControllerRules('raw')
    np.testing.assert_array_equal(raw.apply(action, dying, 1), action)
    assert raw.report() == dict(version=VERSION, mode='raw', decisions=1, changed_decisions=0, override_rate=0.)
    assisted = ControllerRules('assisted')
    assisted.apply(action, dying, 1)
    assert assisted.report()['changed_decisions'] == 1
    with pytest.raises(ValueError):
        ControllerRules('helpful')


def test_perspective_swaps_projectile_ownership_without_touching_the_world():
    """Swapping players alone left P2 reading its own laser as an incoming one."""
    g = state(player(0, 0, on_ground=True), player(40, 0, on_ground=True))
    shot = melee.Projectile()
    shot.owner = 1
    g.projectiles = [shot]
    view = perspective(g, 2)
    assert view.players[1] is g.players[2] and view.players[2] is g.players[1]
    assert view.projectiles[0].owner == 2
    assert g.projectiles[0].owner == 1 and g.players[1] is not view.players[1]
    assert perspective(g, 1) is g


def manifest(tmp_path, entries):
    path = tmp_path / 'league.json'
    write_json(path, dict(version=1, opponents=entries))
    return path


def test_league_refuses_a_panel_that_changed_underneath_it(tmp_path):
    import hashlib
    weights = tmp_path / 'opponent-0.zip'
    weights.write_bytes(b'policy weights')
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    entry = dict(type='policy', weight=.5, path='opponent-0.zip', sha256=digest, character='FOX')
    cpu = dict(type='cpu', weight=.5, character='MARTH', level=9)
    League(manifest(tmp_path, [entry, cpu]))
    weights.write_bytes(b'different weights')
    with pytest.raises(ValueError):
        League(manifest(tmp_path, [entry, cpu]))
    weights.write_bytes(b'policy weights')
    for broken in ([dict(entry, weight=0), cpu], [dict(cpu, level=12)], []):
        with pytest.raises(ValueError):
            League(manifest(tmp_path, broken))


def test_league_samples_opponents_at_their_declared_weights(tmp_path):
    entries = [dict(type='cpu', weight=3., character='FOX', level=9),
               dict(type='cpu', weight=1., character='MARTH', level=7)]
    league = League(manifest(tmp_path, entries))
    rng = np.random.default_rng(0)
    draws = [league.sample(rng)['character'] for _ in range(4000)]
    assert abs(draws.count('FOX') / len(draws) - .75) < .03
    assert league.sample(rng).get('checkpoint') is None


def controller_checkpoint(root, run='source', name='latest', steps=100, body=b''):
    config = Config(action_set='controller', action_frames=1, character='FOX')
    directory = root / 'runs' / run
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + '.zip')
    with zipfile.ZipFile(path, 'w') as out:
        out.writestr('data', json.dumps({'num_timesteps': steps}))
        if body: out.writestr('weights', body)
    from melee_lab.state import SCHEMA, OBS_SIZE
    write_json(path.with_suffix('.json'), dict(schema=SCHEMA, observation_size=OBS_SIZE,
        actions=action_contract(config), config=asdict(config), steps=steps))
    return path


def test_league_snapshot_freezes_its_panel_and_keeps_cpu_coverage(tmp_path):
    from melee_lab.slot import COMPETITIVE_ROSTER
    catalog = Catalog(tmp_path)
    champion = controller_checkpoint(tmp_path, 'champion')
    history = controller_checkpoint(tmp_path, 'history', body=b'older')
    config = Config(action_set='controller', action_frames=1, character='FOX')
    path = snapshot_league(catalog, tmp_path / 'league', str(champion.relative_to(tmp_path)),
                           [str(history.relative_to(tmp_path))], config)
    entries = json.loads(path.read_text())['opponents']
    shares = {role: sum(e['weight'] for e in entries if e['role'] == role) for role in ('champion', 'history', 'coverage')}
    assert shares == pytest.approx({'champion': .4, 'history': .4, 'coverage': .2})
    assert {e['character'] for e in entries if e['type'] == 'cpu'} == set(COMPETITIVE_ROSTER)
    assert {e['level'] for e in entries if e['type'] == 'cpu'} == {7, 8, 9}
    # Copies, not references: retraining the source cannot alter a running league.
    League(path)
    champion.write_bytes(champion.read_bytes() + b'retrained')
    League(path)


def test_comparison_plan_plays_every_pair_in_both_ports():
    jobs = make_jobs(3, 10, roster=True, modes=('assisted', 'raw'))
    duels = {(j['p1'], j['p2']) for j in jobs if j['kind'] == 'policy'}
    assert duels == {(a, b) for a in range(3) for b in range(3)}
    cpu = {(j['character'], j['level']) for j in jobs if j['kind'] == 'cpu'}
    assert ('FOX', 8) in cpu and ('FOX', 9) in cpu and len(cpu) == 7
    assert {j['mode'] for j in jobs} == {'assisted', 'raw'}
    assert len([j for j in jobs if j['kind'] == 'cpu']) == 3 * 7 * 2
    # Two candidates against Fox 8 and 9, two self mirrors, and the pair in both ports.
    assert len(make_jobs(2, 10, roster=False, modes=('assisted',))) == 4 + 2 + 2


def report(mode='assisted', results=('win',) * 10, version=VERSION, strict=True):
    return dict(results=list(results), execution_version=version, execution_mode=mode,
                strict_cpu_level=strict, execution=[])


def test_a_job_counts_only_when_its_report_answers_the_question_asked():
    job = dict(kind='cpu', p1=0, p2=None, character='FOX', level=9, mode='assisted', episodes=10)
    assert summarize_job(job, report())['valid']
    assert summarize_job(job, report())['wins'] == 10
    # Wrong execution rules, an unverifiable build, a truncated run, an outcome the
    # evaluator never writes, and a match played at a level nobody asked for.
    assert not summarize_job(job, report(mode='raw'))['valid']
    assert not summarize_job(job, report(version='legacy'))['valid']
    assert not summarize_job(job, report(results=('win',) * 9))['valid']
    assert not summarize_job(job, report(results=('win',) * 9 + ('crash',)))['valid']
    assert not summarize_job(job, report(strict=False))['valid']
    assert not summarize_job(job, {})['valid']


def gate_rows(episodes, incumbent_wins, candidate_wins, duel_wins, modes=('assisted',)):
    """Fabricate a complete two-candidate suite at a chosen level of dominance."""
    jobs = make_jobs(2, episodes, roster=False, modes=modes)
    rows = []
    for job in jobs:
        if job['kind'] == 'cpu':
            wins = candidate_wins if job['p1'] else incumbent_wins
        elif job['p1'] == job['p2']:
            wins = episodes // 2
        else:
            wins = duel_wins if job['p1'] == 1 else episodes - duel_wins
        rows.append(summarize_job(job, report(job['mode'], ('win',) * wins + ('loss',) * (episodes - wins))))
    return rows, jobs


def test_promotion_needs_more_than_a_winning_record():
    modes = ('assisted',)
    # A ten-game screen is never enough, however lopsided it looks.
    rows, jobs = gate_rows(10, 5, 10, 10, modes)
    gate = promotion_gate(rows, 2, modes, jobs)
    assert gate['automatic_promotion'] is False
    assert not gate['candidates'][0]['eligible']
    assert any('insufficient self-mirror' in r for r in gate['candidates'][0]['reasons'])
    # The same margin over enough games clears it.
    rows, jobs = gate_rows(120, 60, 100, 96, modes)
    assert promotion_gate(rows, 2, modes, jobs)['candidates'][0]['eligible']
    # Beating the incumbent head to head does not excuse losing to the CPU it beat.
    rows, jobs = gate_rows(120, 110, 30, 96, modes)
    reasons = promotion_gate(rows, 2, modes, jobs)['candidates'][0]['reasons']
    assert any('non-regression unproven' in r for r in reasons)
    # An unfinished suite is never eligible, whatever the finished jobs say.
    rows, jobs = gate_rows(120, 60, 100, 96, modes)
    assert not promotion_gate(rows[:-1], 2, modes, jobs)['candidates'][0]['eligible']


def test_threshold_does_not_move_as_the_suite_progresses():
    """The family size is the plan, not the completed rows: the same evidence must
    not pass or fail depending on when the gate happens to be read."""
    rows, jobs = gate_rows(120, 60, 100, 96)
    early = promotion_gate(rows[:3], 2, ('assisted',), jobs)
    late = promotion_gate(rows, 2, ('assisted',), jobs)
    assert early['comparisons'] == late['comparisons']


class FakeWorker:
    """Stands in for melee_lab.worker, which needs an emulator."""
    outcomes = {}
    commands = []

    def __init__(self, command, **kwargs):
        FakeWorker.commands.append(command)
        run_dir = Path(command[command.index('--run-dir') + 1])
        episodes = int(command[command.index('--episodes') + 1])
        config = json.loads((run_dir / 'config.json').read_text())
        wins = FakeWorker.outcomes.get(run_dir.name, episodes)
        write_json(run_dir / 'evaluation.json', report(config['execution_mode'],
                   ('win',) * wins + ('loss',) * (episodes - wins)))
        self.returncode = 0

    def poll(self): return 0
    def wait(self, timeout=None): return 0
    def terminate(self): pass


@pytest.fixture
def suite(tmp_path, monkeypatch):
    monkeypatch.setattr('melee_lab.gauntlet.subprocess.Popen', FakeWorker)
    monkeypatch.setattr('melee_lab.gauntlet.time.sleep', lambda *a: None)
    FakeWorker.commands = []; FakeWorker.outcomes = {}
    catalog = Catalog(tmp_path)
    incumbent = controller_checkpoint(tmp_path, 'incumbent')
    candidate = controller_checkpoint(tmp_path, 'candidate', body=b'newer')
    directory = tmp_path / 'runs' / 'compare'
    names = [str(p.relative_to(tmp_path)) for p in (incumbent, candidate)]
    request = prepare(catalog, directory, names, Config(action_set='controller', action_frames=1),
                      episodes=4, roster=False, modes=('assisted',))
    write_json(directory / 'control.json', {})
    return directory, request


def test_comparison_runs_every_job_against_frozen_copies(suite):
    directory, request = suite
    run_suite(directory)
    results = json.loads((directory / 'results.json').read_text())
    assert len(results['rows']) == len(request['jobs']) == len(FakeWorker.commands)
    assert json.loads((directory / 'status.json').read_text())['status'] == 'completed'
    # Every job played a copy the suite owns, never the run that is still training.
    for command in FakeWorker.commands:
        for flag in ('--checkpoint', '--opponent-checkpoint'):
            if flag in command:
                assert Path(command[command.index(flag) + 1]).parent == directory / 'candidates'
    assert results['gate']['automatic_promotion'] is False


def test_comparison_resumes_without_replaying_finished_jobs(suite):
    directory, request = suite
    run_suite(directory)
    played = len(FakeWorker.commands)
    FakeWorker.commands = []
    run_suite(directory)
    assert FakeWorker.commands == []
    # A job whose evidence no longer answers its question is played again.
    case = directory / 'job-0000'
    write_json(case / 'evaluation.json', report(version='legacy'))
    run_suite(directory)
    assert len(FakeWorker.commands) == 1 and played > 1


def test_comparison_accepts_valid_report_after_worker_cleanup_error(suite, monkeypatch):
    directory, request = suite
    class CleanupWorker(FakeWorker):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            run_dir = Path(command[command.index('--run-dir') + 1])
            write_json(run_dir / 'status.json', {'status': 'completed'})
            self.returncode = 1
    monkeypatch.setattr('melee_lab.gauntlet.subprocess.Popen', CleanupWorker)
    run_suite(directory)
    assert json.loads((directory / 'status.json').read_text())['status'] == 'completed'
    assert len(json.loads((directory / 'results.json').read_text())['rows']) == len(request['jobs'])


def test_comparison_stops_on_request_and_keeps_what_it_measured(suite):
    directory, request = suite
    write_json(directory / 'control.json', {'stop': True})
    run_suite(directory)
    status = json.loads((directory / 'status.json').read_text())
    assert status['status'] == 'stopped' and FakeWorker.commands == []
    assert not (directory / 'results.json').exists()


def test_comparison_refuses_candidates_that_cannot_be_compared(tmp_path):
    catalog = Catalog(tmp_path)
    fox = controller_checkpoint(tmp_path, 'fox')
    config = Config(action_set='controller', action_frames=1)
    names = [str(fox.relative_to(tmp_path))]
    with pytest.raises(ValueError):
        prepare(catalog, tmp_path / 'one', names, config)
    with pytest.raises(ValueError):
        prepare(catalog, tmp_path / 'same', names * 2, config)
    with pytest.raises(ValueError):
        prepare(catalog, tmp_path / 'zero', names * 2, config, episodes=0)


def test_compare_endpoint_freezes_candidates_and_serves_progress(lab, monkeypatch):
    from types import SimpleNamespace
    import os
    manager, client = lab
    controller_checkpoint(manager.root, 'incumbent')
    controller_checkpoint(manager.root, 'candidate', body=b'newer')
    monkeypatch.setattr('melee_lab.manager.subprocess.Popen',
                        lambda *a, **kw: SimpleNamespace(pid=os.getpid(), poll=lambda: None))
    body = dict(candidates=['runs/incumbent/latest.zip', 'runs/candidate/latest.zip'],
                episodes=5, roster=False, modes=['assisted'])
    response = client.post('/api/compare', json=body)
    assert response.status_code == 200, response.text
    run_id = response.json()['id']
    directory = manager.runs / run_id
    assert (directory / 'candidates/candidate-0.zip').exists()
    assert (directory / 'candidates/candidate-1.zip').exists()
    # The frozen copies stay out of the checkpoint library, which lists trained policies.
    assert not any(c['path'].startswith(f'runs/{run_id}/') for c in client.get('/api/state').json()['checkpoints'])
    payload = client.get(f'/api/compare/{run_id}').json()
    assert payload['jobs'] == len(json.loads((directory / 'comparison.json').read_text())['jobs'])
    assert payload['rows'] == [] and payload['episodes'] == 5
    assert client.get('/api/state').json()['runs'][0]['mode'] == 'compare'
    # One candidate, a repeated candidate, and an unknown path are all rejected.
    for bad in (dict(body, candidates=body['candidates'][:1]),
                dict(body, candidates=[body['candidates'][0]] * 2),
                dict(body, candidates=['runs/incumbent/latest.zip', '../escape.zip'])):
        assert client.post('/api/compare', json=bad).status_code in (400, 409, 422)
    assert client.get('/api/compare/missing').status_code == 404
