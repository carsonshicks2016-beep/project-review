"""Dashboard report endpoints on a synthetic run: no server process, no emulators."""
from __future__ import annotations

import json

import pytest

from puffbot import actions, server


def row(t, kind='cpu', result='win', level=9, opponent='FOX', offstage=0.3, **kw):
    acts = [0] * actions.COUNT
    acts[actions.INDEX['up'] if offstage > 0.5 else actions.INDEX['rest']] = 10
    return {'time': t, 'opponent_kind': kind, 'opponent': opponent, 'result': result, 'cpu_level': level,
            'frames': 3600, 'offstage_frames': int(offstage * 3600), 'techs': 1, 'missed_techs': 1,
            'damage_dealt': 100, 'damage_received': 50, 'self_destructs': 0, 'stocks_taken': 4 if result == 'win' else 1,
            'stocks_lost': 1 if result == 'win' else 4, 'stocks_left': 3 if result == 'win' else 0,
            'version': int(t // 100), 'actions': acts, **kw}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    server._games_cache.clear()
    d = tmp_path / 'r1'
    d.mkdir()
    t0 = 1_790_000_000 - 1_790_000_000 % 3600
    games = [row(t0 + 60 * i) for i in range(40)]                                       # hour 1: healthy
    games += [row(t0 + 3600 + 60 * i, result='loss', offstage=0.7) for i in range(40)]  # hour 2: stalling
    games += [row(t0 + 3600 + 60 * i + 30, kind='phillip', opponent='FALCO', result='win' if i == 3 else 'loss')
              for i in range(10)]
    (d / 'games.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in games))
    (d / 'events.jsonl').write_text(json.dumps({'worker': -1, 'time': t0 + 4000, 'error': 'Stalling ALARM: x'}) + '\n')
    (d / 'config.json').write_text(json.dumps({'stall_offstage_warn': 0.5}))
    return d


def test_report_reads_like_a_morning_summary(run):
    r = server.report('r1', hours=0)
    assert [h['cpu9_rate'] for h in r['hours']] == [1.0, 0.0]
    text = ' '.join(r['headline'])
    assert 'went from 100% to 0%' in text
    assert 'won 1 of 10' in text and 'Falco' in text
    assert 'Possible stalling' in text and 'from' in text
    assert 'Health: 0 emulator errors' in text
    assert r['events'][0]['error'].startswith('Stalling ALARM')


def test_log_is_read_incrementally(run):
    assert len(server._all_games(run)) == 90
    with open(run / 'games.jsonl', 'a') as f:
        f.write(json.dumps(row(1_800_000_000)) + '\n' + '{"half a li')
    assert len(server._all_games(run)) == 91, 'a line still being written is not parsed yet'


def test_move_mix_and_phillip_panel(run):
    m = server.moves('r1', buckets=2)
    up = m['names'].index('up')
    assert m['size'] == 50, 'blocks are never smaller than 50 games'
    assert m['buckets'][0]['share'][up] < 0.25 < m['buckets'][-1]['share'][up]
    p = server.phillip_panel('r1')
    assert p['overall']['games'] == 10 and p['overall']['wins'] == 1
    assert p['characters']['FALCO']['wins'] == 1 and len(p['wins']) == 1


def test_scorecards_include_legacy_reports_and_skip_other_protocols(run, tmp_path):
    ev = tmp_path / 'evals'
    base = {'level': 9, 'games_each': 10, 'opponents': server.SCORECARD['opponents'],
            'checkpoint': str(run / 'checkpoints' / 'policy-00048.17M-v3304.pt'),
            'overall': {'wins': 43, 'games': 60, 'win_rate': 0.717, 'win_rate_95ci': [0.59, 0.82]}}
    for name, rep in (('a-old', {**base, 'state': 'finished'}),              # no 'kind', old state name
                      ('b-watch', {**base, 'kind': 'cpu', 'games_each': 3, 'state': 'complete'}),
                      ('c-phillip', {**base, 'kind': 'phillip', 'level': None, 'state': 'complete'})):
        (ev / name).mkdir(parents=True)
        (ev / name / 'report.json').write_text(json.dumps(rep))
    cards = server._scorecards()
    assert [(c['id'], c['state'], c['version'], c['run']) for c in cards] == [('a-old', 'complete', 3304, 'r1')]


def test_impressiveness_rewards_phillip_far_more_than_cpus():
    perfect_cpu = {'games': 60, 'wins': 60, 'taken': 4.0, 'lost': 0.0}
    assert server.impressiveness(perfect_cpu, None)['score'] == 60.0
    some_phillip = {'games': 24, 'wins': 1, 'taken': 2.0, 'lost': 3.9}
    s = server.impressiveness(perfect_cpu, some_phillip)
    assert s['phillip_points'] == 13.1 and s['score'] == 73.1


def test_showcase_ranks_checkpoints_and_pools_repeat_tests(run, tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'CHAMPIONS', tmp_path / 'champions')
    ev = tmp_path / 'evals'

    def report(name, sha, kind, wins, games=6, opponents=None, level=9, games_each=10):
        rows = [{'opponent': 'FALCO', 'result': 'win' if i < wins else 'loss', 'stocks_taken': 4 if i < wins else 1,
                 'stocks_lost': 1 if i < wins else 4} for i in range(games)]
        rep = {'state': 'complete', 'kind': kind, 'level': level if kind == 'cpu' else None, 'games_each': games_each,
               'opponents': opponents or server.SCORECARD['opponents'], 'checkpoint_sha256': sha,
               'checkpoint': str(run / 'checkpoints' / f'policy-{sha}-v1.pt'), 'games': rows}
        (ev / name).mkdir(parents=True)
        (ev / name / 'report.json').write_text(json.dumps(rep))
    report('1', 'aaa', 'cpu', 6)
    report('2', 'bbb', 'cpu', 5)                                   # fewer CPU wins, but beats Phillip
    report('3', 'bbb', 'phillip', 1, opponents=['FALCO'], games_each=6)
    report('4', 'bbb', 'phillip', 1, opponents=['FALCO'], games_each=6)
    report('5', 'ccc', 'cpu', 6, games_each=3)                     # a watch-length test: not a scorecard
    rows = server.showcase()['rows']
    assert [r['sha'] for r in rows] == ['bbb', 'aaa']
    assert rows[0]['phillip']['games'] == 12 and rows[0]['phillip']['wins'] == 2
    assert "🤖 Beat Phillip's Falco ×2" in rows[0]['badges']
