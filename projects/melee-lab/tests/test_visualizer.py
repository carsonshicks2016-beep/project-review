"""Companion reads real run contracts without touching the trainer."""
import json
import os

from melee_lab import layout, visualizer


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_eight_emulators_leave_a_wide_nonoverlapping_strip():
    boxes = layout.grid(8)
    space = layout.visualizer_space(boxes)
    assert space['width'] == layout.SCREEN[0]-2*layout.MARGIN
    assert space['height'] >= 300
    for b in boxes:
        assert (space['x'] >= b['x']+b['width']+layout.GAP or
                space['x']+space['width'] <= b['x']-layout.GAP or
                space['y'] >= b['y']+b['height']+layout.TITLE_BAR+layout.GAP or
                space['y']+space['height'] <= b['y']-layout.GAP)
    assert space['y']+space['height'] <= layout.SCREEN[1]


def test_full_layout_reports_no_space_and_custom_gaps_are_used():
    assert layout.visualizer_space(layout.grid(12)) is None
    space = layout.visualizer_space([dict(x=4, y=32, width=400, height=700)])
    assert space['x'] == 412
    assert space['width'] > 1000


def test_lineage_uses_frozen_resume_metadata_not_mutated_latest(tmp_path):
    parent, child = tmp_path/'runs/parent', tmp_path/'runs/child'
    put(parent/'status.json', dict(steps=9999))
    put(parent/'latest.json', dict(steps=9999))
    put(child/'status.json', dict(steps=140))
    put(child/'request.json', dict(checkpoint='runs/parent/latest.zip',
                                 checkpoint_identity=dict(source='runs/parent/latest.zip', steps=100, sha256='abc')))
    put(child/'input-policy.json', dict(steps=100, imitation_samples=500))
    nodes = visualizer.ancestry(child, tmp_path)
    assert [n['id'] for n in nodes] == ['parent', 'child']
    assert nodes[0]['steps'] == 100
    assert nodes[0]['sha256'] == 'abc'
    assert nodes[0]['kind'] == 'Demonstration learning'
    assert nodes[1]['current'] and nodes[1]['steps'] == 140


def test_cycle_terminates_and_missing_source_is_explicit(tmp_path):
    a, b = tmp_path/'runs/a', tmp_path/'runs/b'
    put(a/'request.json', dict(checkpoint='runs/b/latest.zip'))
    assert visualizer.ancestry(a, tmp_path)[0]['missing']
    put(b/'request.json', dict(checkpoint='runs/a/latest.zip'))
    assert len(visualizer.ancestry(a, tmp_path)) == 2


def test_legacy_ancestry_without_a_snapshot_counter_stays_unknown(tmp_path):
    parent, child = tmp_path/'runs/parent', tmp_path/'runs/child'
    put(parent/'status.json', dict(steps=9999))
    put(child/'request.json', dict(checkpoint='runs/parent/latest.zip'))
    assert visualizer.ancestry(child, tmp_path)[0]['steps'] is None


def test_snapshot_marks_stale_slots_and_excludes_scripted_matches(tmp_path):
    directory = tmp_path/'runs/active'
    put(directory/'status.json', dict(status='running', pid=os.getpid(), envs=2, steps=130, initial_steps=100,
        history=[dict(kind='scripted demonstration', result='win'),dict(kind='ppo', result='loss')]))
    put(directory/'request.json', dict(steps=1000))
    put(directory/'env0/status.json', dict(action='A', frame=10))
    os.utime(directory/'env0/status.json', (10, 10))
    result = visualizer.snapshot(directory, tmp_path, now=30)
    assert result['session_steps'] == 30
    assert all(s['stale'] for s in result['slots'])
    assert len(result['recent']) == 1 and result['recent'][0]['result'] == 'loss'
    assert result['slots'][1]['frame'] is None


def test_launch_is_optional_and_nonblocking(tmp_path, monkeypatch):
    monkeypatch.setattr(visualizer.sys, 'platform', 'darwin')
    calls = []
    monkeypatch.setattr(visualizer.subprocess, 'Popen', lambda *a, **kw: calls.append((a, kw)))
    directory = tmp_path/'runs/run'; directory.mkdir(parents=True)
    visualizer.launch(directory)
    assert len(calls) == 1 and calls[0][1]['start_new_session']
    assert visualizer.read(tmp_path/'.runtime/visualizer-request.json')['directory'] == str(directory)
    put(tmp_path/'visualizer-settings.json', dict(enabled=False))
    visualizer.launch(directory)
    assert len(calls) == 1


def test_launch_failure_does_not_fail_training(tmp_path, monkeypatch):
    monkeypatch.setattr(visualizer.sys, 'platform', 'darwin')
    def fail(*args, **kwargs): raise OSError('fixture spawn failure')
    monkeypatch.setattr(visualizer.subprocess, 'Popen', fail)
    directory = tmp_path/'runs/run'; directory.mkdir(parents=True)
    visualizer.launch(directory)
