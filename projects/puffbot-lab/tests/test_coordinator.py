"""The evaluation coordinator with fake workers and a fake clock: every way a batch can
go missing, without emulators or processes."""
from __future__ import annotations

import pytest

from puffbot.evaluate import Coordinator, Quota


class Proc:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def kill(self):
        self.alive = False

    def join(self, timeout=None):
        pass


class Box(list):
    put = list.append


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def rig(games=10, workers=2, **kw):
    clock = Clock()
    spawned = []

    def spawn(i):
        p, box = Proc(), Box()
        spawned.append((i, p, box))
        return p, box
    make = lambda bid, o, n: {'batch': bid, 'opponent': o, 'games': n}
    c = Coordinator(Quota(['FOX'], games, per_setup=5), 9, make, spawn, workers, clock=clock, **kw)
    c.start()
    return c, clock, spawned


def play(c, i, item, results=None, done=True):
    """Worker i plays `item`: reports each result, then (optionally) says it is done."""
    results = results or ['win'] * item['games']
    for n, r in enumerate(results, 1):
        c.handle({'kind': 'game', 'worker': i, 'batch': item['batch'], 'game_no': n, 'opponent': 'FOX',
                  'opponent_kind': 'cpu', 'cpu_level': 9, 'result': r})
    if done:
        c.handle({'kind': 'setup_done', 'worker': i, 'batch': item['batch'], 'requested': item['games']})


def test_worker_dying_with_a_batch_does_not_hang_the_evaluation():
    """GPT step-1 review R1, its exact reproduction: one worker takes a batch and dies,
    the other finishes its own batch and waits. This used to wait forever at 5/10."""
    c, clock, spawned = rig()
    (_, p0, box0), (_, p1, box1) = spawned
    p0.alive = False                       # died holding batch 1, never reported
    play(c, 1, box1[0])                    # worker 1 finishes batch 2
    assert c.state() is None
    c.tick()                               # the dead worker's lease is reclaimed
    assert c.restarts[0] == 1
    item = [b for _, _, box in spawned[2:] for b in box] + box1[1:]
    assert len(item) == 1 and item[0]['games'] == 5, 'the lost five games are handed out again'
    worker = next(i for i, _, box in spawned if item[0] in box)
    play(c, worker, item[0])
    assert c.state() == 'complete' and c.quota.done['FOX'] == 10


def test_no_restarts_left_means_the_survivor_takes_the_work():
    c, clock, spawned = rig(max_restarts=0)
    (_, p0, box0), (_, p1, box1) = spawned
    p0.alive = False
    play(c, 1, box1[0])
    c.tick()
    assert 0 in c.retired and len(box1) == 2
    play(c, 1, box1[1])
    assert c.state() == 'complete'


def test_silent_worker_holding_a_batch_is_killed_and_its_work_replanned():
    c, clock, spawned = rig(games=5, workers=1, stall=240)
    _, p0, box0 = spawned[0]
    clock.t = 100
    c.handle({'kind': 'status', 'worker': 0, 'frames': 50})
    clock.t = 341                          # 241 s since the last frame progress
    c.tick()
    assert not p0.alive and 'no progress' in c.errors[-1]['error']
    c.tick()                               # dead now: lease reclaimed, worker restarted
    assert c.restarts[0] == 1 and len(spawned) == 2 and spawned[1][2]


def test_heartbeats_without_frame_progress_do_not_count_as_progress():
    """GPT review R6: status messages alone used to keep a stuck worker 'alive'."""
    c, clock, spawned = rig(games=5, workers=1, stall=240)
    _, p0, _ = spawned[0]
    for t in range(0, 300, 10):
        clock.t = t
        c.handle({'kind': 'status', 'worker': 0, 'frames': 1234})   # same count every time
        c.tick()
    assert not p0.alive


def test_duplicate_deliveries_count_once():
    c, clock, spawned = rig(games=5, workers=1)
    item = spawned[0][2][0]
    play(c, 0, item, done=False)
    play(c, 0, item, done=False)           # the same five results arrive again
    assert c.quota.done['FOX'] == 5 and sum(r['rejected'] == 'duplicate delivery' for r in c.rejected) == 5


def test_legitimate_timeouts_never_exhaust_the_budget():
    """GPT review R2: every batch ending after one valid Sudden Death game used to stop
    a ten-game evaluation at 6/10."""
    c, clock, spawned = rig(games=10, workers=1)
    box = spawned[0][2]
    for _ in range(30):
        if c.state():
            break
        item = box[-1]
        play(c, 0, item, results=['draw'])     # one valid game, then the setup ends
    assert c.state() == 'complete' and c.quota.done['FOX'] == 10 and c.quota.wasted['FOX'] == 0


def test_batches_that_produce_nothing_do_exhaust_it():
    c, clock, spawned = rig(games=5, workers=1)
    box = spawned[0][2]
    while c.state() is None:
        play(c, 0, box[-1], results=['capped'])
    assert c.state() == 'incomplete' and c.quota.done['FOX'] == 0


def test_everyone_dead_ends_as_failed_not_a_hang():
    c, clock, spawned = rig(max_restarts=0)
    for _, p, _ in spawned:
        p.alive = False
    c.tick()
    assert c.state() == 'failed'


def test_deadline_ends_an_evaluation_that_cannot_finish():
    c, clock, spawned = rig(games=5, workers=1, deadline=3600)
    clock.t = 3601
    c.handle({'kind': 'status', 'worker': 0, 'frames': 1})
    c.tick()
    assert c.state() == 'incomplete'
