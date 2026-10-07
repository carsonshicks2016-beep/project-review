"""What a saved policy means, and whether this code can load it.

A checkpoint is a set of weights plus the assumptions they were trained under. Loading
it under different assumptions does not crash, it silently plays a different game, so
every policy file records them:

* strict (must match or loading refuses): the macro names and order, and the
  observation layout. Weights for a different vocabulary or feature layout are wrong,
  not merely worse.
* recorded (a mismatch is allowed but reported): frames per decision and the executor
  version. Bug fixes to how macros are played out should not orphan a trained policy,
  but anyone reading an evaluation should be able to see that they happened.

Files saved before this module existed carry only {'space', 'names'}; they were all
trained on observation layout v1 and are upgraded as such.
"""
from __future__ import annotations

from . import actions, observation

STRICT = ('space', 'names', 'observation')


class IncompatibleCheckpoint(ValueError):
    pass


def current(act_every: int | None) -> dict:
    return {
        'space': 'puff-macros-v1',
        'names': list(actions.NAMES),
        'observation': observation.layout(),
        'act_every': None if act_every is None else int(act_every),
        'executor': actions.EXECUTOR_VERSION,
    }


LEGACY_KEYS = {'space', 'names'}


def upgrade(saved: dict | None) -> dict:
    """Fill in what a genuine pre-contract file implies. Only that exact legacy shape is
    assumed to be layout v1; anything else missing its layout is refused by check()
    (GPT step-1 review: a newer-looking contract without a layout used to pass)."""
    saved = dict(saved or {})
    if 'observation' not in saved and set(saved) == LEGACY_KEYS and saved.get('space') == 'puff-macros-v1':
        saved['observation'] = observation.LAYOUT_V1
    saved.setdefault('act_every', None)
    saved.setdefault('executor', 'v1')
    return saved


def check(saved: dict | None, act_every: int | None = None, what: str = 'checkpoint',
          migrate: bool = False) -> list[str]:
    """Raise if `saved` cannot be loaded by this code; return warnings otherwise.
    With migrate=True a step-1 (v1) policy is accepted, to be converted by migrate.py."""
    from . import migrate as mig
    notes = []
    if migrate and mig.is_v1(saved):
        notes.append(f'{what} was {mig.NOTE}.')
        saved = {**upgrade(saved), 'space': 'puff-macros-v1', 'names': list(actions.NAMES),
                 'observation': observation.layout()}
    saved = upgrade(saved)
    here = current(act_every)
    for key in STRICT:
        if saved.get(key) != here[key]:
            raise IncompatibleCheckpoint(f'{what} has a different {key} than this code '
                                         f'({saved.get(key)!r} vs {here[key]!r}).')
    if act_every is not None and saved['act_every'] not in (None, act_every):
        notes.append(f'{what} was trained at {saved["act_every"]} frames per decision, running at {act_every}.')
    if saved['executor'] != here['executor']:
        notes.append(f'{what} was trained with executor {saved["executor"]}; this code uses {here["executor"]}.')
    return notes
