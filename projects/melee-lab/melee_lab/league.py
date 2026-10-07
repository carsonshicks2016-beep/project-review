"""Immutable opponent panel sampled once per episode."""
import hashlib
import json
from pathlib import Path
import numpy as np


class League:
    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text())
        if data.get('version') != 1 or not data.get('opponents'):
            raise ValueError('Invalid league manifest.')
        self.entries = data['opponents']
        weights = np.asarray([e['weight'] for e in self.entries], dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('League weights must be positive and finite.')
        self.probabilities = weights/weights.sum()
        for entry in self.entries:
            if entry['type'] == 'policy':
                artifact = self.path.parent/entry['path']
                if hashlib.sha256(artifact.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError('League checkpoint changed: '+str(artifact))
            elif entry['type'] != 'cpu' or not 1 <= entry['level'] <= 9:
                raise ValueError('Invalid league opponent.')

    def sample(self, rng):
        entry = dict(self.entries[int(rng.choice(len(self.entries), p=self.probabilities))])
        if entry['type'] == 'policy':
            entry['checkpoint'] = str((self.path.parent/entry['path']).resolve())
        return entry


def snapshot_league(catalog, directory, champion, historical, config):
    """40% champion, 30% history, 20% CPU roster, 10% history challenge pool.

    Until matchup evidence identifies exploiters, the challenge allocation samples
    history uniformly. No invented weakness scores influence matchmaking.
    """
    from dataclasses import replace
    from .storage import validate_checkpoint, write_json
    from .slot import COMPETITIVE_ROSTER
    directory = Path(directory)
    entries = []
    historical = list(dict.fromkeys(historical))
    if not historical:
        historical = [champion]
    for i, (source, weight, role) in enumerate(
        [(champion, .4, 'champion')] + [(p, .4/len(historical), 'history') for p in historical]):
        source = catalog.resolve(source)
        meta = json.loads(source.with_suffix('.json').read_text())
        opponent_config = replace(config, character=meta['config']['character'], randomize_opponent=True)
        validate_checkpoint(source, opponent_config)
        target = directory/f'opponent-{i}.zip'
        identity = catalog.snapshot(source, target)
        entries.append(dict(type='policy', role=role, weight=weight, path=target.name,
                            sha256=identity['sha256'], character=opponent_config.character,
                            source=identity['source']))
    for character in COMPETITIVE_ROSTER:
        for level in (7, 8, 9):
            entries.append(dict(type='cpu', role='coverage', weight=.2/(len(COMPETITIVE_ROSTER)*3),
                                character=character, level=level))
    path = directory/'league.json'
    write_json(path, dict(version=1, execution_mode=config.execution_mode, opponents=entries,
                         challenge_policy='Uniform historical sampling until validated weakness evidence exists.'))
    return path
