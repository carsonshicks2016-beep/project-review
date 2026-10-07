"""Incrementally read completed match records; never mix demonstrations with agent play."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import threading

KINDS = ('ppo', 'evaluation', 'scripted demonstration', 'human challenge', 'self-play')
OUTCOMES = ('win', 'loss', 'draw', 'timeout', 'interrupted', 'unexpected_restart')


def wilson(wins, n):
    if not n:
        return None
    z = 1.96
    p = wins / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0, center-radius), min(1, center+radius)]


def summarize(rows):
    counts = Counter(r.get('result') for r in rows)
    rewards = [r['return'] for r in rows if isinstance(r.get('return'), (float, int)) and math.isfinite(r['return'])]
    n = len(rows)
    return dict(episodes=n, wins=counts['win'], outcomes={k:counts[k] for k in OUTCOMES},
                win_rate=counts['win']/n if n else None, win_rate_95_interval=wilson(counts['win'], n),
                mean_return=sum(rewards)/len(rewards) if rewards else None)


class MatchIndex:
    def __init__(self):
        self.cache = {}
        self.lock = threading.RLock()

    def read(self, path):
        path = Path(path)
        with self.lock:
            try:
                stat = path.stat()
            except FileNotFoundError:
                self.cache.pop(str(path), None)
                return [], 0
            key = str(path)
            entry = self.cache.get(key)
            if entry is None or entry['inode'] != stat.st_ino or stat.st_size < entry['size'] or (stat.st_size == entry['size'] and stat.st_mtime_ns != entry['mtime']):
                entry = dict(inode=stat.st_ino, offset=0, rows=[], invalid=0, size=0, mtime=0)
                self.cache[key] = entry
            if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime']):
                with path.open('rb') as handle:
                    handle.seek(entry['offset'])
                    block = handle.read()
                complete = block.rfind(b'\n') + 1
                for line in block[:complete].splitlines():
                    if not line.strip(): continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or row.get('kind') not in KINDS or row.get('result') not in OUTCOMES:
                            raise ValueError('Invalid match record')
                        row = dict(row, number=len(entry['rows'])+1)
                        entry['rows'].append(row)
                    except (ValueError, UnicodeDecodeError):
                        entry['invalid'] += 1
                entry.update(offset=entry['offset']+complete, size=stat.st_size, mtime=stat.st_mtime_ns)
            return entry['rows'][:], entry['invalid']

    def query(self, path, kind='ppo', cpu=None, outcome=None, offset=0, limit=25):
        rows, invalid = self.read(path)
        selected = [r for r in rows if (kind == 'all' or r['kind'] == kind) and (cpu is None or r.get('cpu_level') == cpu)]
        groups = defaultdict(list)
        for row in selected:
            groups[(row['kind'], row.get('cpu_level'))].append(row)
        summaries = [dict(kind=k, cpu_level=level, **summarize(values)) for (k,level),values in sorted(groups.items(), key=lambda item:str(item[0]))]
        # One series per policy kind AND CPU level. The chart never averages across these conditions.
        series = []
        for (k,level),values in groups.items():
            width = max(1, math.ceil(len(values)/400))
            points = []
            for i in range(0, len(values), width):
                part = values[i:i+width]
                stats = summarize(part)
                points.append(dict(match=i+len(part), count=len(part), reward=stats['mean_return'], time=part[-1].get('time')))
            series.append(dict(kind=k, cpu_level=level, bin_size=width, points=points))
        filtered = [r for r in selected if outcome is None or r['result'] == outcome]
        reverse = list(reversed(filtered))
        return dict(total=len(filtered), recorded=len(rows), invalid_records=invalid,
                    offset=offset, limit=limit, rows=reverse[offset:offset+limit], groups=summaries,
                    series=series, revision=f'{len(rows)}:{invalid}',
                    denominator='All recorded matches in this policy/CPU group, including timeouts and interruptions.')
