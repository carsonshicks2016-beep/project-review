"""Predeclared sided odor orientation using unchanged physical measurements."""
from .causal_metrics import trajectory_metrics


def odor_metrics(initial,frames,rows,cue):
    if cue not in ('odor_left','odor_right'):raise ValueError('Sided odor cue required')
    result=trajectory_metrics(initial,frames,rows,'loom_left' if cue=='odor_left' else 'loom_right')
    result['toward_heading_radians']=-result.pop('away_heading_radians')
    return result
