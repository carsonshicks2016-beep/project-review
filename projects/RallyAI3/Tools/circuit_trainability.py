"""Mark an imported circuit as trainable once the physical probe suite has passed on the current player.

    .venv/bin/python Tools/circuit_trainability.py --course ID --prefix stage-c-probe

Reads the probe summaries written by Tools/circuit_probes.py, refuses unless every case
ran against the current build and passed, then records manifest["trainability"]. The
existing /api/courses/{id}/review endpoint still has to be used to mark it reviewed.
"""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rallylab import core

# Every case Tools/circuit_probes.py runs. A partial suite is not a validation.
CASES=['karussell-channel','karussell-bypass','mini-channel','mini-bypass','right-kerb-forward','right-kerb-reverse',
       'left-kerb-forward','left-kerb-reverse','grass-return','flugplatz-crest','fuchsrohre-dip','pflanzgarten-crest',
       'repaired-hillside','seam','karussell-entry-channel','karussell-entry-bypass','grass-depart-return','barrier']
DRIVEN_OUTCOMES={'TimedOut','Finished'}


def evaluate(results,build_hash):
    """results: {case: (summary or None, probe build source hash or None)} -> list of failures."""
    failures=[]
    for case in CASES:
        summary,probe_build=results.get(case,(None,None))
        if summary is None:failures.append(f'{case}: no probe summary');continue
        if probe_build!=build_hash:failures.append(f'{case}: probed on a different player build')
        if summary.get('missingExpectedRegions'):failures.append(f"{case}: never touched {summary['missingExpectedRegions']}")
        if summary.get('wrongSurfaceSamples'):failures.append(f"{case}: {summary['wrongSurfaceSamples']} wrong-surface contact samples")
        outcomes=[r.get('outcome') for r in summary.get('records',[])]
        if not outcomes:failures.append(f'{case}: no episode record');continue
        if case=='barrier':
            if 'HitObstacle' not in outcomes or not summary.get('barrierContactEvidence'):
                failures.append(f'barrier: expected a recorded barrier contact, got {outcomes}')
        elif any(o not in DRIVEN_OUTCOMES for o in outcomes):
            failures.append(f'{case}: unexpected outcome {outcomes}')
    return failures


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--course',required=True);p.add_argument('--prefix',default='stage-c-probe');a=p.parse_args()
    build=core.build_status()
    if build['status']!='current':raise SystemExit('Managed player is not current; rebuild and re-run the probes')
    path=next((p for p in (core.DATA/'courses').glob('*/manifest.json') if core.read(p,{}).get('id')==a.course),None)
    if path is None:raise SystemExit('Unknown course')
    manifest=core.read(path)
    if manifest.get('definition',{}).get('topology')!='circuit':raise SystemExit('Not an imported circuit')
    if not core.course_asset_valid(manifest):raise SystemExit('Course asset failed its identity check')
    results={}
    for case in CASES:
        folder=ROOT/'.rally/circuit-review'/f'{a.prefix}-{case}'
        summary=core.read(folder/'probe-summary.json',None)
        probe_build=(core.read(folder/'build.json',{}) or {}).get('source',{}).get('hash')
        course=(core.read(folder/'course.json',{}) or {}).get('id')
        if summary is not None and course!=a.course:summary=None
        results[case]=(summary,probe_build)
    failures=evaluate(results,build['source']['hash'])
    if failures:raise SystemExit('Not trainable:\n  '+'\n  '.join(failures))
    manifest['trainability']={'stage':'validated','validatedAt':time.time(),'build':build['source']['hash'],
                              'probePrefix':a.prefix,'probes':CASES,
                              'scope':'physical surface/contact probes on the current player; distributed training starts. '
                                      'Not the full runtime fixture matrix or ranked-benchmark eligibility.'}
    core.write(path,manifest);print(json.dumps(manifest['trainability'],indent=2))


if __name__=='__main__':main()
