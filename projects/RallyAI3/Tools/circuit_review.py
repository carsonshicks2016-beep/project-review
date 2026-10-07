"""Bounded, unranked native circuit review. No training and no policy substitution."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import ExitStack

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rallylab import core


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--course',required=True)
    parser.add_argument('--camera',default='ThirdPerson',choices=('ThirdPerson','Driver','Hood','Trackside','Helicopter','Cinematic'))
    parser.add_argument('--height',type=int,default=720,choices=(720,1080))
    parser.add_argument('--landmark',default='Karussell')
    parser.add_argument('--sequence',action='store_true')
    parser.add_argument('--barrier-contact',action='store_true')
    parser.add_argument('--performance-only',action='store_true')
    args=parser.parse_args()
    if not args.label.replace('-','').replace('_','').isalnum():parser.error('Use a simple unique review label')
    build=core.build_status()
    if build['status']!='current':parser.error('Prepare a current player')
    if any(j['state'] in core.ACTIVE for j in core.jobs()):parser.error('Another managed job is active')
    course=next(c for c in core.courses() if c['id']==args.course)
    if not core.course_asset_valid(course) or course['definition'].get('topology')!='circuit':parser.error('Select an intact imported circuit')
    landmark=next(l for l in course['definition']['landmarks'] if l['name']==args.landmark)
    folder=ROOT/'.rally/circuit-review'/args.label
    folder.mkdir(exist_ok=False)
    spec={'schema':1,'mode':'specialist','viewer':True,'evaluation':False,'controlProbe':True,
          'controlMode':'fixed-input' if args.barrier_contact else 'waypoint-follow','controlTargetSpeed':8,
          'controlSteer':.25 if args.barrier_contact else 0,'controlDrive':.2,'controlProbeSeconds':32,'seed':2026,'startingGear':'neutral',
          'courseBundle':course['bundle'],'courseId':course['id'],
          'courseName':course['name']+' / diagnostic reconstruction review','courseFamily':'nordschleife',
          'runId':'circuit-review-'+args.label,'output':str(folder/'attempts'),'timeScale':1,
          'episodeSeconds':course['definition']['episodeSeconds'],'attempts':1}
    core.write(folder/'launch.json',spec);core.write(folder/'build.json',build);core.write(folder/'course.json',course)
    env=dict(os.environ,RALLY_LAB_LAUNCH=str(folder/'launch.json'),RALLY_PRESENTATION_REVIEW=str(folder),
             RALLY_CIRCUIT_AUDIT=str(folder/'circuit-audit.json'),RALLY_REVIEW_CAMERA=args.camera,
             RALLY_CIRCUIT_REVIEW_STATION=str(max(0,landmark['station']-150)),
             RALLY_REVIEW_SEQUENCE='1' if args.sequence else '0',RALLY_FOREST_V3='1')
    if args.performance_only:env.pop('RALLY_CIRCUIT_AUDIT',None)
    with ExitStack() as stack:
        for name in ('compute.lock','viewer.lock'):
            lock=stack.enter_context((core.DATA/name).open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (folder/'stdout.log').open('w') as log:
            result=subprocess.run([build['executable'],'-screen-width',str(1280 if args.height==720 else 1920),
                                   '-screen-height',str(args.height),'-screen-fullscreen','0','-logFile',str(folder/'player.log')],
                                  env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode:raise RuntimeError('Native player failed; inspect player.log')
    if args.performance_only:return
    audit=core.read(folder/'circuit-audit.json')
    if not audit:raise RuntimeError('Native audit did not run')
    print(json.dumps(audit,indent=2))
    if any(audit[k] for k in ('centerFailures','kerbFailures','barrierFailures','projectionFailures','wrongSurfaceContacts','groundFailures','stackedGroundContacts','meshColliderMismatches')):
        raise RuntimeError('Native geometry audit failed')


if __name__=='__main__':main()
