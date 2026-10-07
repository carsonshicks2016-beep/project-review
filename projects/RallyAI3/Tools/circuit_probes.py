"""Bounded physical contact probes, using the real tyre/suspension/vehicle model."""
import json,os,subprocess,fcntl,sys
from pathlib import Path
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rallylab import core


def run(course_id,label,station,lateral=0,reverse=False,mode='waypoint-follow',seconds=14,returning=False,sweep=False):
    b=core.build_status()
    if b['status']!='current' or any(j['state'] in core.ACTIVE for j in core.jobs()):raise RuntimeError('Prepare idle, current player')
    c=next(c for c in core.courses() if c['id']==course_id);folder=ROOT/'.rally/circuit-review'/label;folder.mkdir(exist_ok=False)
    spec={'schema':1,'mode':'specialist','viewer':True,'evaluation':False,'controlProbe':True,'controlMode':mode,'controlTargetSpeed':6,'controlSteer':.25 if mode=='fixed-input' else 0,'controlDrive':.2,'controlProbeSeconds':seconds,'seed':2026,'startingGear':'neutral','courseBundle':c['bundle'],'courseId':c['id'],'courseName':c['name']+' / UNRANKED PHYSICAL PROBE','courseFamily':'nordschleife','runId':label,'output':str(folder/'attempts'),'timeScale':1,'episodeSeconds':1800,'attempts':1}
    core.write(folder/'launch.json',spec);core.write(folder/'build.json',b);core.write(folder/'course.json',c)
    env=dict(os.environ,RALLY_LAB_LAUNCH=str(folder/'launch.json'),RALLY_PRESENTATION_REVIEW=str(folder),RALLY_CIRCUIT_REVIEW_STATION=str(station),RALLY_CIRCUIT_PROBE_LATERAL=str(lateral),RALLY_CIRCUIT_PROBE_REVERSE='1' if reverse else '0',RALLY_CIRCUIT_PROBE_RETURN='1' if returning else '0',RALLY_CIRCUIT_PROBE_SWEEP='1' if sweep else '0',RALLY_CIRCUIT_AUDIT=str(folder/'circuit-audit.json'),RALLY_CIRCUIT_AUDIT_COARSE='1',RALLY_CIRCUIT_PHYSICAL_PROBE=str(folder/'wheel-contacts.jsonl'),RALLY_FOREST_V3='1')
    with ExitStack() as stack:
        for name in ('compute.lock','viewer.lock'):
            lock=stack.enter_context((core.DATA/name).open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (folder/'stdout.log').open('w') as log:
            p=subprocess.run([b['executable'],'-screen-width','1280','-screen-height','720','-screen-fullscreen','0','-logFile',str(folder/'player.log')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=100)
    if p.returncode:raise RuntimeError('Physical probe player failed')
    rows=[json.loads(l) for l in (folder/'wheel-contacts.jsonl').read_text().splitlines()]
    counts={};bad=0
    for sample in rows:
        for contact in sample['contacts']:
            if not contact['grounded']:continue
            region=contact['region'];counts[region]=counts.get(region,0)+1
            if region in ('asphalt','concrete','kerb') and (abs(contact['grip']-1.25)>.001 or contact['looseness']!=0):bad+=1
            if region=='grass' and abs(contact['grip']-.62)>.001:bad+=1
    records=[json.loads(l) for p in (folder/'attempts').glob('episodes-*.jsonl') for l in p.read_text().splitlines()]
    summary={'label':label,'station':station,'lateral':lateral,'reverse':reverse,'contacts':counts,'wrongSurfaceSamples':bad,'records':records,'physics':'real vehicle/suspension/tyre dynamics; heuristic diagnostic only'}
    expected=[]
    if 'channel' in label:expected=['concrete']
    elif 'bypass' in label:expected=['asphalt']
    elif 'kerb' in label:expected=['kerb']
    elif 'grass' in label:expected=['grass','asphalt']
    else:expected=['asphalt']
    summary['expectedRegions']=expected;summary['missingExpectedRegions']=[r for r in expected if not counts.get(r)]
    summary['maxSpeed']=max((r['speed'] for r in rows),default=0)
    summary['barrierContactEvidence']=label.endswith('barrier') and (folder/'barrier-contacts.jsonl').exists()
    core.write(folder/'probe-summary.json',summary);print(json.dumps(summary),flush=True);return folder

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--course',required=True);p.add_argument('--prefix',default='stage-c-probe');a=p.parse_args()
    c=next(c for c in core.courses() if c['id']==a.course);landmarks=c['definition']['landmarks'];at=lambda name:next(m['station'] for m in landmarks if m['name']==name)
    cases=[('karussell-channel',at('Karussell')+40,-3.8,False),('karussell-bypass',at('Karussell')+40,2,False),('mini-channel',at('Mini-Karussell')-60,-3.7,False),('mini-bypass',at('Mini-Karussell')-60,2,False),('right-kerb-forward',at('Aremberg')+60,5.2,False),('right-kerb-reverse',at('Aremberg')+130,5.2,True),('left-kerb-forward',at('Schwedenkreuz')+60,-5.2,False),('left-kerb-reverse',at('Schwedenkreuz')+130,-5.2,True),('grass-return',at('Döttinger Höhe')+600,7.2,False),('flugplatz-crest',at('Flugplatz')-100,0,False),('fuchsrohre-dip',at('Fuchsröhre')+200,0,False),('pflanzgarten-crest',at('Pflanzgarten')-50,0,False),('repaired-hillside',at('Karussell')-150,0,False),('seam',c['definition']['circuitLength']-25,0,False)]
    for name,s,l,r in cases:run(a.course,a.prefix+'-'+name,s,l,r,seconds=34 if 'channel' in name or 'bypass' in name else 14,returning=name=='grass-return')
    for name,lateral in [('karussell-entry-channel',-3.8),('karussell-entry-bypass',2)]:run(a.course,a.prefix+'-'+name,at('Karussell')-105,lateral,seconds=34)
    run(a.course,a.prefix+'-grass-depart-return',at('Döttinger Höhe')+600,7.2,seconds=24,sweep=True)
    run(a.course,a.prefix+'-barrier',at('Karussell')-150,mode='fixed-input')
