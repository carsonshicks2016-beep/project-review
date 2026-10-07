"""Serialized native route inspection. Camera traversal never creates ranked attempts."""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rallylab import core


def run(label,course_id,shots,height=720,capture=True,lifecycle=False):
    build=core.build_status()
    if build['status']!='current':raise ValueError('Prepare a current standalone player')
    if any(j['state'] in core.ACTIVE for j in core.jobs()):raise ValueError('Managed compute work is active')
    course=next(c for c in core.courses() if c['id']==course_id)
    if not core.course_asset_valid(course):raise ValueError('Course integrity check failed')
    if not lifecycle:shots=[{'label':'audit-warmup','start':shots[0]['start'],'end':shots[0]['start'],'duration':12,'camera':'ThirdPerson','overview':False}]+shots
    folder=ROOT/'.rally/circuit-review'/label;folder.mkdir(exist_ok=False)
    core.write(folder/'inspection-plan.json',{'shots':shots,'capture':capture,'captureFps':12})
    spec={'schema':1,'mode':'specialist','viewer':True,'evaluation':False,'controlProbe':True,'controlMode':'waypoint-follow','controlTargetSpeed':8,'controlProbeSeconds':0,'seed':2026,'startingGear':'neutral','courseBundle':course['bundle'],'courseId':course_id,'courseName':course['name']+' / CAMERA INSPECTION','courseFamily':'nordschleife','runId':label,'output':str(folder/'diagnostic-only'),'timeScale':1,'episodeSeconds':1800,'attempts':1}
    core.write(folder/'launch.json',spec);core.write(folder/'build.json',build);core.write(folder/'course.json',course)
    env=dict(os.environ,RALLY_LAB_LAUNCH=str(folder/'launch.json'),RALLY_PRESENTATION_REVIEW=str(folder),RALLY_CIRCUIT_INSPECTION=str(folder/'inspection-plan.json'),RALLY_REVIEW_NO_SCREENSHOTS='1',RALLY_CIRCUIT_AUDIT=str(folder/'circuit-audit.json'),RALLY_FOREST_V3='1')
    if lifecycle:env['RALLY_CIRCUIT_LIFECYCLE']=str(folder/'lifecycle.json')
    timeout=max(180,sum(s['duration']+2 for s in shots)+120) if not lifecycle else 600
    with ExitStack() as stack:
        for name in ('compute.lock','viewer.lock'):
            lock=stack.enter_context((core.DATA/name).open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (folder/'stdout.log').open('w') as log:
            result=subprocess.run([build['executable'],'-screen-width',str(1280 if height==720 else 1920),'-screen-height',str(height),'-screen-fullscreen','0','-logFile',str(folder/'player.log')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError('Native inspection failed: '+str(folder))
    audit=core.read(folder/'circuit-audit.json')
    if not audit or any(audit[k] for k in ('centerFailures','kerbFailures','barrierFailures','groundFailures','stackedGroundContacts','meshColliderMismatches','wrongSurfaceContacts','projectionFailures','degenerateFaces','nonfiniteVertices','vegetationIntrusions','vegetationColliders')):raise RuntimeError('Native geometry audit failed: '+str(audit))
    if any((folder/'diagnostic-only').glob('episodes*.jsonl')):raise RuntimeError('Camera traversal unexpectedly wrote episode records')
    if not lifecycle:
        rows=[json.loads(l) for l in (folder/'coverage.jsonl').read_text().splitlines()]
        for i,s in enumerate(shots):
            seen=[r['station'] for r in rows if r['shot']==i]
            if not seen or abs(min(seen)-min(s['start'],s['end']))>35 or abs(max(seen)-max(s['start'],s['end']))>35:raise RuntimeError('Incomplete capture coverage: '+s['label'])
        coverage=[{'label':s['label'],'requestedStart':s['start'],'requestedEnd':s['end'],'inspectedStart':min(r['station'] for r in rows if r['shot']==i),'inspectedEnd':max(r['station'] for r in rows if r['shot']==i)} for i,s in enumerate(shots)]
        for kind in ('road','overview'):
            regions=[r for r in coverage if r['label'].startswith('region-') and r['label'].endswith('-'+kind)]
            for previous,current in zip(regions,regions[1:]):
                if previous['inspectedEnd']-current['inspectedStart']<100:raise RuntimeError('Region overlap below 100 metres')
        core.write(folder/'measured-coverage.json',coverage)
        if capture:package(folder,shots)
    print(folder,flush=True);return folder


def package(folder,shots):
    from PIL import Image,ImageDraw
    output=folder/'videos';output.mkdir();contact=folder/'contact-sheets';contact.mkdir()
    for i,shot in enumerate(shots):
        frames=sorted((folder/'inspection').glob(f'shot-{i:02}-frame-*.png'))
        rows=[json.loads(l) for l in (folder/'coverage.jsonl').read_text().splitlines() if json.loads(l)['shot']==i]
        listing=folder/f'concat-{i:02}.txt';text=[]
        for j,p in enumerate(frames):
            duration=rows[min(j+1,len(rows)-1)]['wallTime']-rows[min(j,len(rows)-1)]['wallTime'] if j+1<len(frames) else 1/12
            text.append("file '"+str(p)+"'\nduration "+str(max(.001,duration))+"\n")
        listing.write_text(''.join(text))
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-vf','format=yuv420p','-c:v','libx264','-preset','veryfast','-threads','4','-crf','22','-movflags','+faststart',str(output/f'{i:02}-{shot["label"]}.mp4')],check=True)
        # Contact sheets aid review; videos retain continuous traversal.
        selected=frames[::max(1,len(frames)//12)][:12];im=Image.new('RGB',(1280,780),'#202020');draw=ImageDraw.Draw(im)
        for n,p in enumerate(selected):
            pic=Image.open(p);pic.thumbnail((320,240));x=n%4*320;y=n//4*260;im.paste(pic,(x,y));idx=frames.index(p);draw.text((x+4,y+242),f'{shot["label"]} / {rows[min(idx,len(rows)-1)]["station"]:.0f}m',fill='white')
        im.save(contact/f'{i:02}-{shot["label"]}.jpg')
    # PNG frames are retained for inspection until explicitly pruned after delivery.


def main():
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--course',required=True);p.add_argument('--kind',choices=('route','signature','matrix','lifecycle'),default='route');p.add_argument('--height',type=int,choices=(720,1080),default=720);p.add_argument('--no-capture',action='store_true');args=p.parse_args()
    c=next(c for c in core.courses() if c['id']==args.course);L=c['definition']['circuitLength'];marks=c['definition']['landmarks'];shots=[]
    if args.kind=='route':
        for i in range(16):
            for overview in (False,True):shots.append({'label':f'region-{i+1:02}-'+('overview' if overview else 'road'),'start':i*L/16-100,'end':(i+1)*L/16+100,'duration':22,'camera':'ThirdPerson','overview':overview})
    elif args.kind=='signature':
        for name in ('Hatzenbach','Flugplatz','Fuchsröhre','Adenauer Forst','Bergwerk','Karussell','Brünnchen','Pflanzgarten','Mini-Karussell','Galgenkopf','Döttinger Höhe','T13'):
            s=next(m['station'] for m in marks if m['name']==name)
            for camera in ('ThirdPerson','Hood','Driver'):shots.append({'label':name.replace(' ','-')+'-'+camera,'start':s-100,'end':s+200,'duration':8,'camera':camera,'overview':False})
    elif args.kind=='matrix':
        for name in ('Karussell','Mini-Karussell','Hatzenbach','Fuchsröhre','Pflanzgarten','T13'):
            s=next(m['station'] for m in marks if m['name']==name)
            for camera in ('ThirdPerson','Helicopter','Hood','Driver','Trackside','Cinematic'):shots.append({'label':name+'-'+camera,'start':s-80,'end':s+280,'duration':18 if camera=='Cinematic' else 8,'camera':camera,'overview':False})
    else:shots=[{'label':'lifecycle','start':0,'end':0,'duration':500,'camera':'ThirdPerson','overview':False}]
    run(args.label,args.course,shots,args.height,not args.no_capture,args.kind=='lifecycle')
if __name__=='__main__':main()
