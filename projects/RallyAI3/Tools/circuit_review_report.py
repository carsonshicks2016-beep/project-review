"""Package inspected native evidence without changing course eligibility.
Run only after all native capture/probe/lifecycle folders are complete.
"""
import argparse, json, shutil, hashlib, html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'art-source/circuits/nordschleife/stage-c-review'
DATA=ROOT/'.rally/circuit-review'
def read(p):return json.loads(p.read_text())
def package(prefix):
    OUT.mkdir(exist_ok=True)
    sets=['route-720','signatures-720','cameras-720','cameras-1080','lifecycle','performance-720','performance-1080']
    folders={kind:DATA/(prefix+'-'+kind) for kind in sets}
    identity=read(folders['route-720']/'course.json');build=read(folders['route-720']/'build.json')
    reports={};shots=[]
    for kind,folder in folders.items():
        if not folder.exists():raise ValueError('Missing native evidence: '+str(folder))
        if read(folder/'course.json')['id']!=identity['id'] or read(folder/'build.json')!=build:raise ValueError('Mixed candidate/build identities')
        target=OUT/kind;target.mkdir(exist_ok=True)
        for name in ('circuit-audit.json','performance.json','lifecycle.json','coverage.jsonl','inspection-plan.json','build.json','course.json','launch.json','probe-summary.json','measured-coverage.json'):
            if (folder/name).exists():shutil.copy2(folder/name,target/name)
        for name in ('videos','contact-sheets','dense-review','camera-review'):
            if (folder/name).exists():shutil.copytree(folder/name,target/name,dirs_exist_ok=True)
        if (folder/'inspection-plan.json').exists() and kind!='lifecycle':
            for i,shot in enumerate(read(folder/'inspection-plan.json')['shots']):
                if shot['label']=='audit-warmup':continue
                video=f'{kind}/videos/{i:02}-{shot["label"]}.mp4'
                if not (OUT/video).exists():raise ValueError('Missing moving evidence '+video)
                shots.append(dict(shot,group=kind,video=video,contact=f'{kind}/contact-sheets/{i:02}-{shot["label"]}.jpg'))
        reports[kind]={p.name:read(p) for p in target.glob('*.json') if p.name in ('circuit-audit.json','performance.json','lifecycle.json')}
        if kind.startswith('performance-'):
            steady=DATA/(prefix+'-steady-'+kind)
            if read(steady/'course.json')['id']!=identity['id'] or read(steady/'build.json')!=build:
                raise ValueError('Mixed steady performance identity')
            shutil.copy2(target/'performance.json',target/'performance-with-startup-audit.json')
            shutil.copy2(steady/'performance.json',target/'performance.json')
            shutil.copy2(steady/'launch.json',target/'steady-launch.json')
            reports[kind]['performance-with-startup-audit.json']=reports[kind]['performance.json']
            reports[kind]['performance.json']=read(target/'performance.json')
    probes=[]
    for folder in sorted(DATA.glob(prefix+'-probe-*')):
        summary=read(folder/'probe-summary.json');probes.append(summary)
        if read(folder/'course.json')['id']!=identity['id'] or read(folder/'build.json')!=build:raise ValueError('Mixed physical probe identity')
        target=OUT/'physical'/folder.name;target.mkdir(parents=True,exist_ok=True)
        for p in folder.glob('*.json*'):shutil.copy2(p,target/p.name)
        for p in folder.glob('drive-*.png'):shutil.copy2(p,target/p.name)
    if len(probes)<18:raise ValueError('Missing physical probes')
    if any(p['wrongSurfaceSamples'] for p in probes):raise ValueError('Incorrect physical surface response')
    if any(p.get('missingExpectedRegions') for p in probes):raise ValueError('Physical probe did not exercise its intended surface')
    shutil.copytree(DATA/(prefix+'-physical-review'),OUT/'physical-review',dirs_exist_ok=True)
    findings=read(DATA/(prefix+'-inspection-findings.json'))
    if findings['course']!=identity['id'] or findings['revision']!=identity['definition']['circuitRevision']:
        raise ValueError('Visual findings belong to another circuit revision')
    audit=reports['route-720']['circuit-audit.json']
    for key in ('centerFailures','kerbFailures','barrierFailures','projectionFailures','wrongSurfaceContacts','groundFailures','stackedGroundContacts','meshColliderMismatches','degenerateFaces','nonfiniteVertices','vegetationIntrusions','vegetationColliders'):
        if audit.get(key,0):raise ValueError('Native audit failed: '+key)
    if not audit.get('vegetationSamples') or audit.get('failures'):
        raise ValueError('Complete native ground and vegetation audit is required')
    if len(findings['regions'])!=16 or any(r['status']!='passed' for r in findings['regions']):raise ValueError('Complete route visual inspection is required')
    if len(findings['signatureViews'])!=12 or len(findings['cameraMatrices'])!=12 or any(r['status']!='passed' for r in findings['signatureViews']+findings['cameraMatrices']):raise ValueError('Signature and camera matrix inspection required')
    if findings['lifecycleReview'].get('status')!='passed' or findings['performanceReview'].get('status')!='measured':raise ValueError('Lifecycle/performance findings required')
    if findings.get('physicalReview',{}).get('status')!='passed':raise ValueError('Physical probe findings required')
    shutil.copy2(DATA/(prefix+'-inspection-findings.json'),OUT/'inspection-findings.json')
    shutil.copy2(DATA/'stage-c-regression.json',OUT/'regression.json')
    shutil.copy2(DATA/(prefix+'-clearance-audit.json'),OUT/'clearance-audit.json')
    shutil.copy2(DATA/(prefix+'-data-validation.json'),OUT/'data-validation.json')
    for source,target in ((prefix+'-web-validation.json','web-validation.json'),(prefix+'-review-page.png','review-page.png')):
        if (DATA/source).exists():shutil.copy2(DATA/source,OUT/target)
    for kind in ('baseline-performance-720','baseline-performance-1080','procedural'):
        folder=DATA/(prefix+'-'+kind);target=OUT/kind;target.mkdir(exist_ok=True)
        for p in folder.glob('*.json*'):shutil.copy2(p,target/p.name)
    for name in ('profiles.json','reference-ledger.json','import-report.json','terrain-report.json','source.json','SOURCE-ATTRIBUTION.md'):
        shutil.copy2(OUT.parent/name,OUT/name)
    shutil.copytree(DATA/(prefix+'-horizon-v7')/'contact-sheets',OUT/'horizon-repair',dirs_exist_ok=True)
    manifest={'course':identity,'build':build,'shots':shots,'reports':reports,'physicalProbes':probes,'findings':findings,'review':'visual review only; training eligibility remains unreviewed','limitations':['Source centerline/elevation retained; no laser scan or surveyed timing line.','Road dimensions, kerb extents, barrier placement, bowls and overpass dimensions are estimated.','Surrounding terrain is reconstructed from source road elevations, not a measured terrain model.','Historical reference imagery does not establish current circuit geometry.','Camera traversal is kinematic inspection, not driving or learning evidence.']}
    requirements=[
        {'requirement':'Immutable source, source geometry, metre station and legacy preservation','evidence':['import-report.json','data-validation.json','regression.json','profiles.json'],'status':'verified'},
        {'requirement':'Whole-route authored profiles, references and confidence','evidence':['profiles.json','reference-ledger.json'],'status':'verified with estimated numerical dimensions'},
        {'requirement':'Conforming terrain, source-edge constraints, shoulder boundaries and distant perimeter','evidence':['terrain-report.json','route-720/circuit-audit.json','inspection-findings.json'],'status':'verified analytically and in native inspection'},
        {'requirement':'Deterministic incremental forest and complete nearby-road clearance','evidence':['clearance-audit.json','inspection-findings.json','lifecycle/lifecycle.json'],'status':'conservative canopy envelope plus native visual inspection'},
        {'requirement':'Complete moving route, overlap, signature sections and six camera modes at 720p/1080p','evidence':['route.svg','inspection-findings.json','evidence.json'],'status':'verified; diagnostic kinematic traversal'},
        {'requirement':'Kerbs, both channels/bypasses, grass, crests/dips, seam and barrier contacts','evidence':['physical/','evidence.json'],'status':'verified bounded diagnostic probes'},
        {'requirement':'Eight reconstruction cycles and three bundle unload/reload cycles','evidence':['lifecycle/lifecycle.json'],'status':'verified native resource counts; see findings for baseline allocation differences'},
        {'requirement':'Frame-time, memory, collider and available rendering counters','evidence':['performance-720/performance.json','performance-1080/performance.json'],'status':'measured; unavailable counters explicitly identified'},
        {'requirement':'Python, checkpoint defaults, dashboard and native compilation/standalone','evidence':['regression.json','route-720/build.json'],'status':'passed'},
        {'requirement':'Stage C stops before training eligibility, sustained training, resume and remote push','evidence':['evidence.json','regression.json'],'status':'training eligibility remains unreviewed'},
    ]
    detailed=[
        ('Preserve the continuous-terrain frozen baseline and all original procedural bundles', ['regression.json'], 'All recorded original file hashes retained'),
        ('Independent source snapshot, source path, SHA-256 and attribution', ['source.json','import-report.json'], 'Bundled immutable source retained'),
        ('Preserve source centerline and elevation without stretching', ['import-report.json','regression.json'], 'Periodic source polyline subdivision; floating-point conversion tolerance'),
        ('Cover every horizontal station with evidence/confidence intervals', ['profiles.json','reference-ledger.json'], 'Whole-route interval partition; dimensions remain estimated'),
        ('Record operator, Porsche onboard and Toyota guide access/publication/timestamps', ['reference-ledger.json'], 'Historical references; operator access limitation explicitly recorded'),
        ('Separate left/right road offsets, kerb presence/width/height/bevel, shoulder and barriers', ['profiles.json','import-report.json'], 'Shared resolved representation with explicit legacy migration'),
        ('Smooth continuous dimensions and explicit discrete boundaries, including seam', ['profiles.json','route-720/circuit-audit.json'], 'Profile/chunk boundary sampling plus native inspection'),
        ('Retain zero unestablished crossfall, 11m source width and simulator asphalt preset', ['profiles.json','physical/'], 'Estimated defaults; no invented random banking'),
        ('Split road near 200m and at authored transitions; share station UVs and surface normals', ['route-720/circuit-audit.json','inspection-findings.json'], 'Native mesh/collider audit and visual chunk/seam review'),
        ('Author selective side kerbs with bevels and tapered ends', ['profiles.json','physical/','inspection-findings.json'], 'Physical forward/reverse probes and visual inspection'),
        ('Continuous barrier strips, rails/posts and estimated profile clearance', ['profiles.json','route-720/circuit-audit.json','physical/'], 'Barrier support/contact checks; dimensions estimated'),
        ('Restrain asphalt repairs/wear and keep markings collider-free', ['inspection-findings.json','route-720/circuit-audit.json'], 'Rendered surfaces inspected; collision matches road geometry'),
        ('Both descending concrete channels, transitions and asphalt bypasses', ['profiles.json','physical/','signatures-720/'], 'Independent region contacts and close native views'),
        ('Constrain road, kerb and shoulder perimeter in terrain triangulation', ['terrain-report.json','route-720/circuit-audit.json'], 'Analytical area/boundary agreement plus dense native support'),
        ('Share neighbouring tile boundaries and resolve nearby road influences together', ['terrain-report.json','inspection-findings.json'], 'Shared triangulation/height constraints; no independent overlapping shoulders'),
        ('Refine near-road terrain and keep coarser distant reconstructed ground', ['terrain-report.json','route-720/'], 'Native road/elevated complete-route inspection'),
        ('Extend and skirt distant perimeter beyond reviewed camera envelope', ['terrain-report.json','horizon-repair/','inspection-findings.json'], 'Targeted repaired-horizon recapture and complete-route views'),
        ('One intended ordinary ground contact surface; avoid duplicate coplanar colliders', ['terrain-report.json','route-720/circuit-audit.json'], 'Zero stacked ordinary-ground contacts in dense sampled audit'),
        ('Screen grade separation and explicitly model reference-supported crossing', ['reference-ledger.json','terrain-report.json','inspection-findings.json'], 'Estimated external overpass; no nonlocal source road-envelope intersection'),
        ('Preserve forest, car, cockpit, lighting and existing camera styling', ['signatures-720/','cameras-720/','cameras-1080/','regression.json'], 'Native preserved presentation inspected'),
        ('Decorative forest collider-free; clear all nearby road/kerb/barrier envelopes', ['clearance-audit.json','route-720/circuit-audit.json','inspection-findings.json'], 'Conservative canopy bounds plus native sampled vertices and full-route views'),
        ('Incremental stable forest cells across seam and resets', ['lifecycle/lifecycle.json','inspection-findings.json'], 'Stable wrapped cell identities; entering/leaving cells only'),
        ('Asphalt suppresses gravel; concrete/kerb identifiers retain tarmac values; grass response', ['physical/','regression.json'], 'Expected region contacts and no wrong-surface samples'),
        ('Explicit range/duration/camera inspection without policy driving', ['route-720/inspection-plan.json','route-720/measured-coverage.json'], 'Opt-in kinematic traversal; actual station ranges retained'),
        ('Sixteen inspection regions with at least 100m measured overlap and T13 wrap', ['route-720/measured-coverage.json','route.svg'], 'Inspection regions only; not official timing sectors'),
        ('All signature areas inspected with moving chase, hood and driver views', ['signatures-720/','inspection-findings.json'], 'Twelve named close-view groups'),
        ('All six cameras at both resolutions; banked areas, returns, crests and seam', ['cameras-720/','cameras-1080/','inspection-findings.json'], 'Trackside/cinematic changes inspected'),
        ('Diagnostic traversal cannot rank records or grant training review', ['evidence.json','regression.json'], 'No attempt output permitted by capture harness; final course remains unreviewed'),
        ('Finite profiles and meshes; collision/rendered shape agreement', ['route-720/circuit-audit.json','regression.json'], 'Native finite/nondegenerate/shared-mesh checks'),
        ('Dense 5m ground audit plus authored/chunk/sharp-corner boundaries and lateral joins', ['route-720/circuit-audit.json'], 'Downward support checked separately from visual continuity'),
        ('Bounded kerb, bowl/bypass, barrier, grass departure/return, crest/dip and seam probes', ['physical/','inspection-findings.json'], 'Individual real wheel/contact outcomes reviewed; deliberate bounded stops labelled'),
        ('Explain retained resource counts after eight reconstructions and three true reloads', ['lifecycle/lifecycle.json','inspection-findings.json'], 'Post-settle meshes/materials/textures/colliders/effects/audio reviewed'),
        ('Measure median/p95/p99, memory, colliders and rendering counter availability', ['performance-720/performance.json','performance-1080/performance.json','inspection-findings.json'], '60 FPS is a target; measured results and unavailable counters retained'),
        ('Compare equivalent baseline scene/resolution and preserve procedural behaviour', ['baseline-performance-720/','baseline-performance-1080/','procedural/','regression.json'], 'Same bounded scene/camera; audit instrumentation differences documented'),
        ('Rebuild standalone and tie source/profile/generator/terrain/course/build identities', ['evidence.json','route-720/build.json','import-report.json','terrain-report.json'], 'Exact immutable candidate; captures cannot mix builds'),
        ('Provide editable profiles, map, evidence ledger, moving captures and fidelity limits', ['profiles.json','route.svg','requirements.json','evidence.json'], 'Complete local review package with artifact hashes'),
        ('Stop at complete-circuit visual review; no automatic training promotion', ['evidence.json','regression.json'], 'User-requested separate training stop retained; no restart or remote push'),
    ]
    requirements.extend({'requirement':r,'evidence':e,'status':v} for r,e,v in detailed)
    (OUT/'requirements.json').write_text(json.dumps(requirements,indent=2)+'\n')
    (OUT/'evidence.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data=read(ROOT/'Assets/Resources/Circuits/Nordschleife.json');pts=data['points'];xs=[p['x'] for p in pts];zs=[p['z'] for p in pts];L=data['length']
    scale=min(850/(max(xs)-min(xs)),550/(max(zs)-min(zs)))
    coord=lambda p:(50+(p['x']-min(xs))*scale,40+(max(zs)-p['z'])*scale)
    colors=['#53c7ba','#e8b568','#78a8e5','#d386b2']
    paths=[]
    for i in range(16):
        region=[coord(p) for p,s in zip(pts,data['stations']) if i*L/16<=s<=(i+1)*L/16]
        paths.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in region)}" fill="none" stroke="{colors[i%4]}" stroke-width="4"/>')
        if region:
            x,y=region[len(region)//2];paths.append(f'<text x="{x+6}" y="{y}" fill="white" font-size="16">{i+1}</text>')
    for m in data['landmarks']:
        if m['name'] not in ('T13','Hatzenbach','Flugplatz','Fuchsröhre','Adenauer Forst','Bergwerk','Brünnchen','Galgenkopf','Karussell','Mini-Karussell','Pflanzgarten','Döttinger Höhe'):continue
        p=min(zip(pts,data['stations']),key=lambda pair:abs(pair[1]-m['station']))[0];x,y=coord(p);paths.append(f'<text x="{x+7}" y="{y+16}" fill="#ffdda2" font-size="14">{html.escape(m["name"])}</text>')
    elev=' '.join(f'{50+s/L*850:.1f},{810-(p["y"]-min(q["y"] for q in pts))/3:.1f}' for p,s in list(zip(pts,data['stations']))[::20])
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 950 860"><rect width="950" height="860" fill="#152229"/>'+''.join(paths)+f'<text x="50" y="660" fill="white">Elevation / original source datum removed / 295.7 m range</text><polyline points="{elev}" fill="none" stroke="#53c7ba" stroke-width="2"/><text x="50" y="842" fill="white">0 m — 20,832 m / 16 inspection regions / captures overlap by at least 100 m (measured)</text></svg>'
    (OUT/'route.svg').write_text(svg)
    page='''<!doctype html><meta charset="utf-8"><title>Nordschleife complete detailing review</title><style>body{margin:0;background:#111b22;color:#e6eef2;font:16px system-ui}main{max-width:1200px;margin:auto;padding:32px}h1{font-size:30px}a{color:#77d7cc}select{padding:12px;background:#243540;color:white;border:1px solid #567;width:100%;font:inherit}video{width:100%;background:black;margin-top:16px}img{max-width:100%}small{word-break:break-all}td,th{text-align:left;padding:8px 14px 8px 0}section{margin:32px 0;padding:22px;background:#1c2b34;border-radius:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:800px){.grid{display:block}}</style><main><h1>Nordschleife / complete detailing review</h1><p>Full circuit reconstruction, ready for visual review. Training validation remains pending.</p><div class="grid"><section><a href="route.svg"><img src="route.svg" alt="Circuit map, inspection regions and elevation"></a><p>Each road and elevated pass overlaps its neighbours, including T13.</p></section><section><h2>Inspection footage</h2><select id="shots"></select><video id="video" controls preload="metadata"></video><p id="details"></p><a id="sheet">Open contact sheet</a><p>The inspection camera follows the route independently of vehicle physics. Real vehicle contact checks are recorded separately.</p></section></div><section><h2>Evidence and identity</h2><p><a href="evidence.json">Complete evidence manifest</a> · <a href="reference-ledger.json">Section references and confidence</a> · <a href="profiles.json">Editable reconstruction profiles</a> · <a href="terrain-report.json">Terrain checks</a> · <a href="requirements.json">Requirement evidence ledger</a> · <a href="inspection-findings.json">Inspection findings</a> · <a href="regression.json">Compatibility checks</a></p><small id="identity"></small><h3>Measured viewer performance</h3><table id="performance"></table><p id="performance-note"></p><h3>Fidelity limits</h3><ul id="limits"></ul></section></main><script>fetch('evidence.json').then(r=>r.json()).then(d=>{let s=document.querySelector('#shots');d.shots.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=x.group+' / '+x.label;s.append(o)});function show(){let x=d.shots[+s.value];document.querySelector('#video').src=x.video;document.querySelector('#details').textContent=`${x.start.toFixed(1)}–${x.end.toFixed(1)} m / ${x.duration}s / ${x.camera}`;document.querySelector('#sheet').href=x.contact} s.onchange=show;show();let t=document.querySelector('#performance');t.innerHTML='<tr><th>Resolution</th><th>Median</th><th>95th percentile</th><th>99th percentile</th></tr>';['performance-720','performance-1080'].forEach(k=>{let p=d.reports[k]['performance.json'];let r=document.createElement('tr');r.innerHTML=`<td>${p.width} × ${p.height}</td><td>${p.medianMs.toFixed(2)} ms</td><td>${p.p95Ms.toFixed(2)} ms</td><td>${p.p99Ms.toFixed(2)} ms</td>`;t.append(r)});document.querySelector('#performance-note').textContent=d.findings.performanceReview.observations;document.querySelector('#identity').textContent='Frozen course '+d.course.id+' / revision '+d.course.definition.circuitRevision+' / generator '+d.course.definition.generatorVersion+' / player source '+d.build.source.hash;d.limitations.forEach(x=>{let l=document.createElement('li');l.textContent=x;document.querySelector('#limits').append(l)})})</script>'''
    (OUT/'index.html').write_text(page)
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='artifact-hashes.json'}
    (OUT/'artifact-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(OUT)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='stage-c');a=p.parse_args();package(a.prefix)
