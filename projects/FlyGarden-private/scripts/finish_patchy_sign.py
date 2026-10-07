"""Wait for the frozen batch, audit it, retain a nonpromoted decision receipt."""
import sys,json,time,os,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
OUT=ROOT/'reports/brain-integration/recovery/patchy-sign-factorial-v1'
def main(pid):
    atomic_json(OUT/'audit-job.json',{'status':'waiting_for_trials','simulation_pid':pid,'report_sha256':file_sha(ROOT/'scripts/report_patchy_sign.py')})
    while True:
        progress=json.loads((OUT/'progress.json').read_text())
        if progress['status']=='complete':break
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError('Simulation coordinator exited before completed progress; preserve interrupted data')
        time.sleep(5)
    atomic_json(OUT/'audit-job.json',{'status':'auditing','report_sha256':file_sha(ROOT/'scripts/report_patchy_sign.py')})
    with (OUT/'report-console.json').open('w') as log:subprocess.run([sys.executable,str(ROOT/'scripts/report_patchy_sign.py')],cwd=ROOT,stdout=log,check=True)
    r=json.loads((OUT/'results.json').read_text());p=json.loads((OUT/'protocol.json').read_text());assert all(file_sha(ROOT/n)==h for n,h in p['sources'].items())
    qualified=[m for m in ('patchy_only','combined') if r['response_recovery_passed'][m] and r['sensory_contrast_passed'][m]]
    summary='\n'.join(f"- {m}: response/recovery {r['response_recovery_passed'][m]}; contrast {r['sensory_contrast_passed'][m]}." for m in p['sign_mappings'])
    (OUT/'DECISION.md').write_text(f'''# Patchy-sign factorial result

All33 trials and7,920 recording chunks independently audited. Original, il3-only, patchy-only and combined identities retain full graph topology and absolute weights. Fixed recovery and contrast outcomes:

{summary}

Qualifying new candidates: {', '.join(qualified) or 'none'}. No promotion has occurred. Review the plot and per-neuron metrics before further decisions. Two diagnostic seeds are not body/navigation acceptance. Published nonspiking, graded, compartment-specific and peptide behavior remains absent, so passing these gates would not validate physiology.

Exact checkpoint continuation passed for all four model identities; observer-disabled combined replay matched. All nonselected signed weights, absolute weights, source hashes, owned input events and fixed decoder checks passed. Recording metadata and raw failures are retained.

Next: {'Review qualifying candidate and freeze a separate embodied directional-choice protocol; do not promote automatically.' if qualified else 'Preserve sign-only failures and investigate source-supported compartment/graded-release dynamics before more sign tuning or global sweeps.'}

Plot visual inspection remains pending. Application unchanged; navigation, learning and overall completion remain unproven.
''')
    path=ROOT/'reports/brain-integration/recovery/progress.json';master=json.loads(path.read_text());phase=master['phases']['4'];phase['status']='patchy_sign_factorial_audited_visual_review_pending'
    for name in ('results.json','RESULTS.md','DECISION.md'):
        value='reports/brain-integration/recovery/patchy-sign-factorial-v1/'+name
        if value not in phase['evidence']:phase['evidence'].append(value)
    master.update(updated=time.time(),next_action='Inspect completed patchy-sign plot and detailed gate outcomes; no automatic promotion.',next='Review audited patchy-sign results.');atomic_json(path,master)
    path=ROOT/'reports/brain-integration/acceptance-ledger.json';ledger=json.loads(path.read_text());row=next(x for x in ledger['requirements'] if x['step']==8);row.update(status=phase['status'],completion_proven=False)
    for name in ('RESULTS.md','DECISION.md'):
        value='recovery/patchy-sign-factorial-v1/'+name
        if value not in row['evidence']:row['evidence'].append(value)
    ledger['updated']=time.time();atomic_json(path,ledger)
    atomic_json(OUT/'delivery-receipt.json',{'status':'audited_visual_review_pending','trials':33,'chunks_audited':7920,'frozen_sources_reverified':True,'plot_visually_inspected':False,'promoted':False,'application_changed':False,'full_goal_complete':False,'qualifying_new_candidates':qualified,'wall_seconds_trials':r['wall_seconds_trials'],'peak_rss_bytes':r['peak_rss_bytes'],'storage_bytes':sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file()),'artifacts_sha256':{name:file_sha(OUT/name) for name in ('protocol.json','results.json','RESULTS.md','DECISION.md','comparison.png','model-variants.json')},'report_sha256':r['report_sha256']})
    atomic_json(OUT/'audit-job.json',{'status':'complete','visual_review_pending':True,'completed_at':time.time(),'qualifying_new_candidates':qualified})
if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except BaseException as e:atomic_json(OUT/'audit-job.json',{'status':'failed','error':repr(e),'recordings_preserved':True});raise
