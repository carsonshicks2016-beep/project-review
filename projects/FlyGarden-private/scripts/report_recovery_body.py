"""Audit held-out body evidence without changing its committed thresholds."""
import sys,json,pickle
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.accept_recovery_body import OUT,DT,command,sources
from flygarden.body_trial import read_trial,sha
from flygarden.descending import DescendingDecoder
from flygarden.recording import atomic_json

def metrics(initial,rows):
    p=np.asarray([initial['position']]+[r['body']['position'] for r in rows])
    h=np.unwrap([initial['heading']]+[r['body']['heading'] for r in rows])
    speed=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)/DT
    phases=[]
    for phase in range(6):
        a,b=phase*400,(phase+1)*400
        phases.append({'travel_mm':float(speed[a:b].sum()*DT),
                       'net_mm':float(np.linalg.norm(p[b,:2]-p[a,:2])),
                       'heading_rad':float(h[b]-h[a])})
    slip=[r['physical_contacts']['load_weighted_tangential_speed_mm_per_second'] for r in rows]
    measured=[s for s in slip if s is not None]
    stalls=0
    for a in range(0,len(rows),40):
        motor=np.asarray([r['motor'] for r in rows[a:a+40]])
        stalls+=bool(np.sqrt(np.mean(motor**2))>.1 and speed[a:a+40].sum()*DT<.1)
    finite=bool(np.isfinite(p).all() and np.isfinite(h).all())
    return {'phases':phases,'finite':finite,'flipped_samples':sum(bool(r['body']['flipped']) for r in rows),
            'stop_speed_mm_s':float(speed[1520:1600].mean()),'active_stall_seconds':stalls,
            'contact_coverage':len(measured)/len(rows),
            'mean_load_weighted_tangential_speed_mm_s':float(np.mean(measured)) if measured else None}

def run():
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['sources']==sources()
    trials=[]
    for seed in protocol['seeds']:
        folder=OUT/'trials'/str(seed);m,rows,frames=read_trial(folder)
        assert m['sources']==protocol['sources'] and m['status']=='complete'
        assert len(rows)==2400 and len(frames)==1800 and m['completed_windows']==2400
        assert np.all(np.abs(np.asarray([f['time'] for f in frames])-np.arange(1,1801)/30)<.0005+1e-9)
        decoder=DescendingDecoder()
        for tick,row in enumerate(rows):
            kind,value,phase=command(tick*DT)
            assert row['source_kind']==kind and row['input']==value and row['phase']==phase
            assert abs(row['time']-(tick+1)*DT)<1e-9
            assert np.array_equal(row['motor'],decoder.advance(DT,value))
            for key in ('position','heading','feet','contacts','antennae'):
                assert np.isfinite(row['body'][key]).all(),'Nonfinite physical observation: '+key
            c=row['physical_contacts']
            assert np.isfinite(c['foot_solver_ground_forces_model_units']).all()
            for point in c['contact_points']:
                assert np.isfinite([point['normal_force_model_units'],point['tangential_speed_mm_per_second'],*point['position']]).all()
        for chunk in m['chunks']:
            saved=folder/chunk['checkpoint']['file']
            assert sha(saved)==chunk['checkpoint']['sha256']
            with saved.open('rb') as stream:state=pickle.load(stream)
            assert np.isfinite(state['body']['physics']).all(),'Nonfinite saved physical integration state'
        trials.append({'seed':seed,'wall_seconds':m['wall_seconds'],**metrics(m['initial'],rows)})
    continuation=json.loads((OUT/'continuation/result.json').read_text())
    assert continuation['sources']==protocol['sources']
    gates={'finite_complete':all(t['finite'] for t in trials),
           'forward':sum(all(t['phases'][i]['travel_mm']>10 and t['phases'][i]['net_mm']>10 for i in (1,5)) for t in trials)>=8,
           'turn':sum(t['phases'][2]['heading_rad']>.5 and t['phases'][4]['heading_rad']<-.5 for t in trials)>=8,
           'stopping':sum(t['stop_speed_mm_s']<.1 for t in trials)>=8,
           'no_flips':sum(t['flipped_samples']==0 for t in trials)>=9,
           'continuation':continuation['all_rows_frames_and_physics_checkpoints_exact']}
    result={'status':'passed' if all(gates.values()) else 'failed','gates':gates,'trials':trials,
            'protocol_sha256':sha(OUT/'protocol.json'),'commands_reconstructed':True,'frame_timing_verified':True,
            'auditor_sha256':sha(Path(__file__)),
            'all_physical_observations_and_saved_integration_states_finite':True,
            'scope':protocol['scope']}
    atomic_json(OUT/'results.json',result)
    table='\n'.join(f"| {t['seed']} | {t['phases'][1]['net_mm']:.2f} | {t['phases'][2]['heading_rad']:.2f} | {t['phases'][4]['heading_rad']:.2f} | {t['stop_speed_mm_s']:.4f} | {t['flipped_samples']} |" for t in trials)
    (OUT/'RESULTS.md').write_text('# Held-out body acceptance\n\n'+str(result['status'])+'\n\n'+json.dumps(gates,indent=2)+'\n\n| Seed | Forward mm | Left rad | Right rad | Stopped mm/s | Flips |\n|---|---|---|---|---|---|\n'+table+'\n\n'+protocol['scope']+'\n\nContact velocities exclude adhesion actuator loads. These tests do not establish neural navigation or learning.\n')
    print(json.dumps({'status':result['status'],'gates':gates}))

if __name__=='__main__':run()
