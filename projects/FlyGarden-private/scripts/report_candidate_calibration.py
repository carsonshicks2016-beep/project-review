"""Independent input/decoder/timing audit and preregistered state selection."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.calibrate_continuous_candidate import OUT,sources
from flygarden.candidate_inputs import CandidateInputs,INPUT_KEYS
from flygarden.descending import DescendingDecoder
from flygarden.body_trial import sha
from flygarden.recording import atomic_json

def run():
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['sources']==sources()
    from flygarden.synchronized import SynchronizedBody
    initial={}
    for seed in protocol['seeds']:
        body=SynchronizedBody(seed=seed)
        try:initial[seed]=body.observation()
        finally:body.close()
    trials=[]
    for support in protocol['support_hz']:
        for seed in protocol['seeds']:
            folder=OUT/'trials'/f'{seed}-{support}hz';m=json.loads((folder/'manifest.json').read_text())
            assert m['status']=='complete' and m['sources']==protocol['sources'] and len(m['chunks'])==120
            assert m['controller']['neurons']==138639 and m['controller']['connections']==15091983
            rows=json.loads((folder/'bins.json').read_text());frames=json.loads((folder/'frames.json').read_text())
            assert len(rows)==120 and len(frames)==90
            assert np.all(np.abs(np.asarray([f['time'] for f in frames])-np.arange(1,91)/30)<.0005+1e-9)
            channels=np.asarray(m['controller']['input_channels']);mapping=m['controller']['population_mapping']
            profile=CandidateInputs(float(support));rates=profile.rates([[0.,0.],[0.,0.]],[0.,0.],False)
            values=np.asarray([rates[key] for key in INPUT_KEYS]);rng=np.random.default_rng(seed);decoder=DescendingDecoder();held=np.zeros(2)
            for tick,(chunk,row) in enumerate(zip(m['chunks'],rows)):
                p=folder/chunk['file'];assert sha(p)==chunk['sha256']
                with np.load(p) as z:
                    assert np.array_equal(z['counts'],np.bincount(z['spike_i'],minlength=138639))
                    assert len(z['spike_i'])==chunk['spikes']
                    local,ii=np.nonzero(rng.random((250,len(channels)))<values[channels]*.0001)
                    assert np.array_equal(ii,z['external_i'])
                    assert np.allclose(z['external_t'],tick*.025+local*.0001,atol=1e-12,rtol=0)
                    populations={key:float(z['counts'][[n['index'] for n in neurons]].mean()/.025) if neurons else None for key,neurons in mapping.items()}
                    assert populations==row['population_hz']
                    next_motor=decoder.advance(.025,populations)
                    assert np.array_equal(held,row['applied_motor']) and np.array_equal(held,z['applied_motor'])
                    assert np.array_equal(next_motor,row['next_motor']) and np.array_equal(next_motor,z['next_motor'])
                    assert abs(row['start']-tick*.025)<1e-9 and abs(row['next_available_at']-(tick+1)*.025)<1e-9
                    held=next_motor
                for key in ('position','heading','feet','contacts','antennae'):assert np.isfinite(row['body'][key]).all()
            positions=np.asarray([initial[seed]['position']]+[r['body']['position'] for r in rows])
            heading=np.unwrap([initial[seed]['heading']]+[r['body']['heading'] for r in rows])
            displacement=float(np.linalg.norm(positions[-1,:2]-positions[0,:2]));turn=float(heading[-1]-heading[0])
            flips=sum(r['body']['flipped'] for r in rows)
            assert displacement==m['xy_displacement_mm'] and turn==m['heading_change_rad'] and flips==m['flipped_samples']
            trials.append({'seed':seed,'support_hz':support,'xy_displacement_mm':displacement,
                           'heading_change_rad':turn,'flipped_samples':flips,'finite':bool(np.isfinite(positions).all() and np.isfinite(heading).all()),'wall_seconds':m['wall_seconds']})
    usable=[support for support in protocol['support_hz'] if all(t['xy_displacement_mm']>10 and t['finite'] and t['flipped_samples']==0 for t in trials if t['support_hz']==support)]
    result={'status':'complete','selected_support_hz':min(usable) if usable else None,'trials':trials,
            'protocol_sha256':sha(OUT/'protocol.json'),'auditor_sha256':sha(Path(__file__)),
            'raw_inputs_commands_and_timing_verified':True,'scope':protocol['scope'],
            'initial_pose_provenance':'Independently reconstructed with same pinned body constructor; original calibration manifest omitted initial pose.',
            'promotion':'Operating range only; checkpoint continuation and sensory choice gates remain pending.'}
    atomic_json(OUT/'results.json',result)
    print(json.dumps({'selected_support_hz':result['selected_support_hz'],'trials':len(trials)}))

if __name__=='__main__':run()
