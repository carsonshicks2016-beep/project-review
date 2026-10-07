"""Fresh-process continuation of the native continuous candidate and body."""
import sys,json,pickle,fcntl
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.calibrate_continuous_candidate import OUT as CAL,sources
from flygarden.body_trial import sha
from flygarden.recording import atomic_json,space_check
from flygarden.power import on_ac_power
OUT=CAL/'continuation'

def run():
    import brian2 as b
    from flygarden.continuous_candidate import ContinuousCandidate
    from flygarden.candidate_inputs import CandidateInputs
    from flygarden.synchronized import SynchronizedBody,CausalCoupling
    from scripts.diagnose_recovery_interfaces import mappings
    with (ROOT/'.runtime/experiment.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not on_ac_power():raise RuntimeError('Continuation requires AC power')
        protocol=json.loads((CAL/'protocol.json').read_text());assert protocol['sources']==sources()
        folder=CAL/'trials/9501-65hz';checkpoint=folder/'checkpoint-1.5'
        receipt=json.loads((checkpoint/'driver-integrity.json').read_text())
        assert all(sha(checkpoint/name)==digest for name,digest in receipt.items())
        with (checkpoint/'body-driver.pkl').open('rb') as stream:driver=pickle.load(stream)
        _,mapping=mappings();candidate=ContinuousCandidate(mapping,9501,CandidateInputs(65.))
        candidate.load(checkpoint);body=SynchronizedBody(seed=9501);body.restore(driver['body'])
        coupling=CausalCoupling(.025);coupling.restore(driver['coupling'])
        manifest=json.loads((folder/'manifest.json').read_text());rows=json.loads((folder/'bins.json').read_text())
        original_frames=json.loads((folder/'frames.json').read_text());frames=[]
        OUT.mkdir(exist_ok=False);atomic_json(OUT/'progress.json',{'status':'running','start_time':1.5})
        try:
            for tick in range(60,120):
                assert abs(candidate.time-coupling.time)<1e-9 and abs(body.steps*body.dt-coupling.time)<1e-9
                transition=coupling.advance(lambda dt:candidate.advance(dt,[[0.,0.],[0.,0.]],[0.,0.],False),
                                           lambda dt,m:body.advance(dt,m,capture=lambda t,p:frames.append({'time':t,**p})))
                expected={'time':coupling.time,'population_hz':candidate.last['population_hz'],**transition}
                assert expected==rows[tick],'Brain/body continuation row differs'
                entry=manifest['chunks'][tick];path=folder/entry['file'];assert sha(path)==entry['sha256']
                with np.load(path) as z:
                    actual={'spike_i':candidate.last_spikes[0],'spike_t':candidate.last_spikes[1],
                            'counts':candidate.last['spike_counts'],'external_i':candidate.last['external_indices'],
                            'external_t':candidate.last['external_times'],'applied_motor':transition['applied_motor'],'next_motor':transition['next_motor']}
                    for key,value in actual.items():assert np.array_equal(value,z[key]),'Continuation differs: '+key
                path=OUT/f'window-{tick:03d}.npz';space_check(OUT,16*1024**2)
                with path.open('wb') as stream:
                    np.savez_compressed(stream,**actual,neural_v_mV=np.asarray(candidate.brain.neurons.v[:]/b.mV),
                                        neural_g_mV=np.asarray(candidate.brain.neurons.g[:]/b.mV),physics=body.snapshot()['physics'])
            assert frames==[f for f in original_frames if f['time']>1.5+1e-9]
            atomic_json(OUT/'results.json',{'status':'passed','fresh_process':True,'resumed_time':1.5,'final_time':3.,
                                          'raw_spikes_inputs_counts_commands_body_observations_and_poses_exact':True,
                                          'scope':'Supported native candidate at65Hz. Original calibration did not record all-neuron state arrays for every window,so full state-array equality is not asserted here.',
                                          'sources':sources(),'checkpoint_integrity_sha256':sha(checkpoint/'integrity.json'),
                                          'auditor_sha256':sha(Path(__file__))})
            atomic_json(OUT/'progress.json',{'status':'complete','final_time':3.})
            print('Candidate fresh-process continuation passed',flush=True)
        finally:body.close()

if __name__=='__main__':run()
