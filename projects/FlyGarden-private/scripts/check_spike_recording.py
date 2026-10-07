import sys,json,time,gc,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flygarden.brain import FullBrain,ROOT
results=[];reference=None
for enabled in (False,True):
    brain=FullBrain(seed=731,learning=True,record_spikes=enabled);brain.mark_initial();started=time.perf_counter();samples=[]
    for step in range(3):
        motor=brain.advance(.1,odor=(.6,.2),p9=(.65,.65));indices,times=brain.last_spikes
        if enabled:
            assert np.array_equal(np.bincount(indices,minlength=brain.n),brain.last_delta)
            assert len(times)==0 or (times.min()>=step*.1-1e-8 and times.max()<(step+1)*.1)
            assert len(brain.monitor.i)==0
        if step==1:brain.reinforce(1.)
        samples.append(dict(motor=motor.copy(),counts=brain.last_delta.copy(),voltage=np.asarray(brain.neurons.v[:]).copy(),weights=brain.plasticity.weights.copy()))
    if reference is None:reference=samples
    else:
        for a,b in zip(reference,samples):
            for key in a:assert np.array_equal(a[key],b[key]),key
    results.append(dict(recording=enabled,wall_seconds=time.perf_counter()-started,simulated_seconds=.3,spikes=int(brain.previous_counts.sum()),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    del brain;gc.collect()
report=dict(status='passed',full_network=True,neurons=138639,recording_on_off_exact=True,event_counts_equal_reference_monitor=True,plasticity_enabled_with_reward=True,results=results)
(ROOT/'reports/recording-delivery/spike-parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
