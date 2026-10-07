"""Independent native Brian2 event-driven release check; no full brain."""
import sys, json
from pathlib import Path
import numpy as np
import brian2 as b
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from flygarden.olfactory_release_reference import ReleaseReference
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
OUT = ROOT/'reports/brain-integration/recovery/olfactory-spiking-mechanism-v1'
b.prefs.codegen.target = 'numpy'
b.start_scope(); b.defaultclock.dt = .1*b.ms
rng = np.random.default_rng(11201)
ticks = np.sort(rng.choice(np.arange(1,40000),size=300,replace=False))
times = ticks*.0001
source = b.SpikeGeneratorGroup(1,np.zeros(len(times),dtype=int),times*b.second)
target = b.NeuronGroup(1,'total : 1')
syn = b.Synapses(source,target,model='''
dx/dt = (1-x)/(0.1*second) : 1 (event-driven)
du/dt = -u/(0.05*second) : 1 (event-driven)
''',on_pre='''
u += 0.24*(1-u)
total_post += u*x
x *= 1-u
''')
syn.connect(); syn.x = 1; syn.u = 0
monitor = b.StateMonitor(target,'total',record=True,when='end')
network = b.Network(source,target,syn,monitor)
network.run(2*b.second)
network.store('middle',filename=str(OUT/'brian-reference-checkpoint.pkl'))
network.run(2*b.second)
full = np.asarray(monitor.total[0]).copy(); full_x=float(syn.x[0]);full_u=float(syn.u[0])
network.restore('middle',filename=str(OUT/'brian-reference-checkpoint.pkl'))
network.run(2*b.second)
restored = np.asarray(monitor.total[0])
assert np.array_equal(full,restored) and float(syn.x[0])==full_x and float(syn.u[0])==full_u
reference=ReleaseReference(); expected=np.zeros(40000); cumulative=0.; last=0
for tick,t in zip(ticks,times):
    expected[last:tick]=cumulative
    cumulative+=reference.event(float(t));last=tick
expected[last:]=cumulative
error=float(np.max(np.abs(full-expected)))
assert error<1e-9 and abs(full_x-reference.x)<1e-9 and abs(full_u-reference.u)<1e-9
atomic_json(OUT/'brian-validation.json',{'status':'passed','event_count':len(times),
    'delivery_clock_seconds':.0001,'max_cumulative_release_error':error,
    'native_checkpoint_continuation_exact':True,'brian_version':b.__version__,
    'codegen':'numpy','source_sha256':file_sha(Path(__file__)),
    'scope':'One isolated synapse, externally supplied events, PI absent. Not whole-brain checkpoint parity.'})
print(json.dumps({'status':'passed','error':error}))
