import json,time,resource,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.brain import FullBrain,ROOT
start=time.perf_counter()
print('Building full Brian2 network',flush=True)
brain=FullBrain()
setup=time.perf_counter()-start
print(f'Built {brain.n} neurons, {brain.edges} connections; setup {setup:.1f}s',flush=True)
# Warm-up includes initial Cython compilation separately.
t=time.perf_counter();brain.advance(.01,p9=(1,1));warmup=time.perf_counter()-t
print(f'Warm-up {warmup:.1f}s',flush=True)
t=time.perf_counter()
for i in range(10):
 brain.advance(1,p9=(1,1))
 print(f'{i+1}/10 simulated seconds; rates {brain.last_rates}',flush=True)
wall=time.perf_counter()-t
report=dict(status='completed',neurons=brain.n,aggregated_connections=brain.edges,anatomical_synapses=brain.anatomical_synapses,simulated_seconds=10,wall_seconds=wall,throughput=10/wall,setup_seconds=setup,warmup_seconds=warmup,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rates=brain.last_rates,alterations=['Undefined upstream neuron reset variable w removed','Annotated ORN_DM1/DM2 and DNp09 Poisson inputs','Spike counts only; no complete spike-train storage'],learning='blocked_pending_mapping_validation')
(ROOT/'reports/brain-benchmark.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
