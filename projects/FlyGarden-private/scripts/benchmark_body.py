import sys,time,json,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.body import Body
root=Path(__file__).resolve().parents[1]
print('Building physical fly',flush=True);t=time.perf_counter();b=Body();setup=time.perf_counter()-t
print('Body built',setup,b.observation(),flush=True)
t=time.perf_counter();trace=[]
for i in range(100):
 trace.append(b.advance(.1,[1.,1.]))
 if i%10==9: print(f'{(i+1)/10} simulated seconds',b.observation()['position'],flush=True)
wall=time.perf_counter()-t
r=dict(status='completed',simulated_seconds=10,wall_seconds=wall,throughput=10/wall,setup_seconds=setup,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,final=b.observation(),controller='FlyGym HybridTurningController supplied leg control',physics_dt=b.dt)
(root/'reports/body-benchmark.json').write_text(json.dumps(r,indent=2));(root/'reports/body-trace.json').write_text(json.dumps(trace));print(json.dumps(r,indent=2),flush=True)
b.close()
