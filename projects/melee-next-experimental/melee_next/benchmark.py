from dataclasses import replace
from pathlib import Path
import time
from .engine import train
from .storage import atomic_json,read

def benchmark(config,directory,counts=(1,2,4),seconds=45):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    report=dict(config=config.asdict(),status='running',backend=config.backend,seconds_per_count=seconds,results=[],started=time.time(),
                metric='accepted PPO training decisions per second after first fragment; includes learner updates and resets')
    atomic_json(directory/'benchmark.json',report)
    for count in counts:
        if read(directory/'control.json').get('stop'):break
        c=replace(config,workers=count,total_steps=100000000,batch_steps=512,fragment=64,checkpoint_every=1000)
        state=train(c,directory/f'{count}-workers',seconds=seconds,mode='benchmark')
        report['results'].append(dict(workers=count,status=state['status'],accepted=state['accepted'],
            decisions_per_second=state.get('accepted_per_second',0),startup_seconds=state.get('startup_seconds'),
            measurement_seconds=state.get('measurement_seconds',0),stale=state['stale'],error=state.get('error')))
        atomic_json(directory/'benchmark.json',report)
        if state['status']=='stopped':break
    valid=[r for r in report['results'] if r['status']=='completed' and r['accepted']>0]
    base=next((r['decisions_per_second'] for r in valid if r['workers']==1),None)
    for row in report['results']:row['speedup']=row['decisions_per_second']/base if base else None
    report.update(status='completed' if len(valid)==len(counts) else 'partial',recommended_workers=max(valid,key=lambda r:r['decisions_per_second'])['workers'] if valid else None,
                  finished=time.time(),scope='Short local throughput measurement; not playing-strength evidence. Repeat longer before a major training run.')
    atomic_json(directory/'benchmark.json',report)
    return report
