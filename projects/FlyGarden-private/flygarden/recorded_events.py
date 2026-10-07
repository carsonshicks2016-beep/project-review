"""Timeline markers derived only from committed recorded observations and events."""
import json,math
from .recording import metadata,run_folder

ODOR_ON=.15
ODOR_OFF=.10

def markers_from_frames(frames,extra=(),status='complete',end_label=None):
    markers=[];seen=set();odor=[False,False];threat=False;near=False;last_sample=None;first=None;last=None
    def add(stamp,kind,label,source,value=None):
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp):return
        key=(round(stamp,8),kind,label)
        if key in seen:return
        seen.add(key);row=dict(time=float(stamp),kind=kind,label=label,source=source)
        if value is not None:row['value']=value
        markers.append(row)
    for frame in frames:
        clock=frame.get('world',{}).get('time')
        if clock is None:continue
        if first is None:first=clock
        last=clock
        for event in frame.get('events',[]):
            add(event.get('time'),event.get('kind','event'),event.get('message',event.get('kind','Event')),'recorded world event',event.get('value'))
        sensory=frame.get('sensory',{});stamp=frame.get('sensory_time',clock)
        signature=(stamp,json.dumps(sensory,sort_keys=True))
        if not sensory or signature==last_sample:continue
        last_sample=signature
        for channel,value in enumerate(sensory.get('odor',[])[:2]):
            if not odor[channel] and value>=ODOR_ON:
                odor[channel]=True;add(stamp,'odor',f"Odor {'AB'[channel]} exposure ≥ {ODOR_ON:.2f}",'recorded sensory concentration',value)
            elif odor[channel] and value<=ODOR_OFF:
                odor[channel]=False;add(stamp,'odor_end',f"Odor {'AB'[channel]} exposure falls ≤ {ODOR_OFF:.2f}",'recorded sensory concentration',value)
        value=sensory.get('threat',0)
        if value>.25 and not threat:add(stamp,'threat','Visible threat signal > 0.25','recorded engineered threat signal',value)
        threat=value>.25
        ranges=sensory.get('obstacle_ranges',[])
        proximity=bool(ranges) and min(ranges)<2
        if proximity and not near:add(stamp,'obstacle','Obstacle within 2 mm on a sensory ray','recorded engineered range signal',min(ranges))
        near=proximity
    for event in extra:
        add(event.get('time'),event.get('kind','event'),event.get('message',event.get('kind','Event')),'explicit recorded event',event.get('value'))
    if first is not None:
        add(first,'start','Recording begins','recording boundary')
        if status not in ('recording','incomplete'):add(last,'termination',f'Recording ends · {end_label or status}','recording boundary')
        markers=[m for m in markers if first-1e-8<=m['time']<=last+1e-8]
    return sorted(markers,key=lambda m:(m['time'],m['kind']))

def recorded_markers(identity):
    meta=metadata(identity);folder=run_folder(identity);limit=meta['recording'].get('frames')
    def frames():
        with (folder/'frames.jsonl').open() as f:
            for i,line in enumerate(f):
                if limit is not None and i>=limit:break
                try:yield json.loads(line)
                except ValueError:break
    extra=[];path=folder/'events.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:extra.append(json.loads(line))
            except ValueError:break
    return dict(events=markers_from_frames(frames(),extra,meta['recording']['status'],(meta['recording'].get('outcome') or {}).get('termination')),odor_thresholds=dict(on=ODOR_ON,off=ODOR_OFF),scope='Odor/range/threat markers are engineered threshold crossings; contact and capture retain recorded event timestamps.')
