"""Arena mechanics in millimetres; observations expose local signals only."""
from dataclasses import dataclass,field,asdict
import math,copy,heapq
import numpy as np

def angle(x):return (x+math.pi)%(2*math.pi)-math.pi

def intersects(a,b,block,margin=0.):
    low=np.array([block['x']-block['w']/2-margin,block['y']-block['h']/2-margin]);high=np.array([block['x']+block['w']/2+margin,block['y']+block['h']/2+margin])
    a=np.asarray(a,float)[:2];d=np.asarray(b,float)[:2]-a;t0,t1=0.,1.
    for k in range(2):
        if abs(d[k])<1e-12:
            if a[k]<low[k] or a[k]>high[k]:return False
        else:
            u,v=sorted(((low[k]-a[k])/d[k],(high[k]-a[k])/d[k]));t0=max(t0,u);t1=min(t1,v)
            if t0>t1:return False
    return True

def default_arena(level=1):
    blocks=[]
    if level>=2:blocks=[dict(id='wall1',x=10.,y=0.,w=2.,h=12.,z=4.)]
    if level>=6:blocks += shelter(-12,12)
    # Boundary is shared between physics and navigation.
    blocks += [dict(id='boundN',x=0.,y=31.,w=82.,h=2.,z=5.),dict(id='boundS',x=0.,y=-31.,w=82.,h=2.,z=5.),dict(id='boundW',x=-41.,y=0.,w=2.,h=60.,z=5.),dict(id='boundE',x=41.,y=0.,w=2.,h=60.,z=5.)]
    return dict(spawn=[-22.,0.],blocks=blocks,foods=[dict(id='foodA',x=0.,y=0.,odor=0,units=3),dict(id='foodB',x=22.,y=12.,odor=1,units=3)],predator=dict(enabled=level>=5,x=28.,y=-18.,heading=2.5,spawn=[28.,-18.],spawn_heading=2.5,speed=5.,state='patrol',last_seen=None,search_time=0.,velocity=0.))

def shelter(x,y):
    # Open shelter; no invisible safety zone.
    return [dict(id=f'shelter_{x}_{y}_{i}',x=x+dx,y=y+dy,w=w,h=h,z=4.,kind='shelter') for i,(dx,dy,w,h) in enumerate([(-4,0,1,8),(4,0,1,8),(0,4,9,1),(-2.9,-4,3.2,1),(2.9,-4,3.2,1)])]

@dataclass
class World:
    seed:int=0
    level:int=1
    mode:str='sandbox'
    arena:dict|None=None
    time:float=0.
    energy:float=100.
    collected:int=0
    stalls:int=0
    status:str='running'
    rewarded_odor:int=0
    valid:bool=True
    events:list=field(default_factory=list)
    def __post_init__(self):
        self.arena=copy.deepcopy(self.arena or default_arena(self.level));self.rng=np.random.default_rng(self.seed);self.cooldown=0.;self.travel=0.;self.previous=None;self.captures=0;self.escape_count=0
    def visible(self,a,b):return not any(intersects(a,b,v) for v in self.arena['blocks'] if v.get('kind')!='terrain')
    def blocked(self,p,r=.8):return any(intersects(p,p,v,r) for v in self.arena['blocks'] if v.get('kind')!='terrain')
    def concentrations(self,point):
        concentrations=np.zeros(2)
        for f in self.arena['foods']:
            if f['units']>0:concentrations[f['odor']]+=math.exp(-np.linalg.norm(np.asarray(point)[:2]-[f['x'],f['y']])/12.)
        return np.clip(concentrations,0,1)
    def obstacle_ranges(self,pos,heading):
        # Engineered three-ray range proxy, in mm. No object coordinates exposed.
        ranges=[]
        for offset in (.7,0.,-.7):
            direction=np.array([math.cos(heading+offset),math.sin(heading+offset)])
            distance=8.
            for sample in np.arange(.25,8.01,.25):
                if self.blocked(np.asarray(pos)[:2]+direction*sample,.6):
                    distance=float(sample);break
            ranges.append(distance)
        return ranges
    def sensory(self,body):
        pos=np.asarray(body['position'][:2]);h=body['heading'];left=np.array([-math.sin(h),math.cos(h)])*.35;front=np.array([math.cos(h),math.sin(h)])*.7
        antennae=body.get('antennae')
        ol=self.concentrations(antennae[0] if antennae else pos+front+left);orr=self.concentrations(antennae[1] if antennae else pos+front-left)
        p=self.arena['predator'];threat=0.;bearing=0.;distance=None
        if p['enabled']:
            delta=np.array([p['x'],p['y']])-pos;d=float(np.linalg.norm(delta));bearing=angle(math.atan2(delta[1],delta[0])-h)
            if d<25 and abs(bearing)<math.pi*.8 and self.visible(pos,[p['x'],p['y']]):threat=max(0.,1-d/25);distance=d
        return dict(obstacle_ranges=self.obstacle_ranges(pos,h),odor=((ol+orr)/2).tolist(),odor_left=ol.tolist(),odor_right=orr.tolist(),threat=threat,threat_bearing=bearing,threat_distance=distance,contact=float(np.linalg.norm(body['contacts'])),energy=self.energy/100)
    def route(self,start,goal,r=1.6):
        if self.visible(start,goal) and not any(intersects(start,goal,b,r) for b in self.arena['blocks'] if b.get('kind')!='terrain'):return np.array(goal,float)
        def cell(p):return tuple(int(round(v/2)) for v in p)
        s,g=cell(start),cell(goal);queue=[(0,s)];cost={s:0};parent={}
        for _ in range(2400):
            if not queue:break
            _,u=heapq.heappop(queue)
            if u==g:
                while parent.get(u)!=s and u in parent:u=parent[u]
                return np.asarray(u,float)*2
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                v=(u[0]+dx,u[1]+dy);p=np.asarray(v)*2
                if abs(p[0])>38 or abs(p[1])>28 or self.blocked(p,r):continue
                if any(intersects(np.asarray(u)*2,p,b,r) for b in self.arena['blocks'] if b.get('kind')!='terrain'):continue
                nc=cost[u]+1
                if nc<cost.get(v,1e9):cost[v]=nc;parent[v]=u;heapq.heappush(queue,(nc+abs(v[0]-g[0])+abs(v[1]-g[1]),v))
        return np.asarray(start,float)
    def predator_step(self,dt,body):
        p=self.arena['predator']
        if not p['enabled']:return
        start=np.array([p['x'],p['y']]);fly=np.asarray(body['position'][:2]);delta=fly-start;dist=float(np.linalg.norm(delta));seen=dist<25 and abs(angle(math.atan2(delta[1],delta[0])-p['heading']))<math.pi/3 and self.visible(start,fly)
        if seen:p['state']='pursuit';p['last_seen']=fly.tolist();p['search_time']=0.
        elif p['state']=='pursuit':p['state']='search';self.event('lost_sight','Predator searches last observed location')
        if p['state']=='search':
            p['search_time']+=dt
            if p['search_time']>5:p['state']='return';self.escape_count+=1
        if p['state']=='return' and np.linalg.norm(start-p.get('spawn',[28,-18]))<2:p['state']='patrol'
        goal=fly if seen else np.array(p['last_seen'] if p['state']=='search' and p['last_seen'] else (p.get('spawn',[28,-18]) if p['state']=='return' else [25*math.cos(self.time/8),18*math.sin(self.time/8)]))
        target=self.route(start,goal);d=target-start;desired=math.atan2(d[1],d[0]);p['heading']+=float(np.clip(angle(desired-p['heading']),-2*dt,2*dt))
        p['velocity']=min(p['speed'],p['velocity']+12*dt) if np.linalg.norm(d)>.3 else 0.
        dest=start+np.array([math.cos(p['heading']),math.sin(p['heading'])])*p['velocity']*dt
        if not any(intersects(start,dest,b,1.6) for b in self.arena['blocks'] if b.get('kind')!='terrain'):p['x'],p['y']=map(float,dest)
        else:p['velocity']=0.
    def event(self,kind,message):self.events.append(dict(time=round(self.time,4),kind=kind,message=message));self.events=self.events[-200:]
    def advance(self,dt,body):
        if self.status!='running':return 0.
        self.time+=dt;self.energy=max(0,self.energy-dt*.65);self.cooldown=max(0,self.cooldown-dt);pos=np.asarray(body['position'][:2]);self.predator_step(dt,body)
        if self.previous is not None:self.travel+=float(np.linalg.norm(pos-self.previous))
        self.previous=pos.copy();p=self.arena['predator']
        # Capture always precedes food in a shared step.
        if p['enabled'] and np.linalg.norm(pos-[p['x'],p['y']])<2. and self.visible(pos,[p['x'],p['y']]):
            self.status='captured';self.captures+=1;self.event('capture','Trial ended; learned weights retained on reset');return -1.
        if body['position'][2]<-.5 or body.get('flipped',False):self.status='fallen';self.event('fall','Body fell or flipped; no teleportation');return 0.
        reinforcement=0.
        if self.cooldown==0:
            for f in self.arena['foods']:
                if f['units']>0 and np.linalg.norm(pos-[f['x'],f['y']])<1.8:
                    f['units']-=1;self.cooldown=1.
                    if f['odor']==self.rewarded_odor:
                        self.collected+=1;self.energy=min(100,self.energy+18);reinforcement=1.;self.event('food','Food contact: positive modeled reinforcement')
                    else:self.event('empty','Unrewarded odor source contacted')
                    break
        if self.energy<=0:self.status='depleted';self.event('end','Energy depleted')
        elif self.time>=120:self.status='timeout';self.event('end','120 simulated seconds elapsed')
        elif self.mode=='challenge' and ((self.level==5 and self.escape_count>=1) or (self.level!=5 and self.collected>=3)):
            self.status='success';self.event('success','Pursuit escape objective reached' if self.level==5 else 'Challenge food objective reached')
        return reinforcement
    def trial_arena(self):
        arena=copy.deepcopy(self.arena)
        for food in arena['foods']:food['units']=3
        p=arena['predator'];spawn=p.get('spawn',[28.,-18.]);p.update(x=spawn[0],y=spawn[1],heading=p.get('spawn_heading',2.5),state='patrol',velocity=0.,last_seen=None,search_time=0.)
        return arena
    def snapshot(self):
        d=copy.deepcopy(self.__dict__);d['rng']=copy.deepcopy(self.rng.bit_generator.state);return d
    def restore(self,d):
        d=copy.deepcopy(d);rng=d.pop('rng');self.__dict__.update(d);self.rng=np.random.default_rng();self.rng.bit_generator.state=rng
