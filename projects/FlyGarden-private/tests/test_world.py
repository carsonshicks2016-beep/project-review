import numpy as np
from flygarden.world import World,default_arena,intersects

def observation(x=0,y=0):return dict(position=[x,y,1],heading=0,contacts=[[0,0,0]],flipped=False)
def test_capture_precedes_food():
 w=World();w.arena['predator'].update(enabled=True,x=0.,y=0.,speed=0.);w.advance(.1,observation());assert w.status=='captured';assert w.collected==0;assert w.arena['foods'][0]['units']==3

def test_food_depletes_and_refractory_contact():
 w=World();w.advance(.1,observation());assert w.collected==1
 w.advance(.1,observation());assert w.collected==1
 for _ in range(25):w.advance(.1,observation())
 assert w.collected==3;assert w.arena['foods'][0]['units']==0

def test_no_hidden_coordinates_in_sensory():
 w=World();s=w.sensory(observation());assert 'foods' not in s and 'predator' not in s and 'position' not in s
 assert len(s['odor'])==2;assert min(s['odor'])>=0;assert max(s['odor'])<=1

def test_occlusion_and_route():
 w=World();w.arena['blocks']=[dict(x=0,y=0,w=2,h=8,z=3)]
 assert not w.visible([-10,0],[10,0]);assert w.visible([-10,10],[10,10]);assert not np.allclose(w.route([-10,0],[10,0]),[10,0])

def test_predator_limited_detection():
 w=World();w.arena['predator'].update(enabled=True,x=0.,y=0.,heading=0.,speed=0.)
 w.predator_step(.1,observation(-5,0));assert w.arena['predator']['state']=='patrol'
 w.predator_step(.1,observation(5,0));assert w.arena['predator']['state']=='pursuit'

def test_snapshot_rng_and_world_continuation():
 w=World(seed=7);w.advance(.1,observation(3,2));s=w.snapshot();a=w.rng.random();w.advance(.1,observation(4,2));expected=w.snapshot();w.restore(s);assert w.rng.random()==a;w.advance(.1,observation(4,2));assert w.time==expected['time'];assert w.travel==expected['travel']

def test_intersection_parallel_and_crossing():
 b=dict(x=0,y=0,w=2,h=2);assert intersects([-2,0],[2,0],b);assert not intersects([-2,2],[2,2],b);assert intersects([0,0],[0,0],b)

def test_shelter_occlusion_blocks_capture():
 w=World(level=6);w.arena['blocks']=[dict(x=0,y=0,w=.2,h=8,z=4)];w.arena['predator'].update(x=.8,y=0.,speed=0.)
 w.advance(.1,observation(-.8,0));assert w.status=='running';assert w.sensory(observation(-.8,0))['threat']==0

def test_unreachable_food_times_out_without_score():
 w=World(mode='challenge');w.arena['foods'][0].update(x=35,y=25);w.arena['foods'][1]['units']=0
 for _ in range(1201):w.advance(.1,observation(-20,-20))
 assert w.status=='timeout';assert w.collected==0

def test_energy_depletion_ends_trial():
 w=World();w.energy=.01;w.advance(.1,observation(-20,-20));assert w.status=='depleted'

def test_fall_is_recorded_without_rescue():
 w=World();obs=observation(-20,0);obs['flipped']=True;w.advance(.1,obs);assert w.status=='fallen';assert w.events[-1]['kind']=='fall'

def test_escape_challenge_requires_an_escape_event():
 w=World(level=5,mode='challenge');w.collected=3;w.advance(.1,observation(-20,0));assert w.status=='running'
 w.escape_count=1;w.advance(.1,observation(-20,0));assert w.status=='success'

def test_trial_arena_retains_layout_and_resets_dynamic_state():
 w=World();w.arena['custom']=True;w.arena['foods'][0].update(x=-15,units=0);w.arena['predator'].update(spawn=[-10,5],x=2,y=3,state='pursuit',velocity=4,last_seen=[1,1]);arena=w.trial_arena()
 assert arena['foods'][0]['x']==-15 and arena['foods'][0]['units']==3;assert arena['custom'];assert arena['predator']['x']==-10 and arena['predator']['y']==5;assert arena['predator']['state']=='patrol';assert w.arena['foods'][0]['units']==0


def test_shallow_terrain_is_traversable_navigation_geometry():
 w=World();w.arena['blocks']=[dict(x=0,y=0,w=2,h=12,z=.15,kind='terrain')]
 assert w.visible([-10,0],[10,0]);assert not w.blocked([0,0])
 assert np.allclose(w.route([-10,0],[10,0]),[10,0])
 w.arena['blocks'][0].pop('kind');w.arena['blocks'][0]['z']=3
 assert not w.visible([-10,0],[10,0]);assert w.blocked([0,0])


def test_obstacle_proxy_is_egocentric_and_bounded():
 w=World();w.arena['blocks']=[dict(x=4,y=0,w=1,h=12,z=3)]
 front=w.sensory(observation())['obstacle_ranges'];assert 0<front[1]<4
 obs=observation();obs['heading']=np.pi;assert w.sensory(obs)['obstacle_ranges']==[8.,8.,8.]
 from flygarden.baseline import motor_command
 s=w.sensory(observation());motor=motor_command(s,0);assert motor[1]>motor[0]
 assert np.isfinite(motor).all() and min(motor)>=0 and max(motor)<=1.2
