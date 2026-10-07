"""World-coordinate observations and raw GameCube input. No recovery overrides."""
from collections import deque
import math
import numpy as np
import melee

SCHEMA = 'melee-next-2-history4-raw-pad1'
AXES = (49, 9, 2, 2, 2, 2, 4)
BUTTONS = ('BUTTON_A', 'BUTTON_B', 'BUTTON_Y', 'BUTTON_Z')
TRIGGERS = (0., .35, .7, 1.)
CONTINUOUS = 54
FRAME = CONTINUOUS + 5
HISTORY = 4
OBS = FRAME * HISTORY

def polar(index, angles=16, radii=(.4, .7, 1.)):
    if index == 0: return (.5, .5)
    angle, radius = divmod(int(index) - 1, len(radii))
    theta = 2 * math.pi * angle / angles
    return .5 + .5 * radii[radius] * math.cos(theta), .5 + .5 * radii[radius] * math.sin(theta)

def apply(controller, action):
    action = np.asarray(action, dtype=int)
    if action.shape != (7,) or np.any(action < 0) or np.any(action >= AXES): raise ValueError('Invalid controller vector')
    controller.release_all()
    controller.tilt_analog(melee.Button.BUTTON_MAIN, *polar(action[0]))
    controller.tilt_analog(melee.Button.BUTTON_C, *polar(action[1], 8, (1.,)))
    for name, on in zip(BUTTONS, action[2:6]):
        if on: controller.press_button(melee.Button[name])
    trigger = TRIGGERS[action[6]]
    if trigger:
        controller.press_shoulder(melee.Button.BUTTON_L, trigger)
        if trigger == 1: controller.press_button(melee.Button.BUTTON_L)

def from_controller(controller):
    def nearest(target, n, resolve):
        return min(range(n), key=lambda i: sum((x-y)**2 for x,y in zip(target,resolve(i))))
    pressed = {b.name for b, on in controller.button.items() if on}
    if 'BUTTON_X' in pressed: pressed.add('BUTTON_Y')
    trigger = max(controller.l_shoulder, controller.r_shoulder)
    if pressed & {'BUTTON_L', 'BUTTON_R'}: trigger = 1.
    return np.array([nearest(controller.main_stick,49,polar), nearest(controller.c_stick,9,lambda i:polar(i,8,(1.,))),
                     *[int(b in pressed) for b in BUTTONS], min(range(4),key=lambda i:abs(TRIGGERS[i]-trigger))], np.int64)

def enum_value(item): return int(getattr(item, 'value', item))

def features(p, g, port):
    x,y = p.position.x,p.position.y
    projectiles = [q for q in g.projectiles if q.owner != port]
    q = min(projectiles,key=lambda q:(q.position.x-x)**2+(q.position.y-y)**2,default=None)
    return [x/200,y/150,p.percent/300,p.stock/3,float(p.facing),float(p.on_ground),p.jumps_left/5,
            p.shield_strength/60,p.action_frame/120,p.hitstun_frames_left/90,p.hitlag_left/30,float(p.invulnerable),
            p.speed_air_x_self/5,p.speed_ground_x_self/5,p.speed_y_self/5,p.speed_x_attack/10,p.speed_y_attack/10,
            float(p.off_stage),float(p.moonwalkwarning),(q.position.x-x)/200 if q else 0,
            (q.position.y-y)/150 if q else 0,float(q is not None),float(p.is_powershield),float(p.invulnerability_left)/120]

def frame(g, port=1, other=2):
    a,b = g.players[port],g.players[other]
    values = features(a,g,port)+features(b,g,other)+[(b.position.x-a.position.x)/200,(b.position.y-a.position.y)/150,
              min(max(g.frame,0)/28800,1),float(a.facing==(b.position.x>a.position.x)),float(a.off_stage and b.off_stage),1.]
    continuous = np.clip(np.nan_to_num(values),-5,5)
    ids = [min(399,max(0,enum_value(a.action))),min(399,max(0,enum_value(b.action))),
           min(32,max(0,enum_value(a.character))),min(32,max(0,enum_value(b.character))),min(31,max(0,enum_value(g.stage)))]
    return np.concatenate([continuous,ids]).astype(np.float32)

class History:
    def __init__(self, port=1, other=2):
        self.port,self.other=port,other
        self.items=deque(maxlen=HISTORY)
    def push(self,g):
        item=frame(g,self.port,self.other)
        if not self.items: self.items.extend([item]*HISTORY)
        else: self.items.append(item)
        return self.get()
    def get(self): return np.concatenate(self.items)

def players(g):
    return {str(k):dict(character=p.character.name,x=float(p.position.x),y=float(p.position.y),stocks=int(p.stock),
                       percent=float(p.percent),action=getattr(p.action,'name',str(p.action))) for k,p in g.players.items() if k in (1,2)}

class Reward:
    """Damage credit capped per opposing stock; stock and match results dominate."""
    def __init__(self): self.budgets = [2.,2.]
    def step(self, before, after):
        a,b=before.players[1],before.players[2]
        c,d=after.players[1],after.players[2]
        lost=max(0,a.stock-c.stock); taken=max(0,b.stock-d.stock)
        dealt=min(self.budgets[0],max(0,d.percent-b.percent)*.01) if not taken else 0.
        received=min(self.budgets[1],max(0,c.percent-a.percent)*.01) if not lost else 0.
        self.budgets[0]-=dealt; self.budgets[1]-=received
        if taken:self.budgets[0]=2.
        if lost:self.budgets[1]=2.
        result='draw' if c.stock==0 and d.stock==0 else 'win' if d.stock==0 else 'loss' if c.stock==0 else None
        bonus=12. if result=='win' else -12. if result=='loss' else 0.
        return float(4.*(taken-lost)+dealt-received+bonus),result
