"""Small, explicitly scripted demonstration teacher. Never used to override evaluation actions."""
import melee

def choose(g):
    a,b=g.players[1],g.players[2]
    x,y=a.position.x,a.position.y
    dx=b.position.x-x; dy=b.position.y-y
    toward=2 if dx>0 else 1
    inward=1 if x>0 else 2
    jump_in=5 if x>0 else 6
    name=getattr(a.action,'name','')
    if name in ('EDGE_HANGING','EDGE_CATCHING'):
        return inward
    if abs(x)>82 or y < -8:
        if a.hitstun_frames_left>0: return 26 if x>0 else 27
        if name=='SWORD_DANCE_3_LOW': return 26 if x>0 else 27  # Fox up-B charge
        if a.jumps_left>0 and y<15: return jump_in
        if y<-12: return 19
        return inward
    if a.hitstun_frames_left>0:
        return 26 if x>0 else 27
    if not a.on_ground:
        if abs(dx)<22 and abs(dy)<20: return 7
        return inward if abs(x)>65 else toward
    if abs(x)>72 and dx*x>0:
        return 12 if dx<0 else 13
    if abs(dx)<12:
        return 15  # down smash covers both sides
    if abs(dx)<23 and abs(dy)<20:
        return 12 if dx<0 else 13
    if dy>15 and abs(dx)<20:
        return 14
    return toward


def choose_vector(g, actions):
    """The teacher's move expressed as a controller vector."""
    from .controller import from_action
    return from_action(actions[choose(g)])
