"""Controller primitives. The network chooses every one of them, including recovery.

An action is a SEQUENCE of controller frames, not a single frame. Melee's movement
tech is multi-frame by nature -- a wavedash is a jump followed by an air-dodge a few
frames later -- and a one-frame action vocabulary cannot express any of it.

Two rules make the sequences behave like a human's hands:
  * the main stick is analog and is held for the whole window, so holding a direction
    does not require repeating it every frame;
  * buttons are pressed only on the frames their step names, so a repeated press
    registers as a new press rather than a stuck button.
"""
from dataclasses import dataclass, field
import melee

NEUTRAL = (.5, .5)


@dataclass(frozen=True)
class Step:
    """One frame of controller state."""
    stick: tuple = NEUTRAL
    cstick: tuple = NEUTRAL
    buttons: tuple = ()
    triggers: tuple = (0.,0.)


@dataclass(frozen=True)
class Action:
    name: str
    steps: tuple = (Step(),)
    hold: bool = False          # keep the final step applied past the sequence

    def __len__(self): return len(self.steps)


def press(name, stick=NEUTRAL, buttons=(), cstick=NEUTRAL, hold=False):
    """A single-frame input: stick held, buttons pulsed once."""
    return Action(name, (Step(stick, cstick, buttons),), hold)


def combo(name, *steps, hold=False):
    """A multi-frame input. Fox's jumpsquat is 3 frames, so an aerial action that
    follows a jump has to wait those frames out before the character is airborne."""
    return Action(name, tuple(steps), hold)


WAIT = Step()
JUMP = Step(buttons=('BUTTON_Y',))
# A shallow air-dodge angle travels furthest along the ground.
def _airdodge(x): return Step((x, .30), buttons=('BUTTON_L',))

ACTIONS = [
    # --- indices 0-28: the original vocabulary, order preserved so existing
    # --- checkpoints stay readable even though the action count has changed.
    press('Neutral'), press('Run left', (0, .5)), press('Run right', (1, .5)),
    press('Crouch', (.5, 0)),
    press('Jump', buttons=('BUTTON_Y',)),
    press('Jump left', (0, .5), ('BUTTON_Y',)),
    press('Jump right', (1, .5), ('BUTTON_Y',)),
    press('Attack', buttons=('BUTTON_A',)),
    press('Attack left', (0, .5), ('BUTTON_A',)),
    press('Attack right', (1, .5), ('BUTTON_A',)),
    press('Up tilt', (.5, .7), ('BUTTON_A',)),
    press('Down tilt', (.5, 0), ('BUTTON_A',)),
    press('Smash left', cstick=(0, .5)), press('Smash right', cstick=(1, .5)),
    press('Up smash', cstick=(.5, 1)), press('Down smash', cstick=(.5, 0)),
    press('Neutral special', buttons=('BUTTON_B',)),
    press('Special left', (0, .5), ('BUTTON_B',)),
    press('Special right', (1, .5), ('BUTTON_B',)),
    press('Up special', (.5, 1), ('BUTTON_B',)),
    press('Down special', (.5, 0), ('BUTTON_B',)),
    press('Shield', buttons=('BUTTON_L',), hold=True),
    press('Dodge left', (0, .5), ('BUTTON_L',)),
    press('Dodge right', (1, .5), ('BUTTON_L',)),
    press('Grab', buttons=('BUTTON_Z',)),
    press('Fast fall', (.5, 0)),
    press('Aim up-left', (.15, .85)), press('Aim up-right', (.85, .85)),
    press('Aim up', (.5, 1)),

    # --- multi-frame movement tech ---
    # Jump, wait out the 3-frame jumpsquat, then air-dodge into the ground.
    combo('Wavedash left', JUMP, WAIT, WAIT, WAIT, _airdodge(.05)),
    combo('Wavedash right', JUMP, WAIT, WAIT, WAIT, _airdodge(.95)),
    # Holding jump past jumpsquat gives a full hop; the plain 'Jump' pulse is a short hop.
    combo('Full hop', JUMP, JUMP, JUMP, JUMP),
    # Dash dance: reverse direction inside the dash window.
    combo('Dash dance', Step((0, .5)), Step((0, .5)), Step((1, .5)), Step((1, .5))),
    # Short hop then an aerial once actually airborne.
    combo('Short hop attack', JUMP, WAIT, WAIT, WAIT, Step(buttons=('BUTTON_A',))),
    combo('Short hop back-air left', JUMP, WAIT, WAIT, WAIT, Step((0, .5), buttons=('BUTTON_A',))),
    combo('Short hop back-air right', JUMP, WAIT, WAIT, WAIT, Step((1, .5), buttons=('BUTTON_A',))),
    # Shine, then jump out of it -- the entry to waveshine pressure.
    combo('Shine jump', Step((.5, 0), buttons=('BUTTON_B',)), WAIT, WAIT, JUMP),
]

LONGEST = max(len(a) for a in ACTIONS)


def frames_for(index, actions=None):
    """How many frames this action occupies. A macro always runs to completion, so
    action_frames can be lowered for fine control without truncating the tech."""
    return len((actions or ACTIONS)[int(index)])


def apply(controller, index, frame=0, actions=None):
    action = (actions or ACTIONS)[int(index)]
    inside = frame < len(action.steps)
    step = action.steps[frame] if inside else action.steps[-1]
    controller.release_all()
    controller.tilt_analog(melee.Button.BUTTON_MAIN, *step.stick)
    if inside or action.hold:
        controller.tilt_analog(melee.Button.BUTTON_C, *step.cstick)
        for button in step.buttons:
            controller.press_button(melee.Button[button])
        for button,value in zip((melee.Button.BUTTON_L,melee.Button.BUTTON_R),step.triggers):
            if value: controller.press_shoulder(button,value)

# Versioned expansion. Legacy indices are retained, but new heads require explicit
# weight transfer; changing an action count does not make an old checkpoint compatible.
EXTENDED_ACTIONS = list(ACTIONS)
EXTENDED_ACTIONS += [
    press('Walk left',(.28,.5)), press('Walk right',(.72,.5)),
    press('Forward tilt left',(.28,.5),('BUTTON_A',)),
    press('Forward tilt right',(.72,.5),('BUTTON_A',)),
    press('Angled tilt up-left',(.28,.65),('BUTTON_A',)),
    press('Angled tilt up-right',(.72,.65),('BUTTON_A',)),
    press('Angled tilt down-left',(.28,.35),('BUTTON_A',)),
    press('Angled tilt down-right',(.72,.35),('BUTTON_A',)),
    press('Hold attack / charge',buttons=('BUTTON_A',),hold=True),
    press('Hold jump',buttons=('BUTTON_Y',),hold=True),
    press('Spot dodge',(.5,0),('BUTTON_L',)),
    press('Shield grab',buttons=('BUTTON_L','BUTTON_A')),
    press('Shield angle left',(.28,.5),('BUTTON_L',),hold=True),
    press('Shield angle right',(.72,.5),('BUTTON_L',),hold=True),
    press('Shield angle up',(.5,.72),('BUTTON_L',),hold=True),
    press('Shield angle down',(.5,.28),('BUTTON_L',),hold=True),
    press('Air dodge up',(.5,1),('BUTTON_L',)),
    press('Air dodge down',(.5,0),('BUTTON_L',)),
    press('Air dodge up-left',(.15,.85),('BUTTON_L',)),
    press('Air dodge up-right',(.85,.85),('BUTTON_L',)),
    press('DI down-left',(0,0)), press('DI down-right',(1,0)),
    press('Shallow recovery left',(.05,.65)), press('Shallow recovery right',(.95,.65)),
    press('Steep recovery left',(.35,.95)), press('Steep recovery right',(.65,.95)),
    press('Up throw / up input',(.5,1),('BUTTON_A',)),
    press('Down throw / down input',(.5,0),('BUTTON_A',)),
    press('Jump attack left',(0,.5),('BUTTON_Y','BUTTON_A')),
    press('Jump attack right',(1,.5),('BUTTON_Y','BUTTON_A')),
]
EXTENDED_ACTIONS += [Action('Light shield', (Step(triggers=(.35,0)),),True), Action('Medium shield',(Step(triggers=(.65,0)),),True)]

# Single-frame raw controller combinations are composable into character-specific
# tech. At one frame per decision the policy can decide again every game frame.
for direction,stick in [('neutral',NEUTRAL),('left',(0,.5)),('right',(1,.5)),('up',(.5,1)),('down',(.5,0)),('up-left',(.15,.85)),('up-right',(.85,.85)),('down-left',(.15,.15)),('down-right',(.85,.15))]:
    for button in ('A','B','Y','Z','L'):
        EXTENDED_ACTIONS.append(press(f'Raw {button} / {direction}',stick,(f'BUTTON_{button}',),hold=True))


def vocabulary(config):
    return EXTENDED_ACTIONS if getattr(config,'action_set','legacy')=='expanded' else ACTIONS


def describe(config):
    actions=vocabulary(config)
    return dict(action_set=getattr(config,'action_set','legacy'),count=len(actions),
                decision_floor_frames=config.action_frames,decision_floor_ms=round(config.action_frames/60*1000,2),
                max_decisions_per_game_second=60/config.action_frames,
                longest_macro_frames=max(len(a) for a in actions),
                observation_delay_frames=0,
                note='Game logic caps inputs at 60 frames per game second. Macros lock input for their duration; faster hardware cannot skip move startup, hitlag, or recovery.',
                moves=[dict(index=i,name=a.name,frames=max(config.action_frames,len(a)),hold=a.hold,
                            inputs=[dict(stick=s.stick,cstick=s.cstick,buttons=s.buttons,triggers=s.triggers) for s in a.steps]) for i,a in enumerate(actions)])
