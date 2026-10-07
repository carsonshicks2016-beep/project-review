"""A factored controller action space: every reachable GameCube input, not a menu of moves.

A flat list of named moves can only ever contain the moves someone thought to add.
Fox's up-B alone has a full circle of angles, DI is continuous, and real play needs
buttons pressed *together* with an independent C-stick. So the policy picks each axis
of the controller separately and the combinations follow for free:

    main stick   neutral, or one of ANGLES directions at one of MAGNITUDES
    C-stick      neutral, or one of CSTICK_ANGLES directions (always full throw)
    A B Y Z      four independent presses, so any combination is reachable
    trigger      released / light / medium / hard shield

Angles are what a human's thumb controls, so the stick is polar rather than a square
grid: recovery angle and DI angle are the two things that decide whether a Fox lives.
"""
import math
import gymnasium as gym
import numpy as np
import melee

ANGLES = 24                                  # 15 degrees apart
MAGNITUDES = (0.35, 0.60, 0.85, 1.00)        # tilt, walk, run, full throw
CSTICK_ANGLES = 8
TRIGGERS = (0.0, 0.35, 0.65, 1.0)            # released, light, medium, hard shield
BUTTONS = ('BUTTON_A', 'BUTTON_B', 'BUTTON_Y', 'BUTTON_Z')

MAIN_VALUES = 1 + ANGLES * len(MAGNITUDES)   # 97
CSTICK_VALUES = 1 + CSTICK_ANGLES            # 9
DIMENSIONS = (MAIN_VALUES, CSTICK_VALUES) + (2,) * len(BUTTONS) + (len(TRIGGERS),)
NAMES = ('main_stick', 'c_stick') + tuple(b.replace('BUTTON_', '').lower() for b in BUTTONS) + ('trigger',)


def space():
    return gym.spaces.MultiDiscrete(np.array(DIMENSIONS, dtype=np.int64))


def combinations():
    total = 1
    for d in DIMENSIONS: total *= d
    return total


def _polar(index, angles, magnitudes):
    """Index 0 is neutral; the rest walk angle-major through the magnitudes."""
    if index <= 0: return (.5, .5)
    index -= 1
    angle = (index // len(magnitudes)) * (2 * math.pi / angles)
    magnitude = magnitudes[index % len(magnitudes)]
    return (round(.5 + math.cos(angle) * magnitude * .5, 4),
            round(.5 + math.sin(angle) * magnitude * .5, 4))


def main_stick(index): return _polar(int(index), ANGLES, MAGNITUDES)
def c_stick(index): return _polar(int(index), CSTICK_ANGLES, (1.0,))


def decode(vector):
    """Controller state for one action vector."""
    vector = [int(v) for v in np.asarray(vector).reshape(-1)]
    if len(vector) != len(DIMENSIONS):
        raise ValueError(f'Expected {len(DIMENSIONS)} controller axes, got {len(vector)}.')
    for value, limit in zip(vector, DIMENSIONS):
        if not 0 <= value < limit: raise ValueError('Controller axis out of range.')
    return dict(stick=main_stick(vector[0]), cstick=c_stick(vector[1]),
                buttons=tuple(b for b, on in zip(BUTTONS, vector[2:2+len(BUTTONS)]) if on),
                trigger=TRIGGERS[vector[2+len(BUTTONS)]])


def apply(pad, vector, frame=0):
    """Write one controller state. Every axis is re-asserted each frame, so a held
    input stays held and a released one actually releases."""
    state = decode(vector)
    pad.release_all()
    pad.tilt_analog(melee.Button.BUTTON_MAIN, *state['stick'])
    pad.tilt_analog(melee.Button.BUTTON_C, *state['cstick'])
    for button in state['buttons']:
        pad.press_button(melee.Button[button])
    if state['trigger'] > 0:
        pad.press_shoulder(melee.Button.BUTTON_L, state['trigger'])
        # A full press is also the digital click, which is what shields hard.
        if state['trigger'] >= 1.0: pad.press_button(melee.Button.BUTTON_L)


def toward_stage(x, degrees=65.0):
    """Main-stick index for an upward angle aimed back over the stage from position `x`.

    65 degrees rather than straight up: a vertical Firefox regains height but lands the
    agent back in the same place off the side, and a shallow one runs out of travel below
    the ledge. The angle is what a human picks without thinking and what the policy
    currently picks at random."""
    inward = 1.0 if x < 0 else -1.0
    radians = math.radians(degrees)
    return int(nearest(stick=(round(.5 + math.cos(radians) * inward * .5, 4),
                              round(.5 + math.sin(radians) * .5, 4)))[0])


def aims_home(index, x):
    """Is this stick choice survivable from offstage at `x`: upward, and not further out?

    Used to leave a competent choice alone and only overrule a fatal one, so the policy
    keeps whatever recovery angle it has actually learned."""
    sx, sy = main_stick(int(index))
    if sy <= .5:
        return False
    return not ((x < 0 and sx < .45) or (x > 0 and sx > .55))


def nearest(stick=(.5, .5), cstick=(.5, .5), buttons=(), trigger=0.0):
    """The vector closest to an explicit controller state. Lets the scripted teacher
    and any legacy action be expressed in this space."""
    def closest(target, values, resolve):
        return min(range(values), key=lambda i: sum((a - b) ** 2 for a, b in zip(resolve(i), target)))
    vector = [closest(stick, MAIN_VALUES, main_stick), closest(cstick, CSTICK_VALUES, c_stick)]
    vector += [1 if b in buttons else 0 for b in BUTTONS]
    vector.append(min(range(len(TRIGGERS)), key=lambda i: abs(TRIGGERS[i] - trigger)))
    return np.array(vector, dtype=np.int64)


def from_action(action, frame=0):
    """Express one step of a named legacy action as a controller vector."""
    step = action.steps[min(frame, len(action.steps) - 1)]
    triggers = getattr(step, 'triggers', (0., 0.))
    trigger = max(triggers) if triggers else 0.
    if 'BUTTON_L' in step.buttons and trigger == 0: trigger = 1.0
    return nearest(step.stick, step.cstick,
                   tuple(b for b in step.buttons if b != 'BUTTON_L'), trigger)


def describe_vector(vector):
    """Short human-readable label for telemetry."""
    import numpy as _np
    v=[int(x) for x in _np.asarray(vector).reshape(-1)]
    parts=[]
    if v[0]:
        i=v[0]-1
        parts.append(f'stick {int((i//len(MAGNITUDES))*(360/ANGLES))}deg x{MAGNITUDES[i%len(MAGNITUDES)]:.2f}')
    if v[1]: parts.append(f'c-stick {int((v[1]-1)*(360/CSTICK_ANGLES))}deg')
    pressed=[b.replace('BUTTON_','') for b,on in zip(BUTTONS,v[2:2+len(BUTTONS)]) if on]
    if pressed: parts.append('+'.join(pressed))
    t=TRIGGERS[v[2+len(BUTTONS)]]
    if t: parts.append('shield ' + ('hard' if t>=1 else f'{t:.2f}'))
    return ' / '.join(parts) or 'neutral'


def contract():
    """Checkpoint compatibility key: the shape of the controller, not a list of names."""
    return {'space': 'controller-v1', 'axes': list(NAMES), 'dimensions': list(DIMENSIONS),
            'angles': ANGLES, 'magnitudes': list(MAGNITUDES),
            'c_stick_angles': CSTICK_ANGLES, 'triggers': list(TRIGGERS),
            'buttons': list(BUTTONS)}


def describe(config):
    frames = max(1, int(getattr(config, 'action_frames', 1)))
    return dict(action_set='controller', axes=list(NAMES), dimensions=list(DIMENSIONS),
                reachable_controller_states=combinations(),
                policy_outputs=sum(DIMENSIONS),
                stick_angles=ANGLES, stick_angle_degrees=round(360 / ANGLES, 2),
                stick_magnitudes=list(MAGNITUDES), trigger_levels=list(TRIGGERS),
                simultaneous_buttons=True, independent_c_stick=True,
                decision_floor_frames=frames,
                max_decisions_per_game_second=60 / frames,
                note='Every axis is chosen independently, so any combination of stick '
                     'angle, C-stick, buttons and shield depth is reachable.')
