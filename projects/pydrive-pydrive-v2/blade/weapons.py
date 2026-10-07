"""Weapon / shield loadouts for BLADE fighters.

Each right-hand armament and the left-arm shield are MJCF snippets nested inside
the corresponding hand body, with realistic mass from geom density.  The striking
surface of every armament is tagged with a geom name containing ``strike`` and a
``{p}weapon_tip`` site marks its business end, so the env's hit/observation code
is weapon-agnostic.

Loadouts combine a right-hand item with an optional shield.
"""
from __future__ import annotations

TIP = '<site name="{p}weapon_tip" pos="{x} {y} {z}" size="0.022" rgba="1 0.35 0.2 0.6"/>'


def _right_hand(p, weapon):
    """MJCF nested inside the right hand body for the given weapon."""
    if weapon == "none":               # bare fist
        return (f'<geom name="{p}strike_fist" type="sphere" pos="0 0 -0.03" size="0.06" density="1100"/>'
                + TIP.format(p=p, x=0, y=0, z=-0.08))
    if weapon == "sword":
        return f'''<body name="{p}weapon" pos="0 0 0">
          <geom name="{p}wpn_grip" type="capsule" fromto="0 0 0.06 0 0 -0.10" size="0.022" density="700"/>
          <geom name="{p}wpn_guard" type="box" pos="0 0 -0.11" size="0.10 0.025 0.02" density="1500"/>
          <geom name="{p}strike_blade" type="box" pos="0 0 -0.52" size="0.026 0.006 0.42" density="1300" rgba="0.82 0.86 0.95 1"/>
          {TIP.format(p=p, x=0, y=0, z=-0.94)}
        </body>'''
    if weapon == "katana":             # longer, thinner, lighter; greater reach
        return f'''<body name="{p}weapon" pos="0 0 0">
          <geom name="{p}wpn_grip" type="capsule" fromto="0 0 0.10 0 0 -0.12" size="0.02" density="600"/>
          <geom name="{p}strike_blade" type="box" pos="0 0 -0.62" size="0.021 0.006 0.50" density="2100" rgba="0.86 0.89 0.96 1"/>
          {TIP.format(p=p, x=0, y=0, z=-1.12)}
        </body>'''
    if weapon == "mace":               # short, head-heavy, huge swing inertia
        return f'''<body name="{p}weapon" pos="0 0 0">
          <geom name="{p}wpn_shaft" type="capsule" fromto="0 0 0.05 0 0 -0.40" size="0.025" density="900"/>
          <geom name="{p}strike_head" type="sphere" pos="0 0 -0.46" size="0.085" density="1000" rgba="0.40 0.40 0.46 1"/>
          {TIP.format(p=p, x=0, y=0, z=-0.46)}
        </body>'''
    if weapon == "axe":                # bladed head offset to one side
        return f'''<body name="{p}weapon" pos="0 0 0">
          <geom name="{p}wpn_shaft" type="capsule" fromto="0 0 0.06 0 0 -0.50" size="0.022" density="800"/>
          <geom name="{p}strike_head" type="box" pos="0.07 0 -0.46" size="0.085 0.02 0.075" density="1400" rgba="0.50 0.50 0.56 1"/>
          {TIP.format(p=p, x=0.12, y=0, z=-0.46)}
        </body>'''
    raise ValueError(f"unknown weapon {weapon!r}")


def _shield(p):
    return (f'<body name="{p}shield" pos="0 0 -0.02" euler="90 0 0">'
            f'<geom name="{p}shield" type="cylinder" fromto="0 0 -0.02 0 0 0.02" size="0.22" '
            f'density="430" rgba="0.46 0.31 0.18 1"/></body>')


# named loadout -> (right-hand weapon, has_shield)
LOADOUTS = {
    "none":         ("none",  False),
    "sword":        ("sword", False),
    "shield":       ("none",  True),
    "sword_shield": ("sword", True),
    "mace":         ("mace",  False),
    "axe":          ("axe",   False),
    "katana":       ("katana", False),
}
LOADOUT_NAMES = list(LOADOUTS.keys())


def right_hand_xml(p, loadout):
    return _right_hand(p, LOADOUTS[loadout][0])


def shield_xml(p, loadout):
    return _shield(p) if LOADOUTS[loadout][1] else ""


def has_shield(loadout):
    return LOADOUTS[loadout][1]
