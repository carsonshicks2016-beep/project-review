"""Jigglypuff Rest Specialist Lab.

Provides analysis, telemetry parsing, and presets for training and evaluating
a superhuman Jigglypuff agent focused on the frame-1 instant KO move: Rest (Down-B).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import glob
from collections import Counter

from .config import Config


# Key action names for Jigglypuff and opponents during Rest interactions
PUFF_REST_ACTIONS = {'DOWN_B_STUN', 'DOWN_B_AIR', 'DOWN_B_GROUND', 'DOWN_B_GROUND_START'}
TECH_KNOCKDOWN_ACTIONS = {
    'TECH_MISS_UP', 'TECH_MISS_DOWN', 'LYING_GROUND_UP', 'LYING_GROUND_DOWN',
    'DOWN_BOUND_U', 'DOWN_BOUND_D', 'GROUND_GETUP', 'FORWARD_TECH', 'BACKWARD_TECH', 'NEUTRAL_TECH'
}
UP_THROW_ACTIONS = {'THROWN_UP', 'THROWN_UP_VOLLEY', 'THROW_UP'}
CROUCH_ACTIONS = {'CROUCHING', 'CROUCH_START', 'CROUCH_END'}


def evaluate_rest_telemetry(replays_dir_or_file: str | Path) -> Dict[str, Any]:
    """Parse .slp replay(s) and compute detailed Rest execution metrics.

    Returns:
        Dict containing:
          - total_rests: Total Down-B executions
          - hits: Successful lethal Rest hits
          - whiffs: Missed Rest executions
          - hit_rate_pct: Percentage of attempts that connected
          - confirms: Breakdown of confirms (up_throw, crouch_cancel, tech_chase, drill, raw)
          - safe_wakeups: Number of times Puff returned to actionable state without damage
          - wakeup_punishes: Number of times Puff took damage while sleeping
    """
    import melee

    path = Path(replays_dir_or_file)
    if path.is_file() and path.suffix == '.slp':
        slp_files = [path]
    elif path.is_dir():
        slp_files = sorted(path.rglob('*.slp'))
    else:
        slp_files = [Path(p) for p in sorted(glob.glob(str(replays_dir_or_file)))]

    stats = {
        'replays_analyzed': 0,
        'total_frames': 0,
        'total_rests': 0,
        'hits': 0,
        'whiffs': 0,
        'hit_rate_pct': 0.0,
        'confirms': {
            'up_throw': 0,
            'crouch_cancel': 0,
            'tech_chase': 0,
            'drill': 0,
            'raw': 0
        },
        'safe_wakeups': 0,
        'wakeup_punishes': 0,
    }

    for slp in slp_files:
        try:
            console = melee.Console(path=str(slp), is_dolphin=False, allow_old_version=True)
            console.connect()
        except Exception:
            continue

        stats['replays_analyzed'] += 1
        puff_port: Optional[int] = None
        opp_port: Optional[int] = None
        prev_puff_action = ''
        prev_opp_action = ''
        prev_opp_percent = 0.0
        prev_puff_percent = 0.0
        in_sleep = False
        sleep_start_frame = 0
        took_damage_in_sleep = False

        recent_puff_actions = []
        recent_opp_actions = []

        while True:
            gs = console.step()
            if gs is None:
                break
            stats['total_frames'] += 1

            if puff_port is None:
                for p, pl in gs.players.items():
                    if getattr(pl.character, 'name', '') == 'JIGGLYPUFF':
                        puff_port = p
                    else:
                        opp_port = p
                if puff_port is None:
                    break

            if puff_port not in gs.players or opp_port not in gs.players:
                continue

            puff = gs.players[puff_port]
            opp = gs.players[opp_port]

            curr_puff_action = getattr(puff.action, 'name', '')
            curr_opp_action = getattr(opp.action, 'name', '')
            curr_opp_percent = float(opp.percent)
            curr_puff_percent = float(puff.percent)

            is_rest = curr_puff_action in PUFF_REST_ACTIONS
            was_rest = prev_puff_action in PUFF_REST_ACTIONS

            # Track Rest initiation
            if is_rest and not was_rest:
                stats['total_rests'] += 1
                in_sleep = True
                sleep_start_frame = gs.frame
                took_damage_in_sleep = False

                # Did Rest hit?
                delta_dmg = curr_opp_percent - prev_opp_percent
                opp_hit = (
                    delta_dmg >= 25.0 or
                    getattr(opp, 'hitstun_frames_left', 0) > 20 or
                    'FLY' in curr_opp_action or 'DEAD' in curr_opp_action
                )

                if opp_hit:
                    stats['hits'] += 1
                    # Classify confirm
                    if any(a in UP_THROW_ACTIONS for a in recent_opp_actions[-30:]):
                        stats['confirms']['up_throw'] += 1
                    elif any(a in CROUCH_ACTIONS for a in recent_puff_actions[-10:]):
                        stats['confirms']['crouch_cancel'] += 1
                    elif any(a in TECH_KNOCKDOWN_ACTIONS for a in recent_opp_actions[-30:]):
                        stats['confirms']['tech_chase'] += 1
                    elif any(a in ('DAIR', 'LANDING') for a in recent_puff_actions[-15:]):
                        stats['confirms']['drill'] += 1
                    else:
                        stats['confirms']['raw'] += 1
                else:
                    stats['whiffs'] += 1

            # Track sleep vulnerability
            if in_sleep:
                if curr_puff_percent > prev_puff_percent:
                    took_damage_in_sleep = True

                # Transitioned out of sleep
                if was_rest and not is_rest:
                    in_sleep = False
                    if took_damage_in_sleep:
                        stats['wakeup_punishes'] += 1
                    else:
                        stats['safe_wakeups'] += 1

            recent_puff_actions.append(curr_puff_action)
            if len(recent_puff_actions) > 40:
                recent_puff_actions.pop(0)
            recent_opp_actions.append(curr_opp_action)
            if len(recent_opp_actions) > 40:
                recent_opp_actions.pop(0)

            prev_puff_action = curr_puff_action
            prev_opp_action = curr_opp_action
            prev_opp_percent = curr_opp_percent
            prev_puff_percent = curr_puff_percent

    if stats['total_rests'] > 0:
        stats['hit_rate_pct'] = round((stats['hits'] / stats['total_rests']) * 100.0, 1)

    return stats


def get_puff_config(opponent: str = 'FOX', stage: str = 'FINAL_DESTINATION', cpu_level: int = 1) -> Config:
    """Create a Config configured specifically for Jigglypuff Rest training."""
    cfg = Config.load()
    cfg.character = 'JIGGLYPUFF'
    cfg.opponent = opponent.upper()
    cfg.stage = stage
    cfg.cpu_level = cpu_level
    cfg.action_set = 'controller'
    cfg.action_frames = 1
    return cfg
