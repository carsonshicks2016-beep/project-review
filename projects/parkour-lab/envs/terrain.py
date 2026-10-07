"""Procedural terrain generation module for parkour-lab.
Generates structured obstacle courses: flat platforms, boxes, stairs, gaps, and hurdles.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

MAX_GEOMS = 200
TERRAIN_VERSION = "terrain-v2"


def sample_difficulty(rng, difficulty, replay_prob=0.0):
    """Return the actual course level, including explicitly logged rehearsal."""
    if not 0.0 <= replay_prob <= 1.0:
        raise ValueError("replay_prob must be in [0, 1]")
    if difficulty < 0:
        return float(difficulty)  # explicit custom course
    if not 0.0 <= difficulty <= 5.0:
        raise ValueError("difficulty must be in [0, 5], or negative for a custom course")
    # Level zero is a true flat-ground curriculum starting point.
    if difficulty == 0:
        return 0.0
    if difficulty < 1:
        raise ValueError("Use level 0 for flat ground, or a level in [1, 5]")
    if replay_prob and difficulty > 1 and rng.uniform() < replay_prob:
        return float(rng.uniform(1.0, difficulty))
    return float(difficulty)

@dataclass
class Segment:
    """Represents a discrete segment of the parkour course."""
    kind: str
    x_start: float
    x_end: float
    height: float
    geom_specs: List[Dict]


def generate_course(
    rng: np.random.Generator,
    n_segments: Optional[int] = None,
    difficulty: float = 1.0,
    replay_prob: float = 0.0,
) -> Tuple[List[Segment], float]:
    """Generates a procedural parkour course along the +x axis scaled by difficulty.
    
    Args:
        rng: NumPy random generator.
        n_segments: Number of segments to place (default scales with difficulty).
        difficulty: 0 for flat ground, or a curriculum tier in [1.0, 5.0].
        replay_prob: Probability of sampling an earlier difficulty level for rehearsal.
        
    Returns:
        (segments, total_length)
    """
    difficulty = sample_difficulty(rng, difficulty, replay_prob)
    if difficulty < 0.0:
        import json
        import os
        custom_path = os.path.join(os.path.dirname(__file__), '..', 'custom_course.json')
        if os.path.exists(custom_path):
            with open(custom_path, 'r') as f:
                data = json.load(f)
            segments = []
            for s in data['segments']:
                segments.append(Segment(**s))
            return segments, data['length']
        raise FileNotFoundError("A negative difficulty requires custom_course.json")

    if difficulty == 0.0:
        return [Segment('flat', 0.0, 20.0, 0.0, [{
            'name': 'flat_runway', 'pos': [10.0, 0.0, 0.0],
            'size': [10.0, 1.7, 0.05], 'type': 'box',
        }]), Segment('flat', 20.0, 24.0, 0.0, [{
            'name': 'end_flat', 'pos': [22.0, 0.0, 0.0],
            'size': [2.0, 1.7, 0.05], 'type': 'box',
        }])], 24.0

    # Normalize difficulty to alpha in [0.0, 1.0]
    # Difficulty scale: 1.0 (Novice) to 5.0 (Extreme Parkour)
    eff_alpha = float((difficulty - 1.0) / 4.0)

    # Continuous parameter scaling across the curriculum ladder
    gap_min = 0.20 + 0.55 * eff_alpha
    gap_max = 0.35 + 0.55 * eff_alpha

    wall_min = 0.10 + 0.32 * eff_alpha
    wall_max = 0.18 + 0.32 * eff_alpha

    box_min = 0.08 + 0.30 * eff_alpha
    box_max = 0.15 + 0.31 * eff_alpha

    step_rise = 0.04 + 0.08 * eff_alpha
    half_y = (3.4 - 1.2 * eff_alpha) / 2.0  # 3.4m (Level 1) -> 2.2m (Level 5)
    half_z = 0.05                          # 0.1m thick platform

    if n_segments is None:
        n_segments = int(np.round(5 + 2 * eff_alpha))

    segments = []
    x_cursor = 0.0
    current_height = 0.0

    # 1. Always start with a 5.0m flat safe zone (humanoid starts at x=2.5)
    start_len = 5.0
    geom_specs = [{
        'name': 'start_flat',
        'pos': [x_cursor + start_len / 2, 0.0, current_height],
        'size': [start_len / 2, half_y, half_z],
        'type': 'box'
    }]
    segments.append(Segment('flat', x_cursor, x_cursor + start_len, current_height, geom_specs))
    x_cursor += start_len

    segment_types = ['flat', 'boxes', 'stairs_up', 'stairs_down', 'gap', 'low_wall']

    for i in range(1, n_segments):
        prev_kind = segments[-1].kind
        valid_types = list(segment_types)
        
        # Don't place gaps back-to-back or as the final segment
        if prev_kind == 'gap':
            valid_types.remove('gap')
        if i == n_segments - 1 and 'gap' in valid_types:
            valid_types.remove('gap')
            
        # Guarantee flat landing platform after a gap
        if prev_kind == 'gap':
            kind = 'flat'
        else:
            kind = rng.choice(valid_types)

        geom_specs = []

        if kind == 'flat':
            length = float(rng.uniform(3.0, 3.8))
            geom_specs.append({
                'name': f'seg_{i}_flat',
                'pos': [x_cursor + length / 2, 0.0, current_height],
                'size': [length / 2, half_y, half_z],
                'type': 'box'
            })
            segments.append(Segment(kind, x_cursor, x_cursor + length, current_height, geom_specs))
            x_cursor += length

        elif kind == 'boxes':
            length = float(rng.uniform(3.0, 3.8))
            # Main platform
            geom_specs.append({
                'name': f'seg_{i}_plat',
                'pos': [x_cursor + length / 2, 0.0, current_height],
                'size': [length / 2, half_y, half_z],
                'type': 'box'
            })

            # 1-2 stepping/climbing boxes scaled by difficulty
            n_boxes = int(rng.integers(1, 3))
            for j in range(n_boxes):
                box_l = float(rng.uniform(0.6, 1.0))
                box_w = float(rng.uniform(0.8, 1.4))
                box_h = float(rng.uniform(box_min, box_max))
                
                box_x = x_cursor + float(rng.uniform(box_l / 2 + 0.3, length - box_l / 2 - 0.3))
                box_y = float(rng.uniform(-half_y + box_w / 2 + 0.2, half_y - box_w / 2 - 0.2))
                box_z = current_height + half_z + box_h / 2
                
                geom_specs.append({
                    'name': f'seg_{i}_box_{j}',
                    'pos': [box_x, box_y, box_z],
                    'size': [box_l / 2, box_w / 2, box_h / 2],
                    'type': 'box'
                })
            segments.append(Segment(kind, x_cursor, x_cursor + length, current_height, geom_specs))
            x_cursor += length

        elif kind == 'stairs_up':
            n_steps = int(rng.integers(3, 5))
            step_depth = 0.45
            
            for j in range(n_steps):
                step_h = current_height + j * step_rise
                geom_specs.append({
                    'name': f'seg_{i}_step_{j}',
                    'pos': [x_cursor + j * step_depth + step_depth / 2, 0.0, step_h],
                    'size': [step_depth / 2, half_y, half_z],
                    'type': 'box'
                })
            total_l = n_steps * step_depth
            segments.append(Segment(kind, x_cursor, x_cursor + total_l, current_height, geom_specs))
            x_cursor += total_l
            current_height += (n_steps - 1) * step_rise

        elif kind == 'stairs_down':
            n_steps = int(rng.integers(3, 5))
            step_depth = 0.45
            
            for j in range(n_steps):
                step_h = current_height - j * step_rise
                geom_specs.append({
                    'name': f'seg_{i}_step_{j}',
                    'pos': [x_cursor + j * step_depth + step_depth / 2, 0.0, step_h],
                    'size': [step_depth / 2, half_y, half_z],
                    'type': 'box'
                })
            total_l = n_steps * step_depth
            segments.append(Segment(kind, x_cursor, x_cursor + total_l, current_height, geom_specs))
            x_cursor += total_l
            current_height -= (n_steps - 1) * step_rise

        elif kind == 'gap':
            gap_len = float(rng.uniform(gap_min, gap_max))
            # No geoms — empty space over death plane
            segments.append(Segment(kind, x_cursor, x_cursor + gap_len, current_height, []))
            x_cursor += gap_len

        elif kind == 'low_wall':
            length = float(rng.uniform(3.0, 3.8))
            # Main platform
            geom_specs.append({
                'name': f'seg_{i}_wall_plat',
                'pos': [x_cursor + length / 2, 0.0, current_height],
                'size': [length / 2, half_y, half_z],
                'type': 'box'
            })
            # Low hurdle wall across path scaled by difficulty
            wall_h = float(rng.uniform(wall_min, wall_max))
            wall_thick = 0.15
            wall_x = x_cursor + length / 2
            wall_z = current_height + half_z + wall_h / 2
            
            geom_specs.append({
                'name': f'seg_{i}_wall',
                'pos': [wall_x, 0.0, wall_z],
                'size': [wall_thick / 2, half_y, wall_h / 2],
                'type': 'box'
            })
            segments.append(Segment(kind, x_cursor, x_cursor + length, current_height, geom_specs))
            x_cursor += length

    # Final flat landing platform
    end_len = 4.0
    geom_specs = [{
        'name': 'end_flat',
        'pos': [x_cursor + end_len / 2, 0.0, current_height],
        'size': [end_len / 2, half_y, half_z],
        'type': 'box'
    }]
    segments.append(Segment('flat', x_cursor, x_cursor + end_len, current_height, geom_specs))
    x_cursor += end_len


    return segments, x_cursor


def segment_at_x(segments: List[Segment], x: float) -> Optional[Segment]:
    """Finds the course segment containing the x coordinate."""
    for s in segments:
        if s.x_start <= x <= s.x_end:
            return s
    return None
