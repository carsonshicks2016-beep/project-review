from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

from .config import CurriculumConfig

@dataclass
class SegmentResult:
    """Stores the outcome of simulating a single track segment."""
    segment_idx: int
    progress_frac: float     # [0, 1] — fraction of segment completed
    mean_speed: float        # m/s average speed during segment
    pace_ratio: float        # mean_speed / mean(v_ref over segment)
    clean: bool              # True if zero offtrack time
    terminal: bool           # True if crashed/spun/timed out before segment end
    terminal_reason: str     # e.g. 'offtrack_timeout', 'stall', 'backwards'
    offtrack_seconds: float  # total off-track time
    time_elapsed: float      # seconds of simulation
    fitness: float           # computed segment fitness


class SegmentCurriculum:
    """Manages the progressive segment curriculum for Driver 2.0."""
    
    def __init__(self, track: Any, v_ref: np.ndarray, cfg: CurriculumConfig | None = None) -> None:
        """
        Initialize the SegmentCurriculum.
        
        Args:
            track: A supra.track.Track object.
            v_ref: Numpy array of reference speeds at each track point.
            cfg: Curriculum configuration.
        """
        self.track = track
        self.v_ref = v_ref
        self.cfg = cfg if cfg is not None else CurriculumConfig()
        
        self.level = 0
        self.n_segments = self.cfg.n_initial_segments
        self.generations_at_level = 0
        
        # Build initial segments
        n_points = len(self.track.arc)
        indices = np.linspace(0, n_points, self.n_segments + 1, dtype=int)
        
        self.segments: list[tuple[int, int]] = []
        for i in range(self.n_segments):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            if end_idx == n_points:
                end_idx = 0 # wrap around
            self.segments.append((start_idx, end_idx))

    def segment_length(self, seg_idx: int) -> float:
        """Return the arc-length distance of a segment in meters."""
        start_idx, end_idx = self.segments[seg_idx]
        start_arc = self.track.arc[start_idx]
        end_arc = self.track.arc[end_idx] if end_idx != 0 else self.track.length
        
        length = end_arc - start_arc
        if length < 0:
            length += self.track.length
        return float(length)

    def entry_speed(self, seg_idx: int) -> float:
        """Return the entry speed for spawning at a segment start."""
        start_idx, _ = self.segments[seg_idx]
        return float(min(60.0, self.v_ref[start_idx] * 0.90))

    def time_budget(self, seg_idx: int) -> float:
        """Return the episode time budget for a segment."""
        length = self.segment_length(seg_idx)
        return float(length / 30.0 * self.cfg.segment_time_budget_factor)

    def should_condense(self, segment_results: list[SegmentResult]) -> bool:
        """
        Check if the condensation gate is met.
        A segment is mastered if clean == True and pace_ratio >= condense_pace_threshold.
        """
        if not segment_results:
            return False
            
        if self.generations_at_level < self.cfg.min_generations_per_level:
            return False

        mastered_count = 0
        for res in segment_results:
            if res.clean and res.pace_ratio >= self.cfg.condense_pace_threshold:
                mastered_count += 1
                
        fraction_mastered = mastered_count / len(segment_results)
        return fraction_mastered >= self.cfg.condense_mastery_frac

    def condense(self) -> bool:
        """
        Merge adjacent segment pairs. 
        Returns True if condensation happened, False if already at 1 segment.
        """
        if self.n_segments <= 1:
            return False
            
        new_segments: list[tuple[int, int]] = []
        for i in range(0, self.n_segments - 1, 2):
            new_segments.append((self.segments[i][0], self.segments[i+1][1]))
            
        # Handle odd segment counts by keeping the last one unpaired
        if self.n_segments % 2 != 0:
            new_segments.append(self.segments[-1])
            
        self.segments = new_segments
        self.n_segments = len(self.segments)
        self.level += 1
        self.generations_at_level = 0
        return True

    def tick_generation(self) -> None:
        """Increment the generations_at_level counter."""
        self.generations_at_level += 1

    def fitness_weights(self, segment_fitnesses: list[float]) -> np.ndarray:
        """
        Return per-segment weights for the overall genome fitness.
        Segments in the bottom 25% by fitness get the weak_segment_weight.
        """
        n = len(segment_fitnesses)
        weights = np.ones(n, dtype=float)
        
        if n > 0:
            # Find the fitness threshold for the bottom 25%
            threshold = np.percentile(segment_fitnesses, 25)
            
            for i, fit in enumerate(segment_fitnesses):
                if fit <= threshold:
                    weights[i] = self.cfg.weak_segment_weight
                    
            # Normalize so weights sum to 1.0
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
                
        return weights
