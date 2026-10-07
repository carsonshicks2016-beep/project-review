"""Stage authoring and procedural generation.

Both paths go through ``builder.build_stage`` so an authored fixture and a
generated stage are the same kind of object, built the same way.
"""

from .builder import Profile, ProfileBuilder, build_stage, derive_pace_notes
from .optimal import (
    optimal_sector_times_s,
    theoretical_minimum_profile,
    theoretical_minimum_time,
)

__all__ = [
    "Profile",
    "ProfileBuilder",
    "build_stage",
    "derive_pace_notes",
    "optimal_sector_times_s",
    "theoretical_minimum_profile",
    "theoretical_minimum_time",
]
