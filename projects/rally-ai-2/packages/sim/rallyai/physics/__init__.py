"""Vehicle dynamics.

The equations are vendored from Supra Ai 2 under ``vendor/`` and are not edited
here — see ``vendor/VENDOR.md``. Everything RallyAI-specific (car presets, the
coordinate/road-plane bridge) sits alongside it.
"""

from .vendor import CarSpec, Controls, Pacejka, SimSpec, Vehicle

__all__ = ["CarSpec", "Controls", "Pacejka", "SimSpec", "Vehicle"]
