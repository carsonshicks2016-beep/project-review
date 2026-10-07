"""Unpromoted amplitude-only sensory revision; fixed anatomy and decoder."""
from dataclasses import dataclass
from .candidate_inputs import CandidateInputs

@dataclass(frozen=True)
class OdorGainCandidate(CandidateInputs):
    odor_max_hz: float = 5.
    version: str = 'odor-gain-calibration-v2'

    def __post_init__(self):
        if self.support_hz != 65. or self.odor_max_hz not in (1.,5.,10.):
            raise ValueError('Outside registered v2 calibration grid')

    def manifest(self):
        result=super().manifest()
        result['kind']='Unpromoted engineered lower-gain calibration; no navigation claim'
        return result
