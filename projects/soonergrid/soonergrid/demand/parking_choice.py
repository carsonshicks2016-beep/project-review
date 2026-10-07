"""
Multinomial Logit (MNL) discrete choice model for parking lot selection
with real-time capacity tracking and spillover dynamics.
"""

import math
from typing import Dict, List, Tuple, Optional
from soonergrid.data.game_day_zones import ParkingSink, Gateway

# Research-standard behavioral coefficients (calibrated to collegiate stadium access)
BETA_DISTANCE = 0.35    # Disutility per km of road travel
BETA_COST = 0.08        # Disutility per dollar of parking fee
BETA_SHUTTLE = 1.60     # Positive utility bonus for high-frequency express shuttle service
BETA_CROWD = 4.50       # Non-linear penalty exponent for lot occupancy saturation

# Nominal parking fee schedule (USD) by zone type
PARKING_FEES = {
    "satellite_hub": 0.0,       # Lloyd Noble Center is free with shuttle
    "stadium_core": 35.0,       # Premium Heisman/Santee donor lots
    "garage": 20.0,             # Structured parking decks (Elm, Asp, Jenkins)
    "residential_lawn": 20.0,   # Neighborhood yard parking (Chautauqua, Miller)
    "commercial_strip": 15.0,   # Lindsey St commercial plazas
}


class ParkingAllocationManager:
    """
    Tracks real-time parking lot occupancy and computes dynamic
    choice probabilities using a Multinomial Logit (MNL) model.
    """

    def __init__(self, sinks: List[ParkingSink]):
        self.sinks = sinks
        self.sinks_by_id = {s.id: s for s in sinks}
        # Dynamic state: current occupied stalls per sink
        self.occupancy: Dict[str, int] = {s.id: 0 for s in sinks}
        self.spillover_events: List[Dict[str, float]] = []

    def get_remaining_capacity(self, sink_id: str) -> int:
        sink = self.sinks_by_id[sink_id]
        return max(0, sink.capacity_stalls - self.occupancy[sink_id])

    def get_occupancy_ratio(self, sink_id: str) -> float:
        sink = self.sinks_by_id[sink_id]
        if sink.capacity_stalls <= 0:
            return 1.0
        return min(1.0, self.occupancy[sink_id] / sink.capacity_stalls)

    def calculate_sink_utility(self, sink: ParkingSink, dist_km: float) -> float:
        """
        Calculates systematic utility V_ij(t) for choosing a parking sink:
        V = -beta_dist * dist - beta_cost * fee + beta_shuttle * shuttle - beta_crowd * (N/K)^4
        """
        fee = PARKING_FEES.get(sink.zone_type, 15.0)
        shuttle_bonus = BETA_SHUTTLE if sink.shuttle_served else 0.0

        occ_ratio = self.get_occupancy_ratio(sink.id)
        if occ_ratio >= 1.0:
            return -float("inf")  # Completely full, zero probability

        crowding_penalty = BETA_CROWD * (occ_ratio ** 4)
        utility = (-BETA_DISTANCE * dist_km) - (BETA_COST * fee) + shuttle_bonus - crowding_penalty
        return utility

    def get_choice_probabilities(self, gateway_id: str, distances_km: Dict[str, float]) -> Dict[str, float]:
        """
        Computes the Multinomial Logit (MNL) probability distribution over available parking sinks:
        P(j) = exp(V_j) / sum(exp(V_k))
        """
        utilities: Dict[str, float] = {}
        for sink in self.sinks:
            dist = distances_km.get(sink.id, 10.0)
            utilities[sink.id] = self.calculate_sink_utility(sink, dist)

        # Filter out full lots (utility = -inf)
        valid_utilities = {k: v for k, v in utilities.items() if v > -float("inf")}
        if not valid_utilities:
            # Fallback: if all lots are 100% full, allocate evenly to residential overflow
            residential_sinks = [s.id for s in self.sinks if s.zone_type in ["residential_lawn", "commercial_strip"]]
            return {sid: 1.0 / len(residential_sinks) for sid in residential_sinks}

        # Softmax with numerical stability (subtract max utility)
        max_u = max(valid_utilities.values())
        exp_u = {k: math.exp(v - max_u) for k, v in valid_utilities.items()}
        sum_exp = sum(exp_u.values())

        probs = {k: v / sum_exp for k, v in exp_u.items()}
        return probs

    def allocate_vehicle(self, sink_id: str) -> bool:
        """
        Registers a vehicle arriving at sink_id.
        Returns True if parked, False if lot was full (forcing emergency spillover).
        """
        sink = self.sinks_by_id.get(sink_id)
        if not sink:
            return False

        if self.occupancy[sink_id] < sink.capacity_stalls:
            self.occupancy[sink_id] += 1
            return True
        else:
            # Saturated: find nearest available overflow lot
            for alt in self.sinks:
                if self.occupancy[alt.id] < alt.capacity_stalls:
                    self.occupancy[alt.id] += 1
                    return True
            return False

    def discharge_vehicle(self, sink_id: str) -> bool:
        """Removes a vehicle from sink_id during post-game egress."""
        if self.occupancy.get(sink_id, 0) > 0:
            self.occupancy[sink_id] -= 1
            return True
        return False
