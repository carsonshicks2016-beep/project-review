"""
Hydrodynamic Link Transmission Model (LTM) / Cell Transmission Model (CTM)
implementing Daganzo's kinematic wave theory for road networks.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class LinkFlowState:
    edge_id: str
    density_vpm: float         # Vehicles per meter
    vehicles_on_link: float    # Total current count
    speed_mps: float           # Current harmonic mean speed
    speed_mph: float           # Current speed in mph
    sending_capacity_vps: float
    receiving_capacity_vps: float
    queue_length_m: float      # Physical extent of congested queue
    is_spillback: bool         # True if queue exceeds 70% of link length


class LinkTransmissionModel:
    """
    Implements hydrodynamic sending (S_e) and receiving (R_e) capacities
    with finite backward shockwave propagation speed w ≈ 5.0 m/s (11.2 mph).
    """

    def __init__(self, backward_wave_speed_mps: float = 5.0):
        self.w_mps = backward_wave_speed_mps

    def compute_link_state(
        self,
        edge_id: str,
        vehicles_on_link: float,
        length_m: float,
        lanes: int,
        v_free_mps: float,
        capacity_vph: float,
        jam_storage_veh: int,
    ) -> LinkFlowState:
        """
        Calculates instantaneous hydrodynamic state and queue length for a link.
        """
        eff_len = max(1.0, length_m)
        density_vpm = vehicles_on_link / eff_len
        capacity_vps = max(0.05, capacity_vph / 3600.0)
        jam_density_vpm = max(density_vpm, jam_storage_veh / eff_len)

        # Critical density where free-flow transitions to congestion
        rho_crit_vpm = capacity_vps / max(1.0, v_free_mps)

        # 1. Sending Capacity S_e(t) = min(v_free * rho, C)
        sending_vps = min(v_free_mps * density_vpm, capacity_vps)

        # 2. Receiving Capacity R_e(t) = min(C, w * (rho_jam - rho))
        remaining_space_vpm = max(0.0, jam_density_vpm - density_vpm)
        receiving_vps = min(capacity_vps, self.w_mps * remaining_space_vpm)

        # 3. Dynamic Velocity
        if density_vpm <= rho_crit_vpm:
            speed_mps = v_free_mps
        else:
            # Hydrodynamic congested branch
            speed_mps = max(0.8, sending_vps / max(1e-4, density_vpm))
        speed_mph = speed_mps * 2.23694

        # 4. Queue Length (meters of bumper-to-bumper congested shockwave)
        if density_vpm > rho_crit_vpm and jam_density_vpm > rho_crit_vpm:
            queue_fraction = (density_vpm - rho_crit_vpm) / (jam_density_vpm - rho_crit_vpm)
            queue_length_m = min(eff_len, eff_len * max(0.0, min(1.0, queue_fraction)))
        else:
            queue_length_m = 0.0

        is_spillback = queue_length_m >= (eff_len * 0.70)

        return LinkFlowState(
            edge_id=edge_id,
            density_vpm=density_vpm,
            vehicles_on_link=vehicles_on_link,
            speed_mps=speed_mps,
            speed_mph=speed_mph,
            sending_capacity_vps=sending_vps,
            receiving_capacity_vps=receiving_vps,
            queue_length_m=queue_length_m,
            is_spillback=is_spillback,
        )

    def transfer_flow(
        self,
        upstream_sending_vps: float,
        downstream_receiving_vps: float,
        dt_s: float,
        signal_green_ratio: float = 1.0,
    ) -> float:
        """
        Calculates physical vehicle transfer across an intersection boundary:
        q_transfer = min(S_upstream, R_downstream) * green_ratio * dt
        """
        max_flow_rate = min(upstream_sending_vps, downstream_receiving_vps) * signal_green_ratio
        return max(0.0, max_flow_rate * dt_s)
