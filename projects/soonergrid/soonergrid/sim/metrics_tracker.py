"""
Research-grade transportation metrics tracker:
Computes Total System Travel Time (TSTT), Total Delay, I-35 Freeway Spillback,
corridor degradation, and excess fuel emissions.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class SimulationStepMetrics:
    step: int
    time_sec: float
    time_hr: float
    phase_name: str
    active_vehicles_on_network: float
    tstt_step_veh_hrs: float
    cumulative_tstt_veh_hrs: float
    cumulative_delay_veh_hrs: float
    lindsey_corridor_speed_mph: float
    i35_ramp_queue_m: float
    is_i35_spillback_hazard: bool
    fuel_wasted_gallons: float


class MetricsTracker:
    """
    Tracks and aggregates publication-grade transport network performance metrics.
    """

    def __init__(self, dt_s: float = 5.0):
        self.dt_s = dt_s
        self.dt_hrs = dt_s / 3600.0

        self.cumulative_tstt_veh_hrs = 0.0
        self.cumulative_free_flow_tstt_veh_hrs = 0.0
        self.i35_spillback_seconds = 0.0
        self.i35_peak_queue_m = 0.0
        self.time_series: List[SimulationStepMetrics] = []

    def record_step(
        self,
        step: int,
        time_sec: float,
        phase_name: str,
        link_states: Dict[str, Any],
        edge_metadata: Dict[str, Any],
    ) -> SimulationStepMetrics:
        """
        Fast metric update: iterates only over active congested links.
        """
        time_hr = time_sec / 3600.0
        total_veh = 0
        step_tstt_hrs = 0.0
        step_free_flow_tstt_hrs = 0.0

        lindsey_speeds: List[float] = []
        i35_ramp_queues: List[float] = []

        for edge_id, state in link_states.items():
            veh_count = state.vehicles_on_link
            total_veh += veh_count
            step_tstt_hrs += veh_count * self.dt_hrs

            meta = edge_metadata[edge_id]
            # Speed ratio
            fft = meta["free_flow_time_s"]
            actual_time = meta["length_m"] / max(0.8, state.speed_mps)
            step_free_flow_tstt_hrs += veh_count * (fft / max(1.0, actual_time)) * self.dt_hrs

            # Check Lindsey corridor speed
            if meta.get("is_lindsey", False) or "lindsey" in meta["name"].lower():
                lindsey_speeds.append(state.speed_mph)

            # Check I-35 ramp spillback
            if meta.get("is_ramp", False) or "motorway_link" in meta["highway_type"]:
                i35_ramp_queues.append(state.queue_length_m)

        self.cumulative_tstt_veh_hrs += step_tstt_hrs
        self.cumulative_free_flow_tstt_veh_hrs += step_free_flow_tstt_hrs
        cumulative_delay = max(0.0, self.cumulative_tstt_veh_hrs - self.cumulative_free_flow_tstt_veh_hrs)

        avg_lindsey_speed = (sum(lindsey_speeds) / len(lindsey_speeds)) if lindsey_speeds else 35.0
        max_ramp_queue = max(i35_ramp_queues) if i35_ramp_queues else 0.0

        if max_ramp_queue > self.i35_peak_queue_m:
            self.i35_peak_queue_m = max_ramp_queue

        # I-35 off-ramp spillback threshold: 420m of queue on ramp
        is_spillback = max_ramp_queue >= 420.0
        if is_spillback:
            self.i35_spillback_seconds += self.dt_s

        fuel_wasted = cumulative_delay * 0.6

        metrics_obj = SimulationStepMetrics(
            step=step,
            time_sec=time_sec,
            time_hr=time_hr,
            phase_name=phase_name,
            active_vehicles_on_network=total_veh,
            tstt_step_veh_hrs=round(step_tstt_hrs, 2),
            cumulative_tstt_veh_hrs=round(self.cumulative_tstt_veh_hrs, 1),
            cumulative_delay_veh_hrs=round(cumulative_delay, 1),
            lindsey_corridor_speed_mph=round(avg_lindsey_speed, 1),
            i35_ramp_queue_m=round(max_ramp_queue, 1),
            is_i35_spillback_hazard=is_spillback,
            fuel_wasted_gallons=round(fuel_wasted, 1),
        )

        # Snapshot every minute
        if step % max(1, round(60.0 / self.dt_s)) == 0:
            self.time_series.append(metrics_obj)

        return metrics_obj

    def get_summary_report(self) -> Dict[str, Any]:
        """Returns final simulation benchmark report."""
        total_delay = max(0.0, self.cumulative_tstt_veh_hrs - self.cumulative_free_flow_tstt_veh_hrs)
        return {
            "total_system_travel_time_veh_hrs": round(self.cumulative_tstt_veh_hrs, 1),
            "free_flow_baseline_veh_hrs": round(self.cumulative_free_flow_tstt_veh_hrs, 1),
            "total_queue_delay_veh_hrs": round(total_delay, 1),
            "delay_percentage": round((total_delay / max(1.0, self.cumulative_tstt_veh_hrs)) * 100.0, 1),
            "i35_spillback_hazard_minutes": round(self.i35_spillback_seconds / 60.0, 1),
            "i35_peak_ramp_queue_meters": round(self.i35_peak_queue_m, 1),
            "excess_fuel_wasted_gallons": round(total_delay * 0.6, 1),
            "time_series_count": len(self.time_series),
        }
