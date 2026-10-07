"""
Dedicated transit shuttle simulation (Lloyd Noble Center <-> Stadium fleet).
"""

from typing import Dict, List, Any


class LloydNobleShuttleFleet:
    """
    Simulates the high-capacity articulated bus transit line operating between
    the 7,000-stall Lloyd Noble Center parking basin and the stadium South Oval.
    """

    def __init__(
        self,
        fleet_size: int = 35,
        bus_capacity_pax: int = 80,
        cycle_time_min: float = 14.0,  # Round-trip loop duration
    ):
        self.fleet_size = fleet_size
        self.bus_capacity = bus_capacity_pax
        self.cycle_time_min = cycle_time_min
        # Cumulative passenger telemetry
        self.total_passengers_carried = 0
        self.equivalent_car_trips_saved = 0

    def get_shuttle_throughput_pax_per_hr(self, is_peak_surge: bool) -> float:
        """
        Peak frequency: 35 buses cycling every 14 minutes = 150 trips/hour = 12,000 passengers/hour!
        """
        if is_peak_surge:
            trips_per_hour = (60.0 / self.cycle_time_min) * self.fleet_size
            return trips_per_hour * self.bus_capacity
        else:
            # Low-frequency in-game shuttle
            return (self.fleet_size * 0.25) * (60.0 / 20.0) * self.bus_capacity

    def update_shuttle_service(self, elapsed_s: float, is_peak_surge: bool) -> Dict[str, Any]:
        """Calculates passenger throughput and car trips eliminated."""
        throughput_pax_hr = self.get_shuttle_throughput_pax_per_hr(is_peak_surge)
        pax_carried = throughput_pax_hr * (elapsed_s / 3600.0)
        cars_saved = pax_carried / 2.5  # 2.5 passengers per vehicle avoided

        self.total_passengers_carried += int(pax_carried)
        self.equivalent_car_trips_saved += int(cars_saved)

        return {
            "pax_rate_hr": round(throughput_pax_hr, 0),
            "total_pax": self.total_passengers_carried,
            "car_trips_saved": self.equivalent_car_trips_saved,
            "active_buses": self.fleet_size if is_peak_surge else int(self.fleet_size * 0.25),
        }
