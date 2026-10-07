"""
Verification suite for Terrain, Vehicle Dynamics, Suspension Equilibrium, and Sensors.
"""
import sys
import numpy as np
from terrain import ProceduralTerrain
from vehicle import RoverVehicle
from sensors import RoverSensorSuite
from config import SIM_DT

def run_tests():
    print("=== 1. Testing Procedural Terrain Generation ===")
    terrain = ProceduralTerrain(seed=42)
    h_start = terrain.get_height(0.0, 0.0)
    h_hill = terrain.get_height(100.0, 0.0)
    print(f"Start elevation (x=0, y=0): {h_start:.2f} m")
    print(f"Hill elevation (x=100, y=0): {h_hill:.2f} m")
    print(f"Scattered boulders count: {len(terrain.boulders)}")
    assert h_hill > h_start + 4.0, "Terrain elevation gain failed!"
    assert len(terrain.boulders) > 50, "Boulder count insufficient!"

    print("\n=== 2. Testing Rover Suspension & Gravity Settlement ===")
    vehicle = RoverVehicle(terrain, initial_pos=[2.0, 0.0])
    initial_z = vehicle.pos[2]
    print(f"Spawned rover at Z: {initial_z:.3f} m")

    # Step simulation without throttle to check suspension settling
    for _ in range(50):
        vehicle.step(0.0, 0.0, dt=SIM_DT)

    settled_z = vehicle.pos[2]
    print(f"Settled rover at Z: {settled_z:.3f} m")
    print(f"Suspension lengths: {[round(l, 3) for l in vehicle.susp_lengths]} m")
    print(f"Wheel contacts: {vehicle.wheel_contacts}")
    assert all(vehicle.wheel_contacts), "All 4 wheels should be in contact with ground!"
    assert not vehicle.is_flipped(), "Rover should not be flipped at start!"

    print("\n=== 3. Testing Powertrain Acceleration & Traction ===")
    # Apply forward throttle for 1.0 second (100 steps)
    init_x = vehicle.pos[0]
    for _ in range(100):
        vehicle.step(steer_action=0.0, throttle_action=0.8, dt=SIM_DT)

    dist_traveled = vehicle.pos[0] - init_x
    speed = np.linalg.norm(vehicle.vel)
    print(f"Distance traveled under throttle: {dist_traveled:.2f} m")
    print(f"Speed achieved: {speed:.2f} m/s ({speed*3.6:.1f} km/h)")
    assert dist_traveled > 0.5, "Vehicle did not accelerate forward!"
    assert speed > 1.0, "Vehicle forward speed too low!"

    print("\n=== 4. Testing 16-Ray Sensor Array ===")
    sensors = RoverSensorSuite(terrain)
    dists, endpoints, hits, normals = sensors.scan(vehicle)
    print(f"Ray distances normalized (min/max/avg): {dists.min():.2f} / {dists.max():.2f} / {dists.mean():.2f}")
    print(f"Rays hitting ground: {sum(hits)} / {len(hits)}")
    obs = sensors.get_observation(vehicle)
    print(f"Observation vector shape: {obs.shape}, values range: [{obs.min():.2f}, {obs.max():.2f}]")
    assert obs.shape[0] == 30, f"Expected 30 observation dimensions, got {obs.shape[0]}"
    assert sum(hits) >= 10, "At least 10 rays should hit ground on terrain!"

    print("\n[SUCCESS] ALL PHYSICS, SUSPENSION, TERRAIN, AND SENSOR TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
