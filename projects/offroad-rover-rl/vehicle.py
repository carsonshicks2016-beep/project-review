"""
6-DOF Rigid Body Dynamics Engine for Overland 4x4 Off-Road Rover.
Implements independent 4-corner spring-damper suspension, anti-roll bars,
Pacejka-inspired non-linear tire traction/slip, and chassis underbelly collisions.
"""
import numpy as np
from config import (
    SIM_DT, GRAVITY, ROVER_MASS, ROVER_INERTIA, COM_OFFSET,
    WHEEL_ATTACH_POINTS, SUSP_REST_LENGTH, SUSP_MIN_LENGTH, SUSP_MAX_LENGTH,
    SUSP_SPRING_K, SUSP_DAMPER_C, ANTI_ROLL_K, WHEEL_RADIUS,
    TIRE_FRICTION_COEFF, TIRE_LATERAL_STIFFNESS, MAX_DRIVE_TORQUE,
    MAX_BRAKE_TORQUE, MAX_STEER_ANGLE, MAX_STEER_RATE
)

def quat_to_matrix(q):
    """Converts a unit quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]
    ])

def quat_mult(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def normalize_quat(q):
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


class RoverVehicle:
    def __init__(self, terrain, initial_pos=None):
        self.terrain = terrain
        self.reset(initial_pos)

    def reset(self, initial_pos=None):
        """Resets the rover state at the starting gate with settled suspension."""
        if initial_pos is None:
            init_x = 2.0
            init_y = 0.0
        else:
            init_x, init_y = initial_pos[0], initial_pos[1]

        init_z = self.terrain.get_height(init_x, init_y) + SUSP_REST_LENGTH + WHEEL_RADIUS

        self.pos = np.array([init_x, init_y, init_z], dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # Identity (facing +X)
        self.omega = np.zeros(3, dtype=float)  # Body-frame angular velocity (P, Q, R)

        # Actuator States
        self.steer_angle = 0.0
        self.throttle_cmd = 0.0
        self.brake_cmd = 0.0

        # Wheel States: [FL, FR, RL, RR]
        self.susp_lengths = np.full(4, SUSP_REST_LENGTH, dtype=float)
        self.susp_vels = np.zeros(4, dtype=float)
        self.wheel_omega = np.zeros(4, dtype=float)  # Rad/s spin rate
        self.wheel_contacts = [False, False, False, False]
        self.wheel_contact_points = [np.zeros(3) for _ in range(4)]
        self.wheel_forces = [np.zeros(3) for _ in range(4)]  # In world coordinates
        self.tire_slips = np.zeros(4, dtype=float)

        # Belly skid plate contact status
        self.belly_contact = False

        # Settle suspension against ground
        for _ in range(25):
            self.step(0.0, 0.0, dt=SIM_DT)
            self.vel *= 0.5
            self.omega *= 0.5

    def get_euler_angles(self):
        """Returns (roll, pitch, yaw) in radians."""
        R = quat_to_matrix(self.quat)
        # Pitch: rotation around Y
        pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        # Roll: rotation around X
        roll = np.arctan2(R[2, 1], R[2, 2])
        # Yaw: rotation around Z
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    def get_center_of_mass(self):
        """Returns 3D world position of Center of Mass."""
        R = quat_to_matrix(self.quat)
        return self.pos + R @ COM_OFFSET

    def step(self, steer_action, throttle_action, dt=SIM_DT):
        """
        Executes one physics integration step (dt).
        steer_action: [-1.0 (full left), +1.0 (full right)]
        throttle_action: [-1.0 (full reverse / brake), +1.0 (full forward)]
        """
        R = quat_to_matrix(self.quat)
        com = self.pos + R @ COM_OFFSET

        # 1. Update Steering Actuator with realistic slew rate
        target_steer = np.clip(steer_action, -1.0, 1.0) * MAX_STEER_ANGLE
        steer_diff = target_steer - self.steer_angle
        max_delta = MAX_STEER_RATE * dt
        self.steer_angle += np.clip(steer_diff, -max_delta, max_delta)

        # 2. Update Powertrain Commands
        throttle_action = float(np.clip(throttle_action, -1.0, 1.0))
        if throttle_action >= 0.0:
            self.throttle_cmd = throttle_action
            self.brake_cmd = 0.0
        else:
            self.throttle_cmd = 0.0
            self.brake_cmd = abs(throttle_action)

        # 3. Calculate 4-Wheel Suspension & Tire Forces
        total_force = np.array([0.0, 0.0, -ROVER_MASS * GRAVITY])
        total_torque = np.zeros(3)

        # Suspension compressions for anti-roll bar
        compressions = np.zeros(4)
        for i in range(4):
            compressions[i] = SUSP_REST_LENGTH - self.susp_lengths[i]

        # Anti-roll bar torque (FL-FR, RL-RR)
        arb_front = ANTI_ROLL_K * (compressions[0] - compressions[1])
        arb_rear = ANTI_ROLL_K * (compressions[2] - compressions[3])

        # Per-wheel drive torque (4WD 50/50 split)
        drive_torque_per_wheel = (self.throttle_cmd * MAX_DRIVE_TORQUE) / 4.0
        brake_torque_per_wheel = (self.brake_cmd * MAX_BRAKE_TORQUE) / 4.0

        for i in range(4):
            mount_body = WHEEL_ATTACH_POINTS[i]
            mount_world = self.pos + R @ mount_body

            # Velocity of mount point in world
            v_mount = self.vel + R @ np.cross(self.omega, mount_body)

            # Wheel heading direction
            steer = self.steer_angle if i < 2 else 0.0
            # Wheel local rotation in body frame
            c_s, s_s = np.cos(steer), np.sin(steer)
            wheel_forward_body = np.array([c_s, -s_s, 0.0])
            wheel_lateral_body = np.array([s_s, c_s, 0.0])
            wheel_forward_world = R @ wheel_forward_body
            wheel_lateral_world = R @ wheel_lateral_body

            # Raycast downward along body -Z axis to find ground contact
            strut_down = -R[:, 2]
            # Approximate ground query directly beneath wheel mount
            wheel_x = mount_world[0]
            wheel_y = mount_world[1]
            ground_z, ground_normal = self.terrain.get_elevation_and_normal(wheel_x, wheel_y)

            # Distance from mount to ground contact
            dist_to_ground = mount_world[2] - ground_z
            desired_wheel_z = ground_z + WHEEL_RADIUS

            # Check if tire touches ground
            if dist_to_ground <= (SUSP_MAX_LENGTH + WHEEL_RADIUS):
                self.wheel_contacts[i] = True
                susp_length = np.clip(dist_to_ground - WHEEL_RADIUS, SUSP_MIN_LENGTH, SUSP_MAX_LENGTH)
                susp_vel = (susp_length - self.susp_lengths[i]) / dt
                self.susp_lengths[i] = susp_length
                self.susp_vels[i] = susp_vel

                # Spring-damper force calculation
                delta_l = SUSP_REST_LENGTH - susp_length
                f_spring = SUSP_SPRING_K * delta_l - SUSP_DAMPER_C * susp_vel

                # Add anti-roll bar effect
                if i == 0:
                    f_spring += arb_front
                elif i == 1:
                    f_spring -= arb_front
                elif i == 2:
                    f_spring += arb_rear
                elif i == 3:
                    f_spring -= arb_rear

                normal_force = max(0.0, f_spring)

                # Contact patch point
                contact_pt = np.array([wheel_x, wheel_y, ground_z])
                self.wheel_contact_points[i] = contact_pt

                # Tangential tire velocities at contact
                v_long = np.dot(v_mount, wheel_forward_world)
                v_lat = np.dot(v_mount, wheel_lateral_world)

                # Dynamic tire rotation & torque integration
                wheel_linear_speed = self.wheel_omega[i] * WHEEL_RADIUS
                slip_long = (wheel_linear_speed - v_long) / max(0.5, abs(v_long))
                slip_lat = -np.arctan2(v_lat, max(0.3, abs(v_long)))
                self.tire_slips[i] = abs(slip_long)

                # Friction force saturation (Pacejka approximation)
                mu = TIRE_FRICTION_COEFF
                f_x_raw = mu * normal_force * np.tanh(4.5 * slip_long)
                f_y_raw = mu * normal_force * np.tanh(TIRE_LATERAL_STIFFNESS * slip_lat)

                # Combined friction circle constraint
                f_tangent_mag = np.hypot(f_x_raw, f_y_raw)
                max_friction = mu * normal_force
                if f_tangent_mag > max_friction:
                    scale = max_friction / f_tangent_mag
                    f_x = f_x_raw * scale
                    f_y = f_y_raw * scale
                else:
                    f_x = f_x_raw
                    f_y = f_y_raw

                # Add drive torque propulsion
                propulsion_force = drive_torque_per_wheel / WHEEL_RADIUS
                if brake_torque_per_wheel > 0.0:
                    brake_force = np.sign(v_long) * (brake_torque_per_wheel / WHEEL_RADIUS)
                    f_x -= brake_force
                else:
                    f_x += propulsion_force

                # Update wheel rotational inertia
                wheel_net_torque = drive_torque_per_wheel - (f_x * WHEEL_RADIUS)
                if self.brake_cmd > 0.0:
                    wheel_net_torque -= np.sign(self.wheel_omega[i]) * brake_torque_per_wheel
                wheel_alpha = wheel_net_torque / 1.5  # I_wheel
                self.wheel_omega[i] += wheel_alpha * dt

                # Total 3D force generated at contact patch
                contact_force = normal_force * ground_normal + f_x * wheel_forward_world + f_y * wheel_lateral_world
                self.wheel_forces[i] = contact_force

                total_force += contact_force
                # Moment arm relative to Center of Mass
                r_arm = contact_pt - com
                total_torque += np.cross(r_arm, contact_force)
            else:
                self.wheel_contacts[i] = False
                self.susp_lengths[i] = SUSP_MAX_LENGTH
                self.susp_vels[i] = 0.0
                self.wheel_forces[i] = np.zeros(3)
                self.wheel_omega[i] *= 0.98  # Free spin drag

        # 4. Skid Plate / Belly Collision Check
        # Check center and front bumper clearance
        belly_x = self.pos[0]
        belly_y = self.pos[1]
        belly_ground_z, belly_norm = self.terrain.get_elevation_and_normal(belly_x, belly_y)
        clearance = self.pos[2] - belly_ground_z

        if clearance < 0.25:
            self.belly_contact = True
            penetration = 0.25 - clearance
            # Normal restitution + damping
            k_belly = 80000.0
            c_belly = 8000.0
            f_belly_norm = k_belly * penetration - c_belly * self.vel[2]
            f_belly_norm = max(0.0, f_belly_norm)

            # Friction on skid plate
            skid_fric = -0.6 * f_belly_norm * np.tanh(2.0 * self.vel)
            belly_force = f_belly_norm * belly_norm + skid_fric
            total_force += belly_force
            r_belly = np.array([belly_x, belly_y, belly_ground_z]) - com
            total_torque += np.cross(r_belly, belly_force)
        else:
            self.belly_contact = False

        # 5. Aerodynamic Drag
        speed = np.linalg.norm(self.vel)
        if speed > 0.1:
            drag_f = -0.5 * 1.225 * 0.45 * 2.2 * speed * self.vel
            total_force += drag_f

        # 6. Linear & Angular Integration (Semi-Implicit Euler)
        accel = total_force / ROVER_MASS
        self.vel += accel * dt
        self.pos += self.vel * dt

        # Angular Dynamics in Body Coordinates
        torque_body = R.T @ total_torque
        # Euler's equations: I * d_omega/dt = tau - omega x (I * omega)
        I = ROVER_INERTIA
        i_omega = I * self.omega
        gyroscopic = np.cross(self.omega, i_omega)
        alpha_body = (torque_body - gyroscopic) / I
        self.omega += alpha_body * dt

        # Quaternion Update
        omega_world = R @ self.omega
        q_dot = 0.5 * quat_mult(np.array([0.0, omega_world[0], omega_world[1], omega_world[2]]), self.quat)
        self.quat = normalize_quat(self.quat + q_dot * dt)

    def is_flipped(self):
        """Detects if rover has rolled or pitched beyond recovery."""
        roll, pitch, _ = self.get_euler_angles()
        from config import FLIP_ROLL_THRESHOLD, FLIP_PITCH_THRESHOLD
        return abs(roll) > FLIP_ROLL_THRESHOLD or abs(pitch) > FLIP_PITCH_THRESHOLD

    def get_telemetry(self):
        """Returns snapshot of current physical states for HUD and recording."""
        roll, pitch, yaw = self.get_euler_angles()
        R = quat_to_matrix(self.quat)
        speed = float(np.linalg.norm(self.vel))
        forward_speed = float(np.dot(self.vel, R[:, 0]))
        lateral_speed = float(np.dot(self.vel, R[:, 1]))

        return {
            "pos": [round(float(p), 3) for p in self.pos],
            "vel": [round(float(v), 3) for v in self.vel],
            "speed": round(speed, 2),
            "forward_speed": round(forward_speed, 2),
            "lateral_speed": round(lateral_speed, 2),
            "quat": [round(float(q), 4) for q in self.quat],
            "roll_deg": round(float(np.degrees(roll)), 1),
            "pitch_deg": round(float(np.degrees(pitch)), 1),
            "yaw_deg": round(float(np.degrees(yaw)), 1),
            "steer_deg": round(float(np.degrees(self.steer_angle)), 1),
            "throttle": round(float(self.throttle_cmd), 2),
            "brake": round(float(self.brake_cmd), 2),
            "susp_compression": [round(float(SUSP_REST_LENGTH - l), 3) for l in self.susp_lengths],
            "wheel_contacts": [bool(c) for c in self.wheel_contacts],
            "wheel_forces": [[round(float(x), 1) for x in f] for f in self.wheel_forces],
            "wheel_contact_pts": [[round(float(x), 2) for x in pt] for pt in self.wheel_contact_points],
            "com": [round(float(x), 3) for x in self.get_center_of_mass()],
            "flipped": self.is_flipped(),
            "belly_contact": self.belly_contact
        }
