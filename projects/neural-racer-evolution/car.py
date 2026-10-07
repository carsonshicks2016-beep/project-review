import math
import numpy as np
import pygame

# Ray angles relative to car heading (radians)
RAY_ANGLES = [
    -np.pi / 2,
    -np.pi / 3,
    -np.pi / 6,
    -np.pi / 12,
    0.0,
    np.pi / 12,
    np.pi / 6,
    np.pi / 3,
    np.pi / 2,
]
N_RAYS       = len(RAY_ANGLES)
RAY_LENGTH   = 520.0
RAY_STEP     = 16.0
RAY_STEPS    = int(RAY_LENGTH / RAY_STEP)

# ======================================================================
# Vehicle dynamics — bicycle model + simplified Pacejka tires
# ======================================================================
# All in "game units": forces have units of px/s² (since MASS=1 here),
# so force values directly equal the acceleration they produce.

MAX_SPEED         = 540.0    # px/s, used for sensor normalization + safety clamp

# --- Vehicle geometry (px) ---
# Approximate scale: 1 px ≈ 0.10 m (so wheelbase 22 px ≈ 2.2 m, realistic for a sports car)
WHEELBASE         = 22.0
CG_TO_FRONT       = 13.0     # CG closer to rear → rear bears more weight (RWD-friendly)
CG_TO_REAR        =  9.0
CG_HEIGHT         =  3.0     # ratio CG/wheelbase ≈ 0.14 — realistic for low race car

# --- Mass / inertia ---
MASS              = 1.0
INERTIA           = MASS * (WHEELBASE ** 2) / 8.0   # slightly higher than rectangular plate

# Reference gravity in px/s². Treats 1 px = 0.10 m, so 1 g ≈ 100 px/s².
G_REF             = 110.0

# --- Pacejka simplified magic formula: F = D·load·sin(C·atan(B·slip)) ---
TIRE_B            = 9.0      # cornering stiffness
TIRE_C            = 1.5      # shape factor — peak-and-falloff
TIRE_D            = 1.55     # peak friction coefficient (~1.5g grip; race tire on dry tarmac)

# --- Engine + brakes ---
# Engine force is HIGHER than peak grip on purpose: cars can wheelspin if they
# floor it, and the friction circle naturally limits acceleration. This mirrors
# how real race cars are torque-limited at low speed, grip-limited at the wheel.
ENGINE_FORCE      = 320.0    # peak engine push (will be friction-limited in practice)
BRAKE_FORCE       = 600.0    # peak brake force (can lock the wheels)
DRAG_LINEAR       = 0.18     # tuned so terminal velocity ≈ MAX_SPEED with friction-limited drive

# --- Steering ---
MAX_STEER_ANGLE   = 0.55     # rad (~31° lock-to-lock half)
STEER_RATE        = 5.5      # rad/s — how fast the steering wheel can change

# --- Yaw damping (small — real cars are damped by tires, not artificially) ---
YAW_DAMPING       = 0.15     # very low — let the tires do the damping
MAX_YAW_RATE      = 14.0     # rad/s safety cap

# --- Drift detection based on body slip angle (continuous) ---
DRIFT_ANGLE_ENTER = 0.22     # rad (~12.6°) — past peak Pacejka slip, real drift territory
DRIFT_ANGLE_EXIT  = 0.10     # rad (~5.7°) — back to gripping

CAR_W = 12
CAR_H = 22
COLLISION_RADIUS = 13
IDLE_KILL    = 5.0



class Car:
    def __init__(self, x, y, angle, genome):
        self.x      = float(x)
        self.y      = float(y)
        self.angle  = float(angle)
        self.vx     = 0.0
        self.vy     = 0.0
        self.genome = genome

        self.is_drifting  = False
        self.speed        = 0.0
        self.draft_boost  = 0.0   # current slipstream drag reduction (0–1)
        self.prev_angle      = float(angle)
        self.angular_velocity = 0.0   # rad/s — now driven by tire forces directly

        # New realistic-physics state
        self.steer_angle      = 0.0   # current actual steering-wheel angle
        self._long_accel_prev = 0.0   # longitudinal accel last frame (for weight transfer)
        self.front_slip       = 0.0   # cached for visualization / debugging
        self.rear_slip        = 0.0
        self.body_slip        = 0.0

        self.alive            = True
        self.fitness          = 0.0
        self.laps             = 0
        self.distance         = 0.0       # raw |Δposition| — still used for trail/idle
        self.forward_distance = 0.0       # signed progress along track tangent
        self.wrong_way_timer  = 0.0       # seconds of sustained backward motion
        self.time_alive       = 0.0
        self.idle_timer       = 0.0
        self.next_checkpoint  = 0
        self.checkpoints_hit  = 0

        self.fitness_bonus = 0.0   # carries forward across multi-track phases

        self.rays = [RAY_LENGTH] * N_RAYS
        self.color = (200, 80, 80)
        self.trail       = []    # list of (x, y)
        self._trail_accum = 0.0  # distance accumulated since last trail point

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt, track):
        if not self.alive:
            return

        self.time_alive += dt

        # --- Neural net ---
        outputs = self.genome.activate(self._inputs(track))
        steer_in = float(np.tanh(outputs[0])) if len(outputs) > 0 else 0.0
        throttle = (float(np.tanh(outputs[1])) + 1.0) / 2.0 if len(outputs) > 1 else 0.5
        brake    = max(0.0, float(np.tanh(outputs[2]))) if len(outputs) > 2 else 0.0

        pre_x, pre_y = self.x, self.y

        # Sub-step physics for stability — bicycle-model dynamics can be stiff
        SUBSTEPS = 2
        sub_dt   = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            self._physics_step(sub_dt, steer_in, throttle, brake)

        moved = math.hypot(self.x - pre_x, self.y - pre_y)
        self.distance += moved

        # Direction-aware progress: project Δposition onto the track's
        # local forward tangent so backward driving doesn't bank fitness.
        dx = self.x - pre_x
        dy = self.y - pre_y
        fwd = self._track_forward(track)
        if fwd is not None:
            fwd_progress = dx * fwd[0] + dy * fwd[1]
            self.forward_distance += fwd_progress
            # Wrong-way detection — kill car after sustained backward motion
            if fwd_progress < -1.5:
                self.wrong_way_timer += dt
                if self.wrong_way_timer > 2.5:
                    self.alive = False
                    return
            else:
                self.wrong_way_timer = max(0.0, self.wrong_way_timer - dt * 0.5)

        if moved < 0.4:
            self.idle_timer += dt
        else:
            self.idle_timer = max(0.0, self.idle_timer - dt * 0.4)
        if self.idle_timer > IDLE_KILL:
            self.alive = False
            return

        # Trail
        self._trail_accum += moved
        if self._trail_accum >= 9.0 or not self.trail:
            self.trail.append((self.x, self.y))
            self._trail_accum = 0.0
            if len(self.trail) > 55:
                self.trail.pop(0)

        # Sensors + collisions + checkpoints + fitness
        self._cast_rays(track)
        self._check_collision(track)
        if self.alive:
            self._check_checkpoints(track)
        self._compute_fitness()

    # ------------------------------------------------------------------
    # One bicycle-model physics step
    # ------------------------------------------------------------------

    def _physics_step(self, dt, steer_in, throttle, brake):
        # --- Smooth steering toward target ---
        target_steer = steer_in * MAX_STEER_ANGLE
        delta        = target_steer - self.steer_angle
        max_delta    = STEER_RATE * dt
        if   delta >  max_delta: delta =  max_delta
        elif delta < -max_delta: delta = -max_delta
        self.steer_angle += delta

        # --- World velocity → car-frame velocity ---
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        vx_local =  self.vx * cos_a + self.vy * sin_a     # forward (+)
        vy_local = -self.vx * sin_a + self.vy * cos_a     # lateral (left +)

        omega = self.angular_velocity

        # --- Lateral velocity at each axle (includes yaw contribution) ---
        front_vy = vy_local + omega * CG_TO_FRONT
        rear_vy  = vy_local - omega * CG_TO_REAR

        speed_abs = abs(vx_local)
        norm      = max(speed_abs, 5.0)   # numerical floor for atan2
        sign_vx   = 1.0 if vx_local >= 0 else -1.0

        # --- Slip angles ---
        front_slip = math.atan2(front_vy, norm) - self.steer_angle * sign_vx
        rear_slip  = math.atan2(rear_vy,  norm)
        if   front_slip >  1.2: front_slip =  1.2
        elif front_slip < -1.2: front_slip = -1.2
        if   rear_slip  >  1.2: rear_slip  =  1.2
        elif rear_slip  < -1.2: rear_slip  = -1.2
        self.front_slip = front_slip
        self.rear_slip  = rear_slip

        # --- Longitudinal weight transfer ---
        long_accel    = self._long_accel_prev
        weight_shift  = MASS * long_accel * CG_HEIGHT / (CG_TO_FRONT + CG_TO_REAR)
        static_front  = MASS * G_REF * CG_TO_REAR  / (CG_TO_FRONT + CG_TO_REAR)
        static_rear   = MASS * G_REF * CG_TO_FRONT / (CG_TO_FRONT + CG_TO_REAR)
        front_load    = max(0.1 * static_front, static_front - weight_shift)
        rear_load     = max(0.1 * static_rear,  static_rear  + weight_shift)

        # --- Pacejka simplified: F_lat = -D·load·sin(C·atan(B·slip)) ---
        front_lat = -TIRE_D * front_load * math.sin(TIRE_C * math.atan(TIRE_B * front_slip))
        rear_lat  = -TIRE_D * rear_load  * math.sin(TIRE_C * math.atan(TIRE_B * rear_slip))

        # --- Longitudinal force (RWD: engine + brakes on rear axle) ---
        engine_force = throttle * ENGINE_FORCE
        if speed_abs > 1.0:
            brake_long = -brake * BRAKE_FORCE * sign_vx
        else:
            brake_long = 0.0
        drive_long = engine_force + brake_long

        # --- Friction circle on driven axle ---
        peak_rear = TIRE_D * rear_load
        combined  = drive_long * drive_long + rear_lat * rear_lat
        if combined > peak_rear * peak_rear:
            scl         = peak_rear / math.sqrt(combined)
            drive_long *= scl
            rear_lat   *= scl

        # --- Drag (linear, with drafting reduction kept from old game) ---
        effective_drag = DRAG_LINEAR * (1.0 - self.draft_boost * 0.55)
        drag_long      = -effective_drag * vx_local * MASS

        # --- Project front lateral force into car frame (rotated by steer) ---
        cos_s = math.cos(self.steer_angle)
        sin_s = math.sin(self.steer_angle)
        front_fx = -front_lat * sin_s
        front_fy =  front_lat * cos_s

        # --- Net forces in car frame ---
        fx_local = drive_long + front_fx + drag_long
        fy_local = front_fy + rear_lat

        ax_local = fx_local / MASS
        ay_local = fy_local / MASS

        # Save for next frame's weight transfer
        self._long_accel_prev = ax_local

        # --- Integrate velocity in car frame (semi-implicit Euler) ---
        vx_local += ax_local * dt
        vy_local += ay_local * dt

        # --- Yaw torque from tire lateral forces (front through cos(steer)) ---
        yaw_torque = front_lat * cos_s * CG_TO_FRONT - rear_lat * CG_TO_REAR
        ang_accel  = yaw_torque / INERTIA
        self.angular_velocity += ang_accel * dt
        self.angular_velocity *= max(0.0, 1.0 - YAW_DAMPING * dt)
        if   self.angular_velocity >  MAX_YAW_RATE: self.angular_velocity =  MAX_YAW_RATE
        elif self.angular_velocity < -MAX_YAW_RATE: self.angular_velocity = -MAX_YAW_RATE

        # --- Update heading ---
        self.angle += self.angular_velocity * dt

        # --- Convert car-frame velocity back to world frame ---
        new_cos = math.cos(self.angle)
        new_sin = math.sin(self.angle)
        self.vx = vx_local * new_cos - vy_local * new_sin
        self.vy = vx_local * new_sin + vy_local * new_cos

        # --- Top-speed clamp ---
        speed_sq = self.vx * self.vx + self.vy * self.vy
        if speed_sq > MAX_SPEED * MAX_SPEED:
            scl       = MAX_SPEED / math.sqrt(speed_sq)
            self.vx  *= scl
            self.vy  *= scl
            vx_local *= scl
            vy_local *= scl

        self.speed = math.sqrt(vx_local * vx_local + vy_local * vy_local)

        # --- Drift detection: continuous body slip angle ---
        body_slip      = abs(math.atan2(vy_local, norm))
        self.body_slip = body_slip
        if   body_slip > DRIFT_ANGLE_ENTER: self.is_drifting = True
        elif body_slip < DRIFT_ANGLE_EXIT:  self.is_drifting = False

        # --- Integrate position ---
        self.x += self.vx * dt
        self.y += self.vy * dt

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def _inputs(self, track=None):
        rays_norm = [d / RAY_LENGTH for d in self.rays]
        fx, fy = np.cos(self.angle), np.sin(self.angle)
        lx, ly = -np.sin(self.angle), np.cos(self.angle)
        fwd_vel = (self.vx * fx + self.vy * fy) / MAX_SPEED
        lat_vel = (self.vx * lx + self.vy * ly) / MAX_SPEED

        # --- New richer sensors ---
        cp_dist  = 1.0    # 1.0 = far / unknown
        cp_angle = 0.0    # relative bearing to next checkpoint, [-1, 1]
        lap_prog = 0.0
        if track is not None and track.checkpoints:
            cp = track.checkpoints[self.next_checkpoint]
            mx = 0.5 * (cp[0][0] + cp[1][0])
            my = 0.5 * (cp[0][1] + cp[1][1])
            dx = mx - self.x
            dy = my - self.y
            d  = (dx * dx + dy * dy) ** 0.5
            cp_dist  = min(1.0, d / 1500.0)
            world_ang  = np.arctan2(dy, dx)
            rel        = world_ang - self.angle
            # wrap to [-pi, pi]
            if rel >  np.pi: rel -= 2 * np.pi
            if rel < -np.pi: rel += 2 * np.pi
            cp_angle = rel / np.pi
            lap_prog = self.next_checkpoint / max(len(track.checkpoints), 1)

        ang_vel = float(np.clip(self.angular_velocity / 5.0, -1.0, 1.0))

        return rays_norm + [
            fwd_vel, lat_vel,
            1.0 if self.is_drifting else 0.0,
            cp_dist, cp_angle, ang_vel, lap_prog,
        ]

    def _track_forward(self, track):
        """
        Unit forward vector of the track at the car's current position.
        Uses the TrackGrid to find the nearest centerline index, then
        looks a few points ahead for a stable tangent direction.
        Returns None if the car is off-track.
        """
        grid = getattr(track, '_grid', None)
        if grid is None:
            return None
        cell = grid.cell
        key = (int(self.x // cell), int(self.y // cell))
        candidates = grid.grid.get(key)
        if not candidates:
            return None

        # Nearest centerline point among candidates
        best_d2  = float('inf')
        best_idx = candidates[0]
        cl = track.centerline
        for i in candidates:
            cx, cy = cl[i]
            d2 = (self.x - cx) ** 2 + (self.y - cy) ** 2
            if d2 < best_d2:
                best_d2  = d2
                best_idx = i

        # Forward tangent — average direction over a few points ahead
        n = len(cl)
        p1 = cl[best_idx]
        p2 = cl[(best_idx + 6) % n]
        fx, fy = p2[0] - p1[0], p2[1] - p1[1]
        ln = math.hypot(fx, fy)
        if ln < 1e-6:
            return None
        return (fx / ln, fy / ln)

    def _cast_rays(self, track):
        for i, rel_angle in enumerate(RAY_ANGLES):
            angle = self.angle + rel_angle
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            hit = RAY_LENGTH

            for s in range(1, RAY_STEPS + 1):
                d = s * RAY_STEP
                rx = self.x + cos_a * d
                ry = self.y + sin_a * d

                on_road = track.is_on_track(rx, ry)
                if not on_road:
                    hit = d
                    break


            self.rays[i] = hit

    # ------------------------------------------------------------------
    # Collision & checkpoints
    # ------------------------------------------------------------------

    def _check_collision(self, track):
        if not track.is_on_track(self.x, self.y):
            self.alive = False

    def _check_checkpoints(self, track):
        if not track.checkpoints:
            return
        cp = track.checkpoints[self.next_checkpoint]
        if self._near_line(cp[0], cp[1], threshold=16.0):
            self.checkpoints_hit += 1
            self.next_checkpoint = (self.next_checkpoint + 1) % len(track.checkpoints)
            if self.next_checkpoint == 0:
                self.laps += 1

    def _near_line(self, start, end, threshold):
        sx, sy = start
        ex, ey = end
        lx, ly = ex - sx, ey - sy
        lsq = lx * lx + ly * ly
        if lsq < 1e-6:
            return False
        t = max(0.0, min(1.0, ((self.x - sx) * lx + (self.y - sy) * ly) / lsq))
        cx2 = sx + t * lx
        cy2 = sy + t * ly
        return (self.x - cx2) ** 2 + (self.y - cy2) ** 2 < threshold ** 2

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _compute_fitness(self):
        # forward_distance is signed track-tangent progress — driving the
        # wrong way actually DEDUCTS from this term. Cars can no longer
        # farm fitness by reversing or running zig-zags.
        self.fitness = (
            self.laps * 10_000
            + self.checkpoints_hit * 120
            + self.forward_distance * 0.10
            - self.time_alive * 0.3
        )
        self.genome.fitness = self.fitness + self.fitness_bonus

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, surface, offset=(0, 0), draw_rays=False):
        ox, oy = offset
        sx = int(self.x - ox)
        sy = int(self.y - oy)

        # Trail — skip segments that are too long (collision pushes, teleports)
        MAX_SEG_SQ = 80 ** 2
        if len(self.trail) > 1:
            dim = tuple(max(30, int(c * 0.55)) for c in self.color)
            for i in range(1, len(self.trail)):
                p1, p2 = self.trail[i - 1], self.trail[i]
                if (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 > MAX_SEG_SQ:
                    continue
                fade = i / len(self.trail)
                col  = tuple(int(dim[c] * fade) for c in range(3))
                pygame.draw.line(surface, col,
                                 (int(p1[0] - ox), int(p1[1] - oy)),
                                 (int(p2[0] - ox), int(p2[1] - oy)), 1)

        # Rays
        if draw_rays:
            for i, rel_angle in enumerate(RAY_ANGLES):
                angle = self.angle + rel_angle
                ex2 = int(self.x + np.cos(angle) * self.rays[i] - ox)
                ey2 = int(self.y + np.sin(angle) * self.rays[i] - oy)
                hit = self.rays[i] < RAY_LENGTH - 1
                ray_col = (255, 60, 60) if hit else (60, 200, 60)
                pygame.draw.line(surface, ray_col, (sx, sy), (ex2, ey2), 1)
                if hit:
                    pygame.draw.circle(surface, (255, 220, 0), (ex2, ey2), 3)

        # Car body
        corners = self._corners()
        pts = [(int(c[0] - ox), int(c[1] - oy)) for c in corners]
        pygame.draw.polygon(surface, self.color, pts)

        # Heading indicator
        fx, fy = np.cos(self.angle), np.sin(self.angle)
        tip = (int(sx + fx * CAR_H // 2), int(sy + fy * CAR_H // 2))
        pygame.draw.line(surface, (255, 255, 255), (sx, sy), tip, 2)

        # Drift glow
        if self.is_drifting:
            pygame.draw.circle(surface, (255, 180, 0), (sx, sy), 8, 2)

    def _corners(self):
        fx, fy = np.cos(self.angle), np.sin(self.angle)
        lx, ly = -np.sin(self.angle), np.cos(self.angle)
        hw, hh = CAR_W / 2, CAR_H / 2
        return [
            (self.x + fx*hh + lx*hw, self.y + fy*hh + ly*hw),
            (self.x + fx*hh - lx*hw, self.y + fy*hh - ly*hw),
            (self.x - fx*hh - lx*hw, self.y - fy*hh - ly*hw),
            (self.x - fx*hh + lx*hw, self.y - fy*hh + ly*hw),
        ]
