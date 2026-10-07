# Supra Drift Physics Model

The physics engine (`supra/physics.py`) in Supra Drift is a custom, from-scratch 2.5D simulation built to authentically model real-world vehicle dynamics, including combined-slip grip, weight transfer, and airborne ballistics.

## Core Mechanics

### Four-Wheel Dynamics & Pacejka Tyres
- **Independent Contact Patches**: Slip angle and slip ratio are calculated independently for all four corners.
- **Combined-Slip Pacejka**: The model normalizes the friction ellipse so there is only one shared grip budget per tyre. Power-on wheelspin directly bleeds lateral grip, producing genuine throttle-on oversteer (not a scripted effect).
- **Tyre Relaxation Length**: Lateral forces build over distance rather than instantaneously. This produces real transient feel through drift transitions and snap oversteer.
- **Dynamic Load & Friction**: Peak friction is load-sensitive, and weight transfers longitudinally and laterally (roll-stiffness-distributed). Downforce scales with speed and acts on the suspension.

### Drivetrain & Engine
- **Sub-stepped Drivetrain**: The drivetrain simulation resolves at a higher frequency to maintain stability. The power flow is:
  `Engine Flywheel → Clutch → 6-Speed Gearbox → Limited-Slip Diff → Driven Wheels`
- **Turbo Spooling**: Sequential twin-turbo spooling is modeled based on throttle and RPM. 
- **Kinematic Rolling**: Free-rolling wheels are handled kinematically for low-speed stability to prevent numerical blow-ups.

### Environmental Forces
- **Aero Drag & Rolling Resistance**: Aero drag opposes the *full velocity vector*, meaning sideways slides aggressively scrub speed.
- **Off-Track Grip**: Driving off-track applies a strict grip multiplier and increases rolling resistance.

## The 2.5D World: Hills & Jumps

Unlike purely flat 2D simulators, Supra Drift operates in a 2.5D world where tracks feed grade, bank, and vertical curvature into the physics loop (`set_road()`). 

- **Slope Gravity**: Slope gravity acts as a real body force projected onto the car's axes. Sliding sideways down a climb pulls the car laterally down the slope.
- **Dynamic Normal Load**: Normal load scales with the slope and is modified by $v^2$ over crests and dips.
- **Emergent Jumps**: There are no scripted jump zones. When a crest's $v^2$ term fully unloads the car, it takes off. 
  - **Flight**: Ballistic flight conserves planar momentum minus drag. Yaw rate freezes, and the engine free-revs if the throttle is applied.
  - **Landing**: Landing is modeled as a suspension impact whose grip cost directly falls out of the load-sensitive friction model.

*Note: You can bypass the elevation physics and return to byte-identical flat-ground physics by using the `--flat` flag during a run.*

## Physics Validation
The physics engine has been validated against real-world benchmarks:
- **0–100 km/h** takes ~5.3s with realistic turbo lag.
- **Oversteer**: Lifting vs flooring the throttle mid-corner swings the slip angle from ~10° to ~50° (real combined-slip).
- **Hill forces**: The slope pull analytically matches $-g \cdot \text{grade}$. Coast-down energy conservation books to 0.6% error. Takeoff thresholds are within 2% of $\sqrt{g/|v_{\text{curv}}|}$.

Automated tests in `tools/` (e.g., `validate_hills.py`, `validate_jumps.py`, `regression_baseline.py`) enforce that these characteristics remain stable and identical across updates.
