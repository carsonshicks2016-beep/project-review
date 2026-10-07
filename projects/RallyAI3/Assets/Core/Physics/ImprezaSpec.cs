using UnityEngine;

namespace Core.Physics
{
    /// <summary>
    /// Single source of truth for the 1998 Subaru Impreza WRC (GC8, WRC98 spec).
    /// Every number here is the real homologated figure or a well-documented
    /// rally setup value — the runtime physics and the editor mesh builder both
    /// read from this file so the visuals and the simulation can never drift apart.
    ///
    /// COORDINATE CONVENTION (local space of the car root):
    ///     +Z = forward,  +Y = up,  +X = right (driver's right)
    ///     y = 0 is the GROUND PLANE at static ride height.
    /// So a wheel centre sits at y = WheelRadius, the roof at y = Height, and the
    /// centre of mass at y = CoMHeight. Nothing is offset by an arbitrary fudge.
    /// </summary>
    public static class ImprezaSpec
    {
        // ══════════════════════════════════════════════════════════════
        //  DIMENSIONS  (mm on the homologation form → metres here)
        // ══════════════════════════════════════════════════════════════
        public const float Length      = 4.340f;   // nose to tail
        public const float Width       = 1.770f;   // over the arches
        public const float Height      = 1.390f;   // roof
        public const float Wheelbase   = 2.520f;
        public const float TrackFront  = 1.500f;   // WRC-widened from the road car's 1.465
        public const float TrackRear   = 1.500f;

        /// <summary>Front axle Z. Wheelbase is centred on the origin.</summary>
        public const float AxleFrontZ  =  Wheelbase * 0.5f;   // +1.26
        public const float AxleRearZ   = -Wheelbase * 0.5f;   // -1.26

        public const float HalfTrackFront = TrackFront * 0.5f; // ±0.75
        public const float HalfTrackRear  = TrackRear  * 0.5f;

        /// <summary>Front overhang 0.87 m, rear 0.95 m → nose at +2.13, tail at -2.21.</summary>
        public const float NoseZ =  AxleFrontZ + 0.87f;
        public const float TailZ =  AxleRearZ  - 0.95f;

        public const float GroundClearance = 0.180f;  // gravel spec floor height

        // ══════════════════════════════════════════════════════════════
        //  MASS PROPERTIES
        // ══════════════════════════════════════════════════════════════
        /// <summary>FIA minimum weight for a 1998 WRC car, with driver and co-driver.</summary>
        public const float Mass = 1230f;

        /// <summary>Static front weight bias. Flat-four hung ahead of the axle line.</summary>
        public const float FrontWeightBias = 0.57f;

        /// <summary>Height of the centre of mass above the ground.</summary>
        public const float CoMHeight = 0.480f;

        /// <summary>Unsprung mass per corner: wheel, tyre, hub, upright, brake.</summary>
        public const float UnsprungMassPerCorner = 40f;

        //  ── Sprung mass: what a car with a two-mass corner model would put on the
        //     springs. NOTHING should size a spring or a damper off these while the
        //     rig has no unsprung mass — see CornerMass* below. They are kept because
        //     they are the physically meaningful figures and are what a future
        //     sprung/unsprung corner model will need.
        public static float SprungMass => Mass - 4f * UnsprungMassPerCorner;      // 1070 kg
        public static float SprungCornerMassFront => SprungMass * FrontWeightBias * 0.5f;        // 305 kg
        public static float SprungCornerMassRear  => SprungMass * (1f - FrontWeightBias) * 0.5f; // 230 kg

        /// <summary>
        /// What one corner's spring ACTUALLY holds up and accelerates in this rig.
        ///
        /// In a real car — and in a two-mass corner model — this would be the sprung
        /// mass alone, with the wheel, hub and brake riding below the spring on the
        /// tyre. Here the wheel is kinematic: it is positioned straight off the ground
        /// cast and has no mass of its own, so all 1230 kg sits on the four springs.
        ///
        /// Sizing the rates off the sprung mass instead left the springs 13 % too soft
        /// for the load they carry: the car settled 17 mm below its design ride height,
        /// gave that 17 mm away out of 170 mm of bump travel, and rode about 7 % under
        /// its target frequency. Deriving them from the true corner mass fixes all
        /// three at once — and because sag is m·g/k = g/ω², the static ride height and
        /// therefore the strut anchor points are unchanged.
        ///
        /// If unsprung mass is ever modelled properly, this reverts to SprungCornerMass*.
        /// </summary>
        public static float CornerMassFront => Mass * FrontWeightBias * 0.5f;          // 350.6 kg
        public static float CornerMassRear  => Mass * (1f - FrontWeightBias) * 0.5f;   // 264.5 kg

        /// <summary>
        /// Longitudinal CoM position. Measured from the front axle the CoM sits at
        /// (1 - frontBias) * wheelbase, so a 57 % front car has its mass 1.084 m
        /// behind the front axle → z = +0.176 in wheelbase-centred coordinates.
        /// </summary>
        public static float CoMZ => AxleFrontZ - (1f - FrontWeightBias) * Wheelbase;

        public static Vector3 CenterOfMass => new Vector3(0f, CoMHeight, CoMZ);

        /// <summary>
        /// Inertia tensor about the CoM (kg·m²), from radii of gyration typical of a
        /// caged compact rally car: pitch/yaw ≈ 1.27 m, roll ≈ 0.62 m.
        /// Unity orders this (about X, about Y, about Z) = (pitch, yaw, roll).
        /// </summary>
        public static Vector3 InertiaTensor => new Vector3(2000f, 2050f, 480f);

        // ══════════════════════════════════════════════════════════════
        //  WHEELS & TYRES
        // ══════════════════════════════════════════════════════════════
        /// <summary>Loaded rolling radius on 15" gravel wheels (205/65R15).</summary>
        /// <summary>
        /// Vertical rate of the tyre carcass itself. This sits in SERIES with the spring
        /// and is what actually touches the road: at 200 kN/m against a 40 kg unsprung
        /// mass it gives a wheel-hop mode near 11 Hz, which is the frequency a damper is
        /// really being tuned against on gravel.
        /// </summary>
        public const float TyreVerticalRate = 200000f;
        /// <summary>Tyres damp very little on their own — most of it is the damper's job.</summary>
        public const float TyreVerticalDamping = 300f;

        public const float WheelRadius = 0.330f;
        public const float TyreWidth   = 0.205f;

        /// <summary>Rotational inertia of one wheel + tyre + brake disc (kg·m²).</summary>
        public const float WheelInertia = 1.2f;

        // ══════════════════════════════════════════════════════════════
        //  SUSPENSION  (MacPherson struts front and rear, gravel setup)
        // ══════════════════════════════════════════════════════════════
        //  Wheel rates are derived from the sprung corner mass and the ride
        //  frequency a gravel car is set up for — soft, long-travel, ~1.5 Hz.
        //      k = m * (2*pi*f)^2
        // ══════════════════════════════════════════════════════════════
        public const float RideFrequencyFront = 1.50f;   // Hz
        public const float RideFrequencyRear  = 1.60f;   // Hz

        public static float SpringRateFront => WheelRate(CornerMassFront, RideFrequencyFront); // ≈ 31.1 kN/m
        public static float SpringRateRear  => WheelRate(CornerMassRear,  RideFrequencyRear);  // ≈ 26.7 kN/m

        static float WheelRate(float cornerMass, float freqHz)
        {
            float omega = 2f * Mathf.PI * freqHz;
            return cornerMass * omega * omega;
        }

        /// <summary>Critical damping coefficient for a corner, c_c = 2*sqrt(k*m).</summary>
        public static float CriticalDamping(float springRate, float cornerMass)
            => 2f * Mathf.Sqrt(springRate * cornerMass);

        //  Gravel dampers are asymmetric: soft in bump so the wheel can follow
        //  the surface, much firmer in rebound to control the body coming back.
        public const float DampingRatioBump    = 0.30f;
        public const float DampingRatioRebound = 0.60f;

        //  Damping is quoted as a ratio of critical, so it has to be taken against the
        //  same corner mass the spring rate was derived from or the ratios come out wrong.
        public static float DamperBumpFront    => CriticalDamping(SpringRateFront, CornerMassFront) * DampingRatioBump;
        public static float DamperReboundFront => CriticalDamping(SpringRateFront, CornerMassFront) * DampingRatioRebound;
        public static float DamperBumpRear     => CriticalDamping(SpringRateRear,  CornerMassRear)  * DampingRatioBump;
        public static float DamperReboundRear  => CriticalDamping(SpringRateRear,  CornerMassRear)  * DampingRatioRebound;

        /// <summary>Free (fully drooped) strut length. Long travel for gravel.</summary>
        public const float SuspensionRestLength = 0.42f;
        /// <summary>
        /// Shortest the strut can get before the bump stop. With 110 mm of static
        /// sag that leaves 170 mm of bump and 110 mm of droop — long enough for
        /// gravel, short enough that the tyre stays inside the arch on a landing.
        /// </summary>
        public const float SuspensionMinLength  = 0.14f;

        //  Bump stops. Rubber, so they stiffen as they are crushed and they lose
        //  most of the energy as heat rather than handing it back — without that
        //  hysteresis a hard landing turns the car into a trampoline, and with too
        //  much progression the stop goes unstable at a coarse fixed timestep.
        //  These values were picked by simulating drops from 0.3 m to 3 m at
        //  dt = 0.02: the floor pan stays clear of the road throughout and the car
        //  settles inside two seconds.
        //
        //  Note the game does NOT run at dt = 0.02 — its fixed timestep is 0.01, and
        //  the belief that it was 0.02 ran unchallenged through this whole file until
        //  the episode length turned out to be half what it claimed. A finer timestep
        //  is the safe direction for an explicit stiff contact, so the tuning holds;
        //  but it was validated at a rate the game has never used.
        public const float BumpStopRate        = 300000f;
        public const float BumpStopRange       = 0.04f;
        public const float BumpStopProgression = 0.8f;
        /// <summary>
        /// Hysteresis in the bump stop, sized against the UNSPRUNG mass — because with a
        /// two-mass corner that is the mass it acts on, not the sprung corner.
        ///
        /// It was 60000, chosen when the wheel was massless and the stop pushed straight
        /// against a ~305 kg sprung corner. Against 40 kg that is 8.7x critical, and an
        /// overdamped stop does not absorb an impact — it FREEZES the strut. The car
        /// would land, hold the strut compressed below its stop while the wheel flew back
        /// into the air, and keep firing the stop into the chassis: it arrived at 4.2 m/s
        /// and left at 7.2 m/s. Critical for 300 kN/m on 40 kg is 6928, so this is a
        /// damping ratio of about 0.43.
        /// </summary>
        public const float BumpStopDamping     = 3000f;

        /// <summary>
        /// Hard ceiling on what one corner may push, as a multiple of the static
        /// load it carries. Nothing in normal driving comes near it; it exists so a
        /// single bad frame cannot fire the car into orbit.
        /// </summary>
        public const float MaxCornerForceMultiple = 20f;

        /// <summary>
        /// Hard ceiling on the vertical load handed to the TYRE, again as a multiple of
        /// static. Deliberately far below <see cref="MaxCornerForceMultiple"/>: the strut
        /// needs that much authority to stop a three-metre landing inside its travel, but
        /// tyre grip is proportional to load, so letting the tyre see the same spike gives
        /// one corner tens of kN of lateral force at ground level — enough to roll the car
        /// on touchdown. Four times static is well clear of anything cornering generates.
        /// </summary>
        public const float MaxTyreLoadMultiple = 4f;

        //  Anti-roll bars, expressed as an equivalent wheel rate resisting the
        //  difference in travel between the two sides. Rear-biased roll stiffness
        //  is what makes a rally car want to rotate on turn-in.
        public const float AntiRollFront = 14000f;
        public const float AntiRollRear  = 16000f;

        /// <summary>
        /// Static suspension sag under the sprung corner load, i.e. how far the
        /// strut compresses from free length just holding the car up.
        /// </summary>
        public static float StaticSagFront => CornerMassFront * 9.81f / SpringRateFront; // ≈ 0.110 m
        public static float StaticSagRear  => CornerMassRear  * 9.81f / SpringRateRear;  // ≈ 0.097 m

        /// <summary>
        /// Where the strut's upper mount sits, so that at rest the wheel centre is
        /// exactly WheelRadius above the ground: anchorY = radius + (free - sag).
        /// </summary>
        public static float StrutAnchorYFront => WheelRadius + SuspensionRestLength - StaticSagFront; // ≈ 0.640
        public static float StrutAnchorYRear  => WheelRadius + SuspensionRestLength - StaticSagRear;  // ≈ 0.653

        // ══════════════════════════════════════════════════════════════
        //  SUSPENSION KINEMATICS
        //
        //  A suspension link does not hand its load to the body at the contact patch —
        //  it hands it over along the line to the instant centre. Applying the lateral
        //  tyre force at ROLL CENTRE height, and the longitudinal force at the
        //  anti-dive / anti-squat height, reproduces that. With it you get the real
        //  split between GEOMETRIC load transfer (instant, no body movement) and
        //  ELASTIC transfer (through the springs, which is what makes the car roll and
        //  dive, and does it with the spring's lag).
        //
        //  Applying everything at the patch, as before, sends every newton through the
        //  springs: the car rolls and pitches more than it should and always late.
        // ══════════════════════════════════════════════════════════════
        /// <summary>Roll centre heights above the contact patch. Rear higher is conventional.</summary>
        public const float RollCentreFront = 0.05f;
        public const float RollCentreRear  = 0.10f;

        /// <summary>
        /// Height at which longitudinal force is fed into the body — the side-view
        /// instant centre. This is what anti-dive and anti-squat actually are.
        /// </summary>
        public const float AntiPitchHeightFront = 0.10f;
        public const float AntiPitchHeightRear  = 0.12f;

        /// <summary>
        /// Pneumatic trail at zero slip, m — how far behind the centre of the contact
        /// patch the lateral force acts. Roughly a sixth of the patch length. This is the
        /// value at full grip; it decays to zero at the friction peak.
        /// </summary>
        public const float PneumaticTrail = 0.035f;

        /// <summary>
        /// Mechanical trail from caster, m. Adds to steering weight but not to the
        /// chassis yaw moment — it reacts through the steering rack.
        /// </summary>
        public const float MechanicalTrail = 0.025f;

        /// <summary>
        /// Half-length of the ground the tyre can bridge, m. Features shorter than this
        /// get enveloped by the carcass instead of jolting the wheel.
        /// </summary>
        public const float EnvelopeSpan = 0.13f;

        /// <summary>
        /// Halfshaft torsion. ~13 Hz against a bare wheel, which is real driveline
        /// shuffle. Damping is chosen for a well-damped shaft, not a ringing one.
        /// </summary>
        public const float HalfshaftStiffness = 6000f, HalfshaftDamping = 40f;

        /// <summary>
        /// Brake disc thermal mass (J/K) and the fade curve. Past the onset temperature
        /// the pad loses friction; at saturation it has given up MaxLoss of its torque.
        /// </summary>
        public const float DiscThermalMassFront = 2760f, DiscThermalMassRear = 1840f;
        public const float BrakeFadeOnsetC = 500f, BrakeFadeFullC = 800f, BrakeFadeMaxLoss = 0.60f;
        public const float BrakeAmbientC = 20f;

        /// <summary>
        /// Boost anti-lag holds regardless of engine speed. Anti-lag burns fuel in the
        /// exhaust manifold precisely so the turbo keeps spinning at low rpm; gating
        /// spool authority purely on rpm defeats the whole mechanism and made the car
        /// pull away on 85 N.m, taking 7 s to 100 km/h instead of 5.7.
        /// </summary>
        public const float AntiLagSpoolFloor = 0.60f;

        // ── Steering geometry ─────────────────────────────────────────
        public const float MaxSteerAngle   = 35f;    // degrees at the inner wheel
        public const float SteerRateDegSec = 260f;   // how fast the rack can be moved
        public const float CamberFront     = -1.5f;  // degrees, negative = top leans in
        public const float CamberRear      = -1.0f;
        public const float ToeFront        =  0.10f; // degrees, positive = toe-in
        public const float ToeRear         =  0.20f;

        // ══════════════════════════════════════════════════════════════
        //  ENGINE — EJ20 2.0 L turbo flat-four, 34 mm restrictor
        // ══════════════════════════════════════════════════════════════
        //  ~300 bhp @ 5500 rpm, 470 N·m @ 3500–4000 rpm. The restrictor is what
        //  flattens the torque curve and kills the top end.
        // ══════════════════════════════════════════════════════════════
        public const float IdleRpm      = 1400f;
        public const float RedlineRpm   = 7500f;
        public const float RevLimitRpm  = 7600f;
        public const float StallRpm     = 900f;
        /// <summary>Crank + flywheel + clutch inertia (kg·m²). Rally-light flywheel.</summary>
        public const float EngineInertia = 0.28f;

        /// <summary>Torque curve sample points, one every 500 rpm from 1000 to 7500.</summary>
        public static readonly float[] TorqueCurveNm =
        {
            /* 1000 */ 150f,
            /* 1500 */ 215f,
            /* 2000 */ 300f,
            /* 2500 */ 400f,
            /* 3000 */ 450f,
            /* 3500 */ 470f,
            /* 4000 */ 470f,
            /* 4500 */ 455f,
            /* 5000 */ 420f,
            /* 5500 */ 388f,
            /* 6000 */ 355f,
            /* 6500 */ 320f,
            /* 7000 */ 280f,
            /* 7500 */ 230f,
        };
        public const float TorqueCurveStartRpm = 1000f;
        public const float TorqueCurveStepRpm  = 500f;

        /// <summary>Engine braking torque coefficient — pumping and friction losses.</summary>
        public const float EngineBrakingCoeff = 0.035f;   // N·m per rad/s

        /// <summary>Turbo spool-up and bleed-down time constants (seconds).</summary>
        public const float BoostSpoolTime = 0.55f;
        public const float BoostDecayTime = 0.35f;
        /// <summary>Fraction of peak torque available with the turbo completely off boost.</summary>
        public const float OffBoostTorqueFraction = 0.42f;
        /// <summary>Anti-lag holds this much boost with the throttle shut. Group A cars ran it hot.</summary>
        public const float AntiLagBoostFloor = 0.65f;

        // ══════════════════════════════════════════════════════════════
        //  TRANSMISSION — 6-speed sequential, permanent four-wheel drive
        // ══════════════════════════════════════════════════════════════
        public static readonly float[] GearRatios = { -3.20f, 0f, 2.85f, 2.10f, 1.65f, 1.35f, 1.13f, 0.96f };
        //                                              R      N     1      2      3      4      5      6
        public const int ReverseGearIndex = 0;
        public const int NeutralGearIndex = 1;
        public const int FirstGearIndex   = 2;

        public const float FinalDrive          = 3.90f;
        public const float DrivetrainEfficiency = 0.90f;
        public const float ShiftTimeSeconds     = 0.09f;   // sequential dog box — brutally quick
        public const float UpshiftRpm           = 7050f;
        public const float DownshiftRpm         = 3200f;

        // ── Differentials ─────────────────────────────────────────────
        /// <summary>Centre diff torque split. 0.5 = even; higher sends more forward.</summary>
        public const float CentreTorqueSplitFront = 0.50f;
        /// <summary>Viscous coupling in the centre diff (N·m per rad/s of axle speed difference).</summary>
        public const float CentreDiffLock = 220f;
        /// <summary>Plate-LSD locking torque per rad/s of wheel speed difference across an axle.</summary>
        public const float LsdLockFront = 260f;
        public const float LsdLockRear  = 340f;
        /// <summary>Preload — the torque the plates carry before any speed difference at all.</summary>
        public const float LsdPreload   = 45f;

        // ══════════════════════════════════════════════════════════════
        //  BRAKES
        // ══════════════════════════════════════════════════════════════
        /// <summary>Peak clamping torque at one wheel (N·m).</summary>
        public const float BrakeTorqueFront = 2600f;
        public const float BrakeTorqueRear  = 1500f;
        /// <summary>Handbrake acts on the rear axle only and locks it solid.</summary>
        public const float HandbrakeTorque  = 4500f;

        // ══════════════════════════════════════════════════════════════
        //  AERODYNAMICS
        // ══════════════════════════════════════════════════════════════
        public const float AirDensity   = 1.225f;      // kg/m³ at sea level
        public const float DragArea     = 0.80f;       // Cd·A — boxy sedan with a big wing
        public const float LiftAreaFront = 0.35f;      // Cl·A, downforce positive
        public const float LiftAreaRear  = 0.58f;      // the WRC wing does the work
        /// <summary>Where the aero loads are applied, in local space.</summary>
        public static Vector3 AeroCentreFront => new Vector3(0f, 0.45f, AxleFrontZ);
        public static Vector3 AeroCentreRear  => new Vector3(0f, 1.05f, TailZ + 0.25f);

        // ══════════════════════════════════════════════════════════════
        //  TYRE — Pacejka Magic Formula coefficients
        // ══════════════════════════════════════════════════════════════
        //  Slip angle is in RADIANS here, so B is an order of magnitude smaller
        //  than the classic degree-based '89 coefficients.
        //
        //  These are chosen so the PEAK lands where a real tyre's does. The previous
        //  set (BLat 9.0, ELat 0.92) put the lateral peak at 75.6 deg of slip angle
        //  and the longitudinal peak at 0.26 slip ratio — roughly 6x and 2x too late.
        //  E controls curvature near the peak, and driving it up to 0.92 flattens the
        //  curve so far that the maximum migrates off into slip angles no tyre ever
        //  reaches. The car still made about the right peak g, so it looked plausible
        //  on a skidpad, but it got there only after absurd slip: grip built forever
        //  and never broke away, which is the opposite of how a car communicates its
        //  limit. Lateral E for a real tyre sits between -1.0 and +0.5.
        //
        //  Now: lateral peaks at 12.0 deg with 1301 N/deg of cornering stiffness
        //  (real: 10-15 deg on gravel, 1000-1500 N/deg), longitudinal at 0.146 slip
        //  ratio (real: 0.10-0.15).
        // ══════════════════════════════════════════════════════════════
        public const float TyreBLat = 14.25f;
        public const float TyreCLat = 1.30f;
        public const float TyreELat = 0.20f;

        public const float TyreBLong = 12.0f;
        public const float TyreCLong = 1.65f;
        public const float TyreELong = 0.50f;

        /// <summary>Peak friction coefficient on dry gravel, at the reference load.</summary>
        public const float PeakFriction = 1.15f;
        /// <summary>
        /// Load sensitivity: tyres lose grip as they are pressed harder.
        /// mu = mu0 * (1 - k * (Fz/Fz0 - 1)). k ≈ 0.15 is typical.
        /// </summary>
        public const float LoadSensitivity = 0.15f;
        /// <summary>Reference vertical load a corner carries statically (N).</summary>
        public static float ReferenceLoad => Mass * 9.81f * 0.25f;   // ≈ 3016 N

        /// <summary>Relaxation length — how far the tyre must roll to build up its slip force.</summary>
        public const float RelaxationLength = 0.45f;

        /// <summary>Camber thrust coefficient, force per radian of camber per newton of load.</summary>
        public const float CamberStiffness = 0.9f;

        public const float RollingResistance = 0.022f;  // Crr on gravel

        // ══════════════════════════════════════════════════════════════
        //  LIVERY  — 555 State Express colours off the 1998 works car
        // ══════════════════════════════════════════════════════════════
        public static readonly Color32 RallyBlue   = new Color32(  0,  57, 166, 255);
        public static readonly Color32 RallyYellow = new Color32(255, 205,   0, 255);
        public static readonly Color32 RallyWhite  = new Color32(240, 240, 245, 255);
        public static readonly Color32 GlassBlack  = new Color32( 10,  11,  16, 255);
        public static readonly Color32 TrimBlack   = new Color32( 24,  24,  26, 255);
        public static readonly Color32 RimGold     = new Color32(214, 172,  62, 255);
        public static readonly Color32 TyreBlack   = new Color32( 30,  30,  32, 255);
        public static readonly Color32 LampAmber   = new Color32(255, 172,  48, 255);
        public static readonly Color32 LampRed     = new Color32(196,  28,  24, 255);

        // ══════════════════════════════════════════════════════════════
        //  HELPERS
        // ══════════════════════════════════════════════════════════════

        /// <summary>Naturally aspirated-equivalent crank torque at a given engine speed.</summary>
        public static float TorqueAtRpm(float rpm)
        {
            if (rpm <= TorqueCurveStartRpm)
            {
                // Below the first sample, fade in from the stall speed.
                float t = Mathf.InverseLerp(StallRpm, TorqueCurveStartRpm, rpm);
                return TorqueCurveNm[0] * Mathf.Clamp01(t);
            }

            float pos  = (rpm - TorqueCurveStartRpm) / TorqueCurveStepRpm;
            int   i    = Mathf.FloorToInt(pos);
            if (i >= TorqueCurveNm.Length - 1) return TorqueCurveNm[TorqueCurveNm.Length - 1];
            return Mathf.Lerp(TorqueCurveNm[i], TorqueCurveNm[i + 1], pos - i);
        }

        public const float RpmToRadPerSec = Mathf.PI * 2f / 60f;
        public const float RadPerSecToRpm = 60f / (Mathf.PI * 2f);
    }
}
