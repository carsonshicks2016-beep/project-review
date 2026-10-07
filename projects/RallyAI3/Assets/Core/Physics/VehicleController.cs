using UnityEngine;

namespace Core.Physics
{
    /// <summary>
    /// The car itself: mass properties, engine, gearbox, differentials, brakes,
    /// steering and aerodynamics for a 1998 Subaru Impreza WRC.
    ///
    /// Everything downstream of here is torque. The engine makes crank torque from
    /// its curve and boost state, the gearbox multiplies it, the centre diff splits
    /// it front to rear, the two limited-slip diffs split it left to right, and each
    /// <see cref="PacejkaTireModel"/> turns its share into a force on the ground.
    /// Nothing fakes a velocity — the car goes where the tyres push it.
    ///
    /// Runs before the suspension and the tyres so that this step's steering and
    /// torque requests are the ones they solve against.
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    [DefaultExecutionOrder(-100)]
    public class VehicleController : MonoBehaviour
    {
        // ══════════════════════════════════════════════════════════════
        //  INPUTS
        // ══════════════════════════════════════════════════════════════
        [Header("Inputs")]
        [Range(-1f, 1f)] public float steeringInput;
        [Range(0f, 1f)]  public float throttleInput;
        [Range(0f, 1f)]  public float brakeInput;
        [Range(0f, 1f)]  public float handbrakeInput;

        [Tooltip("Let the keyboard take over while a key is held. Turn off for headless ML training.")]
        public bool allowKeyboardFallback = true;

        [Tooltip("How long after the last keypress the policy gets the car back.")]
        public float keyboardHandbackSeconds = 1.0f;

        // ══════════════════════════════════════════════════════════════
        //  SETUP
        // ══════════════════════════════════════════════════════════════
        [Header("Mass")]
        public float mass = ImprezaSpec.Mass;
        public Vector3 centerOfMass = new Vector3(0f, ImprezaSpec.CoMHeight, 0.176f);
        public Vector3 inertiaTensor = new Vector3(2000f, 2050f, 480f);

        [Header("Steering")]
        public float maxSteerAngle = ImprezaSpec.MaxSteerAngle;
        public float steerRateDegPerSec = ImprezaSpec.SteerRateDegSec;
        [Tooltip("0 = both front wheels parallel, 1 = full Ackermann geometry.")]
        [Range(0f, 1f)] public float ackermannFactor = 0.6f;
        [Tooltip("Fraction of full lock still available flat out. Keeps a high-speed twitch from spinning the car.")]
        [Range(0.2f, 1f)] public float highSpeedSteerLimit = 0.42f;

        [Header("Engine")]
        public float idleRpm = ImprezaSpec.IdleRpm;
        public float revLimitRpm = ImprezaSpec.RevLimitRpm;
        public float engineInertia = ImprezaSpec.EngineInertia;
        [Tooltip("Group A anti-lag: keeps the turbo lit with the throttle shut. Loud, hot, and very fast out of hairpins.")]
        public bool antiLag = true;
        [Tooltip("Scales the whole torque curve. 1.0 is the homologated 34 mm restrictor engine.")]
        public float enginePowerScale = 1.0f;

        [Header("Clutch")]
        [Tooltip("Torque the plates can hold while slipping. Must comfortably exceed peak " +
                 "engine torque (470 N·m) or the clutch never stops slipping.")]
        public float clutchCapacityNm = 700f;
        [Tooltip("How long the pedal takes to come up on a standing start.")]
        public float clutchEngageTime = 0.45f;
        [Tooltip("Slip below which the friction torque tapers off instead of switching " +
                 "sign abruptly. Keeps the driveline from chattering as the plates meet.")]
        public float clutchBlendRpm = 400f;

        [Header("Transmission")]
        public bool automaticGearbox = true;
        [Tooltip("Lets the car select reverse by holding the brake at a standstill.")]
        public bool allowAutoReverse = true;
        public float finalDrive = ImprezaSpec.FinalDrive;
        public float drivetrainEfficiency = ImprezaSpec.DrivetrainEfficiency;
        public float shiftTime = ImprezaSpec.ShiftTimeSeconds;
        public float upshiftRpm = ImprezaSpec.UpshiftRpm;
        public float downshiftRpm = ImprezaSpec.DownshiftRpm;

        [Header("Differentials")]
        [Range(0f, 1f)] public float centreSplitFront = ImprezaSpec.CentreTorqueSplitFront;
        public float centreDiffLock = ImprezaSpec.CentreDiffLock;
        public float lsdLockFront = ImprezaSpec.LsdLockFront;
        public float lsdLockRear = ImprezaSpec.LsdLockRear;
        public float lsdPreload = ImprezaSpec.LsdPreload;

        [Header("Brakes")]
        public float brakeTorqueFront = ImprezaSpec.BrakeTorqueFront;
        public float brakeTorqueRear = ImprezaSpec.BrakeTorqueRear;
        public float handbrakeTorque = ImprezaSpec.HandbrakeTorque;

        [Tooltip("Halfshafts wind up under torque instead of delivering it instantly.")]
        public bool drivelineCompliance = true;
        [Tooltip("Brake discs heat up and lose torque past ~500 degC.")]
        public bool brakeFade = true;

        [Header("Aerodynamics")]
        public float dragArea = ImprezaSpec.DragArea;
        public float liftAreaFront = ImprezaSpec.LiftAreaFront;
        public float liftAreaRear = ImprezaSpec.LiftAreaRear;

        // ══════════════════════════════════════════════════════════════
        //  TELEMETRY  (read-only, safe for the HUD, the AI and the audio)
        // ══════════════════════════════════════════════════════════════
        [Header("Telemetry (Read Only)")]
        public float currentSpeedKmh;
        public float currentRpm;
        /// <summary>Display gear: -1 reverse, 0 neutral, 1..6 forward.</summary>
        public int currentGear;
        public float currentBoost;          // 0..1
        public float engineTorqueNm;
        public float wheelTorqueNm;
        /// <summary>Chassis slip angle in degrees — how far sideways the car is travelling.</summary>
        public float driftAngleDeg;
        public float lateralG;
        public float longitudinalG;
        public bool isShifting;
        public int groundedWheels;

        public float NormalisedRpm => Mathf.Clamp01((currentRpm - idleRpm) / (revLimitRpm - idleRpm));
        public float ForwardSpeed { get; private set; }
        public Rigidbody Body => rb;
        public DynamicSuspension[] Suspensions => suspensions;
        public PacejkaTireModel[] Tires => tires;

        // ══════════════════════════════════════════════════════════════
        //  INTERNALS
        // ══════════════════════════════════════════════════════════════
        Rigidbody rb;
        DynamicSuspension[] suspensions;
        PacejkaTireModel[] tires;

        PacejkaTireModel tireFL, tireFR, tireRL, tireRR;
        DynamicSuspension susFL, susFR, susRL, susRR;

        int gearIndex = ImprezaSpec.NeutralGearIndex;
        float shiftTimer;
        float engineRpm;
        float boost;
        float clutchEngage;
        bool clutchLocked;
        float currentSteerDeg;
        Vector3 lastVelocity;
        float lastExternalInputTime = -999f;
        float lastKeyboardTime = -999f;

        /// <summary>
        /// How long an external SetInputs call keeps the keyboard locked out. It has
        /// to outlast an ML-Agents decision period, otherwise the fallback would
        /// fight the policy on every step between decisions.
        /// </summary>
        const float ExternalInputHoldSeconds = 0.5f;

        void Awake()
        {
            rb = GetComponent<Rigidbody>();
            ConfigureRigidbody();
            ConfigureBodywork();
            CollectCorners();
            engineRpm = idleRpm;
            gearIndex = ImprezaSpec.NeutralGearIndex;
        }

        /// <summary>
        /// Mass, centre of mass and inertia tensor. Unity's automatic tensor comes
        /// from the collider shape and is wildly wrong for a car — a box collider
        /// gives a car that rolls like a brick and yaws like a shopping trolley.
        /// These are measured-car figures.
        /// </summary>
        public void ConfigureRigidbody()
        {
            if (rb == null) rb = GetComponent<Rigidbody>();

            rb.mass = mass;
            rb.centerOfMass = centerOfMass;
            rb.inertiaTensor = inertiaTensor;
            rb.inertiaTensorRotation = Quaternion.identity;
            rb.linearDamping = 0f;          // all drag is modelled explicitly
            rb.angularDamping = 0.05f;
            rb.interpolation = RigidbodyInterpolation.Interpolate;
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            rb.maxAngularVelocity = 12f;
        }

        /// <summary>
        /// Bodywork wants to skate along a rock face, not grab it — high-friction
        /// panels turn every glancing hit into a barrel roll. Created at runtime so
        /// there is no extra asset to keep in step with the rest of the setup.
        /// </summary>
        void ConfigureBodywork()
        {
            var col = GetComponent<Collider>();
            if (col == null || col.sharedMaterial != null) return;

            col.sharedMaterial = new PhysicsMaterial("Impreza_Bodywork")
            {
                dynamicFriction = 0.15f,
                staticFriction = 0.15f,
                bounciness = 0.05f,
                frictionCombine = PhysicsMaterialCombine.Minimum,
                bounceCombine = PhysicsMaterialCombine.Minimum
            };
        }

        void CollectCorners()
        {
            suspensions = GetComponentsInChildren<DynamicSuspension>();
            tires = GetComponentsInChildren<PacejkaTireModel>();

            foreach (var sus in suspensions)
            {
                switch (sus.corner)
                {
                    case DynamicSuspension.Corner.FrontLeft:  susFL = sus; break;
                    case DynamicSuspension.Corner.FrontRight: susFR = sus; break;
                    case DynamicSuspension.Corner.RearLeft:   susRL = sus; break;
                    case DynamicSuspension.Corner.RearRight:  susRR = sus; break;
                }
            }

            // Anti-roll bars link the two sides of an axle.
            if (susFL && susFR) { susFL.oppositeWheel = susFR; susFR.oppositeWheel = susFL; }
            if (susRL && susRR) { susRL.oppositeWheel = susRR; susRR.oppositeWheel = susRL; }

            // Take over the corners' solve so the two passes stay in step. Only at
            // runtime — in the editor the corners keep solving themselves so gizmos and
            // the builder still work with no controller running.
            if (Application.isPlaying)
                foreach (var sus in suspensions)
                    if (sus != null) sus.solvedByController = true;

            foreach (var tire in tires)
            {
                var sus = tire.GetComponent<DynamicSuspension>();
                if (sus == null) continue;

                tire.isDriveWheel = true;                 // permanent four-wheel drive
                tire.isSteeringWheel = sus.IsFront;

                switch (sus.corner)
                {
                    case DynamicSuspension.Corner.FrontLeft:  tireFL = tire; break;
                    case DynamicSuspension.Corner.FrontRight: tireFR = tire; break;
                    case DynamicSuspension.Corner.RearLeft:   tireRL = tire; break;
                    case DynamicSuspension.Corner.RearRight:  tireRR = tire; break;
                }
            }
        }

        // ══════════════════════════════════════════════════════════════
        //  MAIN LOOP
        // ══════════════════════════════════════════════════════════════
        void FixedUpdate()
        {
            if (rb == null) return;
            float dt = Time.fixedDeltaTime;

            ReadFallbackInput();

            Vector3 velocity = rb.linearVelocity;
            ForwardSpeed = Vector3.Dot(velocity, transform.forward);
            currentSpeedKmh = velocity.magnitude * 3.6f;

            UpdateSteering(dt);
            UpdateGearbox(dt);
            float driveTorque = UpdateEngine(dt);
            DistributeTorque(driveTorque);
            ApplyBrakes();
            SolveSuspension();
            ApplyAerodynamics(velocity);
            UpdateTelemetry(velocity, dt);
        }

        /// <summary>
        /// Keyboard is only consulted when nothing has driven the car recently, so
        /// an ML-Agents policy never has to fight the Input Manager for control.
        /// </summary>
        void ReadFallbackInput()
        {
            if (!allowKeyboardFallback) return;

            float v, h;
            bool pressing;
            try
            {
                v = Input.GetAxis("Vertical");
                h = Input.GetAxis("Horizontal");

                // Explicit KEYS decide who is driving, never axis magnitude.
                //
                // GetAxis folds together every device Unity has bound to "Horizontal",
                // and this project defines a joystick axis alongside the keyboard one.
                // Anything that reads slightly off centre — a drifting stick, a device
                // mis-enumerated as a joystick — then looks exactly like a driver asking
                // to steer, and the fallback takes the car off the policy and holds it
                // there for the whole run. That is what made a perfectly good trained
                // policy appear to "hang left" in the editor while scoring 107 in a
                // built player. A key is unambiguous; an axis is not.
                pressing = Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.S)
                        || Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.D)
                        || Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.DownArrow)
                        || Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.RightArrow)
                        || Input.GetKey(KeyCode.Space);
            }
            catch { return; }   // Input Manager axes not configured — leave the inspector values alone.

            if (pressing) lastKeyboardTime = Time.time;

            // Control always comes BACK. Holding a key takes the car; letting go hands it
            // to the policy again a moment later. The old version had no route back.
            bool driverHasControl = Time.time - lastKeyboardTime < keyboardHandbackSeconds;
            if (!driverHasControl && Time.time - lastExternalInputTime < ExternalInputHoldSeconds) return;

            throttleInput = Mathf.Clamp01(v);
            brakeInput = Mathf.Clamp01(-v);
            steeringInput = Mathf.Clamp(h, -1f, 1f);
            handbrakeInput = Input.GetKey(KeyCode.Space) ? 1f : 0f;
        }

        // ── Steering ──────────────────────────────────────────────────
        void UpdateSteering(float dt)
        {
            // Full lock is there when you need it at walking pace; flat out the rack
            // is progressively wound down so a twitch cannot spin the car.
            float speedFactor = Mathf.Clamp01(Mathf.Abs(ForwardSpeed) / 45f);
            float allowed = maxSteerAngle * Mathf.Lerp(1f, highSpeedSteerLimit, speedFactor * speedFactor);

            float target = steeringInput * allowed;
            currentSteerDeg = Mathf.MoveTowards(currentSteerDeg, target, steerRateDegPerSec * dt);

            // Ackermann: on a turn, the inside wheel traces a tighter circle than the
            // outside one, so it has to point further in or it just scrubs.
            float inner = currentSteerDeg;
            float outer = currentSteerDeg;

            if (Mathf.Abs(currentSteerDeg) > 0.05f)
            {
                float rad = Mathf.Abs(currentSteerDeg) * Mathf.Deg2Rad;
                float turnRadius = ImprezaSpec.Wheelbase / Mathf.Tan(rad);
                float half = ImprezaSpec.HalfTrackFront;

                float ackInner = Mathf.Atan(ImprezaSpec.Wheelbase / Mathf.Max(0.1f, turnRadius - half)) * Mathf.Rad2Deg;
                float ackOuter = Mathf.Atan(ImprezaSpec.Wheelbase / (turnRadius + half)) * Mathf.Rad2Deg;

                float sign = Mathf.Sign(currentSteerDeg);
                inner = sign * Mathf.Lerp(Mathf.Abs(currentSteerDeg), ackInner, ackermannFactor);
                outer = sign * Mathf.Lerp(Mathf.Abs(currentSteerDeg), ackOuter, ackermannFactor);
            }

            // Turning right (+) makes the right wheel the inside one.
            bool turningRight = currentSteerDeg > 0f;
            if (susFL) susFL.steerAngle = turningRight ? outer : inner;
            if (susFR) susFR.steerAngle = turningRight ? inner : outer;
        }

        // ── Gearbox ───────────────────────────────────────────────────
        void UpdateGearbox(float dt)
        {
            if (shiftTimer > 0f)
            {
                shiftTimer -= dt;
                isShifting = shiftTimer > 0f;
                return;
            }
            isShifting = false;

            if (!automaticGearbox) return;

            bool nearlyStopped = Mathf.Abs(ForwardSpeed) < 0.8f;

            // Reverse: hold the brake with the car stopped and it drops into R.
            if (allowAutoReverse && nearlyStopped && brakeInput > 0.5f && throttleInput < 0.1f)
            {
                if (gearIndex != ImprezaSpec.ReverseGearIndex) ShiftTo(ImprezaSpec.ReverseGearIndex);
                return;
            }

            if (gearIndex == ImprezaSpec.ReverseGearIndex)
            {
                if (throttleInput > 0.2f && ForwardSpeed > -0.5f) ShiftTo(ImprezaSpec.FirstGearIndex);
                return;
            }

            if (gearIndex == ImprezaSpec.NeutralGearIndex)
            {
                if (throttleInput > 0.05f || !nearlyStopped) ShiftTo(ImprezaSpec.FirstGearIndex);
                return;
            }

            int topGear = ImprezaSpec.GearRatios.Length - 1;

            if (engineRpm > upshiftRpm && gearIndex < topGear && throttleInput > 0.15f)
            {
                ShiftTo(gearIndex + 1);
            }
            else if (engineRpm < downshiftRpm && gearIndex > ImprezaSpec.FirstGearIndex)
            {
                ShiftTo(gearIndex - 1);
            }
        }

        void ShiftTo(int newIndex)
        {
            newIndex = Mathf.Clamp(newIndex, 0, ImprezaSpec.GearRatios.Length - 1);
            if (newIndex == gearIndex) return;
            gearIndex = newIndex;
            shiftTimer = shiftTime;
            isShifting = true;
        }

        public void ShiftUp()   => ShiftTo(gearIndex + 1);
        public void ShiftDown() => ShiftTo(gearIndex - 1);

        public void PrepareFirstGearStandingStart()
        {
            if (Mathf.Abs(ForwardSpeed) > 0.1f) return;
            gearIndex = ImprezaSpec.FirstGearIndex;
            shiftTimer = 0f;
            isShifting = false;
        }

        // ── Engine ────────────────────────────────────────────────────
        /// <summary>Returns the torque arriving at the driveline, already geared up.</summary>
        float UpdateEngine(float dt)
        {
            float gearRatio = ImprezaSpec.GearRatios[gearIndex];
            float totalRatio = gearRatio * finalDrive;
            bool inGear = !Mathf.Approximately(gearRatio, 0f) && !isShifting;

            // What the wheels are asking the crank to do, through the gears.
            float avgDrivenOmega = AverageDrivenWheelOmega();
            float drivelineRpm = Mathf.Abs(avgDrivenOmega * totalRatio) * ImprezaSpec.RadPerSecToRpm;

            // ── Clutch engagement, on a TIMER.
            //    It used to be keyed off engine speed, which deadlocks: engaging drags
            //    the revs down, dropping the revs backs the engagement off, and the car
            //    parks at ~1500 rpm on 4 % of clutch forever — enough torque to hold
            //    station against rolling resistance and not a newton more.
            bool wantsDrive = inGear && (throttleInput > 0.05f || drivelineRpm > idleRpm * 0.9f);
            clutchEngage = Mathf.MoveTowards(clutchEngage, wantsDrive ? 1f : 0f,
                                             dt / Mathf.Max(0.05f, wantsDrive ? clutchEngageTime : 0.15f));
            if (!inGear) { clutchEngage = 0f; clutchLocked = false; }
            float clutch = clutchEngage;

            // Turbo. Nothing below about 1800 rpm, spools over half a second, and with
            // anti-lag it stays lit when the driver lifts.
            // Anti-lag keeps the turbo spinning by burning fuel in the exhaust manifold,
            // which does not care what the engine is doing. Gating spool authority purely
            // on rpm defeats it, and the car pulls away on off-boost torque.
            float spoolAuthority = Mathf.Clamp01((engineRpm - 1800f) / 1800f);
            if (antiLag) spoolAuthority = Mathf.Max(spoolAuthority, ImprezaSpec.AntiLagSpoolFloor);
            float boostTarget = (throttleInput > 0.15f ? 1f : (antiLag ? ImprezaSpec.AntiLagBoostFloor : 0f)) * spoolAuthority;
            float boostTau = boostTarget > boost ? ImprezaSpec.BoostSpoolTime : ImprezaSpec.BoostDecayTime;
            boost = Mathf.Lerp(boost, boostTarget, Mathf.Clamp01(dt / Mathf.Max(0.01f, boostTau)));

            // Fuel cut on the limiter, so the engine bounces off it rather than sailing past.
            bool onLimiter = engineRpm >= revLimitRpm;
            float throttleForTorque = (onLimiter || isShifting) ? 0f : throttleInput;

            float boostMultiplier = Mathf.Lerp(ImprezaSpec.OffBoostTorqueFraction, 1f, boost);
            float crankTorque = ImprezaSpec.TorqueAtRpm(engineRpm) * boostMultiplier * enginePowerScale * throttleForTorque;

            // Pumping and friction losses — this is what engine braking actually is.
            float engineOmega = engineRpm * ImprezaSpec.RpmToRadPerSec;
            float dragTorque = ImprezaSpec.EngineBrakingCoeff * engineOmega * Mathf.Lerp(1f, 0.25f, throttleInput);
            float netEngineTorque = crankTorque - dragTorque;

            engineTorqueNm = netEngineTorque;
            currentBoost = boost;

            if (!inGear)
            {
                // Free-revving against its own inertia, nothing attached.
                float neutralRpm = engineRpm + netEngineTorque / Mathf.Max(0.01f, engineInertia) * dt * ImprezaSpec.RadPerSecToRpm;
                if (throttleInput < 0.05f) neutralRpm = Mathf.Lerp(neutralRpm, idleRpm, 2.5f * dt);
                engineRpm = Mathf.Max(Mathf.Clamp(neutralRpm, ImprezaSpec.StallRpm, revLimitRpm + 120f), idleRpm);
                return 0f;
            }

            float capacity = clutchCapacityNm * clutch;

            if (clutchLocked)
            {
                // Plates home: engine and driveline are one shaft, so the engine simply
                // follows the gearing and the whole crank torque reaches the wheels.
                engineRpm = Mathf.Clamp(drivelineRpm, ImprezaSpec.StallRpm, revLimitRpm + 120f);
                if (drivelineRpm < idleRpm * 0.95f || Mathf.Abs(netEngineTorque) > capacity)
                {
                    clutchLocked = false;
                    engineRpm = Mathf.Max(engineRpm, idleRpm);
                }
                return netEngineTorque * totalRatio * drivetrainEfficiency;
            }

            // Slipping: the plates carry a friction torque capped at their capacity and
            // the ENGINE keeps its own speed. That is the whole point of a clutch — it
            // is what lets the engine sit at 4000 rpm while the wheels are still
            // crawling, which is what makes a standing start possible at all.
            float slipRpm = engineRpm - drivelineRpm;
            float transmitted = capacity * Mathf.Clamp(slipRpm / Mathf.Max(1f, clutchBlendRpm), -1f, 1f);

            // One-step implicit solve, the same trick IntegrateWheel uses: the clutch
            // torque moves as the engine speed moves, and against a 0.28 kg·m² crank
            // that feedback is far too stiff to integrate explicitly at any timestep
            // this simulation would use.
            float dClutch = Mathf.Abs(slipRpm) < clutchBlendRpm ? capacity / Mathf.Max(1f, clutchBlendRpm) : 0f;
            float beta = dt * dClutch * ImprezaSpec.RadPerSecToRpm / Mathf.Max(0.01f, engineInertia);

            float rpmNext = engineRpm + (netEngineTorque - transmitted)
                          / Mathf.Max(0.01f, engineInertia) * dt * ImprezaSpec.RadPerSecToRpm / (1f + beta);
            if (throttleInput < 0.05f && clutch < 0.5f)
                rpmNext = Mathf.Lerp(rpmNext, idleRpm, 2.5f * dt);       // idle control valve

            engineRpm = Mathf.Max(Mathf.Clamp(rpmNext, ImprezaSpec.StallRpm, revLimitRpm + 120f), idleRpm);

            // Speeds have converged and the plates can hold the load — let it bite.
            if (clutch > 0.999f && Mathf.Abs(engineRpm - drivelineRpm) < 120f
                && Mathf.Abs(netEngineTorque) < capacity && drivelineRpm > idleRpm)
                clutchLocked = true;

            return transmitted * totalRatio * drivetrainEfficiency;
        }

        float AverageDrivenWheelOmega()
        {
            float sum = 0f;
            int n = 0;
            foreach (var tire in tires)
            {
                if (tire == null || !tire.isDriveWheel) continue;
                sum += tire.angularVelocity;
                n++;
            }
            return n > 0 ? sum / n : 0f;
        }

        // ── Differentials ─────────────────────────────────────────────
        /// <summary>
        /// Splits driveline torque front/rear through the centre diff, then left/right
        /// through a plate LSD on each axle. The LSDs are the reason a wheel dangling
        /// in the air does not simply eat all the drive.
        /// </summary>
        void DistributeTorque(float drivelineTorque)
        {
            wheelTorqueNm = drivelineTorque;

            // Tell each driven wheel how much of the engine's rotating mass it is
            // dragging around through the current gear. Without this a wheel has only
            // its own 1.2 kg·m² to resist a driveline spike and the whole thing rings.
            float ratio = ImprezaSpec.GearRatios[gearIndex] * finalDrive;
            float reflected = engineInertia * ratio * ratio * Mathf.Clamp01(clutchEngage) * 0.25f;
            if (tires != null)
                foreach (var tire in tires)
                    if (tire != null) tire.drivelineInertia = tire.isDriveWheel ? reflected : 0f;

            float frontOmega = AxleOmega(tireFL, tireFR);
            float rearOmega  = AxleOmega(tireRL, tireRR);

            float frontAxle = drivelineTorque * centreSplitFront;
            float rearAxle  = drivelineTorque * (1f - centreSplitFront);

            // Viscous centre coupling: whichever axle is spinning faster hands
            // torque to the one that still has grip.
            float axleSlip = rearOmega - frontOmega;
            float maxTransfer = Mathf.Abs(drivelineTorque) * 0.45f + 60f;
            float transfer = Mathf.Clamp(axleSlip * centreDiffLock, -maxTransfer, maxTransfer);
            frontAxle += transfer;
            rearAxle  -= transfer;

            ApplyAxleTorque(tireFL, tireFR, frontAxle, lsdLockFront);
            ApplyAxleTorque(tireRL, tireRR, rearAxle,  lsdLockRear);
        }

        static float AxleOmega(PacejkaTireModel left, PacejkaTireModel right)
        {
            if (left == null && right == null) return 0f;
            if (left == null) return right.angularVelocity;
            if (right == null) return left.angularVelocity;
            return (left.angularVelocity + right.angularVelocity) * 0.5f;
        }

        void ApplyAxleTorque(PacejkaTireModel left, PacejkaTireModel right, float axleTorque, float lockRate)
        {
            if (left == null || right == null)
            {
                if (left != null) left.motorTorqueRequest = axleTorque;
                if (right != null) right.motorTorqueRequest = axleTorque;
                return;
            }

            float half = axleTorque * 0.5f;
            float omegaDiff = left.angularVelocity - right.angularVelocity;

            // Salisbury-style ramp: the plates carry a preload plus a share of
            // whatever torque is passing through them.
            float maxLock = lsdPreload + Mathf.Abs(axleTorque) * 0.45f;
            float lockTorque = Mathf.Clamp(omegaDiff * lockRate, -maxLock, maxLock);

            left.motorTorqueRequest  = Halfshaft(left,  half - lockTorque);
            right.motorTorqueRequest = Halfshaft(right, half + lockTorque);
        }

        /// <summary>
        /// Halfshaft torsional compliance. The shaft is a spring, not a rod: it winds up
        /// under torque and unwinds as the wheel catches up. That both filters torque
        /// spikes and produces the shuffle you feel on a snatchy throttle pickup. Feeding
        /// engine torque straight to the contact patch makes throttle response
        /// instantaneous in a way no real driveline is.
        ///
        /// The engine side is far heavier than the wheel, so its speed is treated as
        /// prescribed and the wheel winds against it. Damping is solved implicitly, so
        /// the spring is stable at any stiffness.
        /// </summary>
        float Halfshaft(PacejkaTireModel wheel, float demand)
        {
            if (!drivelineCompliance || wheel == null) return demand;

            float dt = Time.fixedDeltaTime;
            float k = ImprezaSpec.HalfshaftStiffness, damp = ImprezaSpec.HalfshaftDamping;

            float targetTwist = demand / k;
            float rate = (targetTwist - wheel.halfshaftTwist) / Mathf.Max(1e-4f, dt);
            wheel.halfshaftTwist += rate * dt / (1f + damp * dt / k);

            return k * wheel.halfshaftTwist + damp * rate;
        }

        // ── Suspension ────────────────────────────────────────────────
        /// <summary>
        /// Solves all four corners in two passes: measure every strut, then let every
        /// strut push.
        ///
        /// The split is not cosmetic. An anti-roll bar force is a function of the travel
        /// DIFFERENCE across an axle, so the two ends only produce an equal-and-opposite
        /// pair if they are computed from the same set of travels. Letting each corner
        /// measure-and-push in its own FixedUpdate meant the first corner of each axle
        /// sized its bar force against the other side's previous step; the leftover is
        /// proportional to the other wheel's travel speed, which is negative damping on
        /// the roll mode. It pumped the car over onto its side in about a second, from a
        /// standstill, on flat ground.
        /// </summary>
        void SolveSuspension()
        {
            if (suspensions == null) return;

            foreach (var sus in suspensions)
                if (sus != null) sus.MeasureCorner();

            foreach (var sus in suspensions)
                if (sus != null) sus.ApplyCornerForces();
        }

        // ── Brakes ────────────────────────────────────────────────────
        void ApplyBrakes()
        {
            float front = brakeInput * brakeTorqueFront;
            float rear  = brakeInput * brakeTorqueRear;
            float hand  = handbrakeInput * handbrakeTorque;

            // Hold the car still rather than letting it roll away on a slope.
            bool coasting = throttleInput < 0.02f && brakeInput < 0.02f && handbrakeInput < 0.02f;
            if (coasting && Mathf.Abs(ForwardSpeed) < 0.6f)
            {
                front = Mathf.Max(front, 900f);
                rear  = Mathf.Max(rear, 900f);
            }

            if (tireFL) tireFL.brakeTorqueRequest = front;
            if (tireFR) tireFR.brakeTorqueRequest = front;
            if (tireRL) tireRL.brakeTorqueRequest = rear + hand;
            if (tireRR) tireRR.brakeTorqueRequest = rear + hand;

            UpdateBrakeTemperatures();
        }

        /// <summary>
        /// Disc temperature and fade. Braking turns the car's kinetic energy into heat in
        /// a lump of steel with finite thermal mass; past about 500 degC the pad loses
        /// friction and the pedal goes long. On a long descent that is the difference
        /// between a stage finish and a retirement, and it is a consequence the policy
        /// should be able to feel rather than a number that never moves.
        /// </summary>
        void UpdateBrakeTemperatures()
        {
            if (!brakeFade) return;

            float dt = Time.fixedDeltaTime;
            float speed = Mathf.Abs(ForwardSpeed);

            for (int i = 0; i < 4; i++)
            {
                PacejkaTireModel wheel = i == 0 ? tireFL : i == 1 ? tireFR : i == 2 ? tireRL : tireRR;
                if (wheel == null) continue;

                bool isFront = i < 2;
                float mass = isFront ? ImprezaSpec.DiscThermalMassFront : ImprezaSpec.DiscThermalMassRear;

                // Heat in is the power this corner actually removes from the car: braking
                // force at the contact patch times road speed. Using the brake torque
                // REQUEST overstates it badly — past the grip limit the wheel just locks
                // and the surplus torque does no work.
                float power = wheel.brakeTorqueRequest > 1f ? Mathf.Abs(wheel.longitudinalForce * speed) : 0f;

                // Convective cooling rises with airflow over the disc.
                float hA = (isFront ? 14f : 10f) + 1.2f * speed;
                float cooling = hA * (wheel.discTemperatureC - ImprezaSpec.BrakeAmbientC);

                wheel.discTemperatureC = Mathf.Max(
                    ImprezaSpec.BrakeAmbientC,
                    wheel.discTemperatureC + (power - cooling) * dt / mass);

                float over = (wheel.discTemperatureC - ImprezaSpec.BrakeFadeOnsetC)
                           / (ImprezaSpec.BrakeFadeFullC - ImprezaSpec.BrakeFadeOnsetC);
                wheel.brakeFadeFactor = 1f - Mathf.Clamp01(over) * ImprezaSpec.BrakeFadeMaxLoss;
                wheel.brakeTorqueRequest *= wheel.brakeFadeFactor;
            }
        }

        // ── Aerodynamics ──────────────────────────────────────────────
        void ApplyAerodynamics(Vector3 velocity)
        {
            float speed = velocity.magnitude;
            if (speed < 0.5f) return;

            float q = 0.5f * ImprezaSpec.AirDensity * speed * speed;
            rb.AddForce(-velocity.normalized * q * dragArea, ForceMode.Force);

            // Downforce only comes from air going over the car front to back, and
            // it is applied where the bodywork actually generates it — the splitter
            // and the wing — so it trims the balance, not just the ride height.
            float vf = Vector3.Dot(velocity, transform.forward);
            if (vf <= 0.5f) return;

            float qf = 0.5f * ImprezaSpec.AirDensity * vf * vf;
            rb.AddForceAtPosition(-transform.up * qf * liftAreaFront,
                                  transform.TransformPoint(ImprezaSpec.AeroCentreFront), ForceMode.Force);
            rb.AddForceAtPosition(-transform.up * qf * liftAreaRear,
                                  transform.TransformPoint(ImprezaSpec.AeroCentreRear), ForceMode.Force);
        }

        // ── Telemetry ─────────────────────────────────────────────────
        void UpdateTelemetry(Vector3 velocity, float dt)
        {
            currentRpm = engineRpm;
            currentGear = gearIndex == ImprezaSpec.ReverseGearIndex ? -1
                        : gearIndex == ImprezaSpec.NeutralGearIndex ? 0
                        : gearIndex - ImprezaSpec.FirstGearIndex + 1;

            Vector3 localVel = transform.InverseTransformDirection(velocity);
            driftAngleDeg = velocity.magnitude > 1.5f
                ? Mathf.Atan2(localVel.x, Mathf.Abs(localVel.z)) * Mathf.Rad2Deg
                : 0f;

            Vector3 accel = (velocity - lastVelocity) / Mathf.Max(1e-5f, dt);
            lastVelocity = velocity;
            lateralG = Vector3.Dot(accel, transform.right) / 9.81f;
            longitudinalG = Vector3.Dot(accel, transform.forward) / 9.81f;

            groundedWheels = 0;
            foreach (var sus in suspensions)
                if (sus != null && sus.isGrounded) groundedWheels++;
        }

        // ══════════════════════════════════════════════════════════════
        //  EXTERNAL CONTROL
        // ══════════════════════════════════════════════════════════════

        /// <summary>Drives the car from an ML-Agents policy or a scripted driver.</summary>
        public void SetInputs(float steering, float throttle, float brake)
            => SetInputs(steering, throttle, brake, 0f);

        public void SetInputs(float steering, float throttle, float brake, float handbrake)
        {
            steeringInput  = Mathf.Clamp(steering, -1f, 1f);
            throttleInput  = Mathf.Clamp01(throttle);
            brakeInput     = Mathf.Clamp01(brake);
            handbrakeInput = Mathf.Clamp01(handbrake);
            lastExternalInputTime = Time.time;
        }

        /// <summary>
        /// Puts the whole driveline back to a cold start. Call this whenever the car
        /// is teleported — otherwise wheel speeds, gear and boost survive the reset
        /// and the first second of the new episode is garbage.
        /// </summary>
        public void ResetVehicle()
        {
            if (rb != null)
            {
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }

            engineRpm = idleRpm;
            boost = 0f;
            gearIndex = ImprezaSpec.NeutralGearIndex;
            shiftTimer = 0f;
            isShifting = false;
            clutchEngage = 0f;
            clutchLocked = false;
            currentSteerDeg = 0f;
            lastVelocity = Vector3.zero;
            steeringInput = throttleInput = brakeInput = handbrakeInput = 0f;

            if (tires != null)
                foreach (var tire in tires)
                    if (tire != null) tire.ResetTire();

            if (suspensions != null)
                foreach (var sus in suspensions)
                    if (sus != null) sus.ResetSuspension();
        }

        /// <summary>Re-reads the spec sheet into every corner. Used by the editor builder.</summary>
        public void ApplySpecToAllCorners()
        {
            CollectCorners();
            foreach (var sus in suspensions)
            {
                if (sus == null) continue;
                sus.ApplySpec();
            }
            foreach (var tire in tires)
            {
                if (tire == null) continue;
                var sus = tire.GetComponent<DynamicSuspension>();
                tire.ApplySpec(sus != null && sus.IsFront, true);
            }
        }
    }
}
