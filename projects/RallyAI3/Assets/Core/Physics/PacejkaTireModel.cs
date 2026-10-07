using UnityEngine;

namespace Core.Physics
{
    /// <summary>
    /// A single tyre, solved with the Pacejka Magic Formula.
    ///
    /// What it actually does each physics step:
    ///   1. reads the vertical load and the contact frame the strut published,
    ///   2. works out slip ratio and slip angle at the contact patch, passed
    ///      through a relaxation-length filter so the forces build up over
    ///      distance rolled rather than instantly,
    ///   3. evaluates the Magic Formula separately for longitudinal and lateral,
    ///      then combines them on a friction ellipse — a tyre that is already
    ///      spinning up under power has very little grip left to steer with,
    ///   4. integrates the wheel's own rotation from drive torque, brake torque
    ///      and the road's reaction, and
    ///   5. pushes the resulting force into the chassis at the contact point.
    ///
    /// Slip angles are in RADIANS throughout, which is why the B coefficients look
    /// small next to the textbook degree-based ones.
    /// </summary>
    [RequireComponent(typeof(DynamicSuspension))]
    [DefaultExecutionOrder(0)]
    public class PacejkaTireModel : MonoBehaviour
    {
        [Header("Wheel Configuration")]
        public bool isDriveWheel = true;
        public bool isSteeringWheel = false;

        [Header("Magic Formula — Lateral")]
        public float Blat = ImprezaSpec.TyreBLat;
        public float Clat = ImprezaSpec.TyreCLat;
        public float Elat = ImprezaSpec.TyreELat;

        [Header("Magic Formula — Longitudinal")]
        public float Blong = ImprezaSpec.TyreBLong;
        public float Clong = ImprezaSpec.TyreCLong;
        public float Elong = ImprezaSpec.TyreELong;

        [Header("Friction")]
        [Tooltip("Peak friction coefficient at the reference load, on a dry surface.")]
        public float peakFriction = ImprezaSpec.PeakFriction;

        [Tooltip("Global grip scale. WeatherSystem writes rain/snow multipliers here.")]
        public float D = 1.0f;

        [Tooltip("How fast grip falls away as the tyre is loaded past its reference load.")]
        public float loadSensitivity = ImprezaSpec.LoadSensitivity;

        [Tooltip("Load this corner carries standing still, used to normalise load sensitivity.")]
        public float referenceLoad = 3016f;

        [Tooltip("Distance the tyre must roll to build its full slip force.")]
        public float relaxationLength = ImprezaSpec.RelaxationLength;

        [Header("Tyre Condition")]
        [Range(0f, 1f)] public float tireWearPercent = 0f;
        [Tooltip("Wear accrued per megajoule of energy dumped into sliding.")]
        public float wearRate = 0.020f;
        [Tooltip("Grip left on a completely worn tyre.")]
        [Range(0.2f, 1f)] public float wornGripFloor = 0.55f;

        public bool enableThermalModel = true;
        public float ambientTempC = 12f;
        public float optimalTempC = 85f;
        [Tooltip("Grip lost when the tyre is stone cold or badly overheated.")]
        [Range(0f, 0.5f)] public float thermalGripSwing = 0.14f;

        [Header("Solver")]
        [Tooltip("Below this speed the tyre switches to a static-friction model so the car does not creep or buzz.")]
        public float lowSpeedThreshold = 2.0f;
        [Tooltip("Rotational inertia of wheel + tyre + brake disc.")]
        public float wheelInertia = ImprezaSpec.WheelInertia;

        // ── Torque requests, written by the VehicleController each step ──
        [HideInInspector] public float motorTorqueRequest;
        [HideInInspector] public float brakeTorqueRequest;
        [HideInInspector] public float steerAngleRequest;

        /// <summary>
        /// Engine and gearbox inertia reflected through the gearing to this wheel, written
        /// by the VehicleController. A driven wheel is bolted to the engine through the
        /// ratio, so it carries the crank's inertia multiplied by that ratio SQUARED — in
        /// first gear 0.28 * 11.1² = 34.6 kg·m² against the wheel's own 1.2. Leaving it out
        /// lets a driveline torque spike spin a wheel to an absurd speed in one step, and
        /// the whole driveline rings.
        /// </summary>
        [HideInInspector] public float drivelineInertia;

        /// <summary>What this wheel's rotation actually has to accelerate.</summary>
        public float EffectiveInertia => wheelInertia + Mathf.Max(0f, drivelineInertia);

        // ── Telemetry ──
        public float lateralSlipAngle { get; private set; }
        public float longitudinalSlipRatio { get; private set; }
        public float angularVelocity { get; private set; }          // rad/s, + is forward
        public float longitudinalForce { get; private set; }        // N, + drives the car forward
        public float lateralForce { get; private set; }             // N, + pushes the car right
        public float normalLoad => suspension != null ? suspension.normalLoad : 0f;
        public float currentGripMultiplier { get; private set; } = 1f;
        public float temperatureC { get; private set; } = 12f;
        /// <summary>0 = well within the envelope, 1 = at or past the limit of grip.</summary>
        public float slipUtilisation { get; private set; }
        /// <summary>Speed of the contact patch sliding over the ground, m/s. Drives dust and skid audio.</summary>
        public float slipSpeed { get; private set; }
        public bool isGrounded => suspension != null && suspension.isGrounded;

        Rigidbody rb;
        DynamicSuspension suspension;

        // Slip states, filtered by relaxation length.
        float slipAngleState;
        float slipRatioState;

        // Peak-slip locations of the current coefficient set, cached for the
        // combined-slip normalisation.
        float peakSlipRatio = 0.20f;
        float peakSlipAngle = 0.18f;

        /// <summary>
        /// Torque this tyre feeds back up the steering, N.m. Steering FEEL: it includes
        /// mechanical trail, which acts about the steering axis and never on the chassis.
        /// Meaningful on the steered axle only. Read this for force feedback.
        /// </summary>
        public float selfAligningTorque { get; private set; }

        /// <summary>Current pneumatic trail, m. Falls to zero as the tyre saturates.</summary>
        public float pneumaticTrail { get; private set; }

        /// <summary>Halfshaft wind-up, rad. Written by the VehicleController's compliance model.</summary>
        [HideInInspector] public float halfshaftTwist;

        /// <summary>Brake disc temperature, degC, and the torque multiplier it causes.</summary>
        [HideInInspector] public float discTemperatureC = ImprezaSpec.BrakeAmbientC;
        [HideInInspector] public float brakeFadeFactor = 1f;

        void Awake()
        {
            rb = GetComponentInParent<Rigidbody>();
            suspension = GetComponent<DynamicSuspension>();
            temperatureC = ambientTempC;
            RecomputePeaks();
        }

        void OnValidate() => RecomputePeaks();

        /// <summary>
        /// Locates the peak of each Magic Formula curve numerically. Those two
        /// numbers are what the combined-slip normalisation divides by, so they
        /// have to track whatever coefficients are dialled in.
        /// </summary>
        void RecomputePeaks()
        {
            peakSlipRatio = FindPeak(Blong, Clong, Elong, 1.5f);
            peakSlipAngle = FindPeak(Blat, Clat, Elat, 0.6f);
        }

        static float FindPeak(float b, float c, float e, float searchRange)
        {
            float best = 0f, bestVal = 0f;
            const int steps = 200;
            for (int i = 1; i <= steps; i++)
            {
                float s = searchRange * i / steps;
                float v = MagicFormula(s, b, c, e);
                if (v > bestVal) { bestVal = v; best = s; }
            }
            return Mathf.Max(best, 0.02f);
        }

        /// <summary>Applies the spec-sheet setup for this corner.</summary>
        public void ApplySpec(bool front, bool driven)
        {
            isDriveWheel = driven;
            isSteeringWheel = front;
            peakFriction = ImprezaSpec.PeakFriction;
            loadSensitivity = ImprezaSpec.LoadSensitivity;
            relaxationLength = ImprezaSpec.RelaxationLength;
            wheelInertia = ImprezaSpec.WheelInertia;
            referenceLoad = ImprezaSpec.Mass * 9.81f *
                            (front ? ImprezaSpec.FrontWeightBias : 1f - ImprezaSpec.FrontWeightBias) * 0.5f;
            Blat = ImprezaSpec.TyreBLat;   Clat = ImprezaSpec.TyreCLat;   Elat = ImprezaSpec.TyreELat;
            Blong = ImprezaSpec.TyreBLong; Clong = ImprezaSpec.TyreCLong; Elong = ImprezaSpec.TyreELong;
            RecomputePeaks();
        }

        void Reset()
        {
            var sus = GetComponent<DynamicSuspension>();
            if (sus != null) ApplySpec(sus.IsFront, true);
        }

        // ══════════════════════════════════════════════════════════════
        //  SOLVE
        // ══════════════════════════════════════════════════════════════
        void FixedUpdate()
        {
            if (rb == null || suspension == null) return;
            float dt = Time.fixedDeltaTime;

            if (!suspension.isGrounded)
            {
                SolveAirborne(dt);
                return;
            }

            Vector3 contactPoint  = suspension.tireHit.point;
            Vector3 contactNormal = suspension.tireHit.normal;
            Vector3 fwd = suspension.ContactForward;
            Vector3 right = suspension.ContactRight;
            float radius = suspension.wheelRadius;

            // ── Contact patch velocity, split into the wheel's own axes ──
            Vector3 patchVel = rb.GetPointVelocity(contactPoint);
            float vx = Vector3.Dot(patchVel, fwd);      // rolling direction
            float vy = Vector3.Dot(patchVel, right);    // sideways
            float speed = new Vector2(vx, vy).magnitude;

            // ── Slip ratio and slip angle at this instant ──
            float vRef = Mathf.Max(Mathf.Abs(vx), 1.0f);   // keeps the ratio finite at a standstill
            float slipRatioTarget = (angularVelocity * radius - vx) / vRef;
            slipRatioTarget = Mathf.Clamp(slipRatioTarget, -4f, 4f);
            float slipAngleTarget = Mathf.Atan2(-vy, Mathf.Max(Mathf.Abs(vx), 0.6f));

            // ── Relaxation: a tyre is a spring, its force builds over distance
            //    rolled, not over time. This is what stops the lateral force
            //    ringing at high frame rates and near zero speed.
            float rollRate = (Mathf.Abs(vx) + 0.7f) * dt / Mathf.Max(0.05f, relaxationLength);
            float lagLat = Mathf.Clamp01(rollRate);
            float lagLong = Mathf.Clamp01(rollRate * 2.2f);   // longitudinal relaxes ~2x quicker
            slipAngleState = Mathf.Lerp(slipAngleState, slipAngleTarget, lagLat);
            slipRatioState = Mathf.Lerp(slipRatioState, slipRatioTarget, lagLong);

            lateralSlipAngle = slipAngleState;
            longitudinalSlipRatio = slipRatioState;

            // ── Available friction at this load ──
            float Fz = suspension.normalLoad;
            float mu = EffectiveFriction(Fz);
            float muFz = mu * Fz;

            // ── Magic Formula, each axis on its own, then combined ──
            float kappaN = slipRatioState / peakSlipRatio;
            float alphaN = slipAngleState / peakSlipAngle;
            float combined = Mathf.Sqrt(kappaN * kappaN + alphaN * alphaN);
            slipUtilisation = Mathf.Clamp01(combined);

            float Fx0 = muFz * MagicFormulaSigned(slipRatioState, Blong, Clong, Elong);
            float Fy0 = muFz * MagicFormulaSigned(slipAngleState, Blat, Clat, Elat);

            float Fx, Fy;
            if (combined > 1e-4f)
            {
                // Friction-ellipse split: whichever axis is doing more of the
                // slipping gets more of the budget, and the total never exceeds
                // what a single axis alone could produce.
                Fx = Fx0 * Mathf.Abs(kappaN) / combined;
                Fy = Fy0 * Mathf.Abs(alphaN) / combined;
            }
            else
            {
                Fx = Fx0;
                Fy = Fy0;
            }

            // ── Camber thrust: a leaned-over tyre pulls toward the lean.
            //    Derived from the geometry so the sign is right on both sides.
            float lean = Vector3.Dot(suspension.KnuckleRotation * Vector3.up, right);
            Fy += ImprezaSpec.CamberStiffness * lean * Fz;

            // ── At walking pace the Magic Formula is meaningless; swap in a
            //    static-friction model that cancels the residual contact velocity
            //    outright, clamped to what the tyre could actually hold.
            float stickBlend = 0f, stickRate = 0f;
            if (speed < lowSpeedThreshold)
            {
                float blend = 1f - speed / lowSpeedThreshold;
                float effectiveMass = Fz / 9.81f;                  // mass this corner carries
                stickBlend = blend;
                stickRate = effectiveMass / dt;

                // Longitudinally the tyre resists SLIP — the tread sliding across the
                // road — not the car's own motion. Opposing vx instead pins the chassis
                // to the ground with a ~17 kN per m/s spring, and no amount of drive
                // torque can pull away from it: the balance point is about 3 cm/s, so
                // the car simply sits there with the engine revving.
                float slipVx = angularVelocity * radius - vx;
                float stickX = slipVx * effectiveMass / dt;
                float stickY = -vy * effectiveMass / dt;

                float stickMag = new Vector2(stickX, stickY).magnitude;
                if (stickMag > muFz && stickMag > 1e-4f)
                {
                    float k = muFz / stickMag;
                    stickX *= k; stickY *= k;
                }

                Fx = Mathf.Lerp(Fx, stickX, blend);
                Fy = Mathf.Lerp(Fy, stickY, blend);
            }

            // ── Rolling resistance always opposes the direction of travel ──
            if (Mathf.Abs(vx) > 0.1f)
                Fx -= Mathf.Sign(vx) * suspension.surfaceRollingDrag * Fz;

            longitudinalForce = Fx;
            lateralForce = Fy;

            // ── Push it into the chassis, flattened onto the contact plane so a tyre
            //    can never generate force into or out of the ground.
            //
            //    Lateral and longitudinal enter the body at DIFFERENT heights, because a
            //    suspension link transmits load along the line to its instant centre, not
            //    from the contact patch. Lateral goes in at the roll centre; longitudinal
            //    at the side-view instant centre. That is what anti-dive, anti-squat and
            //    geometric load transfer physically are. Feeding both in at the patch
            //    routes every newton through the springs, so the car rolls and dives more
            //    than it should and always a spring-lag late.
            Vector3 up = suspension.transform.up;
            Vector3 lateral      = Vector3.ProjectOnPlane(right * Fy, contactNormal);
            Vector3 longitudinal = Vector3.ProjectOnPlane(fwd * Fx, contactNormal);
            rb.AddForceAtPosition(lateral,      contactPoint + up * suspension.rollCentreHeight);
            rb.AddForceAtPosition(longitudinal, contactPoint + up * suspension.antiPitchHeight);

            // ── Self-aligning moment ──
            //    Lateral force does not act at the centre of the contact patch. It acts a
            //    little BEHIND it, by the pneumatic trail, and that offset is what tries to
            //    straighten a steered wheel.
            //
            //    The part that matters is that the trail COLLAPSES as the tyre approaches
            //    its limit: the patch goes from mostly gripping to mostly sliding and the
            //    force centroid walks forward. Torque therefore peaks BEFORE peak lateral
            //    force and falls through it, which is why the steering goes light just
            //    before the front lets go. Full trail at zero slip, zero at the friction
            //    peak, slightly negative past it.
            float saturation  = Mathf.Clamp(Mathf.Abs(slipAngleState) / peakSlipAngle, 0f, 1.3f);
            pneumaticTrail    = ImprezaSpec.PneumaticTrail * Mathf.Cos(saturation * Mathf.PI * 0.5f);

            // The chassis feels the offset as a yaw moment about the contact normal.
            rb.AddTorque(contactNormal * (-pneumaticTrail * Fy));

            // Steering feel carries the MECHANICAL trail from caster as well, which reacts
            // through the steering rack rather than on the chassis, so it is reported only.
            selfAligningTorque = -Fy * (pneumaticTrail + ImprezaSpec.MechanicalTrail);

            // ── Wheel rotation ──
            //    The road reaction is fed back semi-implicitly. Without this the
            //    wheel/tyre pair is a stiff spring even at the project's 100 Hz
            //    timestep and the whole car buzzes; with it, the wheel settles
            //    instead of ringing, and the
            //    damping fades out past the friction peak so wheelspin is still free.
            float slopeStep = 0.01f;
            float dMF = (MagicFormulaSigned(slipRatioState + slopeStep, Blong, Clong, Elong)
                       - MagicFormulaSigned(slipRatioState - slopeStep, Blong, Clong, Elong)) / (2f * slopeStep);
            float combinedShare = combined > 1e-4f ? Mathf.Abs(kappaN) / combined : 1f;
            float dFx_dOmega = Mathf.Max(0f, muFz * dMF * combinedShare) * radius / vRef;

            // The stick model is a far stiffer function of wheel speed than the Magic
            // Formula is, so its slope has to go into the implicit term as well or the
            // wheel rings itself apart the moment the tyre starts sticking.
            if (stickBlend > 0f)
                dFx_dOmega = Mathf.LerpUnclamped(dFx_dOmega, stickRate * radius, stickBlend);

            IntegrateWheel(dt, Fx, radius, vx, dFx_dOmega);

            // ── Wear and heat ──
            slipSpeed = new Vector2(angularVelocity * radius - vx, vy).magnitude;
            UpdateCondition(dt, Mathf.Abs(Fx * (angularVelocity * radius - vx)) + Mathf.Abs(Fy * vy));

            suspension.spinAngle += angularVelocity * Mathf.Rad2Deg * dt;
        }

        void SolveAirborne(float dt)
        {
            // Free-spinning wheel: engine torque still spins it up, and bearing
            // drag plus any brake pressure slows it down.
            float netTorque = motorTorqueRequest - Mathf.Sign(angularVelocity) * 4f;
            angularVelocity += netTorque / EffectiveInertia * dt;
            ApplyBrake(dt, brakeTorqueRequest);

            longitudinalSlipRatio = 0f;
            lateralSlipAngle = 0f;
            longitudinalForce = 0f;
            lateralForce = 0f;
            slipUtilisation = 0f;
            slipSpeed = 0f;
            slipAngleState = Mathf.Lerp(slipAngleState, 0f, 8f * dt);
            slipRatioState = Mathf.Lerp(slipRatioState, 0f, 8f * dt);

            if (enableThermalModel)
                temperatureC = Mathf.Lerp(temperatureC, ambientTempC, 0.15f * dt);

            if (suspension != null)
                suspension.spinAngle += angularVelocity * Mathf.Rad2Deg * dt;
        }

        /// <summary>
        /// Integrates the wheel's spin. The road's reaction torque is -Fx*R, which
        /// is what makes a wheel bog down under load and spin up when it lets go.
        /// </summary>
        void IntegrateWheel(float dt, float Fx, float radius, float vx, float dFx_dOmega)
        {
            float roadTorque = -Fx * radius;
            float inertia = EffectiveInertia;
            // beta is how much the road torque itself would change if the wheel
            // sped up this step. Dividing the increment by (1 + beta) is a
            // one-step implicit solve of the wheel's equation of motion — it
            // leaves the equilibrium untouched and only kills the overshoot.
            float beta = dt * radius * dFx_dOmega / inertia;
            angularVelocity += (motorTorqueRequest + roadTorque) / inertia * dt / (1f + beta);

            ApplyBrake(dt, brakeTorqueRequest);

            // Numerical guard: without drive or brake torque a wheel cannot end up
            // spinning wildly faster than the ground it is rolling on.
            if (Mathf.Approximately(motorTorqueRequest, 0f) && brakeTorqueRequest < 1f)
            {
                float freeRolling = vx / Mathf.Max(0.05f, radius);
                angularVelocity = Mathf.Lerp(angularVelocity, freeRolling, Mathf.Clamp01(6f * dt));
            }
        }

        /// <summary>
        /// Brakes bleed off wheel speed but can never drive it past zero — that is
        /// the difference between a locked wheel and a wheel spinning backwards.
        /// </summary>
        void ApplyBrake(float dt, float brakeTorque)
        {
            if (brakeTorque <= 0f) return;

            float inertia = EffectiveInertia;
            float stoppingTorque = Mathf.Abs(angularVelocity) * inertia / dt;
            float applied = Mathf.Min(brakeTorque, stoppingTorque);
            angularVelocity -= Mathf.Sign(angularVelocity) * applied / inertia * dt;

            if (Mathf.Abs(angularVelocity) < 0.01f) angularVelocity = 0f;
        }

        /// <summary>
        /// Peak friction after everything that spoils it: the surface, the weather,
        /// how hard the tyre is being squashed, how worn it is and how hot it is.
        /// </summary>
        float EffectiveFriction(float Fz)
        {
            // Load sensitivity — a tyre carrying twice its reference load does NOT
            // make twice the grip. This single term is what produces load transfer
            // understeer and why lifting mid-corner rotates the car.
            float loadRatio = Fz / Mathf.Max(1f, referenceLoad);
            float loadFactor = Mathf.Clamp(1f - loadSensitivity * (loadRatio - 1f), 0.55f, 1.35f);

            float wearFactor = Mathf.Lerp(1f, wornGripFloor, tireWearPercent);

            float thermalFactor = 1f;
            if (enableThermalModel)
            {
                // A shallow parabola around the optimal window: cold and cooked
                // both cost grip, the middle 30 degrees barely matter.
                float t = (temperatureC - optimalTempC) / 60f;
                thermalFactor = 1f - thermalGripSwing * Mathf.Clamp01(t * t);
            }

            currentGripMultiplier = loadFactor * wearFactor * thermalFactor * suspension.surfaceGrip * D;
            return peakFriction * currentGripMultiplier;
        }

        void UpdateCondition(float dt, float slipPowerWatts)
        {
            slipPowerWatts = Mathf.Abs(slipPowerWatts);

            if (enableThermalModel)
            {
                // Heat in from sliding, heat out to the air — faster the quicker you go.
                float speed = rb.linearVelocity.magnitude;
                float heating = slipPowerWatts * 1.1e-3f;
                float cooling = (temperatureC - ambientTempC) * (0.06f + speed * 0.010f);
                temperatureC = Mathf.Clamp(temperatureC + (heating - cooling) * dt, -20f, 220f);
            }

            // Wear tracks energy dissipated, not time — a tyre only dies when it slides.
            tireWearPercent = Mathf.Clamp01(tireWearPercent + slipPowerWatts * 1e-6f * wearRate * dt);
        }

        /// <summary>Resets the tyre to a fresh, cold state. Called on episode reset.</summary>
        public void ResetTire()
        {
            angularVelocity = 0f;
            slipAngleState = 0f;
            slipRatioState = 0f;
            lateralSlipAngle = 0f;
            longitudinalSlipRatio = 0f;
            longitudinalForce = 0f;
            lateralForce = 0f;
            slipUtilisation = 0f;
            slipSpeed = 0f;
            tireWearPercent = 0f;
            temperatureC = ambientTempC;
            motorTorqueRequest = 0f;
            brakeTorqueRequest = 0f;
        }

        /// <summary>Forces the wheel speed, used when the driveline locks the axle together.</summary>
        public void SetAngularVelocity(float omega) => angularVelocity = omega;

        // ══════════════════════════════════════════════════════════════
        //  THE MAGIC FORMULA
        //      F(s) = sin( C * atan( B*s - E*(B*s - atan(B*s)) ) )
        //  Normalised so it peaks at 1.0; the caller multiplies by mu*Fz.
        // ══════════════════════════════════════════════════════════════
        static float MagicFormulaSigned(float slip, float b, float c, float e)
        {
            float bs = b * slip;
            return Mathf.Sin(c * Mathf.Atan(bs - e * (bs - Mathf.Atan(bs))));
        }

        static float MagicFormula(float slip, float b, float c, float e)
            => Mathf.Abs(MagicFormulaSigned(slip, b, c, e));
    }
}
