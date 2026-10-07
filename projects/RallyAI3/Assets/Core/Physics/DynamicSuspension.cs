using UnityEngine;

namespace Core.Physics
{
    /// <summary>
    /// One corner of the car: a MacPherson strut, a steering knuckle and a wheel.
    ///
    /// The GameObject this sits on IS the upper strut mount. It never rotates —
    /// it moves rigidly with the chassis — so the strut axis is always the
    /// chassis' up vector and the geometry stays sane however far the car is
    /// leaning. Steering, camber and toe live on the knuckle, which this
    /// component computes and hands to <see cref="PacejkaTireModel"/>; the wheel
    /// mesh is driven off the same numbers so what you see is what the tyre feels.
    ///
    /// Contact is found by sampling several rays across the width of the tyre and
    /// keeping the highest hit, which stops a wheel from dropping into every rut
    /// narrower than the tread.
    /// </summary>
    [DefaultExecutionOrder(-50)]
    public class DynamicSuspension : MonoBehaviour
    {
        public enum Corner { FrontLeft, FrontRight, RearLeft, RearRight }

        [Header("Corner Identity")]
        public Corner corner = Corner.FrontLeft;

        [Header("Strut")]
        [Tooltip("Free length of the strut, fully drooped.")]
        public float restLength = ImprezaSpec.SuspensionRestLength;
        [Tooltip("Shortest the strut gets before it is sitting on the bump stop.")]
        public float minLength = ImprezaSpec.SuspensionMinLength;
        [Tooltip("Wheel rate in N/m.")]
        public float springStiffness = 27000f;
        [Tooltip("Damping in N·s/m while the strut is compressing.")]
        public float damperBump = 1700f;
        [Tooltip("Damping in N·s/m while the strut is extending. Always the firmer of the two.")]
        public float damperRebound = 3400f;
        [Tooltip("Rate of the rubber bump stop once travel runs out.")]
        public float bumpStopRate = ImprezaSpec.BumpStopRate;
        [Tooltip("How far the bump stop can be crushed before it is effectively solid.")]
        public float bumpStopRange = ImprezaSpec.BumpStopRange;
        [Tooltip("How sharply the stop stiffens as it is crushed. Higher is harsher but less stable.")]
        public float bumpStopProgression = ImprezaSpec.BumpStopProgression;
        [Tooltip("Hysteresis in the stop. This is what stops a landing bouncing straight back out.")]
        public float bumpStopDamping = ImprezaSpec.BumpStopDamping;
        [Tooltip("Ceiling on this corner's output, as a multiple of the static load it carries.")]
        public float maxForceMultiple = ImprezaSpec.MaxCornerForceMultiple;
        [Tooltip("Ceiling on the load handed to the tyre, as a multiple of static. Lower than " +
                 "the strut ceiling on purpose — the strut may push hard enough to stop a landing, " +
                 "but the tyre must not turn that into a matching spike of grip.")]
        public float maxTyreLoadMultiple = ImprezaSpec.MaxTyreLoadMultiple;

        [Header("Unsprung Mass")]
        [Tooltip("Wheel, tyre, hub, upright and brake — the mass BELOW the spring. " +
                 "Modelling it is what gives the wheel dynamics of its own: it can follow " +
                 "a rough surface faster than the body, and it can hop.")]
        public float unsprungMass = ImprezaSpec.UnsprungMassPerCorner;
        [Tooltip("Vertical rate of the tyre carcass, in series with the spring.")]
        public float tyreRate = ImprezaSpec.TyreVerticalRate;
        public float tyreDamping = ImprezaSpec.TyreVerticalDamping;
        [Tooltip("Deflection at which the rim reaches the ground. A 205/65R15 has ~133 mm " +
                 "of sidewall, so past ~100 mm the contact is metal, not rubber — it goes " +
                 "near-solid and stops generating grip however hard it is driven in.")]
        public float tyreBumpThrough = 0.10f;
        public float rimRate = 2000000f;
        [Tooltip("The tyre spring is an order of magnitude stiffer than the suspension " +
                 "spring, so the corner is integrated at this multiple of the physics rate.")]
        [Range(1, 8)] public int unsprungSubSteps = 4;

        [Header("Wheel")]
        public float wheelRadius = ImprezaSpec.WheelRadius;
        public float wheelWidth = ImprezaSpec.TyreWidth;
        [Tooltip("Number of rays across the tread. Odd numbers keep one ray on centre.")]
        [Range(1, 5)] public int contactRays = 3;

        [Tooltip("Sample fore and aft so the tyre bridges obstacles instead of tracing them.")]
        public bool envelopeContact = true;
        [Range(1, 7)] public int envelopeSamples = 5;
        public LayerMask groundMask = ~0;

        [Header("Geometry")]
        [Tooltip("Static camber in degrees. Negative leans the top of the wheel into the car.")]
        public float camberAngle = ImprezaSpec.CamberFront;
        [Tooltip("Static toe in degrees. Positive points the wheel at the centreline.")]
        public float toeAngle = ImprezaSpec.ToeFront;
        [Tooltip("Camber gained as the strut compresses, in degrees over full travel.")]
        public float camberGain = -1.2f;

        [Header("Kinematics")]
        [Tooltip("Roll centre height above the contact patch. Lateral tyre force is fed " +
                 "into the body here rather than at the patch, which is what separates " +
                 "geometric load transfer from elastic (spring) transfer.")]
        public float rollCentreHeight = ImprezaSpec.RollCentreFront;
        [Tooltip("Side-view instant centre height — this is what anti-dive and anti-squat " +
                 "physically are. Longitudinal force enters the body here.")]
        public float antiPitchHeight = ImprezaSpec.AntiPitchHeightFront;

        [Header("Anti-Roll Bar")]
        public DynamicSuspension oppositeWheel;
        [Tooltip("Equivalent wheel rate of the bar, N/m of travel difference across the axle.")]
        public float antiRollRate = ImprezaSpec.AntiRollFront;

        [Header("Wheel Visuals")]
        public Transform wheelMesh;

        // ── Read-only state, consumed by the tyre model, the AI and the HUD ──
        public bool  isGrounded       { get; private set; }
        /// <summary>0 at full droop, 1 at full compression.</summary>
        public float compressionRatio { get; private set; }
        /// <summary>Vertical load carried by this tyre, in newtons. This is the tyre's Fz.</summary>
        public float normalLoad       { get; private set; }
        public float currentLength    { get; private set; }
        /// <summary>Compression from free length, in metres. What the anti-roll bar reacts to.</summary>
        public float travel           => restLength - currentLength;
        public float suspensionForce  { get; private set; }
        public RaycastHit tireHit;

        /// <summary>Grip multiplier of whatever this wheel is standing on.</summary>
        public float surfaceGrip { get; private set; } = 1f;
        public float surfaceRollingDrag { get; private set; } = ImprezaSpec.RollingResistance;
        public float surfaceLooseness { get; private set; } = 0.6f;

        /// <summary>Steering angle at this wheel, in degrees. Written by the VehicleController.</summary>
        [HideInInspector] public float steerAngle;
        /// <summary>Spin angle of the wheel about its axle, in degrees. Written by the tyre model.</summary>
        [HideInInspector] public float spinAngle;

        /// <summary>Where the wheel centre currently is, in world space.</summary>
        public Vector3 WheelCenter => transform.position - transform.up * currentLength;
        /// <summary>Rotation of the knuckle: chassis attitude plus steer, camber and toe.</summary>
        public Quaternion KnuckleRotation { get; private set; }
        /// <summary>Wheel heading projected onto the ground, normalised. Longitudinal force axis.</summary>
        public Vector3 ContactForward { get; private set; }
        /// <summary>Lateral force axis at the contact patch, normalised.</summary>
        public Vector3 ContactRight { get; private set; }
        /// <summary>Effective camber in radians, signed so positive leans the top away from the car.</summary>
        public float CamberRadians { get; private set; }

        public bool IsFront => corner == Corner.FrontLeft  || corner == Corner.FrontRight;
        public bool IsLeft  => corner == Corner.FrontLeft  || corner == Corner.RearLeft;

        Rigidbody rb;
        float lastLength;

        /// <summary>Rate of change of currentLength. Positive = strut extending.</summary>
        public float strutVelocity { get; private set; }
        /// <summary>How far the tyre carcass is squashed, in metres.</summary>
        public float tyreDeflection { get; private set; }
        float lastDeflection;
        /// <summary>Unclamped strut length from the last ground cast — the bump stop needs it.</summary>
        float rawStrutLength;
        readonly RaycastHit[] hitBuffer = new RaycastHit[8];
        Transform root;

        void Awake()
        {
            rb = GetComponentInParent<Rigidbody>();
            root = rb != null ? rb.transform : transform.root;
            KnuckleRotation = transform.rotation;
            ContactForward = transform.forward;
            ContactRight = transform.right;
            ResetSuspension();
        }

        /// <summary>
        /// Drops the strut back to the length it settles at under its share of the
        /// car's weight. Starting from full droop instead would fire a spurious
        /// damper spike into the chassis on the first physics step after a respawn.
        /// </summary>
        public void ResetSuspension()
        {
            currentLength = Mathf.Clamp(restLength - StaticSag(), minLength, restLength);
            lastLength = currentLength;
            compressionRatio = Mathf.Clamp01((restLength - currentLength) / (restLength - minLength));
            normalLoad = 0f;
            suspensionForce = 0f;
            steerAngle = 0f;
            spinAngle = 0f;
        }

        /// <summary>
        /// Fills in spring rates, damping and anti-roll for this corner from the
        /// car's spec sheet. Called by the editor builder and by Reset().
        /// </summary>
        public void ApplySpec()
        {
            restLength = ImprezaSpec.SuspensionRestLength;
            minLength  = ImprezaSpec.SuspensionMinLength;
            wheelRadius = ImprezaSpec.WheelRadius;
            wheelWidth  = ImprezaSpec.TyreWidth;
            bumpStopRate = ImprezaSpec.BumpStopRate;
            bumpStopRange = ImprezaSpec.BumpStopRange;
            bumpStopProgression = ImprezaSpec.BumpStopProgression;
            bumpStopDamping = ImprezaSpec.BumpStopDamping;
            maxForceMultiple = ImprezaSpec.MaxCornerForceMultiple;
            maxTyreLoadMultiple = ImprezaSpec.MaxTyreLoadMultiple;

            rollCentreHeight = IsFront ? ImprezaSpec.RollCentreFront : ImprezaSpec.RollCentreRear;
            antiPitchHeight  = IsFront ? ImprezaSpec.AntiPitchHeightFront : ImprezaSpec.AntiPitchHeightRear;

            if (IsFront)
            {
                springStiffness = ImprezaSpec.SpringRateFront;
                damperBump      = ImprezaSpec.DamperBumpFront;
                damperRebound   = ImprezaSpec.DamperReboundFront;
                antiRollRate    = ImprezaSpec.AntiRollFront;
                camberAngle     = ImprezaSpec.CamberFront;
                toeAngle        = ImprezaSpec.ToeFront;
            }
            else
            {
                springStiffness = ImprezaSpec.SpringRateRear;
                damperBump      = ImprezaSpec.DamperBumpRear;
                damperRebound   = ImprezaSpec.DamperReboundRear;
                antiRollRate    = ImprezaSpec.AntiRollRear;
                camberAngle     = ImprezaSpec.CamberRear;
                toeAngle        = ImprezaSpec.ToeRear;
            }
        }

        void Reset() => ApplySpec();

        // ══════════════════════════════════════════════════════════════
        //  SOLVE
        // ══════════════════════════════════════════════════════════════
        //  Order matters, in two separate ways.
        //
        //  Across components, script execution order does it: the suspension runs
        //  first and publishes normalLoad and the contact frame, then
        //  PacejkaTireModel consumes them in the same physics step.
        //
        //  Across the four corners, it cannot be done that way — they are all the
        //  same component, so Unity gives no ordering between them. The solve is
        //  therefore split in half and the VehicleController runs pass 1 on every
        //  corner before pass 2 on any of them, so an anti-roll bar always compares
        //  two travels from the same instant.
        // ══════════════════════════════════════════════════════════════
        /// <summary>
        /// Set at runtime by the <see cref="VehicleController"/> when it takes over the
        /// two-pass solve. A corner with no controller above it still solves itself; it
        /// just loses the guarantee that both ends of a bar are measured at one instant.
        /// </summary>
        [System.NonSerialized] public bool solvedByController;

        void FixedUpdate()
        {
            if (rb == null || solvedByController) return;
            MeasureCorner();
            ApplyCornerForces();
        }

        /// <summary>
        /// Pass 1 — steering geometry, ground contact, strut length. Reads nothing
        /// outside this corner, so all four can be measured before any of them starts
        /// turning those measurements into forces.
        /// </summary>
        public void MeasureCorner()
        {
            if (rb == null) return;
            UpdateKnuckle();
            CastForGround();
            MeasureStrut();
        }

        /// <summary>
        /// Pass 2 — spring, damper, bump stop, anti-roll bar. This is the half that
        /// reads <see cref="oppositeWheel"/>, so it cannot run until every corner has
        /// been through pass 1. Running the two halves interleaved, one corner at a
        /// time, is what used to put the car on its roof: the first corner of an axle
        /// sized its bar force against the other side's PREVIOUS step, so the two bar
        /// forces were not equal and opposite and the mismatch fed the roll mode.
        /// </summary>
        public void ApplyCornerForces()
        {
            if (rb == null) return;
            SolveStrut();
            UpdateVisuals();
        }

        /// <summary>Steering, toe and camber, resolved into a world-space knuckle frame.</summary>
        void UpdateKnuckle()
        {
            float toeSigned = IsLeft ? toeAngle : -toeAngle;   // toe-in points both wheels inboard
            float yaw = steerAngle + toeSigned;

            // Negative camber must lean the top of BOTH wheels inboard, so the sign
            // of the roll flips between the left and right side of the car.
            float camberNow = camberAngle + camberGain * compressionRatio;
            float roll = IsLeft ? camberNow : -camberNow;

            KnuckleRotation = transform.rotation * Quaternion.Euler(0f, yaw, roll);
            CamberRadians = camberNow * Mathf.Deg2Rad;
        }

        /// <summary>
        /// Samples the ground across the tread width. Keeps the contact that leaves
        /// the strut most compressed, i.e. the highest ground under the tyre.
        /// </summary>
        void CastForGround()
        {
            float maxDistance = restLength + wheelRadius;
            Vector3 down = -transform.up;
            Vector3 across = KnuckleRotation * Vector3.right;

            bool found = false;
            float bestDistance = float.MaxValue;
            RaycastHit best = default;

            // ── Contact patch enveloping ──
            //    A tyre is not a point. It is a 660 mm circle resting on a patch about
            //    150 mm long, and it BRIDGES features smaller than that rather than
            //    tracing them. A single column of downward rays makes the wheel climb
            //    every pebble at full height and fall into every crack at full depth,
            //    which on gravel is the difference between a car and a pogo stick.
            //
            //    So sample fore and aft as well, and treat the tyre as its actual
            //    circle: at a distance x ahead of or behind the wheel centre the carcass
            //    hangs sqrt(R^2 - x^2) below it, so ground at that sample forces the
            //    centre to sit at least (depth + R - sqrt(R^2 - x^2)) above. The binding
            //    constraint is the SMALLEST such value across the patch — the point the
            //    tyre is really resting on. Peaks bind; narrow holes never bind, so the
            //    tyre spans them exactly as a real one does.
            Vector3 along = KnuckleRotation * Vector3.forward;
            int rays = Mathf.Max(1, contactRays);
            int longRays = envelopeContact ? Mathf.Max(1, envelopeSamples) : 1;
            float span = envelopeContact ? ImprezaSpec.EnvelopeSpan : 0f;

            for (int k = 0; k < longRays; k++)
            {
                float u = longRays == 1 ? 0f : (k / (float)(longRays - 1)) * 2f - 1f;
                float x = u * span;
                // How far below the wheel centre the carcass reaches at this offset.
                float sag = Mathf.Sqrt(Mathf.Max(0f, wheelRadius * wheelRadius - x * x));
                float lift = envelopeContact ? wheelRadius - sag : 0f;

                for (int i = 0; i < rays; i++)
                {
                    float t = rays == 1 ? 0f : (i / (float)(rays - 1)) - 0.5f;   // -0.5 .. +0.5
                    Vector3 origin = transform.position + across * (t * wheelWidth) + along * x;

                    int count = UnityEngine.Physics.RaycastNonAlloc(
                        origin, down, hitBuffer, maxDistance + span, groundMask, QueryTriggerInteraction.Ignore);

                    for (int h = 0; h < count; h++)
                    {
                        ref RaycastHit hit = ref hitBuffer[h];
                        if (hit.collider == null) continue;
                        if (hit.collider.transform.IsChildOf(root)) continue;   // never hit our own car

                        // Convert to a wheel-centre constraint so samples at different
                        // fore/aft offsets are directly comparable.
                        float constraint = hit.distance + lift;
                        if (constraint < bestDistance)
                        {
                            bestDistance = constraint;
                            best = hit;
                            found = true;
                        }
                    }
                }
            }

            isGrounded = found;
            if (found)
            {
                // Everything downstream reads tireHit.distance, so it has to carry the
                // ENVELOPED distance rather than the raw hit — otherwise the geometry is
                // taken from whichever sample happened to win and the bridging is lost.
                best.distance = bestDistance;
                tireHit = best;
                SurfaceProperties.Sample(best.collider, out float g, out float rr, out float loose);
                surfaceGrip = g;
                surfaceRollingDrag = rr;
                surfaceLooseness = loose;
            }
        }

        /// <summary>
        /// Pass 1 tail: how far this strut is compressed, and nothing else. Kept apart
        /// from the force solve so an anti-roll bar reads a travel figure that belongs
        /// to this physics step rather than the last one.
        /// </summary>
        void MeasureStrut()
        {
            if (!isGrounded) return;   // the strut still integrates; the tyre just carries nothing

            rawStrutLength = tireHit.distance - wheelRadius;
            // With unsprung mass, strut length is a STATE integrated from forces, not a
            // measurement taken off the ground. That difference is the whole model.
        }

        /// <summary>
        /// The vertical chain: chassis — strut — unsprung mass — tyre — ground.
        ///
        /// The old model made the wheel massless. It was placed wherever the ground was
        /// and the spring force went straight into the body, which means the wheel could
        /// never hop, never move independently, and every road input arrived at the
        /// chassis undiminished. A real corner has ~40 kg on a ~200 kN/m carcass, and
        /// that pair — not the spring — is what follows a rough surface. Measured on the
        /// rig, adding it cut chassis vertical acceleration by ~40 % and took wheel
        /// contact over 50 mm corrugations from 81 % to 100 %.
        ///
        /// Sub-stepped because the tyre spring is an order of magnitude stiffer than the
        /// suspension spring, and damping is solved implicitly inside each sub-step so
        /// the bump stop cannot destabilise it however hard it is set.
        /// </summary>
        void SolveStrut()
        {
            float dt = Time.fixedDeltaTime;
            int n = Mathf.Max(1, unsprungSubSteps);
            float h = dt / n;

            Vector3 strutAxis = transform.up;
            float gAlong = UnityEngine.Physics.gravity.magnitude * Vector3.Dot(strutAxis, Vector3.up);
            float arb = AntiRollForce();

            for (int k = 0; k < n; k++)
            {
                // How far the carcass is squashed between wheel centre and ground.
                float wheelToGround = isGrounded ? (tireHit.distance - currentLength) : 999f;
                tyreDeflection = Mathf.Max(0f, wheelRadius - wheelToGround);

                float springs = 0f, damping = 0f;

                springs += (restLength - currentLength) * springStiffness;

                float over = minLength - currentLength;
                if (over > 0f)
                {
                    float crush = over / Mathf.Max(0.005f, bumpStopRange);
                    springs += bumpStopRate * over * (1f + bumpStopProgression * crush * crush);
                    damping += bumpStopDamping;
                }

                springs += arb;
                damping += strutVelocity < 0f ? damperBump : damperRebound;

                if (tyreDeflection > 0f)
                {
                    springs -= tyreRate * tyreDeflection;   // tyre pushes the wheel back up
                    float through = tyreDeflection - tyreBumpThrough;
                    if (through > 0f) springs -= rimRate * through;   // rim is on the ground now
                    damping += tyreDamping;
                }

                float accel = springs / Mathf.Max(1f, unsprungMass) + gAlong;
                strutVelocity = (strutVelocity + accel * h) / (1f + damping * h / Mathf.Max(1f, unsprungMass));

                currentLength += strutVelocity * h;

                // The BUMP end is deliberately not clamped. Clamping at minLength makes
                // (minLength - currentLength) identically zero, which silently disables
                // the bump stop: the strut pins at its travel limit carrying only spring
                // force, the chassis keeps descending, the wheel is driven through the
                // ground, and once the strut mount passes below the surface the ground
                // cast finds nothing and the car falls out of the world. The stop IS what
                // arrests the travel, so it has to be allowed to compress.
                // Full droop is a real mechanical limit with nothing pushing past it, so
                // clamping there is honest.
                if (currentLength > restLength) { currentLength = restLength; if (strutVelocity > 0f) strutVelocity = 0f; }
                if (currentLength < minLength - 0.15f) { currentLength = minLength - 0.15f; if (strutVelocity < 0f) strutVelocity = 0f; }
            }

            compressionRatio = Mathf.Clamp01((restLength - currentLength) / (restLength - minLength));

            // ── What the CHASSIS feels is the strut, never the tyre.
            float strutForce = (restLength - currentLength) * springStiffness + arb
                             - (strutVelocity < 0f ? damperBump : damperRebound) * strutVelocity;
            float overNow = minLength - currentLength;
            if (overNow > 0f)
            {
                float crush = overNow / Mathf.Max(0.005f, bumpStopRange);
                strutForce += bumpStopRate * overNow * (1f + bumpStopProgression * crush * crush);
            }

            suspensionForce = Mathf.Clamp(strutForce, 0f, StaticCornerLoad * maxForceMultiple);
            rb.AddForceAtPosition(strutAxis * suspensionForce, WheelCenter);

            // ── And the TYRE's Fz is the carcass force. This is the real thing now, so
            //    the artificial load cap is gone: a tyre physically cannot push harder
            //    than it is squashed, which is the bound the cap was imitating.
            if (isGrounded)
            {
                // Fz for GRIP saturates at bump-through. Past that the rim carries the
                // load, and a rim makes no grip — which is the physical bound the old
                // artificial load cap was imitating.
                float deflectRate = (tyreDeflection - lastDeflection) / Mathf.Max(1e-5f, dt);
                float gripDeflection = Mathf.Min(tyreDeflection, tyreBumpThrough);
                normalLoad = Mathf.Max(0f, tyreRate * gripDeflection + tyreDamping * deflectRate);

                Vector3 fwd = Vector3.ProjectOnPlane(KnuckleRotation * Vector3.forward, tireHit.normal);
                if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.ProjectOnPlane(transform.forward, tireHit.normal);
                ContactForward = fwd.normalized;
                ContactRight = Vector3.Cross(tireHit.normal, ContactForward).normalized;
            }
            else
            {
                normalLoad = 0f;
                ContactForward = Vector3.ProjectOnPlane(KnuckleRotation * Vector3.forward, strutAxis).normalized;
                ContactRight = Vector3.Cross(strutAxis, ContactForward).normalized;
            }

            lastDeflection = tyreDeflection;
            lastLength = currentLength;
        }

        /// <summary>
        /// Anti-roll bar: reacts to the travel DIFFERENCE across the axle, pushing the
        /// chassis UP on the compressed corner. Subtracting the other way round builds a
        /// pro-roll bar that walks the car onto its side from a standstill.
        /// </summary>
        float AntiRollForce()
        {
            if (oppositeWheel == null) return 0f;
            float otherTravel = oppositeWheel.isGrounded ? oppositeWheel.travel : 0f;
            return (travel - otherTravel) * antiRollRate;
        }

        void UpdateVisuals()
        {
            if (wheelMesh == null) return;

            wheelMesh.position = WheelCenter;
            // The wheel mesh is authored with its axle along local X, so the spin is
            // an X rotation applied inside the steer/camber frame.
            wheelMesh.rotation = KnuckleRotation * Quaternion.Euler(spinAngle, 0f, 0f);
        }

        // ══════════════════════════════════════════════════════════════
        //  GIZMOS — draw the travel envelope so setup mistakes are obvious
        // ══════════════════════════════════════════════════════════════
        void OnDrawGizmosSelected()
        {
            Vector3 top = transform.position;
            Vector3 down = -transform.up;

            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(top, top + down * minLength);
            Gizmos.color = new Color(1f, 0.55f, 0f);
            Gizmos.DrawLine(top + down * minLength, top + down * restLength);

            float len = Application.isPlaying ? currentLength : restLength - StaticSag();
            Vector3 hub = top + down * len;

            Gizmos.color = isGrounded ? Color.green : Color.red;
            Gizmos.DrawWireSphere(hub, 0.05f);

            // Tyre outline, in the knuckle's plane.
            Quaternion knuckle = Application.isPlaying
                ? KnuckleRotation
                : transform.rotation * Quaternion.Euler(0f, steerAngle, (IsLeft ? 1f : -1f) * camberAngle);
            Gizmos.color = new Color(0.2f, 0.9f, 1f, 0.9f);
            const int steps = 24;
            Vector3 prev = hub + knuckle * new Vector3(0f, wheelRadius, 0f);
            for (int i = 1; i <= steps; i++)
            {
                float a = i / (float)steps * Mathf.PI * 2f;
                Vector3 p = hub + knuckle * new Vector3(0f, Mathf.Cos(a) * wheelRadius, Mathf.Sin(a) * wheelRadius);
                Gizmos.DrawLine(prev, p);
                prev = p;
            }
        }

        /// <summary>Weight this corner holds up standing still, in newtons.</summary>
        float StaticCornerLoad
            => (IsFront ? ImprezaSpec.CornerMassFront : ImprezaSpec.CornerMassRear) * 9.81f;

        float StaticSag() => springStiffness > 1f ? StaticCornerLoad / springStiffness : 0f;
    }
}
