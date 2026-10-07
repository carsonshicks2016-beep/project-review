using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Core.Physics;
using Core.Environment;

namespace Core.ML
{
    [RequireComponent(typeof(VehicleController))]
    public class RallyAgent : Agent
    {
        private VehicleController vehicle;
        private Rigidbody rb;
        private TrackGenerator track;
        private WeatherSystem weather;
        private PacejkaTireModel[] tires;
        private DynamicSuspension[] struts;
        private RayPerceptionSensorComponent3D[] raySensors;

        /// <summary>Each fan's configured range, kept so weather can scale it without losing it.</summary>
        private float[] rayBaseLengths;

        /// <summary>
        /// Exactly four tyre slots, FL/FR/RL/RR, so <see cref="CollectObservations"/> always
        /// writes the same number of floats. Looping over whatever GetComponentsInChildren
        /// happened to find makes the observation vector's LENGTH depend on the hierarchy,
        /// and ML-Agents throws the moment it stops matching Behaviour Parameters.
        /// </summary>
        private readonly PacejkaTireModel[] tyreSlots = new PacejkaTireModel[4];

        /// <summary>
        /// Size of the vector observation this agent emits. Behaviour Parameters must be
        /// set to exactly this. Kept next to CollectObservations so the two stay in step.
        ///     1  forward speed
        ///     3  velocity direction, car-local
        ///    12  four tyres x (slip ratio, slip angle, wear)
        ///     2  chassis attitude (up-dot, roll)
        ///     3  direction to the next waypoint, car-local
        ///     1  distance to it
        ///     1  signed lateral offset from the road centreline
        ///     1  heading error against the road tangent
        ///     4  how far the road turns over the next 10 / 25 / 50 / 100 m
        ///
        /// The last six are the road itself, and they were missing entirely. Of the original
        /// 22, twelve described the tyres and NONE described the road — the only road-related
        /// input was a bearing to a centreline point up to 100 m away. The ray sensor could
        /// not fill the gap either: its rays lie in a horizontal plane at 0.70 m, and a road
        /// edge is a change of surface rather than a wall, so there is nothing there to hit.
        ///
        /// A policy given no way to perceive the road can only memorise one, which is what
        /// rally05 did — 85 % stage completion on a fixed seed, and 0 waypoints once anything
        /// moved.
        /// </summary>
        public const int VectorObservationSize = 28;

        /// <summary>
        /// Continuous actions this agent expects: [0] steering, [1] drive.
        ///
        /// Drive is ONE axis — positive is throttle, negative is brake — rather than a
        /// separate throttle and a separate brake. With two independent pedals an
        /// untrained Gaussian policy presses both at once, and the brakes out-torque the
        /// engine so heavily at low throttle that the car simply sits there. Measured on
        /// the physics rig: 8-25 m covered in 120 s with separate pedals, 32-48 m with
        /// one axis. Zero reward means zero gradient, so a car that cannot move cannot
        /// learn to move.
        /// </summary>
        public const int ContinuousActionCount = 2;

        public Transform nextWaypoint;
        private int currentWaypointIndex = 0;

        [Header("Spawning")]
        [Tooltip("How far above the road surface the car is dropped. The rig's origin IS " +
                 "its ground plane, so this is literally the gap under the tyres. Keep it small: " +
                 "a 0.30 m drop is right on the limit of what the struts can absorb, and anything " +
                 "that bottoms them out fires the bump stops at their force ceiling.")]
        public float spawnHeight = 0.05f;

        // ══════════════════════════════════════════════════════════════
        //  REWARD
        //
        //  The ordering these have to produce, best to worst:
        //      finish the stage  >  get a long way down it  >  crash  >  sit still
        //
        //  The last pair is the one that went wrong. With a -5 terminal penalty and no
        //  cost for time, doing NOTHING scored 0 while trying scored a risk of -5, so
        //  the policy correctly learned to park: timeouts went to 100 %, progress stuck
        //  at 4 % of the stage.
        //
        //  Note what actually punishes a crash. It is not the explicit penalty, it is
        //  the FORFEITED REST OF THE STAGE — ending at 300 m of a course worth ~110
        //  throws away ~70. That is already large and automatic, so the explicit penalty
        //  only needs to break ties, not to dominate.
        // ══════════════════════════════════════════════════════════════
        [Header("Reward")]
        [Tooltip("Cost per second of merely existing. This is what makes timing out worse " +
                 "than crashing, and therefore what stops 'park and wait' being optimal. " +
                 "0.05/s over the 120 s episode is -6, against 95 for a finished stage.")]
        public float timePenaltyPerSecond = 0.05f;

        [Tooltip("Paid per waypoint reached. Seventeen of them, so this is the dominant " +
                 "term on a completed stage and progress outranks caution.")]
        public float waypointBonus = 5f;

        [Tooltip("Paid once for reaching the end of the stage.")]
        public float finishBonus = 10f;

        [Tooltip("Charged for a crash, a rollover, or leaving the stage. Deliberately " +
                 "small — the real cost is the stage the car no longer gets to finish.")]
        public float failurePenalty = 1f;

        [Header("Episode Termination")]
        [Tooltip("Impact severity that counts as a crash, measured as change in velocity (m/s). " +
                 "Collision impulse scales with mass, so a raw newton-second threshold means " +
                 "something completely different on a 1230 kg car than on a default rigidbody.")]
        public float crashDeltaV = 4f;

        [Tooltip("Collisions and rollovers are ignored for this long after a respawn, so simply " +
                 "settling onto the track can never end the episode.")]
        public float respawnGraceSeconds = 1.0f;

        [Tooltip("Metres from the road centreline before the car counts as having left the " +
                 "stage. The road is 12 m wide, so 18 m leaves about 12 m of shoulder and " +
                 "verge to run wide onto — enough that a fast line clipping the dirt survives, " +
                 "while the deep scenery stops being somewhere the car can live. This is a " +
                 "constraint, not a penalty: nothing rewards staying on the road, because the " +
                 "surface grip and the trees already price that correctly.")]
        public float offStageDistance = 18f;

        // ══════════════════════════════════════════════════════════════
        //  STALL
        //
        //  A parked car still generates a full episode of experience — 12000 steps of
        //  identical, uninformative samples. Measured on a real run: 745k steps bought only
        //  124 episodes, and just 3 of those involved the car moving at all. Ending a dead
        //  episode early is worth roughly an order of magnitude in episode diversity per
        //  step, which matters far more than the wall clock does.
        // ══════════════════════════════════════════════════════════════
        [Header("Stall")]
        [Tooltip("Speed below which the car counts as not moving, m/s.")]
        public float stallSpeedThreshold = 1.0f;

        [Tooltip("Seconds below that speed before the episode is abandoned. 0 disables it.")]
        public float stallTimeoutSeconds = 5f;

        /// <summary>Last time the car was actually moving. Reset on every episode.</summary>
        private float lastMotionTime;

        [Tooltip("Backstop only. With terrain under everything the car has nothing to fall " +
                 "off, so this should almost never fire — but the landform can drop well " +
                 "below the road, so keep it far below the lowest ground.")]
        public float fallOutY = -60f;

        [Tooltip("Log the reason each episode ended to the Console. Handy while setting the " +
                 "scene up; turn it off before a long training run.")]
        public bool logEpisodeEnds = true;

        [Tooltip("Append one JSON line per episode to results/episodes/. Unlike the Console " +
                 "log this is meant to be left ON during long headless runs — it is the only " +
                 "per-episode record such a run produces, and it carries the stage seed so a " +
                 "bad episode can be reproduced exactly.")]
        public bool logEpisodesToFile = true;

        // ══════════════════════════════════════════════════════════════
        //  STAGE VARIETY
        //
        //  TrackGenerator.GenerateTrack runs ONCE, from its own Awake. OnEpisodeBegin only
        //  calls it if there are no waypoints at all. So TrackGenerator.randomiseSeed alone
        //  buys almost nothing for training: it draws one random stage per PROCESS, which
        //  with --num-envs 6 is six stages for a ten-million-step run. Six stages are as
        //  memorisable as one.
        //
        //  Rebuilding the stage every few episodes is what actually forces a policy to read
        //  the road rather than recall it. It is not free — the rebuild re-cooks two
        //  MeshColliders, and the finer tessellation the affine-mapping fix introduced made
        //  the terrain mesh ~51k vertices — so it is amortised over a block of episodes
        //  instead of paying it every reset.
        // ══════════════════════════════════════════════════════════════
        [Header("Stage Variety")]
        [Tooltip("Draw a brand-new stage every N episodes. 0 disables it and the stage is " +
                 "fixed for the whole run, which is only what you want while debugging.")]
        public int stageRefreshEpisodes = 15;

        [Tooltip("Name of the ML-Agents environment parameter that drives rock density, so " +
                 "a YAML curriculum can ramp obstacles in as the policy gets competent. " +
                 "Ignored when no trainer is attached.")]
        public string obstacleDensityParameter = "obstacle_density";

        private int episodesSinceStageChange;

        private float episodeStartTime;

        /// <summary>
        /// Our own running total for the episode.
        ///
        /// GetCumulativeReward() cannot be used for this: ML-Agents zeroes it inside
        /// EndEpisode(), which runs BEFORE OnEpisodeBegin(), which is where the episode
        /// is booked. Reading it there always returned 0.0 — which is exactly what the
        /// monitor was displaying while TensorBoard showed the reward climbing past 25.
        /// </summary>
        private float episodeReward;

        /// <summary>Adds reward and keeps our own tally of it.</summary>
        private void Reward(float amount)
        {
            episodeReward += amount;
            AddReward(amount);
        }

        // ══════════════════════════════════════════════════════════════
        //  FAILURE FORENSICS
        //
        //  Two failure modes have resisted everything tried against them: obstacle strikes
        //  stuck at 45-53 % across three runs and a sensor upgrade that tripled detection
        //  range, and rollovers pinned at 15-20 % across every run and every difficulty.
        //
        //  Neither has ever been measured. The log recorded one word — "HitObstacle",
        //  "RolledOver" — and every competing explanation produces that same word. Was the
        //  rock avoidable, or was the stage blocked? Was it even a road rock, or a tree the
        //  car only met because it had already run wide? Was the rollover too much speed
        //  for the corner, a bad landing, or the centre of mass?
        //
        //  What follows is the smallest set of measurements that separates those. It is
        //  gathered continuously at PoseHz and read out once, at the moment the episode
        //  ends. Nothing here changes the reward or the observations — the policy cannot
        //  see any of it and does not train on it.
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// Rate the forensic snapshot and the pose trail are sampled at. Deliberately not the
        /// physics rate: at 100 Hz this would call SampleRoadFrame a hundred times a second
        /// for a number only read when an episode ends, and 50 ms of staleness is invisible
        /// against a rollover that takes the better part of a second.
        /// </summary>
        const float PoseHz = 20f;

        /// <summary>
        /// How much pose history is kept, in samples. At <see cref="PoseHz"/> this is 6.4 s,
        /// comfortably longer than the approach to anything the 60 m forward fan can see —
        /// 2.4 s at 25 m/s — so a warning time is almost never clipped by the buffer.
        /// </summary>
        const int PoseSamples = 128;

        /// <summary>
        /// Above this up-dot the car counts as level, and the forensic snapshot keeps
        /// updating. 0.7 is about 45 degrees of lean: past that it is going over, and what
        /// it does on the way is a consequence rather than a cause.
        /// </summary>
        const float LevelUprightDot = 0.7f;

        /// <summary>
        /// Stage either side of a strike that counts as "the same place" when measuring the
        /// gap the car had. Roughly a car length plus the distance covered while committed
        /// to a line; a rock 20 m further on is a separate decision.
        /// </summary>
        const float GapWindowMetres = 6f;

        /// <summary>
        /// What TrackGenerator names a rock on the driveable surface. Scenery is "Tree" or
        /// "Boulder"; all three carry the Obstacle tag, so the name is the only thing that
        /// separates a rock in the racing line from a tree reached by leaving the road.
        /// </summary>
        const string RoadRockName = "RoadRock";

        private struct Pose
        {
            public float time;
            public Vector3 position;
            public Vector3 forward;
        }

        private readonly Pose[] poseTrail = new Pose[PoseSamples];
        private int poseWrites;
        private float nextSampleTime;

        /// <summary>
        /// The episode's state as of the last moment the car was level. For every outcome
        /// but a rollover this is the current frame; for a rollover it is the frame that
        /// explains it.
        /// </summary>
        private EpisodeRecord snapshot;

        private int wheelsGrounded;
        private float airborneSeconds;
        private float longestFlight;
        private float peakSpeed;
        private float lastSteer, lastDrive;
        private int labTicks, labAttempt;
        private bool labInvalid;
        private string circuitInvalidReason = "";
        // Training-only random starting point along an imported circuit. Such an episode is
        // never a ranked attempt, but it is not "invalid driving" either: time-attack rewards
        // must still pay a clean finish, so it is tracked apart from labInvalid.
        private bool distributedStart;
        private float spawnStation = -1f;
        private int startGate = -1;
        private System.Random spawnStations;
        private Vector3 labPrevious;
        private Vector3 labStartGate;
        private int labStartTick;
        private bool labStartCrossed;
        private bool episodeStarted;
        private readonly System.Collections.Generic.List<float> labSplits = new System.Collections.Generic.List<float>();
        private System.IO.StreamWriter labTrajectory;

        /// <summary>What the episode ended on, filled in by OnCollisionEnter. Empty otherwise.</summary>
        private string hitWhat;
        private bool hitRoadRock;
        private float rockLateral, gapWidth, gapCentre, warningSeconds;

        /// <summary>Road-relative offset of whatever was struck, metres, signed right.</summary>
        private float struckLateral;
        private float struckDistance;
        private float struckStation;

        /// <summary>Clears everything the forensics carry, so no reading survives into the next episode.</summary>
        private void ResetForensics()
        {
            snapshot = default;
            poseWrites = 0;
            nextSampleTime = 0f;
            wheelsGrounded = 0;
            airborneSeconds = 0f;
            longestFlight = 0f;
            peakSpeed = 0f;
            lastSteer = lastDrive = 0f;
            hitWhat = "";
            hitRoadRock = false;
            rockLateral = gapWidth = gapCentre = struckLateral = 0f;
            struckDistance = struckStation = 0f;
            warningSeconds = 0f;
        }

        void FixedUpdate()
        {
            if(LabRuntime.CircuitInspection)return;
            // FixedUpdate is a Unity message and does not wait for Agent.Initialize.
            if (rb == null) return;
            if (LabRuntime.Enabled)
            {
                labTicks++;
                if (LabRuntime.Config.controlProbe && LabRuntime.Config.controlProbeSeconds > 0f
                    && Time.time - episodeStartTime >= LabRuntime.Config.controlProbeSeconds)
                {
                    Fail("control probe time limit reached", EpisodeOutcome.TimedOut);
                    return;
                }
                if (!labStartCrossed && (LabRuntime.Config.evaluation || LabRuntime.Config.viewer))
                    LabCrossedStart();
                if (track != null)
                {
                    var frame = track.SampleRoadFrame(transform.position, currentWaypointIndex);
                    if (track.importedCircuit)
                    {
                        // Footprint corners, not just the car center. Once invalid, always invalid.
                        foreach(float x in new[]{-Core.Physics.ImprezaSpec.Width*.5f,Core.Physics.ImprezaSpec.Width*.5f})
                        foreach(float z in new[]{Core.Physics.ImprezaSpec.TailZ,Core.Physics.ImprezaSpec.NoseZ})
                        {
                            var corner=track.SampleRoadFrame(transform.TransformPoint(new Vector3(x,0,z)),currentWaypointIndex);
                            if(!corner.valid||Mathf.Abs(corner.signedOffset)>track.LegalHalfWidthAt(corner.distanceAlong,corner.signedOffset))
                            {labInvalid=true;circuitInvalidReason="vehicle footprint left asphalt/kerb envelope";}
                        }
                    }
                    else if (frame.valid && Mathf.Abs(frame.signedOffset) > track.roadWidth * 0.5f + 1f) labInvalid = true;
                }
                if (labTrajectory != null && labTicks % 5 == 0)
                {
                    var sample = new LabSample { tick = labTicks, position = transform.position,
                        rotation = transform.rotation, speed = rb.linearVelocity.magnitude, steer = lastSteer, drive = lastDrive };
                    labTrajectory.WriteLine(JsonUtility.ToJson(sample));
                }
            }

            // First, because an episode that should have ended must not spend another
            // physics step accumulating reward or forensics.
            CheckTerminalConditions();

            // Contact and speed are cheap and want the full physics rate: a flight of three
            // physics steps is still a flight, and the peak speed of an episode is exactly
            // the sort of thing a 20 Hz sampler misses.
            int grounded = 0;
            if (struts != null)
                foreach (var strut in struts)
                    if (strut != null && strut.isGrounded) grounded++;
            wheelsGrounded = grounded;

            float dt = Time.fixedDeltaTime;
            if (grounded == 0)
            {
                airborneSeconds += dt;
                if (airborneSeconds > longestFlight) longestFlight = airborneSeconds;
            }
            else airborneSeconds = 0f;

            float speed = rb.linearVelocity.magnitude;
            if (speed > peakSpeed) peakSpeed = speed;

            if (Time.time < nextSampleTime) return;
            nextSampleTime = Time.time + 1f / PoseHz;

            poseTrail[poseWrites % PoseSamples] = new Pose
            {
                time = Time.time,
                position = transform.position,
                forward = transform.forward
            };
            poseWrites++;

            if (Vector3.Dot(transform.up, Vector3.up) >= LevelUprightDot) CaptureSnapshot();
        }

        /// <summary>
        /// Reads the road-relative state into <see cref="snapshot"/>. Only called while the
        /// car is level, which is what makes the recorded speed the speed it CARRIED IN
        /// rather than whatever is left after a roll has scrubbed most of it off.
        /// </summary>
        private void CaptureSnapshot()
        {
            snapshot.speed = vehicle.ForwardSpeed;
            snapshot.uprightDot = Vector3.Dot(transform.up, Vector3.up);
            snapshot.wheelsGrounded = wheelsGrounded;
            snapshot.airborneSeconds = airborneSeconds;

            if (track == null) return;

            TrackGenerator.RoadFrame frame = track.SampleRoadFrame(transform.position, currentWaypointIndex);
            if (!frame.valid) return;

            snapshot.station = frame.distanceAlong;
            snapshot.lateralOffset = frame.signedOffset;
            // 25 m of lookahead, matching the second of the observed curvatures: far enough
            // that a corner is already committed to, near enough to still be this corner.
            snapshot.curvature = track.RoadTurnAhead(frame.distanceAlong, 25f);

            Vector3 carFwd = track.transform.InverseTransformDirection(transform.forward);
            carFwd = new Vector3(carFwd.x, 0f, carFwd.z);
            snapshot.headingError = carFwd.sqrMagnitude < 1e-8f
                ? 0f
                : SignedAngleRad(frame.tangent, carFwd.normalized);
        }

        /// <summary>
        /// How long an object had been inside a ray fan before the car reached it, in
        /// seconds — the warning the policy actually got.
        ///
        /// Walks the pose trail oldest-first and returns the age of the first sample from
        /// which the object was within some fan's range and half-angle. Occlusion is
        /// ignored, so this is the OPTIMISTIC figure: the real warning can only be shorter.
        /// That is the useful direction — a small number here is proof the policy could not
        /// have reacted, while a large one leaves it no excuse.
        ///
        /// Returns -1 when the object was never in any fan, which is the signature of
        /// hitting something while sideways, backwards, or already off the road.
        /// </summary>
        private float WarningSecondsFor(Vector3 worldPoint)
        {
            if (raySensors == null || raySensors.Length == 0 || poseWrites == 0) return -1f;

            int available = Mathf.Min(poseWrites, PoseSamples);
            int oldest = poseWrites - available;

            for (int n = oldest; n < poseWrites; n++)
            {
                Pose pose = poseTrail[n % PoseSamples];
                Vector3 toTarget = worldPoint - pose.position;
                float range = toTarget.magnitude;
                if (range < 1e-3f) continue;

                Vector3 direction = toTarget / range;

                for (int s = 0; s < raySensors.Length; s++)
                {
                    var sensor = raySensors[s];
                    if (sensor == null) continue;
                    // The fan's CURRENT range, not the one it was built with — weather
                    // shortens it, and a warning time has to describe the sensor the policy
                    // actually had.
                    if (range > sensor.RayLength) continue;

                    // The fan is horizontal, so only the horizontal angle decides whether a
                    // ray could have reached it.
                    Vector3 flatFwd = new Vector3(pose.forward.x, 0f, pose.forward.z);
                    Vector3 flatDir = new Vector3(direction.x, 0f, direction.z);
                    if (flatFwd.sqrMagnitude < 1e-8f || flatDir.sqrMagnitude < 1e-8f) continue;

                    float degrees = Vector3.Angle(flatFwd.normalized, flatDir.normalized);
                    if (degrees > sensor.MaxRayDegrees) continue;

                    return Time.time - pose.time;
                }
            }
            return -1f;
        }

        /// <summary>Live progress, read by the training monitor.</summary>
        public int WaypointsReached => track != null
            ? Mathf.Max(0, currentWaypointIndex - StartGate) : 0;
        public int WaypointTarget => track != null
            ? Mathf.Max(1, track.FinishWaypointIndex - StartGate+(track.importedCircuit?1:0)) : 1;
        // Gates skipped by a mid-circuit start are not driven progress.
        int StartGate => startGate >= 0 ? startGate : track.spawnWaypointIndex + 1;

        public override void Initialize()
        {
            vehicle = GetComponent<VehicleController>();
            rb = GetComponent<Rigidbody>();
            track = FindAnyObjectByType<TrackGenerator>();
            weather = FindAnyObjectByType<WeatherSystem>();
            tires = GetComponentsInChildren<PacejkaTireModel>();
            struts = GetComponentsInChildren<DynamicSuspension>();

            // In CHILDREN. The fans live on Sensor_Forward and Sensor_Wide so the eye point
            // can move without dragging the car's origin around, and there are two of them
            // since the ray-spacing fix. A GetComponent on the car root — which is what the
            // weather hook used to do — finds neither.
            raySensors = GetComponentsInChildren<RayPerceptionSensorComponent3D>();
            rayBaseLengths = new float[raySensors.Length];
            for (int i = 0; i < raySensors.Length; i++)
                rayBaseLengths[i] = raySensors[i] != null ? raySensors[i].RayLength : 0f;

            // Bind the four corners to fixed observation slots, by corner identity rather
            // than by whatever order the hierarchy happens to be in.
            foreach (var tire in tires)
            {
                var sus = tire.GetComponent<DynamicSuspension>();
                if (sus == null) continue;
                switch (sus.corner)
                {
                    case DynamicSuspension.Corner.FrontLeft:  tyreSlots[0] = tire; break;
                    case DynamicSuspension.Corner.FrontRight: tyreSlots[1] = tire; break;
                    case DynamicSuspension.Corner.RearLeft:   tyreSlots[2] = tire; break;
                    case DynamicSuspension.Corner.RearRight:  tyreSlots[3] = tire; break;
                }
            }

            if (nextWaypoint == null)
            {
                nextWaypoint = new GameObject("NextWaypointTarget").transform;
            }
            
            if (weather != null)
            {
                weather.OnWeatherChanged += HandleWeatherChanged;

                // And apply what the weather ALREADY is. WeatherSystem raises the event from
                // its own Start, which has long since run by the time an agent initialises,
                // so a scene that opens at night would have subscribed to an event that had
                // already fired and kept full daylight vision until the weather next changed.
                HandleWeatherChanged(weather.currentWeather);
            }

            ReadEvaluationOverrides();
            LabRuntime.Configure(this);

            // Whatever the stage was actually built with, so the logged density is never a
            // claim about a curriculum that has not run yet. It read 0 on every episode
            // before the first stage refresh, on stages that plainly had rocks in them.
            if (track != null) currentObstacleDensity = track.obstaclesPer100m;

            ReportEpisodeClock();
        }

        /// <summary>Said once per process, however many agents there are.</summary>
        private static bool reportedClock;

        /// <summary>
        /// Prints how long an episode actually lasts, in seconds.
        ///
        /// This exists because that number was wrong for the whole project's life and
        /// nothing said so. Every comment in the codebase asserted a 50 Hz fixed timestep;
        /// the project has always run at 100 Hz. MaxStep counts physics steps, so the
        /// episode the code described as 120 s was 60 s — half a stage's worth of clock,
        /// silently, on a task where "ran out of time" and "drove badly" look identical
        /// from the outside.
        ///
        /// A derived quantity that matters this much should be printed, not commented.
        /// </summary>
        private void ReportEpisodeClock()
        {
            if (reportedClock) return;
            reportedClock = true;

            float dt = Time.fixedDeltaTime;
            float episodeSeconds = MaxStep * dt;
            var requester = GetComponent<DecisionRequester>();
            float decisionHz = requester != null && requester.DecisionPeriod > 0
                ? 1f / (dt * requester.DecisionPeriod) : 0f;

            Debug.Log($"[RallyAgent] Physics {1f / dt:0} Hz, decisions {decisionHz:0} Hz, " +
                      $"episode {MaxStep} steps = {episodeSeconds:0.#} s.", this);

            if (episodeSeconds < 100f)
                Debug.LogWarning(
                    $"[RallyAgent] An episode is only {episodeSeconds:0.#} s. A 1 km stage " +
                    "needs about 120 s at a sane pace, so a careful driver will be timing out " +
                    "on the clock rather than on its driving. Raise MaxStep.", this);
        }

        void OnDestroy()
        {
            if (weather != null)
            {
                weather.OnWeatherChanged -= HandleWeatherChanged;
            }
        }

        /// <summary>
        /// Shortens the ray fans at night and restores them otherwise.
        ///
        /// This did nothing at all until now, in two separate ways. It called GetComponent
        /// on the car, but both fans are on CHILD nodes, so it always found null and
        /// returned — night vision was a setting that silently changed nothing. And the
        /// length it wrote was a hardcoded 50 m, which is neither fan's range: the forward
        /// fan is 60 m and the wide one 45 m, so had it ever found a sensor it would have
        /// LENGTHENED the wide fan by 5 m on the way into the dark.
        ///
        /// Each fan is now scaled from its own configured range, captured once in
        /// Initialize, so this is idempotent — repeated calls cannot ratchet a fan down.
        /// </summary>
        private void HandleWeatherChanged(WeatherSystem.WeatherCondition condition)
        {
            if (raySensors == null) return;

            float scale = condition == WeatherSystem.WeatherCondition.Night && weather != null
                ? Mathf.Clamp01(weather.nightVisionTruncation)
                : 1f;

            for (int i = 0; i < raySensors.Length; i++)
            {
                if (raySensors[i] == null) continue;
                raySensors[i].RayLength = rayBaseLengths[i] * scale;
            }
        }

        void StartAtCircuitStation(float station,float lateral,bool reverse,ref Vector3 spawn,ref Vector3 forward)
        {
            spawn=track.transform.TransformPoint(track.CircuitSurfacePoint(station,lateral));
            forward=track.transform.TransformDirection(track.CircuitTangent(station));
            if(reverse)forward=-forward;
            while(currentWaypointIndex<track.FinishWaypointIndex&&track.CircuitGateStation(currentWaypointIndex)<station)
                currentWaypointIndex++;
        }

        public override void OnEpisodeBegin()
        {
            // Book the episode that just ended. Every ending funnels through here,
            // including a MaxStep timeout, which the agent is never told about directly.
            RecordEpisode();
            if (LabRuntime.ViewerPaused) return;
            if (LabRuntime.Enabled)
            {
                labTrajectory?.Dispose(); labTrajectory = null;
                labTicks = 0; labInvalid = false; circuitInvalidReason=""; labSplits.Clear(); labAttempt++;
                distributedStart = false; spawnStation = -1f; startGate = -1;
                if (LabRuntime.Config.evaluation || LabRuntime.Config.viewer)
                {
                    UnityEngine.Random.InitState(unchecked(LabRuntime.Config.seed + labAttempt * 7919));
                    string path = System.IO.Path.Combine(LabRuntime.Config.output, "trajectory-" + labAttempt + ".jsonl");
                    labTrajectory = new System.IO.StreamWriter(path);
                }
            }

            // Reset position, rotation, velocity — and the driveline with it, so the
            // new episode does not inherit last episode's gear, revs and wheel speeds.
            vehicle.ResetVehicle();
            if (LabRuntime.Enabled && LabRuntime.Config.startingGear == "first")
                vehicle.PrepareFirstGearStandingStart();

            episodeStartTime = Time.time;
            episodeStarted = true;
            lastMotionTime = Time.time;      // the stall clock starts from the spawn
            currentWaypointIndex = 0;
            ResetForensics();

            if (track == null) return;

            if (track.waypoints.Count == 0)
            {
                track.GenerateTrack(); // Force generation if we beat TrackGenerator's Awake
            }
            else if (LabRuntime.Enabled && LabRuntime.Config.mode == "generalist" && episodesSinceStageChange == 0 && !track.frozenCourse)
            {
                track.seed = LabRuntime.NextSeed();
                track.randomiseSeed = false;
                track.GenerateTrack();
                episodesSinceStageChange = 1;
            }
            else if (stageRefreshEpisodes > 0 && ++episodesSinceStageChange >= stageRefreshEpisodes)
            {
                episodesSinceStageChange = 0;
                DrawNewStage();
            }

            currentWaypointIndex = track.spawnWaypointIndex + 1;

            if (!track.TryGetSpawn(out Vector3 spawn, out Vector3 forward)) return;
            string reviewStation=System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_REVIEW_STATION");
            if(track.importedCircuit && float.TryParse(reviewStation,System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture,out float rs))
            {
                float lateral=0;float.TryParse(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_PROBE_LATERAL"),System.Globalization.NumberStyles.Float,System.Globalization.CultureInfo.InvariantCulture,out lateral);
                if(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_PROBE_SWEEP")=="1")lateral=0;
                StartAtCircuitStation(rs,lateral,System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_PROBE_REVERSE")=="1",ref spawn,ref forward);
                labInvalid=true;circuitInvalidReason="diagnostic station start; not a ranked attempt";
            }
            else if(track.importedCircuit && LabRuntime.Enabled && LabRuntime.Config.spawnProfile=="distributed"
                    && !LabRuntime.Config.evaluation && !LabRuntime.Config.viewer)
            {
                // A fixed T13 start on a 20 km lap means an early policy only ever sees the
                // first kilometre before it hits a barrier. Spread starts over the whole route,
                // leaving room to cross the finish from the latest one.
                if(spawnStations==null)spawnStations=new System.Random(LabRuntime.NextSeed());
                float last=Mathf.Max(track.circuitStart,track.circuitEnd-300f);
                float station=track.circuitStart+(float)spawnStations.NextDouble()*(last-track.circuitStart);
                StartAtCircuitStation(station,0f,false,ref spawn,ref forward);
                distributedStart=true;spawnStation=station;
            }
            startGate=currentWaypointIndex;

            // The waypoint is the spline CENTRELINE, not the road surface — the mesh is
            // banked around it by up to camberScale/2 degrees. Dropping the car in level
            // lands it on two wheels on a slope, and the corner that touches first takes
            // the whole car's landing on its own. Find the surface and sit on it flat.
            Vector3 up = Vector3.up;
            if (RoadUnder(spawn, out RaycastHit surface))
            {
                spawn = surface.point;
                up = surface.normal;
            }

            forward = Vector3.ProjectOnPlane(forward, up).normalized;
            if (forward.sqrMagnitude < 1e-6f) forward = Vector3.forward;

            labStartCrossed = !LabRuntime.Enabled || (!LabRuntime.Config.evaluation && !LabRuntime.Config.viewer);
            labStartTick = 0;
            if (!labStartCrossed)
            {
                labStartGate = spawn;
                spawn -= forward * 3f;
            }

            rb.position = spawn + up * spawnHeight;
            rb.rotation = Quaternion.LookRotation(forward, up);
            if (track.importedCircuit)
            {
                // Synchronize the new chassis and wheel ray origins before the next
                // decision/contact step, then discard state from the previous pose.
                transform.SetPositionAndRotation(rb.position, rb.rotation);
                UnityEngine.Physics.SyncTransforms();
                vehicle.ResetVehicle();
                if (LabRuntime.Enabled && LabRuntime.Config.startingGear == "first")
                    vehicle.PrepareFirstGearStandingStart();
            }
            labPrevious = rb.position;
            nextWaypoint.position = track.transform.TransformPoint(track.waypoints[currentWaypointIndex]);
        }

        /// <summary>
        /// Finds the road under a point, ignoring the car itself — which is still sitting
        /// wherever the last episode ended it, quite possibly right here.
        /// </summary>
        private bool RoadUnder(Vector3 point, out RaycastHit surface)
        {
            surface = default;
            var hits = UnityEngine.Physics.RaycastAll(
                point + Vector3.up * 10f, Vector3.down, 30f, ~0, QueryTriggerInteraction.Ignore);

            float best = float.MaxValue;
            bool found = false;
            foreach (var hit in hits)
            {
                if (hit.collider.transform.IsChildOf(transform)) continue;
                if(track!=null&&track.importedCircuit&&!hit.collider.name.StartsWith("Road_")&&!hit.collider.name.StartsWith("Kerb_")&&!hit.collider.name.StartsWith("Ground"))continue;
                if (hit.distance >= best) continue;
                best = hit.distance;
                surface = hit;
                found = true;
            }
            return found;
        }

        // ══════════════════════════════════════════════════════════════
        //  EVALUATION MODE
        //
        //  Every episode logs the seed of the stage it drove, which makes a failure
        //  reproducible in principle. In practice it was not: nothing could ask the game to
        //  drive a PARTICULAR stage, so "replay the stages where it died and check whether
        //  the rock was avoidable" meant editing the scene by hand, once per seed.
        //
        //  These four environment variables turn that into a command. Nothing reads them
        //  unless they are set, so training behaves exactly as before.
        //
        //      RALLY_EVAL_SEEDS=a,b,c   drive these stages in turn, instead of random ones
        //      RALLY_EVAL_REFRESH=n     episodes per stage before moving to the next seed
        //      RALLY_EVAL_EPISODES=n    quit the process after n episodes
        //      RALLY_EVAL_ROCKS=x       force rock density, overriding the curriculum
        //
        //  See evaluate.sh, which drives a built player with these set.
        // ══════════════════════════════════════════════════════════════

        private int[] evalSeeds;
        private int evalSeedCursor;
        private int evalEpisodeLimit;
        private int evalEpisodesDone;
        private float evalRockDensity = -1f;
        private bool inEvaluation;

        /// <summary>True when ANY evaluation override is set, not only a seed list.</summary>
        private bool InEvaluation => inEvaluation;

        private bool HasSeedList => evalSeeds != null && evalSeeds.Length > 0;

        /// <summary>
        /// Reads the evaluation overrides. Called from Initialize, before the first episode
        /// begins, so a forced seed is in place for episode one rather than from episode two.
        /// </summary>
        private void ReadEvaluationOverrides()
        {
            string seeds = ReadVar("RALLY_EVAL_SEEDS");
            if (!string.IsNullOrEmpty(seeds))
            {
                var parsed = new System.Collections.Generic.List<int>();
                foreach (string piece in seeds.Split(','))
                    if (int.TryParse(piece.Trim(), out int value)) parsed.Add(value);
                evalSeeds = parsed.ToArray();
            }

            if (int.TryParse(ReadVar("RALLY_EVAL_EPISODES"), out int limit) && limit > 0)
                evalEpisodeLimit = limit;

            if (float.TryParse(ReadVar("RALLY_EVAL_ROCKS"),
                               System.Globalization.NumberStyles.Float,
                               System.Globalization.CultureInfo.InvariantCulture,
                               out float rocks) && rocks >= 0f)
                evalRockDensity = rocks;

            // ANY of the three means this process is measuring rather than training.
            // Keying it on the seed list alone left `evaluate.sh --episodes 60 --rocks 2`
            // running the scene's own stage sixty times over at the scene's own difficulty:
            // both knobs silently ignored, and sixty samples of one course — the exact
            // memorisation trap the stage refresh exists to break.
            inEvaluation = HasSeedList || evalEpisodeLimit > 0 || evalRockDensity >= 0f;
            if (!inEvaluation) return;

            // A fresh stage every episode by default. An evaluation wants each episode
            // labelled with the stage it actually drove, and a measurement taken on one
            // course says nothing about a policy meant to generalise.
            stageRefreshEpisodes = 1;
            if (int.TryParse(ReadVar("RALLY_EVAL_REFRESH"), out int refresh) && refresh > 0)
                stageRefreshEpisodes = refresh;

            // ── Run faster than the wall clock.
            //
            //    Nothing sets the time scale when there is no trainer attached, so an
            //    evaluation ran at 1x: 120 episodes averaging 21 s of simulated time took
            //    three quarters of an hour of real time, which is enough friction that the
            //    measurement does not get taken. mlagents-learn sets 20x during training
            //    for exactly this reason.
            //
            //    This is safe to change ONLY because the terminal checks moved to
            //    FixedUpdate. While they lived in Update they were frame-rate coupled, and
            //    raising the time scale would have quietly changed how quickly a rollover
            //    was noticed — turning a speed knob into a physics knob.
            float scale = 20f;
            if (float.TryParse(ReadVar("RALLY_EVAL_TIMESCALE"),
                               System.Globalization.NumberStyles.Float,
                               System.Globalization.CultureInfo.InvariantCulture,
                               out float requested) && requested > 0f)
                scale = requested;
            Time.timeScale = scale;

            Debug.Log($"[RallyAgent] Evaluation mode: " +
                      $"{(HasSeedList ? evalSeeds.Length + " named seed(s)" : "random seeds")}, " +
                      $"{stageRefreshEpisodes} episode(s) per stage, " +
                      $"{(evalEpisodeLimit > 0 ? evalEpisodeLimit + " episodes then quit" : "no episode limit")}, " +
                      $"{(evalRockDensity >= 0f ? evalRockDensity + " rocks/100 m" : "scene rock density")}, " +
                      $"time scale {Time.timeScale:0.#}x.", this);
        }

        private static string ReadVar(string name)
        {
            // Fully qualified: Core.Environment shadows System.Environment in this namespace.
            try { return System.Environment.GetEnvironmentVariable(name); }
            catch { return null; }
        }

        /// <summary>
        /// Counts the episode against the evaluation budget and shuts the process down once
        /// it is spent. Called after the episode has been written, never before — the last
        /// episode of a run is not worth less than the others.
        /// </summary>
        private void CheckEvaluationBudget()
        {
            if (LabRuntime.Enabled && LabRuntime.Config.viewer)
            {
                int viewerBudget = Mathf.Max(1, LabRuntime.Config.attempts);
                if (++evalEpisodesDone < viewerBudget) return;

                labTrajectory?.Flush();
                labTrajectory?.Dispose();
                labTrajectory = null;
                Debug.Log($"[RallyAgent] Viewer attempt complete: {evalEpisodesDone}/{viewerBudget}. " +
                          "The final pose is frozen for inspection.", this);
                LabRuntime.PauseViewer();
                return;
            }

            if (evalEpisodeLimit <= 0) return;
            if (++evalEpisodesDone < evalEpisodeLimit) return;

            Debug.Log($"[RallyAgent] Evaluation complete: {evalEpisodesDone} episodes. " +
                      $"Episodes written to {EpisodeLog.FilePath}.", this);
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }

        /// <summary>
        /// Draws a brand-new stage: applies the obstacle curriculum, picks a fresh seed, and
        /// rebuilds the track.
        ///
        /// The seed is set EXPLICITLY rather than by switching on
        /// <c>TrackGenerator.randomiseSeed</c>, so the scene asset on disk keeps its fixed
        /// seed and stays reproducible and diffable — only the training run varies.
        /// GenerateTrack ends by calling Random.InitState(seed), so the sequence of stages a
        /// run sees is a deterministic chain from whatever seed the scene was saved with. The
        /// run is still reproducible; it just is not a single stage any more.
        /// </summary>
        private void DrawNewStage()
        {
            if (LabRuntime.Enabled && track.frozenCourse) return;
            ApplyObstacleCurriculum();
            track.seed = LabRuntime.Enabled ? LabRuntime.NextSeed() : HasSeedList
                ? evalSeeds[evalSeedCursor++ % evalSeeds.Length]
                : UnityEngine.Random.Range(int.MinValue, int.MaxValue);
            track.GenerateTrack();
        }

        /// <summary>
        /// Pulls rock density from the trainer's environment parameters so a YAML curriculum
        /// can ramp obstacles in as the policy earns them.
        ///
        /// Obstacles are part of the task from the first step rather than something bolted on
        /// at the end — a rally stage without them is a different and much smaller problem.
        /// What ramps is the DENSITY, not the feature, because a policy that cannot steer yet
        /// hits a rock within seconds of every episode and never strings together enough clean
        /// driving to learn cornering from.
        ///
        /// Note this governs rocks ON THE ROAD only. The stage's scenery — 259 trees and 77
        /// boulders on this seed, all tagged Obstacle and all episode-ending — is controlled
        /// by TrackGenerator.scatterProps and is present at every lesson, including the first.
        ///
        /// With no trainer attached, this leaves whatever the scene was built with.
        /// </summary>
        private void ApplyObstacleCurriculum()
        {
            if (track == null) return;

            // An evaluation fixes the difficulty on purpose: comparing two policies is
            // meaningless if the stage under them is being ramped by a curriculum.
            if (evalRockDensity >= 0f)
            {
                track.obstaclesPer100m = evalRockDensity;
                track.spawnObstacles = evalRockDensity > 0f;
                currentObstacleDensity = evalRockDensity;
                return;
            }

            if (string.IsNullOrEmpty(obstacleDensityParameter)) return;
            if (!Academy.IsInitialized) return;

            float density = Academy.Instance.EnvironmentParameters
                                   .GetWithDefault(obstacleDensityParameter, track.obstaclesPer100m);

            track.obstaclesPer100m = Mathf.Max(0f, density);
            track.spawnObstacles = track.obstaclesPer100m > 0f;
            currentObstacleDensity = track.obstaclesPer100m;
        }

        /// <summary>Density the current stage was built with, republished each episode so the
        /// curriculum's progress is visible in TensorBoard next to the outcomes it changes.</summary>
        private float currentObstacleDensity;

        /// <summary>
        /// Writes exactly <see cref="VectorObservationSize"/> floats, always, in the same
        /// order. Everything directional is expressed in the CAR's frame — a world-space
        /// heading tells the policy nothing it can steer on without first working out
        /// which way the car is pointing.
        /// </summary>
        public override void CollectObservations(VectorSensor sensor)
        {
            // ── Motion, in the car's own frame (4)
            sensor.AddObservation(vehicle.ForwardSpeed / 55f);                       // 1, ~200 km/h full scale
            Vector3 localVel = transform.InverseTransformDirection(rb.linearVelocity);
            sensor.AddObservation(localVel.sqrMagnitude > 1e-4f ? localVel.normalized : Vector3.zero); // 3

            // ── Tyre state (12). Fixed slots: a missing corner still writes its zeros.
            foreach (var tire in tyreSlots)
            {
                if (tire == null) { sensor.AddObservation(0f); sensor.AddObservation(0f); sensor.AddObservation(0f); continue; }
                sensor.AddObservation(Mathf.Clamp(tire.longitudinalSlipRatio, -1f, 1f));
                sensor.AddObservation(Mathf.Clamp(tire.lateralSlipAngle, -1f, 1f));
                sensor.AddObservation(tire.tireWearPercent);
            }

            // ── Attitude (2). Dot products, not Euler angles: localEulerAngles wraps
            //    0..360, so a car a degree either side of level reads 0.003 or 0.997 and
            //    the policy has to learn the discontinuity for nothing.
            sensor.AddObservation(Vector3.Dot(transform.up, Vector3.up));            // 1, 1 = level, 0 = on its side
            sensor.AddObservation(Vector3.Dot(transform.right, Vector3.up));         // 1, signed roll

            // ── Where to go next (4), also car-local.
            if (track != null && track.waypoints.Count > currentWaypointIndex)
            {
                // Waypoints are stored in the track's local space; the car is in world space.
                Vector3 toNext = track.transform.TransformPoint(track.waypoints[currentWaypointIndex])
                                 - transform.position;
                sensor.AddObservation(transform.InverseTransformDirection(toNext.normalized)); // 3
                sensor.AddObservation(Mathf.Clamp01(toNext.magnitude / 100f));                 // 1
            }
            else
            {
                sensor.AddObservation(Vector3.zero);
                sensor.AddObservation(0f);
            }

            // ── The road (6). Where we sit across it, how far off its heading we are, and
            //    what it does next. See VectorObservationSize for why these exist.
            AddRoadObservations(sensor);

            // RayPerceptionSensor3D handles LiDAR on its own component.
            // We now update its length via the OnWeatherChanged event to avoid CPU bottleneck here.
        }

        /// <summary>
        /// Lookahead distances for the road-curvature observations, in metres. Short enough
        /// to matter at corner entry, long enough that the longest reaches past the next
        /// waypoint — waypoints are ~50 m apart, so 100 m is genuinely new information
        /// rather than a restatement of the bearing already observed above.
        /// </summary>
        static readonly float[] TurnLookaheads = { 10f, 25f, 50f, 100f };

        /// <summary>
        /// Writes exactly 6 floats, always. A missing track still writes its zeros, because
        /// the observation vector's LENGTH can never depend on the scene being wired up.
        ///
        /// Everything is normalised into roughly -1..1 rather than left in metres and radians.
        /// The trainer runs with normalize: true so it would eventually rescale these itself,
        /// but a running normaliser is badly disturbed by outliers, and lateral offset in
        /// particular can reach 30 m before the episode ends.
        /// </summary>
        private void AddRoadObservations(VectorSensor sensor)
        {
            if (track == null)
            {
                for (int i = 0; i < 2 + TurnLookaheads.Length; i++) sensor.AddObservation(0f);
                return;
            }

            TrackGenerator.RoadFrame frame = track.SampleRoadFrame(transform.position, currentWaypointIndex);
            if (!frame.valid)
            {
                for (int i = 0; i < 2 + TurnLookaheads.Length; i++) sensor.AddObservation(0f);
                return;
            }

            // Where across the road we are. 1.0 is exactly on the edge; it is deliberately
            // allowed past that, because "two road-widths into the scenery" and "just over
            // the line" call for very different steering.
            float halfWidth = Mathf.Max(0.5f, track.importedCircuit?track.RoadHalfWidthAt(frame.distanceAlong,frame.signedOffset):track.roadWidth * 0.5f);
            sensor.AddObservation(Mathf.Clamp(frame.signedOffset / halfWidth, -3f, 3f));

            // Heading error: how far the car is pointed off the road's own direction.
            // Positive means the car is aimed to the RIGHT of where the road goes.
            Vector3 carFwd = track.transform.InverseTransformDirection(transform.forward);
            carFwd = new Vector3(carFwd.x, 0f, carFwd.z);
            float headingError = carFwd.sqrMagnitude < 1e-8f
                ? 0f
                : SignedAngleRad(frame.tangent, carFwd.normalized);
            sensor.AddObservation(Mathf.Clamp(headingError / (Mathf.PI * 0.5f), -1f, 1f));

            // What the road does next. Normalised so 1.0 is a 90-degree change of direction
            // inside the window, which is far tighter than this generator can produce — the
            // stage report puts the sharpest corner at ~36 deg per 50 m.
            foreach (float lookahead in TurnLookaheads)
            {
                float turn = track.RoadTurnAhead(frame.distanceAlong, lookahead);
                sensor.AddObservation(Mathf.Clamp(turn / (Mathf.PI * 0.5f), -1f, 1f));
            }
        }

        /// <summary>
        /// Signed angle from a to b about world up, radians, positive to the RIGHT so it
        /// matches the steering sign. Both vectors must be normalised and horizontal.
        /// </summary>
        private static float SignedAngleRad(Vector3 a, Vector3 b)
        {
            float dot = Mathf.Clamp(a.x * b.x + a.z * b.z, -1f, 1f);
            float cross = a.x * b.z - a.z * b.x;   // positive when b is LEFT of a
            return -Mathf.Atan2(cross, dot);
        }

        // Diagnostic only: RALLY_OBS_DUMP=<file> records the vector observations, ray hits and
        // chosen action for the first decisions of each episode, to compare courses directly.
        static readonly string obsDumpPath=System.Environment.GetEnvironmentVariable("RALLY_OBS_DUMP");
        int obsDumpDecisions;
        void DumpObservations(ActionBuffers actions)
        {
            if(string.IsNullOrEmpty(obsDumpPath))return;
            if(labTicks<=5)obsDumpDecisions=0;
            if(obsDumpDecisions++>=6)return;
            var sb=new System.Text.StringBuilder();
            var ci=System.Globalization.CultureInfo.InvariantCulture;
            sb.Append("{\"attempt\":").Append(labAttempt).Append(",\"tick\":").Append(labTicks);
            sb.Append(",\"action\":[").Append(actions.ContinuousActions[0].ToString(ci)).Append(',').Append(actions.ContinuousActions[1].ToString(ci)).Append(']');
            sb.Append(",\"vector\":[");
            var obs=GetObservations();
            for(int i=0;i<obs.Count;i++){if(i>0)sb.Append(',');sb.Append(obs[i].ToString("0.####",ci));}
            sb.Append("],\"rays\":[");
            bool first=true;
            foreach(var component in raySensors)
            {
                if(component==null)continue;
                var input=component.GetRayPerceptionInput();
                var output=RayPerceptionSensor.Perceive(input,false);
                for(int r=0;r<output.RayOutputs.Length;r++)
                {
                    var o=output.RayOutputs[r];
                    if(!first)sb.Append(',');first=false;
                    string hit=o.HasHit&&o.HitGameObject!=null?o.HitGameObject.name.Replace("\"",""):"";
                    sb.Append("{\"sensor\":\"").Append(component.SensorName).Append("\",\"ray\":").Append(r)
                      .Append(",\"hit\":\"").Append(hit).Append("\",\"tag\":").Append(o.HitTagIndex)
                      .Append(",\"fraction\":").Append(o.HitFraction.ToString("0.###",ci)).Append('}');
                }
            }
            sb.Append("]}");
            System.IO.File.AppendAllText(obsDumpPath,sb.ToString()+"\n");
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            if(LabRuntime.CircuitInspection)return;
            // Continuous actions: [0] steering, [1] drive (+throttle / -brake)
            if (actions.ContinuousActions.Length < ContinuousActionCount) { WarnActionSpec(); return; }

            DumpObservations(actions);
            float steering = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float drive    = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
            if (track != null && track.importedCircuit && InGracePeriod)
            {
                // Settle on the sampled slope before policy inputs load the tyres.
                // Terminal checks keep their existing grace; no failure is hidden.
                steering = drive = 0f;
            }

            // One pedal at a time, by construction.
            vehicle.SetInputs(steering, Mathf.Max(0f, drive), Mathf.Max(0f, -drive));

            // Kept only so the log can say what the policy was asking for as it went off.
            lastSteer = steering;
            lastDrive = drive;

            float dt = Time.fixedDeltaTime;

            // Time costs something, always. Without this, running out the clock is free
            // and standing still is a perfectly good strategy.
            Reward(-TimePenaltyRatePerSecond * dt);

            // ── Reward shaping ────────────────────────────────────────────────
            //  The continuous terms are scaled by dt so they are worth a fixed amount
            //  PER SECOND rather than per step. Without that they scale with the physics
            //  rate and drown everything else: at 30 m/s the old speed term paid ~15 a
            //  second, so a two-minute episode banked ~1800 from speed alone against ~20
            //  from waypoints. Progress was one percent of the signal, and the policy's
            //  best move was to point somewhere and hold the throttle down.
            if (track != null && track.waypoints.Count > currentWaypointIndex && labStartCrossed)
            {
                Vector3 currentWorldWaypoint = track.transform.TransformPoint(track.waypoints[currentWaypointIndex]);
                Vector3 tangent = (currentWorldWaypoint - transform.position).normalized;
                float forwardVelocity = Vector3.Dot(rb.linearVelocity, tangent);

                // Making progress toward the next waypoint: ~1.0/s at full speed.
                Reward(forwardVelocity / 55f * dt);

                // Sliding sideways instead: worth half as much, so a bit of oversteer on
                // the way through a corner is not punished harder than the speed is paid.
                Vector3 lateralVel = Vector3.ProjectOnPlane(rb.linearVelocity, tangent);
                if (!LabRuntime.Enabled || LabRuntime.Config.reward != "time-attack-no-slip-v1")
                    Reward(-lateralVel.magnitude / 55f * 0.5f * dt);

                // Progress to next waypoint
                int gateLimit=track.importedCircuit?8:1;
                int crossed=0;
                while(crossed<gateLimit && (LabRuntime.Enabled ? LabCrossedGate(currentWorldWaypoint) : Vector3.Distance(transform.position, currentWorldWaypoint) < 10f))
                {
                    crossed++;
                    currentWaypointIndex++;
                    Reward(waypointBonus);

                    // Past the last waypoint with road under it, the stage is finished —
                    // otherwise the car drives off the end of the mesh and falls until
                    // MaxStep, banking reward for a run it never completed.
                    if (track.importedCircuit ? currentWaypointIndex > track.FinishWaypointIndex : currentWaypointIndex >= track.FinishWaypointIndex)
                    {
                        bool timeAttack = LabRuntime.Enabled && LabRuntime.Config.reward.StartsWith("time-attack");
                        if (timeAttack && labInvalid)
                        {
                            Reward(-failurePenalty);
                            Fail("crossed finish outside course limits", EpisodeOutcome.FellOff);
                            return;
                        }

                        Reward(finishBonus);
                        Finish("reached the end of the stage");
                        break;
                    }
                    currentWorldWaypoint=track.transform.TransformPoint(track.waypoints[currentWaypointIndex]);
                }
                if(track.importedCircuit) labPrevious=transform.position;
            }
        }

        [System.Serializable]
        private class LabSample
        {
            public int tick;
            public Vector3 position;
            public Quaternion rotation;
            public float speed, steer, drive;
        }

        private bool LabCrossedGate(Vector3 gate)
        {
            Vector3 previousGate = track.transform.TransformPoint(track.waypoints[Mathf.Max(0, currentWaypointIndex - 1)]);
            Vector3 normal = Vector3.ProjectOnPlane(gate - previousGate, Vector3.up).normalized;
            if(track.importedCircuit)normal=track.transform.TransformDirection(track.CircuitTangent(track.CircuitGateStation(currentWaypointIndex)));
            Vector3 now = transform.position;
            if(track.importedCircuit && Vector3.Distance(labPrevious,now)>Mathf.Max(3,rb.linearVelocity.magnitude*Time.fixedDeltaTime*3+.5f))
            {labInvalid=true;circuitInvalidReason="discontinuous movement across circuit gates";return false;}
            float before = Vector3.Dot(labPrevious - gate, normal), after = Vector3.Dot(now - gate, normal);
            bool crossed = before <= 0 && after > 0;
            if (crossed)
            {
                Vector3 intersection = Vector3.Lerp(labPrevious, now, -before / (after - before));
                float signed=Vector3.Dot(intersection-gate,Vector3.Cross(Vector3.up,normal));float lateral=Mathf.Abs(signed);
                crossed = lateral <= (track.importedCircuit?track.LegalHalfWidthAt(track.CircuitGateStation(currentWaypointIndex),signed):track.roadWidth*.5f+1f) && Mathf.Abs(intersection.y - gate.y) < 5f;
                if (crossed) labSplits.Add(labTicks * Time.fixedDeltaTime);
            }
            if(!track.importedCircuit) labPrevious = now;
            return crossed;
        }

        private void LabCrossedStart()
        {
            Vector3 gateForward = track.transform.TransformDirection(
                track.waypoints[track.spawnWaypointIndex + 1] - track.waypoints[track.spawnWaypointIndex]).normalized;
            if(track.importedCircuit)
                gateForward=track.transform.TransformDirection(track.SampleRoadFrame(labStartGate,currentWaypointIndex).tangent);
            gateForward = Vector3.ProjectOnPlane(gateForward, Vector3.up).normalized;
            Vector3 now = transform.position;
            float before = Vector3.Dot(labPrevious - labStartGate, gateForward);
            float after = Vector3.Dot(now - labStartGate, gateForward);
            if (before <= 0 && after > 0)
            {
                Vector3 crossing = Vector3.Lerp(labPrevious, now, -before / (after - before));
                float signed=Vector3.Dot(crossing-labStartGate,Vector3.Cross(Vector3.up,gateForward));float lateral=Mathf.Abs(signed);
                if (lateral <= (track.importedCircuit?track.LegalHalfWidthAt(track.circuitStart,signed):track.roadWidth*.5f+1f) && Mathf.Abs(crossing.y - labStartGate.y) < 5f)
                {
                    labStartCrossed = true;
                    labStartTick = labTicks;
                    labSplits.Add(labTicks * Time.fixedDeltaTime);
                }
            }
            labPrevious = now;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            if (LabRuntime.Enabled && LabRuntime.Config.controlProbe)
            {
                var probeActions = actionsOut.ContinuousActions;
                if (probeActions.Length < ContinuousActionCount) { WarnActionSpec(); return; }
                if (LabRuntime.Config.controlMode == "steering-step")
                {
                    probeActions[0] = Mathf.Clamp(LabRuntime.Config.controlSteer, -0.25f, 0.25f);
                    float speedError = LabRuntime.Config.controlTargetSpeed - Mathf.Abs(vehicle.ForwardSpeed);
                    probeActions[1] = Mathf.Clamp(speedError * 0.18f, -0.5f, 1f);
                }
                else if (LabRuntime.Config.controlMode == "fixed-input")
                {
                    probeActions[0] = Mathf.Clamp(LabRuntime.Config.controlSteer, -1f, 1f);
                    probeActions[1] = Mathf.Clamp(LabRuntime.Config.controlDrive, -1f, 1f);
                }
                else if (LabRuntime.Config.controlMode == "waypoint-follow" && track != null && nextWaypoint != null)
                {
                    var frame = track.SampleRoadFrame(transform.position, currentWaypointIndex);
                    if (frame.valid)
                    {
                        bool reverse=System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_PROBE_REVERSE")=="1";
                        Vector3 tangent = track.transform.TransformDirection(frame.tangent)*(reverse?-1:1);
                        tangent = Vector3.ProjectOnPlane(tangent, Vector3.up).normalized;
                        float headingError = SignedAngleRad(tangent, transform.forward);
                        float desired=0;float.TryParse(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_PROBE_LATERAL"),System.Globalization.NumberStyles.Float,System.Globalization.CultureInfo.InvariantCulture,out desired);
                        if(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_PROBE_SWEEP")=="1")desired*=Mathf.Sin(Mathf.PI*Mathf.Clamp01((Time.time-episodeStartTime)/20));
                        else if(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_PROBE_RETURN")=="1")desired*=Mathf.Clamp01(1-(Time.time-episodeStartTime)/8);
                        float offsetError = Mathf.Atan2((frame.signedOffset-desired)*(reverse?-1:1), 18f);
                        probeActions[0] = Mathf.Clamp(-(headingError * 1.8f + offsetError), -1f, 1f);
                    }
                    else probeActions[0] = 0f;

                    float speedError = (track.importedCircuit?LabRuntime.Config.controlTargetSpeed:16f) - Mathf.Abs(vehicle.ForwardSpeed);
                    probeActions[1] = Mathf.Clamp(speedError * 0.18f, -0.5f, 1f);
                }
                else
                {
                    probeActions[0] = 0f;
                    probeActions[1] = 1f;
                }
                return;
            }

            // Human-in-the-loop override for testing
            var continuousActionsOut = actionsOut.ContinuousActions;
            if (continuousActionsOut.Length < ContinuousActionCount) { WarnActionSpec(); return; }

            continuousActionsOut[0] = Input.GetAxis("Horizontal");
            continuousActionsOut[1] = (Input.GetKey(KeyCode.W) ? 1.0f : 0.0f)
                                    - (Input.GetKey(KeyCode.S) ? 1.0f : 0.0f);
        }

        private bool warnedActionSpec;

        /// <summary>
        /// Behaviour Parameters defaults to zero continuous actions, and indexing past the
        /// end of the buffer throws every single step — which kills the agent's control of
        /// the car silently. Say so once instead.
        /// </summary>
        private void WarnActionSpec()
        {
            if (warnedActionSpec) return;
            warnedActionSpec = true;
            Debug.LogWarning(
                $"[RallyAgent] Behaviour Parameters needs Continuous Actions = {ContinuousActionCount} " +
                "(steering, drive). The agent cannot drive until it is set. " +
                "Keyboard control still works.", this);
        }

        void OnCollisionEnter(Collision collision)
        {
            if (InGracePeriod) return;

            // Dividing the impulse by the mass turns it into the change in velocity
            // the impact caused, which is what actually decides whether something was
            // a crash. The raw impulse would scale with however heavy the car is.
            float deltaV = collision.impulse.magnitude / Mathf.Max(1f, rb.mass);
            bool hitObstacle = collision.gameObject.CompareTag("Obstacle");

            if (hitObstacle || deltaV > crashDeltaV)
            {
                RecordStrike(collision);
                Reward(-failurePenalty);
                Fail(hitObstacle ? "hit an obstacle" : $"impact at {deltaV:0.0} m/s of delta-v",
                     hitObstacle ? EpisodeOutcome.HitObstacle : EpisodeOutcome.HardImpact);
            }
        }

        /// <summary>
        /// Works out what was hit and whether it could have been avoided.
        ///
        /// The name matters as much as anything else here. Road rocks, trees and boulders
        /// all carry the Obstacle tag and all end the episode identically, so "half of all
        /// failures are obstacle strikes" has never distinguished a rock in the racing line
        /// from a tree the car could only have reached by leaving the road — which is not an
        /// obstacle problem at all, it is a cornering problem wearing an obstacle's clothes.
        /// The generator now names road rocks, so the log can tell them apart.
        ///
        /// For a road rock it also records the widest clear corridor at that point and how
        /// long the rock had been in view. Those two numbers together decide the question
        /// three runs could not: whether ~48 % is a skill ceiling or an unfair stage.
        /// </summary>
        private void RecordStrike(Collision collision)
        {
            GameObject struck = collision.gameObject;
            hitWhat = struck.name;
            warningSeconds = WarningSecondsFor(struck.transform.position);
            if (track == null) return;

            // Where the thing that was hit sits across the road — for EVERYTHING, not only
            // road rocks. Without this the log gave the car's offset and left the object's
            // to be inferred, and the inference was wrong: episodes ending against a tree
            // read 2-4 m of car offset on a stage whose scenery starts at 13 m, and there
            // was no way to tell whether the car's number was lying or the tree's was.
            TrackGenerator.RoadFrame frame =
                track.SampleRoadFrame(struck.transform.position, currentWaypointIndex);
            if (!frame.valid) return;

            struckLateral = frame.signedOffset;
            // Unsigned, and the struck object's own station. The signed offset is a dot
            // against the road's right, so it under-reports anything that is mostly ahead
            // rather than beside; these two say whether that is happening.
            struckDistance = track.LateralOffset(struck.transform.position, currentWaypointIndex);
            struckStation = frame.distanceAlong;

            // By name, not by parentage. The terrain and the road mesh are also children of
            // the track, so "parented to the track" would file a hard landing on the terrain
            // as a rock strike and then measure a gap for it.
            hitRoadRock = struck.name == RoadRockName;
            if (!hitRoadRock) return;

            rockLateral = frame.signedOffset;
            if (!track.WidestGapAt(frame.distanceAlong, GapWindowMetres, out gapWidth, out gapCentre))
            {
                // No rock near the station the strike projected to. That means the hit did
                // not come from the driveable surface — most likely a rock clipped from
                // outside its own segment — so a "gap" here would be a fiction.
                hitRoadRock = false;
            }
        }

        /// <summary>
        /// The checks that end an episode for a reason other than a collision.
        ///
        /// These used to run in Update, which tied them to the FRAME rate and therefore
        /// made them mean different things in different places. ML-Agents runs training at
        /// a time scale of 20, so a frame covered twenty physics steps and a rollover was
        /// noticed up to 0.2 s of simulated time after it happened; the editor at a time
        /// scale of 1 noticed it inside 0.01 s. The same policy on the same stage could end
        /// an episode differently depending on how fast the wall clock was moving, which is
        /// not something that should be true of a physics simulation.
        ///
        /// In FixedUpdate they are deterministic in simulated time, and consistent with
        /// crashes, which have always been detected on the physics clock.
        /// </summary>
        private void CheckTerminalConditions()
        {
            if (InGracePeriod) return;

            // ── Stalled: nothing this episode can still teach us ──
            if (rb.linearVelocity.magnitude > stallSpeedThreshold) lastMotionTime = Time.time;
            if (stallTimeoutSeconds > 0f && Time.time - lastMotionTime > stallTimeoutSeconds)
            {
                Fail($"stalled for {stallTimeoutSeconds:0.#}s", EpisodeOutcome.Stalled);
                return;
            }

            // Check for rollover
            if (Vector3.Dot(transform.up, Vector3.up) < 0.1f)
            {
                Reward(-failurePenalty);
                Fail("rolled over", EpisodeOutcome.RolledOver);
                return;
            }

            // Deep in the scenery. Measured from the centreline rather than by falling
            // off the mesh, because with terrain everywhere there is nothing to fall off.
            if (track != null && track.LateralOffset(transform.position, currentWaypointIndex) > offStageDistance)
            {
                Reward(-failurePenalty);
                Fail("left the stage", EpisodeOutcome.FellOff);
                return;
            }

            bool fellOut=track!=null && track.importedCircuit
                ? transform.position.y < track.transform.TransformPoint(track.CircuitPosition(track.SampleRoadFrame(transform.position,currentWaypointIndex).distanceAlong)).y-30
                : transform.position.y < fallOutY;
            if (fellOut)
            {
                Reward(-failurePenalty);
                Fail("fell out of the world", EpisodeOutcome.FellOff);
            }
        }

        /// <summary>
        /// Publishes the things TensorBoard cannot infer on its own: WHY the episode
        /// ended, and how far down the stage it got.
        ///
        /// This matters more than it looks. The editor Training Monitor cannot exist in a
        /// headless build — there is no editor — and a headless run is how training
        /// actually gets done at speed. Averaged over a summary window each of these
        /// becomes the RATE of that outcome, so "Outcome/RolledOver" reads 0.24 when a
        /// quarter of episodes end on their roof. Same information, but logged,
        /// timestamped and comparable between runs instead of living in a window.
        /// </summary>
        private void ReportToTensorboard(EpisodeOutcome outcome, int waypointsReached)
        {
            var stats = Academy.Instance.StatsRecorder;

            stats.Add("Outcome/Finished",    outcome == EpisodeOutcome.Finished    ? 1f : 0f);
            stats.Add("Outcome/TimedOut",    outcome == EpisodeOutcome.TimedOut    ? 1f : 0f);
            stats.Add("Outcome/RolledOver",  outcome == EpisodeOutcome.RolledOver  ? 1f : 0f);
            stats.Add("Outcome/HitObstacle", outcome == EpisodeOutcome.HitObstacle ? 1f : 0f);
            stats.Add("Outcome/HardImpact",  outcome == EpisodeOutcome.HardImpact  ? 1f : 0f);
            stats.Add("Outcome/FellOff",     outcome == EpisodeOutcome.FellOff     ? 1f : 0f);
            stats.Add("Outcome/Stalled",     outcome == EpisodeOutcome.Stalled     ? 1f : 0f);

            stats.Add("Progress/Waypoints", waypointsReached);
            stats.Add("Progress/StageFraction", waypointsReached / (float)WaypointTarget);

            // The curriculum's current rung, so a change in the outcome rates can be read
            // against the difficulty that caused it rather than looking like a regression.
            stats.Add("Curriculum/ObstacleDensity", currentObstacleDensity);
        }

        /// <summary>
        /// Right after a respawn the car is still dropping onto its suspension. Nothing
        /// that happens while it settles should be read as the driver's fault.
        /// </summary>
        private bool InGracePeriod => Time.time - episodeStartTime < respawnGraceSeconds;

        /// <summary>
        /// Charges the time penalty the rest of the episode would have accrued.
        ///
        /// Without this, abandoning a dead episode EARLY would be a reward for parking. Sitting
        /// out the full clock costs 0.05 x 120 = -6.0; bailing at 5 s costs only -0.25. The
        /// policy would learn that the cheapest available move is to stop moving and get the
        /// episode cut short — inverting the ordering the whole reward is built around
        /// (finish > progress > crash > park), and creating a worse bug than the one early
        /// termination exists to fix.
        ///
        /// Charge failures for the unused episode clock at the same rate as live steps. In
        /// time-attack mode this applies to every early failure, not only stalls; otherwise
        /// a crash could avoid most of its time cost. Baseline runs retain their historical
        /// stall-only accounting.
        /// </summary>
        private float TimePenaltyRatePerSecond =>
            LabRuntime.Enabled && LabRuntime.Config.reward.StartsWith("time-attack")
                ? 0.12f : timePenaltyPerSecond;

        private void ChargeForfeitedTime()
        {
            if (MaxStep <= 0) return;                                  // no clock to forfeit
            float budget = MaxStep * Time.fixedDeltaTime;
            float used = Time.time - episodeStartTime;
            Reward(-TimePenaltyRatePerSecond * Mathf.Max(0f, budget - used));
        }

        /// <summary>
        /// Set whenever this agent ends its own episode. If an episode begins without one
        /// having been set, ML-Agents ended it on MaxStep — i.e. it timed out — and that
        /// is the one outcome the agent never gets told about directly.
        /// </summary>
        private EpisodeOutcome? pendingOutcome;

        private void Fail(string reason, EpisodeOutcome outcome)
        {
            bool timeAttack = LabRuntime.Enabled && LabRuntime.Config.reward.StartsWith("time-attack");
            if (timeAttack || outcome == EpisodeOutcome.Stalled)
                ChargeForfeitedTime();

            if (logEpisodeEnds)
                Debug.Log($"[RallyAgent] Episode ended after {Time.time - episodeStartTime:0.0}s: {reason}.");
            pendingOutcome = outcome;
            EndEpisode();
        }

        private void Finish(string reason)
        {
            if (logEpisodeEnds)
                Debug.Log($"[RallyAgent] Stage complete after {Time.time - episodeStartTime:0.0}s: {reason}.");
            pendingOutcome = EpisodeOutcome.Finished;
            EndEpisode();
        }

        /// <summary>Books the episode that just ended, whatever ended it.</summary>
        public EpisodeRecord? ViewerResult { get; private set; }

        private void RecordEpisode()
        {
            if(LabRuntime.CircuitInspection)return;
            if (!episodeStarted) return;

            EpisodeOutcome outcome = pendingOutcome ?? EpisodeOutcome.TimedOut;
            int reached = Mathf.Max(0, WaypointsReached);
            float seconds = LabRuntime.Enabled ? Mathf.Max(0, labTicks - labStartTick) * Time.fixedDeltaTime : Time.time - episodeStartTime;
            if (LabRuntime.Enabled && LabRuntime.Config.viewer)
                ViewerResult = new EpisodeRecord {managed=true, ticks=labTicks, stageSeed=track!=null?track.CurrentSeed:0,
                    outcome=outcome, valid=!labInvalid && !distributedStart && outcome==EpisodeOutcome.Finished,
                    seconds=seconds, peakSpeed=peakSpeed, attempt=labAttempt, splits=labSplits.ToArray(),
                    waypointsReached=reached, waypointTarget=WaypointTarget};
            labTrajectory?.Flush();

            TrainingStats.Record(outcome, episodeReward, seconds, reached);
            ReportToTensorboard(outcome, reached);

            // On disk as well as in memory. TrainingStats is editor-only and single-process,
            // and TensorBoard only samples every summary_freq steps, so without this a
            // headless run leaves no per-episode record at all — including which stage seed
            // each episode was driving.
            if (logEpisodesToFile)
            {
                // The forensic fields come from the snapshot, which is the last frame the car
                // was level. Everything else is known here.
                EpisodeRecord record = snapshot;
                record.outcome = outcome;
                record.reward = episodeReward;
                record.seconds = seconds;
                record.waypointsReached = reached;
                record.waypointTarget = WaypointTarget;
                record.stageSeed = track != null ? track.CurrentSeed : 0;
                record.obstacleDensity = currentObstacleDensity;
                record.peakSpeed = peakSpeed;
                record.longestFlight = longestFlight;
                record.steer = lastSteer;
                record.drive = lastDrive;
                record.hitWhat = hitWhat;
                record.hitRoadRock = hitRoadRock;
                record.rockLateral = rockLateral;
                record.struckLateral = struckLateral;
                record.struckDistance = struckDistance;
                record.struckStation = struckStation;
                record.gapWidth = gapWidth;
                record.gapCentre = gapCentre;
                record.warningSeconds = warningSeconds;
                record.managed = LabRuntime.Enabled;
                record.valid = LabRuntime.Enabled && !labInvalid && !distributedStart && outcome == EpisodeOutcome.Finished;
                record.ticks = labTicks;
                record.attempt = labAttempt;
                record.splits = labSplits.ToArray();
                if(track!=null && track.importedCircuit)
                {
                    record.circuitRevision=track.circuitRevision;record.circuitStart=track.circuitStart;record.circuitEnd=track.circuitEnd;
                    record.episodeSeconds=MaxStep*Time.fixedDeltaTime;record.spawnProfile=LabRuntime.Enabled?LabRuntime.Config.spawnProfile:"fixed";
                    record.invalidationReason=distributedStart&&string.IsNullOrEmpty(circuitInvalidReason)?"distributed training start; not a ranked attempt":circuitInvalidReason;
                    record.spawnStation=spawnStation;
                }

                EpisodeLog.Append(record);
            }

            EpisodeEnded?.Invoke(outcome, reached);

            pendingOutcome = null;
            episodeReward = 0f;
            episodeStarted = false;

            CheckEvaluationBudget();
        }

        /// <summary>
        /// Raised once per finished episode, after it has been booked. Exists so things that
        /// only watch — the ghost recorder, the evaluation harness — can hook on without
        /// RallyAgent having to know they are there.
        /// </summary>
        public event System.Action<EpisodeOutcome, int> EpisodeEnded;
    }
}
