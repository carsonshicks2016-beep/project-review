using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Core.Environment;
using Core.Physics;

namespace UI
{
    public class SpectatorDirector : MonoBehaviour
    {
        bool resultOrbitActive;
        float resultOrbitStart, resultOrbitAngle;
        Vector3 resultOrbitCenter;
        // Keep the first three values stable: existing scenes serialize them as integers.
        public enum CameraMode
        {
            ThirdPerson = 0,
            Driver = 1,
            Trackside = 2,
            Helicopter = 3,
            Hood = 4,
            Cinematic = 5
        }

        struct TracksideShot
        {
            public Vector3 position;
            public Quaternion rotation;
            public float routeDistance;
        }

        struct GlassState
        {
            public Material material;
            public Color color;
            public string renderType;
            public float mode;
            public int sourceBlend;
            public int destinationBlend;
            public int zWrite;
            public int cull;
            public int renderQueue;
            public bool alphaBlend;
            public bool alphaPremultiply;
            public bool alphaTest;
        }

        static readonly CameraMode[] CinematicShots =
        {
            CameraMode.ThirdPerson,
            CameraMode.Helicopter,
            CameraMode.Trackside,
            CameraMode.Hood
        };

        static readonly CameraMode[] SelectableModes =
        {
            CameraMode.ThirdPerson,
            CameraMode.Helicopter,
            CameraMode.Hood,
            CameraMode.Driver,
            CameraMode.Trackside,
            CameraMode.Cinematic
        };

        public CameraMode currentMode = CameraMode.ThirdPerson;

        [Header("References")]
        public Transform targetCar;
        public Rigidbody carRb;
        public Transform headBoneTarget;

        [Header("Chase")]
        public float orbitDistance = 6.35f;
        public float orbitHeight = 1.82f;
        public float chaseSideOffset = 0.48f;
        public float speedZoom = 0.012f;
        public float followSmoothness = 5.5f;

        [Header("Helicopter")]
        public float helicopterHeight = 11f;
        public float helicopterDistance = 8f;

        [Header("In-Car Cameras")]
        public Vector3 hoodLocalPosition = new Vector3(0f, 1.42f, 1.38f);
        public Vector3 driverLocalPosition = new Vector3(0.36f, 1.04f, -0.12f);
        public float gForceMultiplier = 0.035f;
        public float maxGForceOffset = 0.12f;
        public float returnSpring = 5f;

        [Header("Cinematic")]
        [Min(3f)] public float cinematicShotSeconds = 8f;

        Vector3 currentGForceOffset;
        Vector3 lastVelocity;
        bool hasLastVelocity;
        float chaseRoll;
        float smoothedCameraSpeed;
        bool hasSmoothedCameraSpeed;
        Vector3 lastTargetPosition;
        Vector3 smoothedTargetPosition;
        public CameraMode ActiveShotMode => renderedMode;
        Vector3 stablePlanarForward = Vector3.forward;
        float targetVerticalVelocity;
        bool hasFollowTarget;
        bool hasCameraPose;
        float cinematicClock;
        int cinematicShotIndex;
        TrackGenerator course;
        int courseSeed = int.MinValue;
        readonly List<TracksideShot> tracksideShots = new List<TracksideShot>(8);
        readonly List<GlassState> driverGlass = new List<GlassState>(4);
        float[] routeDistances;
        int tracksideShotIndex = -1;
        int renderedStation = -1;
        CameraMode renderedMode;
        float lastStationChange = -10f;
        Transform glassTarget;
        bool driverGlassEnabled;
        System.IO.StreamWriter frameTrace;
        public int ActiveTracksideStation => tracksideShotIndex;
        public bool CarInSafeFrame { get; private set; }
        int traceFrames;
        Core.ML.RallyAgent episodeSource;
        public event System.Action<CameraMode, bool, bool> AudioPerspectiveChanged;
        bool audioResetPending;

        [System.Serializable] class CameraFrame
        {
            public int frame, mode, shot, interpolation, vsync, fpsCap;
            public double time, fixedTime;
            public float dt, scale, fov;
            public Vector3 car, physics, velocity, camera;
            public Quaternion rotation;
            public int station;
            public bool carInSafeFrame;
        }

        void Start()
        {
            string trace = System.Environment.GetEnvironmentVariable("RALLY_CAMERA_TRACE");
            if (!string.IsNullOrEmpty(trace) && SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null)
                frameTrace = new System.IO.StreamWriter(trace) { AutoFlush = true };
            BindVehicleIfNeeded();
            BindEpisodeEvents();
            if (carRb != null)
            {
                lastVelocity = carRb.linearVelocity;
                hasLastVelocity = true;
                smoothedCameraSpeed = lastVelocity.magnitude;
                hasSmoothedCameraSpeed = true;
            }
            if (course == null) ConfigureCourse(FindObjectOfType<TrackGenerator>());
        }

        void BindVehicleIfNeeded()
        {
            if (targetCar != null) return;
            VehicleController vehicle = FindObjectOfType<Core.Physics.VehicleController>();
            if (vehicle == null) return;
            targetCar = vehicle.transform;
            carRb = vehicle.GetComponent<Rigidbody>();
            BindEpisodeEvents();
        }

        void BindEpisodeEvents()
        {
            var source = targetCar != null ? targetCar.GetComponent<Core.ML.RallyAgent>() : null;
            if (episodeSource == source) return;
            if (episodeSource != null) episodeSource.EpisodeEnded -= ResetEpisodeCamera;
            episodeSource = source;
            if (episodeSource != null) episodeSource.EpisodeEnded += ResetEpisodeCamera;
        }

        void ResetEpisodeCamera(Core.ML.EpisodeOutcome outcome, int reached)
        {
            hasFollowTarget = hasCameraPose = hasSmoothedCameraSpeed = false;
            tracksideShotIndex = renderedStation = -1;
            lastStationChange = -10f;
            cinematicClock = 0f;
            audioResetPending = true;
        }

        public void ConfigureCourse(TrackGenerator track)
        {
            if (course == track && track != null && courseSeed == track.CurrentSeed) return;
            course = track;
            courseSeed = track != null ? track.CurrentSeed : int.MinValue;
            tracksideShots.Clear();
            routeDistances = null;
            tracksideShotIndex = -1;
            if (track == null || track.waypoints == null || track.waypoints.Count < 4) return;

            routeDistances = new float[track.waypoints.Count];
            for (int i = 1; i < track.waypoints.Count; i++)
                routeDistances[i] = routeDistances[i - 1] + Vector3.Distance(
                    track.transform.TransformPoint(track.waypoints[i - 1]),
                    track.transform.TransformPoint(track.waypoints[i]));

            int first = Mathf.Clamp(track.spawnWaypointIndex + 1, 1, track.waypoints.Count - 2);
            int last = Mathf.Clamp(track.FinishWaypointIndex - 1, first, track.waypoints.Count - 2);
            int spacing = Mathf.Max(2, (last - first) / 7);
            int shotNumber = 0;
            for (int i = first; i <= last; i += spacing)
            {
                Vector3 center = track.transform.TransformPoint(track.waypoints[i]);
                Vector3 forward = track.transform.TransformDirection(
                    track.waypoints[Mathf.Min(i + 1, track.waypoints.Count - 1)] -
                    track.waypoints[Mathf.Max(i - 1, 0)]).normalized;
                forward = Vector3.ProjectOnPlane(forward, Vector3.up).normalized;
                if (forward.sqrMagnitude < 0.001f) continue;
                Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;
                float side = shotNumber % 2 == 0 ? 1f : -1f;
                Vector3 position = center + right * side * (track.roadWidth * 0.5f + 2.5f)
                                   + Vector3.up * (2.0f + (shotNumber % 3) * 0.35f);
                if (Physics.Raycast(position + Vector3.up * 100f, Vector3.down, out var groundHit,
                    200f, ~0, QueryTriggerInteraction.Ignore))
                    position.y = Mathf.Max(position.y, groundHit.point.y + 2.5f);
                Vector3 lookPoint = center + forward * 6f + Vector3.up * 0.65f;
                tracksideShots.Add(new TracksideShot
                {
                    position = position,
                    rotation = Quaternion.LookRotation(lookPoint - position, Vector3.up),
                    routeDistance = routeDistances[i]
                });
                shotNumber++;
            }
        }

        public void SetMode(CameraMode mode)
        {
            if (currentMode == mode) return;
            currentMode = mode;
            cinematicShotIndex = 0;
            cinematicClock = 0f;
            tracksideShotIndex = -1;
        }

        public void CycleMode()
        {
            int index = System.Array.IndexOf(SelectableModes, currentMode);
            SetMode(SelectableModes[(index + 1 + SelectableModes.Length) % SelectableModes.Length]);
        }

        void Update()
        {
            if (Input.GetKeyDown(KeyCode.V)) CycleMode();
            if (Input.GetKeyDown(KeyCode.Alpha1)) SetMode(CameraMode.ThirdPerson);
            if (Input.GetKeyDown(KeyCode.Alpha2)) SetMode(CameraMode.Helicopter);
            if (Input.GetKeyDown(KeyCode.Alpha3)) SetMode(CameraMode.Hood);
            if (Input.GetKeyDown(KeyCode.Alpha4)) SetMode(CameraMode.Driver);
            if (Input.GetKeyDown(KeyCode.Alpha5)) SetMode(CameraMode.Trackside);
            if (Input.GetKeyDown(KeyCode.Alpha6)) SetMode(CameraMode.Cinematic);
        }

        void FixedUpdate()
        {
            if (carRb == null) return;
            Vector3 velocity = carRb.linearVelocity;
            if (!hasLastVelocity)
            {
                lastVelocity = velocity;
                hasLastVelocity = true;
                return;
            }

            Vector3 acceleration = (velocity - lastVelocity) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            lastVelocity = velocity;
            Vector3 localGForce = carRb.transform.InverseTransformDirection(-acceleration);
            Vector3 targetOffset = Vector3.ClampMagnitude(localGForce * gForceMultiplier, maxGForceOffset);
            currentGForceOffset = Vector3.Lerp(currentGForceOffset, targetOffset,
                1f - Mathf.Exp(-10f * Time.fixedDeltaTime));
            currentGForceOffset = Vector3.Lerp(currentGForceOffset, Vector3.zero,
                1f - Mathf.Exp(-returnSpring * Time.fixedDeltaTime));
        }

        void LateUpdate()
        {
            if (targetCar == null) BindVehicleIfNeeded();
            if (targetCar == null) return;
            if (Core.ML.LabRuntime.ViewerPaused)
            {
                if (!resultOrbitActive)
                {
                    resultOrbitActive=true; resultOrbitStart=Time.unscaledTime;
                    resultOrbitCenter=targetCar.position+Vector3.up*.65f;
                    resultOrbitAngle=Mathf.Atan2(transform.position.x-resultOrbitCenter.x,
                        transform.position.z-resultOrbitCenter.z);
                }
                float elapsed=Time.unscaledTime-resultOrbitStart;
                float angle=resultOrbitAngle+elapsed*Mathf.PI*2/24f;
                Vector3 desired=resultOrbitCenter+new Vector3(Mathf.Sin(angle)*5f,1.9f,Mathf.Cos(angle)*5f);
                // Pull the shot above nearby terrain instead of orbiting through it.
                if (UnityEngine.Physics.Raycast(desired+Vector3.up*30,Vector3.down,out var hit,60))
                    desired.y=Mathf.Max(desired.y,hit.point.y+.8f);
                float orbitBlend=1-Mathf.Exp(-3f*Time.unscaledDeltaTime);
                transform.position=Vector3.Lerp(transform.position,desired,orbitBlend);
                transform.rotation=Quaternion.Slerp(transform.rotation,
                    Quaternion.LookRotation(resultOrbitCenter-transform.position,Vector3.up),orbitBlend);
                var resultCamera=GetComponent<Camera>();
                if(resultCamera!=null) {resultCamera.nearClipPlane=.05f; resultCamera.fieldOfView=Mathf.Lerp(resultCamera.fieldOfView,48,orbitBlend);}
                return;
            }
            resultOrbitActive=false;

            float deltaTime = Mathf.Max(Time.unscaledDeltaTime, 0.001f);
            bool teleported = UpdateStableTarget(deltaTime);

            CameraMode shotMode = currentMode;
            if (currentMode == CameraMode.Cinematic)
            {
                cinematicClock += Time.unscaledDeltaTime;
                if (cinematicClock >= cinematicShotSeconds)
                {
                    cinematicClock %= cinematicShotSeconds;
                    cinematicShotIndex = (cinematicShotIndex + 1) % CinematicShots.Length;
                }
                shotMode = CinematicShots[cinematicShotIndex];
            }

            if (targetCar.GetComponentInChildren<Core.Presentation.RallyCockpit>() == null)
                SetDriverGlass(shotMode == CameraMode.Driver);

            float speed = carRb != null ? carRb.linearVelocity.magnitude : 0f;
            if (!hasSmoothedCameraSpeed)
            {
                smoothedCameraSpeed = speed;
                hasSmoothedCameraSpeed = true;
            }
            else
            {
                float speedBlend = 1f - Mathf.Exp(-1.8f * deltaTime);
                smoothedCameraSpeed = Mathf.Lerp(smoothedCameraSpeed, speed, speedBlend);
            }

            GetCameraPose(shotMode, out Vector3 desiredPosition, out Quaternion desiredRotation);
            Camera camera = GetComponent<Camera>();
            if (camera != null) camera.nearClipPlane = shotMode == CameraMode.Driver ? .025f : .05f;
            float targetFov = ModeFov(shotMode, smoothedCameraSpeed);
            bool cut = shotMode != renderedMode ||
                (shotMode == CameraMode.Trackside && renderedStation != tracksideShotIndex);
            renderedMode = shotMode;
            renderedStation = tracksideShotIndex;
            if (teleported || !hasCameraPose || cut)
            {
                transform.SetPositionAndRotation(desiredPosition, desiredRotation);
                if (camera != null) camera.fieldOfView = targetFov;
                hasCameraPose = true;
                PublishAudioPerspective(shotMode, true, teleported);
                TraceFrame(shotMode);
                return;
            }

            float smoothness = shotMode == CameraMode.Trackside ? 16f :
                shotMode == CameraMode.Hood || shotMode == CameraMode.Driver ? 28f :
                Mathf.Max(5f, followSmoothness);
            float blend = 1f - Mathf.Exp(-Mathf.Max(0.1f, smoothness) * deltaTime);
            bool onboard = shotMode == CameraMode.Driver || shotMode == CameraMode.Hood;
            // The target pose is already smoothed. A second position filter trails
            // a fast car far enough to put the observer behind the seat.
            transform.position = onboard || shotMode == CameraMode.Trackside && tracksideShotIndex >= 0
                ? desiredPosition : Vector3.Lerp(transform.position, desiredPosition, blend);
            transform.rotation = onboard ? desiredRotation : Quaternion.Slerp(transform.rotation, desiredRotation, blend);
            if (camera != null)
            {
                camera.fieldOfView = Mathf.Lerp(camera.fieldOfView, targetFov, blend);
            }
            PublishAudioPerspective(shotMode, false, false);
            TraceFrame(shotMode);
        }

        void PublishAudioPerspective(CameraMode shotMode, bool cut, bool teleported)
        {
            CameraMode effective = shotMode == CameraMode.Trackside && tracksideShotIndex < 0
                ? CameraMode.Helicopter : shotMode;
            AudioPerspectiveChanged?.Invoke(effective, cut, teleported || audioResetPending);
            audioResetPending = false;
        }

        void TraceFrame(CameraMode shotMode)
        {
            Vector3 viewport = GetComponent<Camera>().WorldToViewportPoint(smoothedTargetPosition + Vector3.up * .8f);
            CarInSafeFrame = viewport.z > 0f && viewport.x > .08f && viewport.x < .92f &&
                viewport.y > .08f && viewport.y < .92f;
            if (frameTrace == null) return;
            frameTrace.WriteLine(JsonUtility.ToJson(new CameraFrame {
                frame = Time.frameCount, mode = (int)currentMode, shot = (int)shotMode,
                interpolation = carRb != null ? (int)carRb.interpolation : -1,
                time = Time.timeAsDouble, fixedTime = Time.fixedTimeAsDouble,
                dt = Time.unscaledDeltaTime, scale = Time.timeScale,
                fov = GetComponent<Camera>().fieldOfView, vsync = QualitySettings.vSyncCount,
                fpsCap = Application.targetFrameRate,
                car = targetCar.position, physics = carRb != null ? carRb.position : targetCar.position,
                velocity = carRb != null ? carRb.linearVelocity : Vector3.zero,
                camera = transform.position, rotation = transform.rotation,
                station = tracksideShotIndex, carInSafeFrame = CarInSafeFrame
            }));
            if (++traceFrames >= 1200) { frameTrace.Dispose(); frameTrace = null; }
        }

        void OnDestroy()
        {
            if (episodeSource != null) episodeSource.EpisodeEnded -= ResetEpisodeCamera;
            frameTrace?.Dispose();
            RestoreDriverGlass();
        }

        bool UpdateStableTarget(float deltaTime)
        {
            Vector3 position = targetCar.position;
            Vector3 forward = Vector3.ProjectOnPlane(targetCar.forward, Vector3.up);
            if (forward.sqrMagnitude < 0.001f)
                forward = Quaternion.Euler(0f, targetCar.eulerAngles.y, 0f) * Vector3.forward;
            forward.Normalize();

            if (!hasFollowTarget)
            {
                lastTargetPosition = smoothedTargetPosition = position;
                stablePlanarForward = forward;
                hasFollowTarget = true;
                return true;
            }

            bool teleported = (position - lastTargetPosition).sqrMagnitude > 144f;
            lastTargetPosition = position;
            if (teleported)
            {
                smoothedTargetPosition = position;
                stablePlanarForward = forward;
                targetVerticalVelocity = 0f;
                tracksideShotIndex = -1;
                return true;
            }

            smoothedTargetPosition.x = position.x;
            smoothedTargetPosition.z = position.z;
            smoothedTargetPosition.y = Mathf.SmoothDamp(smoothedTargetPosition.y, position.y,
                ref targetVerticalVelocity, 0.14f, Mathf.Infinity, deltaTime);
            stablePlanarForward = Vector3.Slerp(stablePlanarForward, forward,
                1f - Mathf.Exp(-10f * deltaTime)).normalized;
            return false;
        }

        void SetDriverGlass(bool enabled)
        {
            if (glassTarget != targetCar)
            {
                RestoreDriverGlass();
                CacheDriverGlass();
            }
            if (driverGlassEnabled == enabled) return;
            driverGlassEnabled = enabled;
            for (int i = 0; i < driverGlass.Count; i++)
            {
                GlassState state = driverGlass[i];
                Material material = state.material;
                if (material == null) continue;
                if (!enabled)
                {
                    material.SetColor("_Color", state.color);
                    if (material.HasProperty("_Mode")) material.SetFloat("_Mode", state.mode);
                    if (material.HasProperty("_SrcBlend")) material.SetInt("_SrcBlend", state.sourceBlend);
                    if (material.HasProperty("_DstBlend")) material.SetInt("_DstBlend", state.destinationBlend);
                    if (material.HasProperty("_ZWrite")) material.SetInt("_ZWrite", state.zWrite);
                    if (material.HasProperty("_Cull")) material.SetInt("_Cull", state.cull);
                    material.SetOverrideTag("RenderType", state.renderType);
                    material.renderQueue = state.renderQueue;
                    SetKeyword(material, "_ALPHABLEND_ON", state.alphaBlend);
                    SetKeyword(material, "_ALPHAPREMULTIPLY_ON", state.alphaPremultiply);
                    SetKeyword(material, "_ALPHATEST_ON", state.alphaTest);
                    continue;
                }

                Color tint = state.color;
                tint.a = 0.24f;
                material.SetColor("_Color", tint);
                if (material.HasProperty("_Mode")) material.SetFloat("_Mode", 2f);
                if (material.HasProperty("_SrcBlend")) material.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
                if (material.HasProperty("_DstBlend")) material.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
                if (material.HasProperty("_ZWrite")) material.SetInt("_ZWrite", 0);
                if (material.HasProperty("_Cull")) material.SetInt("_Cull", 0);
                material.SetOverrideTag("RenderType", "Transparent");
                material.renderQueue = 3000;
                material.DisableKeyword("_ALPHATEST_ON");
                material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                material.EnableKeyword("_ALPHABLEND_ON");
            }
        }

        void CacheDriverGlass()
        {
            driverGlass.Clear();
            glassTarget = targetCar;
            if (targetCar == null) return;
            foreach (Renderer renderer in targetCar.GetComponentsInChildren<Renderer>(true))
            foreach (Material material in renderer.sharedMaterials)
            {
                if (material == null || material.name.IndexOf("Glass", System.StringComparison.OrdinalIgnoreCase) < 0 ||
                    !material.HasProperty("_Color")) continue;
                driverGlass.Add(new GlassState
                {
                    material = material,
                    color = material.color,
                    renderType = material.GetTag("RenderType", false, "Opaque"),
                    mode = material.HasProperty("_Mode") ? material.GetFloat("_Mode") : 0f,
                    sourceBlend = material.HasProperty("_SrcBlend") ? material.GetInt("_SrcBlend") : (int)BlendMode.One,
                    destinationBlend = material.HasProperty("_DstBlend") ? material.GetInt("_DstBlend") : (int)BlendMode.Zero,
                    zWrite = material.HasProperty("_ZWrite") ? material.GetInt("_ZWrite") : 1,
                    cull = material.HasProperty("_Cull") ? material.GetInt("_Cull") : 2,
                    renderQueue = material.renderQueue,
                    alphaBlend = material.IsKeywordEnabled("_ALPHABLEND_ON"),
                    alphaPremultiply = material.IsKeywordEnabled("_ALPHAPREMULTIPLY_ON"),
                    alphaTest = material.IsKeywordEnabled("_ALPHATEST_ON")
                });
            }
        }

        void RestoreDriverGlass()
        {
            if (!driverGlassEnabled)
            {
                driverGlass.Clear();
                return;
            }
            driverGlassEnabled = false;
            for (int i = 0; i < driverGlass.Count; i++)
            {
                GlassState state = driverGlass[i];
                if (state.material == null) continue;
                state.material.SetColor("_Color", state.color);
                if (state.material.HasProperty("_Mode")) state.material.SetFloat("_Mode", state.mode);
                if (state.material.HasProperty("_SrcBlend")) state.material.SetInt("_SrcBlend", state.sourceBlend);
                if (state.material.HasProperty("_DstBlend")) state.material.SetInt("_DstBlend", state.destinationBlend);
                if (state.material.HasProperty("_ZWrite")) state.material.SetInt("_ZWrite", state.zWrite);
                if (state.material.HasProperty("_Cull")) state.material.SetInt("_Cull", state.cull);
                state.material.SetOverrideTag("RenderType", state.renderType);
                state.material.renderQueue = state.renderQueue;
                SetKeyword(state.material, "_ALPHABLEND_ON", state.alphaBlend);
                SetKeyword(state.material, "_ALPHAPREMULTIPLY_ON", state.alphaPremultiply);
                SetKeyword(state.material, "_ALPHATEST_ON", state.alphaTest);
            }
            driverGlass.Clear();
        }

        static void SetKeyword(Material material, string keyword, bool enabled)
        {
            if (enabled) material.EnableKeyword(keyword);
            else material.DisableKeyword(keyword);
        }

        static float ModeFov(CameraMode mode, float speed)
        {
            switch (mode)
            {
                case CameraMode.Helicopter: return 50f;
                case CameraMode.Hood: return 72f;
                case CameraMode.Driver: return 78f;
                case CameraMode.Trackside: return 48f;
                default: return Mathf.Lerp(56f, 68f, Mathf.InverseLerp(8f, 48f, speed));
            }
        }

        void GetCameraPose(CameraMode mode, out Vector3 position, out Quaternion rotation)
        {
            float speed = smoothedCameraSpeed;
            float lead = Mathf.Clamp(2.5f + speed * 0.13f, 2.5f, 11f);
            Vector3 planarForward = stablePlanarForward;
            Vector3 planarRight = Vector3.Cross(Vector3.up, planarForward).normalized;
            Vector3 lookPoint = smoothedTargetPosition + Vector3.up * 0.9f + planarForward * lead;

            switch (mode)
            {
                case CameraMode.Helicopter:
                    float orbit = Mathf.Sin(Time.unscaledTime * 0.16f) * 1.6f;
                    position = smoothedTargetPosition - planarForward * helicopterDistance
                               + planarRight * orbit + Vector3.up * helicopterHeight;
                    rotation = Quaternion.LookRotation(lookPoint - position, Vector3.up);
                    return;

                case CameraMode.Hood:
                    Transform hoodAnchor = targetCar.Find("ViewerCockpit/HoodEye");
                    Vector3 hoodLocal = hoodAnchor != null ? hoodAnchor.localPosition : hoodLocalPosition;
                    // Transform is the Rigidbody's interpolated render pose, not its
                    // fixed-step position. Body, cockpit and observer share this pose.
                    position = targetCar.TransformPoint(hoodLocal);
                    rotation = targetCar.rotation;
                    return;

                case CameraMode.Driver:
                    Transform driverAnchor = targetCar.Find("ViewerCockpit/DriverEye");
                    Vector3 driverLocal = headBoneTarget != null
                        ? targetCar.InverseTransformPoint(headBoneTarget.position)
                        : driverAnchor != null ? driverAnchor.localPosition : driverLocalPosition;
                    position = targetCar.TransformPoint(driverLocal)
                               + targetCar.right * currentGForceOffset.x;
                    rotation = targetCar.rotation;
                    return;

                case CameraMode.Trackside:
                    if (TryGetTracksidePose(out position, out rotation)) return;
                    GetCameraPose(CameraMode.Helicopter, out position, out rotation);
                    return;
            }

            float distance = orbitDistance + Mathf.Min(1.5f, speed * speedZoom);
            position = smoothedTargetPosition - planarForward * distance
                       + planarRight * chaseSideOffset + Vector3.up * orbitHeight;
            rotation = Quaternion.LookRotation(lookPoint - position, Vector3.up);
        }

        bool TryGetTracksidePose(out Vector3 position, out Quaternion rotation)
        {
            position = transform.position;
            rotation = transform.rotation;
            if (tracksideShots.Count == 0) return false;

            float routeDistance = GetRouteDistance();
            Vector3 target = smoothedTargetPosition + Vector3.up * .8f;
            int best = -1;
            bool currentUsable = false;
            float bestScore = float.MaxValue;
            for (int i = 0; i < tracksideShots.Count; i++)
            {
                var candidate = tracksideShots[i];
                Vector3 offset = target - candidate.position;
                if (offset.sqrMagnitude < 100f || Mathf.Abs(candidate.routeDistance - routeDistance) > 110f) continue;
                if (Physics.Raycast(candidate.position, offset.normalized, out var hit,
                    Mathf.Max(0f, offset.magnitude - 2f), ~0, QueryTriggerInteraction.Ignore) &&
                    !hit.transform.IsChildOf(targetCar)) continue;
                if (i == tracksideShotIndex) currentUsable = true;
                float score = Mathf.Abs(candidate.routeDistance - routeDistance);
                if (i == tracksideShotIndex) score -= 22f;
                if (score < bestScore) { bestScore = score; best = i; }
            }
            if (best != tracksideShotIndex)
            {
                bool currentSafe = currentUsable;
                Camera cam = GetComponent<Camera>();
                Vector3 viewport = cam.WorldToViewportPoint(target);
                currentSafe &= viewport.z > 0f && viewport.x > .12f && viewport.x < .88f &&
                    viewport.y > .12f && viewport.y < .88f;
                if (!currentSafe || Time.unscaledTime - lastStationChange > 1.5f)
                {
                    tracksideShotIndex = best;
                    lastStationChange = Time.unscaledTime;
                }
            }
            if (tracksideShotIndex < 0) return false;
            TracksideShot shot = tracksideShots[tracksideShotIndex];
            position = shot.position;
            rotation = Quaternion.LookRotation(target - position, Vector3.up);
            return true;
        }

        float GetRouteDistance()
        {
            if (course == null || routeDistances == null || routeDistances.Length < 2) return 0f;
            Vector3 car = targetCar.position;
            float bestDistanceSq = float.MaxValue;
            float progress = 0f;
            for (int i = 0; i < routeDistances.Length - 1; i++)
            {
                Vector3 a = course.transform.TransformPoint(course.waypoints[i]);
                Vector3 b = course.transform.TransformPoint(course.waypoints[i + 1]);
                Vector2 span = new Vector2(b.x - a.x, b.z - a.z);
                Vector2 relative = new Vector2(car.x - a.x, car.z - a.z);
                float t = Mathf.Clamp01(Vector2.Dot(relative, span) / Mathf.Max(0.001f, span.sqrMagnitude));
                Vector2 nearest = new Vector2(a.x, a.z) + span * t;
                float distanceSq = (new Vector2(car.x, car.z) - nearest).sqrMagnitude;
                if (distanceSq >= bestDistanceSq) continue;
                bestDistanceSq = distanceSq;
                progress = routeDistances[i] + (routeDistances[i + 1] - routeDistances[i]) * t;
            }
            return progress;
        }
    }
}
