using System.IO;
using UnityEditor;
using UnityEngine;
using Audio;
using Core.Physics;
using UI;

namespace EditorScripts
{
    /// <summary>
    /// One-click build of the whole rally car: generates the mesh, livery and
    /// materials as real project assets, then assembles the GameObject hierarchy
    /// with the physics already dialled in from <see cref="ImprezaSpec"/>.
    ///
    /// Everything is idempotent. Run it as many times as you like — existing
    /// generated assets are updated in place rather than replaced, so material
    /// tweaks and scene references survive a rebuild.
    /// </summary>
    public static class RallyCarBuilder
    {
        const string GeneratedFolder = "Assets/Art/Generated";
        const string CarName = "RallyCar";

        static readonly string[] ManagedChildren =
        {
            "VisualChassis", "FL_Wheel", "FR_Wheel", "RL_Wheel", "RR_Wheel",
            "SteeringColumn",
            "Audio_Engine", "Audio_Tyres", "Audio_Wind"
        };

        class CarAssets
        {
            public Mesh Body, Wheel, SteeringWheel;
            public Texture2D Livery;
            public Material BodyMat, GlassMat, TrimMat, LampMat, TailMat, RimMat, TyreMat;

            public Material[] BodyMaterials => new[] { BodyMat, GlassMat, TrimMat, LampMat, TailMat };
            public Material[] WheelMaterials => new[] { RimMat, TyreMat };
        }

        // ══════════════════════════════════════════════════════════════
        //  MENU
        // ══════════════════════════════════════════════════════════════

        [MenuItem("Rally/Build Impreza WRC (Full Rebuild)", false, 0)]
        public static void BuildFull()
        {
            CarAssets assets = GenerateAssets();
            GameObject car = FindOrCreateCar();

            Undo.RegisterFullObjectHierarchyUndo(car, "Build Impreza WRC");

            DeformerSetup deformer = CaptureDeformer(car);
            ClearManagedChildren(car);
            ConfigureChassis(car);
            BuildVisualChassis(car, assets, deformer);
            BuildCorners(car, assets);
            BuildCockpit(car, assets);
            BuildAudio(car);

            VehicleController vc = car.GetComponent<VehicleController>();
            vc.ApplySpecToAllCorners();
            EditorUtility.SetDirty(car);

            Selection.activeGameObject = car;
            // ?. would skip Unity's == overload and call through a fake null.
            SceneView view = SceneView.lastActiveSceneView;
            if (view != null) view.FrameSelected();

            Debug.Log($"[Rally] Built 1998 Impreza WRC — {assets.Body.triangles.Length / 3} body tris, " +
                      $"{assets.Body.vertexCount} verts, {ImprezaSpec.Mass} kg, " +
                      $"{ImprezaSpec.FrontWeightBias * 100f:0}/{(1f - ImprezaSpec.FrontWeightBias) * 100f:0} weight split.");
        }

        [MenuItem("Rally/Rebuild Car Body Mesh Asset", false, 1)]
        public static void RebuildBodyMeshAsset()
        {
            EnsureFolder(GeneratedFolder);
            Mesh body = SaveAsset(AuthoredCarMesh.Body(), $"{GeneratedFolder}/Impreza_Body.asset");
            PrepareCockpitAssets();
            AssetDatabase.SaveAssets();
            Debug.Log($"[Rally] Regenerated body mesh asset: {body.vertexCount} vertices, {body.triangles.Length / 3} triangles.");
        }

        [MenuItem("Rally/Rebuild Visuals Only", false, 20)]
        public static void RebuildVisuals()
        {
            GameObject car = FindCar();
            if (car == null) { Debug.LogError("[Rally] No RallyCar in the scene. Run the full rebuild first."); return; }

            CarAssets assets = GenerateAssets();
            Undo.RegisterFullObjectHierarchyUndo(car, "Rebuild Impreza Visuals");

            DeformerSetup deformer = CaptureDeformer(car);
            Transform chassis = car.transform.Find("VisualChassis");
            if (chassis != null) Undo.DestroyObjectImmediate(chassis.gameObject);
            BuildVisualChassis(car, assets, deformer);

            foreach (var sus in car.GetComponentsInChildren<DynamicSuspension>())
                AttachWheelVisual(sus, assets);

            // The cockpit is visuals too, and rebuilding it in place rather than adding a
            // second column is why SteeringColumn is in ManagedChildren.
            Transform column = car.transform.Find("SteeringColumn");
            if (column != null) Undo.DestroyObjectImmediate(column.gameObject);
            BuildCockpit(car, assets);

            Debug.Log("[Rally] Visuals rebuilt.");
        }

        [MenuItem("Rally/Reapply Physics Spec", false, 21)]
        public static void ReapplySpec()
        {
            GameObject car = FindCar();
            if (car == null) { Debug.LogError("[Rally] No RallyCar in the scene."); return; }

            Undo.RegisterFullObjectHierarchyUndo(car, "Reapply Impreza Spec");

            var vc = car.GetComponent<VehicleController>();
            vc.mass = ImprezaSpec.Mass;
            vc.centerOfMass = ImprezaSpec.CenterOfMass;
            vc.inertiaTensor = ImprezaSpec.InertiaTensor;
            vc.ConfigureRigidbody();
            vc.ApplySpecToAllCorners();
            PositionCorners(car);
            ConfigureCollisionShell(car);

            EditorUtility.SetDirty(car);
            Debug.Log("[Rally] Physics spec reapplied to every corner.");
        }

        [MenuItem("Rally/Print Spec Sheet", false, 40)]
        public static void PrintSpec()
        {
            Debug.Log(
                "1998 SUBARU IMPREZA WRC — as simulated\n" +
                $"  Dimensions      {ImprezaSpec.Length:0.00} x {ImprezaSpec.Width:0.00} x {ImprezaSpec.Height:0.00} m, wheelbase {ImprezaSpec.Wheelbase:0.00} m\n" +
                $"  Mass            {ImprezaSpec.Mass:0} kg, {ImprezaSpec.FrontWeightBias * 100f:0}% front, CoM {ImprezaSpec.CoMHeight:0.000} m up / {ImprezaSpec.CoMZ:+0.000;-0.000} m fore\n" +
                $"  Inertia         pitch {ImprezaSpec.InertiaTensor.x:0} / yaw {ImprezaSpec.InertiaTensor.y:0} / roll {ImprezaSpec.InertiaTensor.z:0} kg.m2\n" +
                $"  Engine          470 N.m @ 3500-4000 rpm, ~300 bhp @ 5500 rpm, limiter {ImprezaSpec.RevLimitRpm:0}\n" +
                $"  Transmission    6-speed sequential, final drive {ImprezaSpec.FinalDrive:0.00}, AWD {ImprezaSpec.CentreTorqueSplitFront * 100f:0}/{(1f - ImprezaSpec.CentreTorqueSplitFront) * 100f:0}\n" +
                $"  Springs         front {ImprezaSpec.SpringRateFront / 1000f:0.0} kN/m ({ImprezaSpec.RideFrequencyFront:0.00} Hz), rear {ImprezaSpec.SpringRateRear / 1000f:0.0} kN/m ({ImprezaSpec.RideFrequencyRear:0.00} Hz)\n" +
                $"  Dampers         front {ImprezaSpec.DamperBumpFront:0}/{ImprezaSpec.DamperReboundFront:0}, rear {ImprezaSpec.DamperBumpRear:0}/{ImprezaSpec.DamperReboundRear:0} N.s/m (bump/rebound)\n" +
                $"  Static sag      front {ImprezaSpec.StaticSagFront * 1000f:0} mm, rear {ImprezaSpec.StaticSagRear * 1000f:0} mm\n" +
                $"  Anti-roll       front {ImprezaSpec.AntiRollFront / 1000f:0.0} kN/m, rear {ImprezaSpec.AntiRollRear / 1000f:0.0} kN/m\n" +
                $"  Tyres           {ImprezaSpec.WheelRadius * 2f:0.00} m dia, peak mu {ImprezaSpec.PeakFriction:0.00} at {ImprezaSpec.ReferenceLoad:0} N\n" +
                $"  Aero            Cd.A {ImprezaSpec.DragArea:0.00}, Cl.A {ImprezaSpec.LiftAreaFront + ImprezaSpec.LiftAreaRear:0.00} " +
                $"({(ImprezaSpec.LiftAreaFront + ImprezaSpec.LiftAreaRear) * 0.5f * ImprezaSpec.AirDensity * 40f * 40f / 9.81f:0} kg of downforce at 144 km/h)");
        }

        // ══════════════════════════════════════════════════════════════
        //  ASSETS
        // ══════════════════════════════════════════════════════════════
        static CarAssets GenerateAssets()
        {
            EnsureFolder(GeneratedFolder);
            PrepareCockpitAssets();

            var a = new CarAssets();

            a.Livery = SaveAsset(AuthoredCarLivery.Build(), $"{GeneratedFolder}/Impreza_555_Livery.asset");
            a.Livery.filterMode = FilterMode.Bilinear;
            a.Livery.wrapMode = TextureWrapMode.Clamp;
            a.Livery.anisoLevel = 4;

            a.Body  = SaveAsset(AuthoredCarMesh.Body(),  $"{GeneratedFolder}/Impreza_Body.asset");
            a.Wheel = SaveAsset(AuthoredCarMesh.Wheel(), $"{GeneratedFolder}/Impreza_Wheel.asset");
            a.SteeringWheel = SaveAsset(ImprezaMeshBuilder.BuildSteeringWheel(),
                                        $"{GeneratedFolder}/Impreza_SteeringWheel.asset");

            // Shared atlas UVs keep the hand-authored livery aligned across every panel;
            // material response separates paint, dark glass, trim, lenses and rubber.
            a.BodyMat  = MakeMaterial("Impreza_Body",     a.Livery, 0.55f, 34f, 0f);
            a.GlassMat = MakeMaterial("Impreza_Glass",    a.Livery, 1.30f, 70f, 0f);
            a.GlassMat.SetFloat("_Mode", 3);
            a.GlassMat.SetColor("_Color", new Color(.7f,.8f,.9f,.18f));
            a.GlassMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
            a.GlassMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            a.GlassMat.SetInt("_ZWrite", 0);
            a.GlassMat.SetOverrideTag("RenderType", "Transparent");
            a.GlassMat.EnableKeyword("_ALPHAPREMULTIPLY_ON");
            a.GlassMat.renderQueue = 3000;
            EditorUtility.SetDirty(a.GlassMat);
            a.TrimMat  = MakeMaterial("Impreza_Trim",     a.Livery, 0.08f, 12f, 0f);
            a.LampMat  = MakeMaterial("Impreza_Lamp",     a.Livery, 0.90f, 50f, 0.55f);
            a.TailMat  = MakeMaterial("Impreza_TailLamp", a.Livery, 0.60f, 40f, 0.75f);
            a.RimMat   = MakeMaterial("Impreza_Rim",      a.Livery, 0.95f, 42f, 0f);
            a.TyreMat  = MakeMaterial("Impreza_Tyre",     a.Livery, 0.05f, 8f,  0f);

            AssetDatabase.SaveAssets();
            return a;
        }

        static Material MakeMaterial(string name, Texture2D tex, float spec, float shininess, float emission)
        {
            Shader shader = Shader.Find("Standard (Specular setup)");
            Material m = new Material(shader) { name = name };
            m.mainTexture = tex;
            m.SetFloat("_Glossiness", Mathf.Clamp01(shininess / 128f));
            m.SetColor("_SpecColor", Color.white * Mathf.Clamp(spec * .08f, .004f, .16f));
            if (emission > 0f)
            {
                m.EnableKeyword("_EMISSION");
                m.SetColor("_EmissionColor", Color.white * emission);
            }

            return SaveAsset(m, $"{GeneratedFolder}/{name}.mat");
        }

        public static void PrepareCockpitAssets()
        {
            EnsureFolder("Assets/Resources/Vehicles");
            SaveAsset(AuthoredCarMesh.Cockpit(), "Assets/Resources/Vehicles/GC8_Cockpit.asset");
            SaveAsset(new Material(Shader.Find("Unlit/Color")), "Assets/Resources/Vehicles/Instrument.asset");
            AssetDatabase.SaveAssets();
        }

        public static void RefreshForReview()
        {
            var scene = UnityEditor.SceneManagement.EditorSceneManager.OpenScene("Assets/Scenes/RallyTraining.unity");
            RebuildVisuals();
            AssetDatabase.SaveAssets();
            // Shared assets update in place; never save physical scene changes here.
        }

        // ══════════════════════════════════════════════════════════════
        //  CHASSIS
        // ══════════════════════════════════════════════════════════════
        static void ConfigureChassis(GameObject car)
        {
            ComponentEnsure.EnsureUndo<Rigidbody>(car);

            var vc = ComponentEnsure.EnsureUndo<VehicleController>(car);
            vc.mass = ImprezaSpec.Mass;
            vc.centerOfMass = ImprezaSpec.CenterOfMass;
            vc.inertiaTensor = ImprezaSpec.InertiaTensor;
            vc.ConfigureRigidbody();

            // A root MeshRenderer is left over from whatever placeholder cube the car
            // used to be. Hide it rather than deleting it — it may be someone's marker.
            var rootRenderer = car.GetComponent<MeshRenderer>();
            if (rootRenderer != null) rootRenderer.enabled = false;

            // Collision shell: the cabin box only. It deliberately stops above the
            // arch line so the body never grounds out before the suspension does.
            ConfigureCollisionShell(car);
        }

        /// <summary>
        /// The crash shell. Its floor sits at 0.34 m, which is high enough that even
        /// a landing hard enough to crush the bump stops — the body drops 0.17 m
        /// relative to the wheels — never puts the floor pan through the road. A
        /// shell that grounds out fires a collision the agent reads as a crash.
        /// The low-friction bodywork material is added at runtime by
        /// VehicleController, so there is no stray asset to keep in sync.
        /// </summary>
        static void ConfigureCollisionShell(GameObject car)
        {
            var box = ComponentEnsure.EnsureUndo<BoxCollider>(car);
            Undo.RecordObject(box, "Configure Collision Shell");
            box.center = new Vector3(0f, 0.84f, (ImprezaSpec.NoseZ + ImprezaSpec.TailZ) * 0.5f);
            box.size = new Vector3(ImprezaSpec.Width, 1.00f, ImprezaSpec.Length);
        }

        static void BuildVisualChassis(GameObject car, CarAssets assets, DeformerSetup deformer)
        {
            var chassis = new GameObject("VisualChassis");
            Undo.RegisterCreatedObjectUndo(chassis, "Create VisualChassis");
            chassis.transform.SetParent(car.transform, false);
            chassis.transform.localPosition = Vector3.zero;
            chassis.transform.localRotation = Quaternion.identity;
            chassis.transform.localScale = Vector3.one;

            chassis.AddComponent<MeshFilter>().sharedMesh = assets.Body;
            var mr = chassis.AddComponent<MeshRenderer>();
            mr.sharedMaterials = assets.BodyMaterials;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;

            deformer.RestoreTo(chassis);
        }

        /// <summary>
        /// A rebuild replaces the whole VisualChassis object, which would silently
        /// take the damage deformer and its compute shader reference with it. Its
        /// settings are lifted off the old object and put back on the new one.
        /// </summary>
        struct DeformerSetup
        {
            public bool present;
            public ComputeShader compute;
            public float impactThreshold, deformationMultiplier, maxDeformation;

            public void RestoreTo(GameObject chassis)
            {
                var d = chassis.AddComponent<DamageDeformer>();

                // Defaults on a first build. This used to restore ONLY what a previous
                // build had, which meant a car that had never been given a deformer by
                // hand never got one — the feature was written, shipped a compute shader,
                // and was attached to nothing. Nobody had put it on the car, so nobody
                // could tell it did not work.
                d.deformationCompute = compute != null ? compute : DefaultComputeShader();
                d.impactThreshold = present ? impactThreshold : 10f;
                d.deformationMultiplier = present ? deformationMultiplier : 0.05f;
                d.maxDeformation = present ? maxDeformation : 0.5f;
            }

            /// <summary>
            /// The deformation kernel that has been sitting in the project unreferenced.
            /// A missing one is not fatal: DamageDeformer disables itself and says so.
            /// </summary>
            static ComputeShader DefaultComputeShader()
            {
                const string path = "Assets/Art/Shaders/Deformation.compute";
                var shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
                if (shader == null)
                    Debug.LogWarning($"[Rally] No compute shader at {path}; damage will stay off.");
                return shader;
            }
        }

        static DeformerSetup CaptureDeformer(GameObject car)
        {
            Transform t = car.transform.Find("VisualChassis");
            DamageDeformer d = t != null ? t.GetComponent<DamageDeformer>() : null;
            if (d == null) return default;

            return new DeformerSetup
            {
                present = true,
                compute = d.deformationCompute,
                impactThreshold = d.impactThreshold,
                deformationMultiplier = d.deformationMultiplier,
                maxDeformation = d.maxDeformation
            };
        }

        // ══════════════════════════════════════════════════════════════
        //  COCKPIT
        //
        //  A steering wheel, and the half of RobotDriverIK that can honestly be wired.
        //
        //  That script has been in the project since the start and has never done
        //  anything, because it needs three transforms — a steering wheel and two hand
        //  targets — and the car had none of them. The wheel is the part that can be
        //  built without inventing a driver: it is geometry the car actually has, and
        //  rotating it against steeringInput makes the steering visible, which matters
        //  when you are watching a policy drive and trying to tell lock from countersteer.
        //
        //  The hand targets stay null. RobotDriverIK returns early without a steering
        //  wheel and simply does not touch hands it has not been given, so this is the
        //  feature working at the fraction of it that has anything to act on. A driver
        //  model is a separate piece of work and is not pretended at here.
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// Column angle from vertical. A rally car's wheel is close to upright, far more
        /// so than a road car's — the driver sits high and holds it near the chest.
        /// </summary>
        const float SteeringColumnTiltDegrees = 22f;

        static void BuildCockpit(GameObject car, CarAssets assets)
        {
            // The column carries the tilt, so RobotDriverIK can keep writing a plain
            // localRotation about Z on the wheel without knowing the column exists.
            var column = new GameObject("SteeringColumn");
            Undo.RegisterCreatedObjectUndo(column, "Create SteeringColumn");
            column.transform.SetParent(car.transform, false);

            // Right-hand drive, seated behind the front axle, wheel at chest height.
            column.transform.localPosition = new Vector3(0.36f, 0.86f, ImprezaSpec.AxleFrontZ - 0.62f);
            column.transform.localRotation = Quaternion.Euler(SteeringColumnTiltDegrees, 0f, 0f);

            var wheel = new GameObject("SteeringWheel");
            Undo.RegisterCreatedObjectUndo(wheel, "Create SteeringWheel");
            wheel.transform.SetParent(column.transform, false);

            wheel.AddComponent<MeshFilter>().sharedMesh = assets.SteeringWheel;
            var mr = wheel.AddComponent<MeshRenderer>();
            mr.sharedMaterials = assets.WheelMaterials;
            // Inside the cabin, behind blacked-out glass: a shadow from it lands on
            // nothing anyone will see and costs a caster in every shadow pass.
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

            var ik = ComponentEnsure.EnsureUndo<RobotDriverIK>(car);
            ik.vehicle = car.GetComponent<VehicleController>();
            ik.steeringWheel = wheel.transform;
            // Lock to lock is 2.5 turns on a real car; this is the VISUAL range for a
            // steeringInput of -1..1, and matching the road wheels' actual travel reads
            // better than a realistic 450 degrees of blur.
            ik.maxWheelRotation = 180f;
        }

        // ══════════════════════════════════════════════════════════════
        //  CORNERS
        // ══════════════════════════════════════════════════════════════
        // ══════════════════════════════════════════════════════════════
        //  AUDIO
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// The three procedural sources, on their own nodes because an AudioSource is one
        /// per GameObject and these need to be heard from different places: the exhaust is
        /// behind you, the tyres are under you, and the wind is everywhere.
        ///
        /// Building them HERE rather than in a scene builder is deliberate — every scene
        /// in the project gets its car from this method, so none of them can end up as the
        /// one that silently has no engine. Training is not a reason to leave them out
        /// either: <see cref="Audio.AudioGate"/> shuts the synthesis off entirely while a
        /// trainer is attached, so the cost of their being present is an idle callback.
        /// </summary>
        static void BuildAudio(GameObject car)
        {
            var vc = car.GetComponent<VehicleController>();

            // At the tailpipe. On a chase camera this is the near end of the car, which is
            // what puts the exhaust in front of the listener where it belongs.
            var engine = AudioNode(car, "Audio_Engine", new Vector3(0f, 0.34f, ImprezaSpec.TailZ + 0.05f));
            var es = engine.AddComponent<EngineSynth>();
            es.vehicle = vc;
            es.volume = EngineSynth.DefaultVolume;
            es.spatialBlend = 0.85f;
            es.maxDistance = 220f;      // a rally car is audible a long way off

            // Under the floor, between the axles: the sum of four contact patches has no
            // one place it comes from, and the middle is the honest answer.
            var tyres = AudioNode(car, "Audio_Tyres", new Vector3(0f, 0.12f, 0f));
            var ts = tyres.AddComponent<TyreSynth>();
            ts.vehicle = vc;
            ts.volume = TyreSynth.DefaultVolume;
            ts.spatialBlend = 0.75f;
            ts.maxDistance = 90f;       // stones carry nothing like as far as an exhaust

            // Wind is the one sound that is not coming from the car — it is the listener's
            // own airflow. Nearly unpositioned, and it does not attenuate with distance the
            // way a point source would.
            var wind = AudioNode(car, "Audio_Wind", new Vector3(0f, 1.05f, 0f));
            var ws = wind.AddComponent<WindSynth>();
            ws.vehicle = vc;
            ws.volume = WindSynth.DefaultVolume;
            ws.spatialBlend = 0.20f;
            ws.maxDistance = 300f;
        }

        static GameObject AudioNode(GameObject car, string name, Vector3 localPosition)
        {
            var node = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(node, "Create Audio Source");
            node.transform.SetParent(car.transform, false);
            node.transform.localPosition = localPosition;
            node.transform.localRotation = Quaternion.identity;
            node.transform.localScale = Vector3.one;
            return node;
        }

        static void BuildCorners(GameObject car, CarAssets assets)
        {
            CreateCorner(car, assets, DynamicSuspension.Corner.FrontLeft,  "FL_Wheel");
            CreateCorner(car, assets, DynamicSuspension.Corner.FrontRight, "FR_Wheel");
            CreateCorner(car, assets, DynamicSuspension.Corner.RearLeft,   "RL_Wheel");
            CreateCorner(car, assets, DynamicSuspension.Corner.RearRight,  "RR_Wheel");
        }

        static void CreateCorner(GameObject car, CarAssets assets, DynamicSuspension.Corner corner, string name)
        {
            var node = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(node, "Create Suspension Corner");
            node.transform.SetParent(car.transform, false);
            node.transform.localRotation = Quaternion.identity;
            node.transform.localScale = Vector3.one;
            node.transform.localPosition = AnchorFor(corner);

            var sus = node.AddComponent<DynamicSuspension>();
            sus.corner = corner;
            sus.ApplySpec();

            var tire = node.AddComponent<PacejkaTireModel>();
            tire.ApplySpec(sus.IsFront, driven: true);

            AttachWheelVisual(sus, assets);
        }

        /// <summary>
        /// The strut's upper mount. Y comes straight out of the spec: wheel radius
        /// plus the free strut length minus the sag it takes up holding the car,
        /// which is what puts the wheel exactly on the ground at rest.
        /// </summary>
        static Vector3 AnchorFor(DynamicSuspension.Corner corner)
        {
            bool front = corner == DynamicSuspension.Corner.FrontLeft || corner == DynamicSuspension.Corner.FrontRight;
            bool left  = corner == DynamicSuspension.Corner.FrontLeft || corner == DynamicSuspension.Corner.RearLeft;

            float x = (left ? -1f : 1f) * (front ? ImprezaSpec.HalfTrackFront : ImprezaSpec.HalfTrackRear);
            float y = front ? ImprezaSpec.StrutAnchorYFront : ImprezaSpec.StrutAnchorYRear;
            float z = front ? ImprezaSpec.AxleFrontZ : ImprezaSpec.AxleRearZ;
            return new Vector3(x, y, z);
        }

        static void PositionCorners(GameObject car)
        {
            foreach (var sus in car.GetComponentsInChildren<DynamicSuspension>())
            {
                Undo.RecordObject(sus.transform, "Position Corner");
                sus.transform.localPosition = AnchorFor(sus.corner);
                sus.transform.localRotation = Quaternion.identity;
            }
        }

        static void AttachWheelVisual(DynamicSuspension sus, CarAssets assets)
        {
            for (int i = sus.transform.childCount - 1; i >= 0; i--)
            {
                var child = sus.transform.GetChild(i).gameObject;
                if (child.name == "WheelMesh" || child.name == "ProceduralWheel")
                    Undo.DestroyObjectImmediate(child);
            }

            var wheel = new GameObject("WheelMesh");
            Undo.RegisterCreatedObjectUndo(wheel, "Create Wheel Mesh");
            wheel.transform.SetParent(sus.transform, false);

            // Static pose for the editor. At runtime DynamicSuspension drives this
            // transform every physics step, camber and all.
            float sag = sus.IsFront ? ImprezaSpec.StaticSagFront : ImprezaSpec.StaticSagRear;
            wheel.transform.localPosition = new Vector3(0f, -(sus.restLength - sag), 0f);
            wheel.transform.localRotation = Quaternion.Euler(0f, 0f, (sus.IsLeft ? 1f : -1f) * sus.camberAngle);
            wheel.transform.localScale = Vector3.one;

            wheel.AddComponent<MeshFilter>().sharedMesh = assets.Wheel;
            wheel.AddComponent<MeshRenderer>().sharedMaterials = assets.WheelMaterials;

            Undo.RecordObject(sus, "Assign Wheel Mesh");
            sus.wheelMesh = wheel.transform;
        }

        // ══════════════════════════════════════════════════════════════
        //  IMPORTED MODEL  (kept from the original workflow)
        // ══════════════════════════════════════════════════════════════
        [MenuItem("Rally/Attach Imported 3D Model", false, 60)]
        static void AttachImportedModel()
        {
            GameObject car = FindCar();
            if (car == null) { Debug.LogError("[Rally] No RallyCar in the scene."); return; }

            const string path = "Assets/Models/SubaruImpreza/ImprezaRally/ImprezaRally.obj";
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (prefab == null) { Debug.LogError($"[Rally] No model at {path}."); return; }

            Transform old = car.transform.Find("VisualChassis");
            if (old != null) Undo.DestroyObjectImmediate(old.gameObject);

            var model = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            Undo.RegisterCreatedObjectUndo(model, "Attach Imported Model");
            model.name = "VisualChassis";
            model.transform.SetParent(car.transform, false);
            model.transform.localRotation = Quaternion.Euler(0f, 180f, 0f);

            // Scale the import to the real car's length, then sit it on the ground —
            // the rig's origin is the ground plane, so the model's floor goes to y=0.
            Bounds bounds = new Bounds();
            bool first = true;
            foreach (var mf in model.GetComponentsInChildren<MeshFilter>())
            {
                if (mf.sharedMesh == null) continue;
                if (first) { bounds = mf.sharedMesh.bounds; first = false; }
                else bounds.Encapsulate(mf.sharedMesh.bounds);
            }

            if (!first && bounds.size.z > 0.01f)
            {
                float scale = ImprezaSpec.Length / bounds.size.z;
                model.transform.localScale = Vector3.one * scale;
                model.transform.localPosition = new Vector3(0f, -bounds.min.y * scale, 0f);
            }

            // Any collider inside the visual model would be hit by the suspension
            // raycasts, and the car would launch itself into orbit.
            foreach (var col in model.GetComponentsInChildren<Collider>())
                Undo.DestroyObjectImmediate(col);

            // Hide the model's baked-in wheels; the rig supplies its own.
            foreach (var mr in model.GetComponentsInChildren<MeshRenderer>())
                if (mr.gameObject.name.ToLower().Contains("wheel")) mr.enabled = false;

            Debug.Log("[Rally] Imported model attached and scaled to 4.34 m. Physics rig untouched.");
        }

        // ══════════════════════════════════════════════════════════════
        //  HELPERS
        // ══════════════════════════════════════════════════════════════
        static GameObject FindCar()
        {
            if (Selection.activeGameObject != null)
            {
                var sel = Selection.activeGameObject.GetComponentInParent<VehicleController>();
                if (sel != null) return sel.gameObject;
            }

            var existing = Object.FindAnyObjectByType<VehicleController>();
            if (existing != null) return existing.gameObject;

            return GameObject.Find(CarName);
        }

        static GameObject FindOrCreateCar()
        {
            GameObject car = FindCar();
            if (car != null) return car;

            car = new GameObject(CarName);
            Undo.RegisterCreatedObjectUndo(car, "Create RallyCar");
            car.transform.position = new Vector3(0f, 0.05f, 0f);
            return car;
        }

        static void ClearManagedChildren(GameObject car)
        {
            for (int i = car.transform.childCount - 1; i >= 0; i--)
            {
                Transform child = car.transform.GetChild(i);
                foreach (string managed in ManagedChildren)
                {
                    if (child.name == managed)
                    {
                        Undo.DestroyObjectImmediate(child.gameObject);
                        break;
                    }
                }
            }
        }

        static void EnsureFolder(string folder)
        {
            if (AssetDatabase.IsValidFolder(folder)) return;

            string parent = Path.GetDirectoryName(folder).Replace('\\', '/');
            string leaf = Path.GetFileName(folder);
            EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, leaf);
        }

        /// <summary>
        /// Writes a generated object to disk. If something is already at that path
        /// its contents are overwritten in place, so every scene and material
        /// reference pointing at it stays valid across rebuilds.
        /// </summary>
        static T SaveAsset<T>(T generated, string path) where T : Object
        {
            T existing = AssetDatabase.LoadAssetAtPath<T>(path);
            if (existing != null && existing != generated)
            {
                EditorUtility.CopySerialized(generated, existing);
                EditorUtility.SetDirty(existing);
                Object.DestroyImmediate(generated);
                return existing;
            }

            AssetDatabase.CreateAsset(generated, path);
            return generated;
        }
    }
}
