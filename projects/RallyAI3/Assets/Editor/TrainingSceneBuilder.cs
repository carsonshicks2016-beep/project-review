using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.InferenceEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Core.Physics;
using Core.Environment;
using Core.ML;
using UI;

namespace EditorScripts
{
    /// <summary>
    /// Builds the whole training scene from nothing and saves it to disk.
    ///
    /// The point is that the scene stops being something that exists only in the editor's
    /// backup folder. Everything here — the track, the car, the agent wiring, the sensor
    /// config — is reconstructed from code, so the scene asset is reproducible, reviewable
    /// in a diff, and impossible to lose by closing Unity without saving.
    ///
    /// Re-running it rebuilds the scene in place. That is the intended way to change the
    /// setup: edit this file, run the menu item, commit the result.
    /// </summary>
    public static class TrainingSceneBuilder
    {
        const string ScenesFolder = "Assets/Scenes";
        const string ScenePath    = ScenesFolder + "/RallyTraining.unity";
        const string GeneratedFolder = "Assets/Art/Generated";

        /// <summary>
        /// The one colour the fog, the horizon and the base of the far hills all have to
        /// agree on. The whole depth illusion is that distant geometry fades into exactly
        /// the colour the sky already is at that height; if these drift apart the ridge
        /// line stops being distant and becomes a wall standing in front of the sky.
        /// </summary>
        static readonly Color Haze = new Color(0.71f, 0.76f, 0.79f);

        /// <summary>Must match the behaviour name in the training YAML.</summary>
        const string BehaviorName = "RallyDriver";

        /// <summary>
        /// Trained policy to drop into Behaviour Parameters, if it is on disk.
        ///
        /// Assigning it HERE rather than by hand is the difference between a scene that
        /// drives itself and one that silently doesn't. With BehaviorType.Default a model
        /// is only consulted when no trainer is attached, so this costs training nothing —
        /// mlagents-learn still takes control the moment it connects — but it means
        /// pressing Play shows the trained driver instead of falling back to the keyboard
        /// heuristic, which is what an unassigned field gets you and gives no warning about.
        ///
        /// Deliberately a FIXED name rather than a run id. A policy only loads into a scene
        /// whose observation shape matches the one it was trained against, and this project
        /// has changed that shape twice — 22 to 28 vector floats, then one ray fan to two.
        /// Pointing this at a specific run means every later change silently wires up a model
        /// that cannot run: rally05 sat here through both changes, so a rebuild was attaching
        /// a 22-float policy to a 28-float scene. Copy whichever policy currently matches the
        /// scene to this path instead, and the builder stays correct across rebuilds.
        /// </summary>
        const string TrainedModelPath = "Assets/ML-Agents/Models/RallyDriver-current.onnx";

        // ══════════════════════════════════════════════════════════════
        //  MENU
        // ══════════════════════════════════════════════════════════════

        [MenuItem("Rally/Training/Build Training Scene", false, 0)]
        public static void BuildAndSave()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                EditorUtility.DisplayDialog("Rally", "Stop play mode before rebuilding the scene.", "OK");
                return;
            }

            // ── Both prompts are skipped in batch mode, and that is not a shortcut.
            //
            //    Unity refuses to show a dialog with no human at the keyboard: it logs
            //    "Canceling DisplayDialog … This should not be called when not controlled
            //    by a human" and returns FALSE. So the guard meant to protect the scene
            //    silently ABORTED every unattended build, exit code 0, no error — the
            //    build looked like it worked and the scene was never rebuilt.
            //
            //    Running `Unity -batchmode -executeMethod …BuildAndSave` is itself the
            //    deliberate act the dialog exists to obtain. Asking again cannot succeed.
            if (!Application.isBatchMode)
            {
                if (File.Exists(ScenePath) &&
                    !EditorUtility.DisplayDialog(
                        "Rebuild training scene?",
                        $"This replaces {ScenePath} with a freshly built scene.\n\n" +
                        "Anything you changed by hand in that scene is lost. Continue?",
                        "Rebuild", "Cancel"))
                    return;

                // NewScene(..., Single) throws away whatever is open WITHOUT asking, and
                // the scene that is open right now has only ever existed in the editor's
                // backup folder. Give the user the save prompt first and honour a cancel.
                if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            }

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            ConfigureProjectForTraining();
            BuildLighting();
            TrackGenerator track = BuildTrack();
            GameObject car = BuildCar();
            ConfigureAgent(car, track);
            BuildCamera(car);
            BuildTelemetryHud(car);

            EnsureFolder(ScenesFolder);
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();

            AddSceneToBuildSettings();

            Selection.activeGameObject = car;
            Debug.Log(
                $"[Rally] Training scene built and saved to {ScenePath}.\n" +
                $"  Behaviour   {BehaviorName}, {RallyAgent.VectorObservationSize} vector obs, {RallyAgent.ContinuousActionCount} continuous actions\n" +
                $"  Track       seed {track.seed}, {track.trackLength:0} m, {track.roadWidth:0} m wide\n" +
                $"  Train with  mlagents-learn config/rally_ppo.yaml --run-id=rally01\n" +
                $"  then press Play.");
        }

        [MenuItem("Rally/Training/Open Training Scene", false, 1)]
        public static void OpenScene()
        {
            if (!File.Exists(ScenePath))
            {
                Debug.LogError($"[Rally] No scene at {ScenePath}. Run Rally/Training/Build Training Scene first.");
                return;
            }
            EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo();
            EditorSceneManager.OpenScene(ScenePath);
        }

        // ══════════════════════════════════════════════════════════════
        //  PIECES
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// Project-wide settings training cannot work without.
        ///
        /// runInBackground is the one that matters. With it off — Unity's default — play mode
        /// PAUSES whenever the editor is not the frontmost application, so training silently
        /// stops the instant you click on anything else and resumes when you click back. It
        /// does not error and it does not disconnect: mlagents-learn sits waiting for
        /// observations that never arrive, and the run looks frozen for no visible reason.
        /// That cost a real run several minutes of apparent hang before the cause was found.
        ///
        /// Set here rather than left as a note in the README because it is not a preference —
        /// no editor training run works correctly without it.
        /// </summary>
        static void ConfigureProjectForTraining()
        {
            if (PlayerSettings.runInBackground) return;
            PlayerSettings.runInBackground = true;
            Debug.Log("[Rally] Enabled Player Settings > Run In Background. Without it, play " +
                      "mode pauses whenever the editor loses focus and training silently stalls.");
        }

        static void BuildLighting()
        {
            var sun = new GameObject("Directional Light");
            var light = sun.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = new Color(1f, 0.96f, 0.88f);
            light.intensity = 1.1f;
            light.shadows = LightShadows.Soft;
            sun.transform.rotation = Quaternion.Euler(42f, -35f, 0f);

            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.55f, 0.62f, 0.75f);
            RenderSettings.ambientEquatorColor = new Color(0.42f, 0.44f, 0.46f);
            RenderSettings.ambientGroundColor = new Color(0.26f, 0.24f, 0.22f);
            RenderSettings.sun = light;

            // Fog, and not for atmosphere's own sake. The ground stops 45 m either side of
            // the centreline and the camera sees out to 1200 m; fog is how hardware that
            // could not draw the difference always dealt with exactly this, and both world
            // shaders have carried multi_compile_fog since they were written without
            // anything ever switching it on.
            //
            // Exponential-squared rather than linear: it leaves the near ground crisp
            // (3% at 45 m) instead of laying haze over the road, and still reaches 76% by
            // the 400 m ridge, which is what turns the backdrop into distance rather than
            // scenery. Nothing here touches training — the agent observes vectors and
            // raycasts, and headless runs do not render at all.
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogDensity = 0.0030f;
            RenderSettings.fogColor = Haze;

            RenderSettings.skybox = SkyboxMaterial();
        }

        /// <summary>
        /// A restrained built-in skybox that matches the live viewer lighting.
        /// </summary>
        static Material SkyboxMaterial()
        {
            const string path = GeneratedFolder + "/Sky.mat";

            Shader sky = Shader.Find("Skybox/Procedural");
            if (sky == null)
            {
                Debug.LogWarning("[Rally] Procedural skybox shader not found — keeping the default sky.");
                return RenderSettings.skybox;
            }

            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat == null)
            {
                mat = new Material(sky) { name = "Sky" };
                EnsureFolder(GeneratedFolder);
                AssetDatabase.CreateAsset(mat, path);
            }
            else if (mat.shader != sky) mat.shader = sky;

            mat.SetFloat("_AtmosphereThickness", 1f);
            mat.SetFloat("_Exposure", 1.05f);
            mat.SetColor("_SkyTint", new Color(0.56f, 0.68f, 0.80f));
            mat.SetColor("_GroundColor", new Color(0.40f, 0.42f, 0.38f));
            EditorUtility.SetDirty(mat);
            return mat;
        }

        static TrackGenerator BuildTrack()
        {
            var go = new GameObject("Track");
            go.transform.position = Vector3.zero;

            // TrackGenerator requires these and fills them in itself on Awake.
            go.AddComponent<MeshFilter>();
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = WorldMaterial("Gravel", WorldTexture("World_Gravel", WorldTextureBuilder.BuildGravel));

            var track = go.AddComponent<TrackGenerator>();
            track.seed = 20260727;
            // Left OFF deliberately, and it is not the same decision it looks like. The scene
            // asset keeps one fixed, diffable, reproducible stage; RallyAgent draws a fresh
            // seed at runtime every stageRefreshEpisodes instead. Switching this on would only
            // randomise the stage ONCE per process — GenerateTrack runs from Awake and never
            // again — which with --num-envs 6 is six stages for a ten-million-step run.
            track.randomiseSeed = false;

            // Rocks ON. They are part of what a rally stage IS, so they are in the task from
            // the first step; what ramps is the DENSITY, via the obstacle_density curriculum
            // in config/rally_ppo.yaml, which RallyAgent reads on each stage refresh.
            //
            // Worth being precise about what this flag does: it governs rocks lying ON THE
            // ROAD. The stage's scenery — 259 trees and 77 boulders on this seed, every one of
            // them tagged Obstacle and episode-ending — is scatterProps below, and is present
            // at every lesson including the first. "Rocks off" was never a clear stage.
            track.spawnObstacles = true;
            // Per 100 m, not per spline sample, so the stage's difficulty no longer moves
            // when the tessellation below does. 2 per 100 m is the density the old
            // per-sample 0.10 worked out to at the resolution it was written for.
            track.obstaclesPer100m = 2f;
            track.obstacleSpawnClearance = 40f;

            // Tessellation, for the affine texture mapping's sake rather than for shape:
            // 1.25 m along the stage and 1.5 m across the road. PS1_Lit interpolates UVs
            // linearly in screen space, and the error in doing so is a function of how much
            // depth a single triangle spans — so big road quads made the gravel bend and swim
            // across the bottom of the frame. Small quads are how the hardware's own games
            // dealt with it.
            track.resolutionPerSegment = 40;
            track.roadLanes = 8;

            // Terrain: the road is cut into a landscape. This is the car's only boundary
            // AND the only thing the ray sensor can range against, so it is not scenery.
            track.buildTerrain = true;
            track.terrainHalfWidth = 45f;
            track.terrainResolution = 16;
            track.terrainRelief = 14f;
            track.terrainFrequency = 0.003f;
            track.dramaFrequency = 0.0015f;
            track.ditchDepth = 0.7f;
            track.shoulderWidth = 3.5f;
            track.terrainRoughness = 1.6f;
            track.terrainMaterial = WorldMaterial("Terrain", WorldTexture("World_Scrub", WorldTextureBuilder.BuildScrub));

            // Texture scale, in metres of ground per repeat. Laid out in real distance
            // rather than per-strip, so the ground does not change scale as it recedes.
            track.roadTileMetres = 6f;
            track.terrainTileMetres = 12f;
            track.rockTileMetres = 1.4f;

            // The far hills. Decorative in the strict sense — no collider, no tag, out of
            // the ray sensor's range — so this changes what the stage looks like without
            // changing the problem the agent is solving or invalidating earlier runs.
            track.buildBackdrop = true;
            track.backdropRadius = 400f;      // ~76% fogged at the density set in BuildLighting
            track.backdropHeight = 70f;
            track.backdropDepth = 300f;
            track.backdropSegments = 48;
            track.backdropMaterial = WorldMaterial("Backdrop", WorldTexture("World_Backdrop", WorldTextureBuilder.BuildBackdrop));

            // Scenery. Collidable and tagged Obstacle, so this is part of the task, not
            // set dressing — it is what stops a drop being merely slow.
            track.scatterProps = true;
            track.propClearance = 13f;
            track.treesPer100m = 26f;
            track.rocksPer100m = 8f;
            track.treeMaterial = WorldMaterial("Foliage", WorldTexture("World_Foliage", WorldTextureBuilder.BuildFoliage));
            track.rockMaterial = WorldMaterial("Boulder", WorldTexture("World_Stone", WorldTextureBuilder.BuildStone));

            // Build it now so the saved scene has waypoints in it and the spawn can be
            // resolved in the editor rather than only at runtime.
            track.GenerateTrack();
            return track;
        }

        /// <summary>
        /// Uses the Built-in Render Pipeline's textured lit material for world surfaces.
        /// </summary>
        static Shader WorldShader()
        {
            Shader standard = Shader.Find("Standard");
            if (standard != null) return standard;
            Debug.LogWarning("[Rally] Standard shader unavailable; using Diffuse for world surfaces.");
            return Shader.Find("Diffuse");
        }

        /// <summary>
        /// A world material with one generated albedo texture.
        ///
        /// Reuses the asset already on disk so scene references survive a rebuild, but
        /// REPAIRS it rather than trusting it. A builder that returned whatever it found
        /// is exactly how four materials stayed silently bound to the wrong shader.
        /// </summary>
        static Material WorldMaterial(string name, Texture2D tex)
        {
            string path = $"{GeneratedFolder}/{name}.mat";
            Shader shader = WorldShader();

            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat == null)
            {
                mat = new Material(shader) { name = name };
                EnsureFolder(GeneratedFolder);
                AssetDatabase.CreateAsset(mat, path);
            }
            else if (mat.shader != shader)
            {
                mat.shader = shader;
            }

            if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", tex);
            // Only reachable on the Standard fallback; harmless and correct there.
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", Color.white);
            if (mat.HasProperty("_Glossiness")) mat.SetFloat("_Glossiness", 0.03f);
            if (mat.HasProperty("_Metallic")) mat.SetFloat("_Metallic", 0f);

            EditorUtility.SetDirty(mat);
            return mat;
        }

        /// <summary>
        /// Generates a world texture and writes it to disk, overwriting in place so that
        /// every material already pointing at that path keeps pointing at it.
        /// </summary>
        static Texture2D WorldTexture(string name, System.Func<Texture2D> build)
        {
            EnsureFolder(GeneratedFolder);
            string path = $"{GeneratedFolder}/{name}.asset";

            Texture2D generated = build();
            var existing = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            if (existing != null)
            {
                EditorUtility.CopySerialized(generated, existing);
                EditorUtility.SetDirty(existing);
                Object.DestroyImmediate(generated);
                return existing;
            }

            AssetDatabase.CreateAsset(generated, path);
            return generated;
        }

        static GameObject BuildCar()
        {
            // RallyCarBuilder owns the rig: mesh, materials, corners, physics spec. Reusing
            // it here means the training scene can never drift from the car the rest of the
            // project builds.
            RallyCarBuilder.BuildFull();

            var vehicle = Object.FindAnyObjectByType<VehicleController>();
            if (vehicle == null)
            {
                Debug.LogError("[Rally] RallyCarBuilder did not produce a VehicleController.");
                return null;
            }
            return vehicle.gameObject;
        }

        static void ConfigureAgent(GameObject car, TrackGenerator track)
        {
            if (car == null) return;

            // ── Auto-reverse OFF, and this is a trap rather than a preference.
            //
            //    The gearbox drops into reverse when the brake is held above 0.5 at a
            //    standstill. The agent's drive axis makes brake = max(0, -drive), so an
            //    untrained policy outputting a slightly negative mean sits on the brake and
            //    selects reverse. Escaping reverse needs throttle > 0.2 AND ForwardSpeed
            //    > -0.5 — but applying throttle in reverse accelerates the car BACKWARDS, so
            //    ForwardSpeed falls through -0.5 and the escape condition stops being true.
            //    Meanwhile reversing away from the next waypoint earns negative progress
            //    reward, so the one action that could get the car out is the one punished.
            //
            //    Observed directly: 121 of 124 episodes timed out with the car stationary in
            //    gear R holding 0.52 of brake. Reverse is not a capability an agent with a
            //    single drive axis can use; it is only somewhere to get stuck.
            var vehicle = car.GetComponent<VehicleController>();
            if (vehicle != null) vehicle.allowAutoReverse = false;

            // ── Behaviour Parameters. The observation size has to match exactly what
            //    CollectObservations writes, which is why RallyAgent owns that constant.
            var bp = ComponentEnsure.Ensure<BehaviorParameters>(car);
            bp.BehaviorName = BehaviorName;
            bp.BrainParameters.VectorObservationSize = RallyAgent.VectorObservationSize;
            bp.BrainParameters.NumStackedVectorObservations = 3;   // gives the policy a sense of rate
            bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(RallyAgent.ContinuousActionCount); // steer, drive
            bp.BehaviorType = BehaviorType.Default;                // trainer if present, else heuristic

            // ── The agent itself.
            var agent = ComponentEnsure.Ensure<RallyAgent>(car);

            // ── 12000, and the old 6000 was a measurement error rather than a choice.
            //
            //    MaxStep counts PHYSICS steps, and this project's fixed timestep is 0.01 s,
            //    not the 0.02 every comment in the codebase claimed. So "6000 = 120 s at
            //    50 Hz" was really 60 s at 100 Hz — every episode ran for half the time it
            //    was designed to, on a 1 km stage, and a policy that brakes properly for
            //    corners was competing against a clock nobody knew was there.
            //
            //    Doubling MaxStep restores the intended 120 s without touching the timestep.
            //    Halving the timestep to 0.02 would also have done it, and would have made
            //    every comment true at a stroke, but it changes the dynamics of a stiff
            //    tyre-and-suspension integration that is currently stable, and it would
            //    invalidate every policy trained so far. Not worth it to fix a comment.
            //
            //    RallyAgent logs the real rate at startup now, so this cannot drift again.
            agent.MaxStep = 12000;             // 120 s at the actual 100 Hz
            agent.spawnHeight = 0.05f;
            agent.respawnGraceSeconds = 1.0f;
            agent.crashDeltaV = 4f;
            agent.timePenaltyPerSecond = 0.05f;
            agent.waypointBonus = 5f;
            agent.finishBonus = 10f;
            agent.failurePenalty = 1f;
            // 18 m rather than 30. A constraint correction, not reward shaping: 30 m left 24 m
            // of open scenery beyond the road edge where no fast line can possibly live, so
            // episodes were free to wander out and explore terrain that could never pay. This
            // still leaves ~12 m of shoulder and verge, so a line that clips the dirt survives.
            agent.offStageDistance = 18f;
            agent.fallOutY = -60f;

            // Abandon an episode after 5 s of not moving. A parked car otherwise burns a full
            // 12000-step episode producing identical, uninformative samples — on the run that
            // exposed this, 745k steps bought 124 episodes of which only 3 involved movement.
            // The forfeited time penalty is charged on the way out, so bailing early is exactly
            // as expensive as sitting there and the incentives do not move.
            agent.stallSpeedThreshold = 1.0f;
            agent.stallTimeoutSeconds = 5f;

            // A brand-new stage every 15 episodes. This is what stops the policy memorising
            // one course — which is exactly what rally05 did, scoring 85 % stage completion on
            // a fixed seed while having no way to perceive the road at all. 15 rather than
            // every episode because a refresh re-cooks two MeshColliders and the terrain mesh
            // is ~51k vertices since the tessellation change.
            agent.stageRefreshEpisodes = 15;
            agent.obstacleDensityParameter = "obstacle_density";
            // Off: with six headless envs this is six processes writing a line per
            // episode to logs nobody reads. Outcomes now go to TensorBoard via
            // StatsRecorder, and the editor Training Monitor covers the live view.
            agent.logEpisodeEnds = false;

            // ── Decisions. Every 5 physics steps, with the last action held in between.
            //    At the real 0.01 s timestep that is 20 Hz, not the 10 Hz the comment here
            //    used to claim — a 50 ms reaction time, which is if anything generous for a
            //    driving task. Left at 5: 20 Hz costs nothing the run cannot afford, and
            //    changing the decision rate changes what an action MEANS, which would
            //    invalidate every policy trained so far for no measured gain.
            var dr = ComponentEnsure.Ensure<DecisionRequester>(car);
            dr.DecisionPeriod = 5;
            dr.TakeActionsBetweenDecisions = true;

            // ══════════════════════════════════════════════════════════════
            //  SIGHT — two fans, because one cannot do both jobs
            //
            //  A single ±80° fan of 17 rays puts 10° between neighbours. Angular gaps grow
            //  with range, so at 10 m adjacent rays are already 1.75 m apart while a 1 m road
            //  rock plus the cast radius is only ~1.7 m wide. Past about 9 m the rocks fall
            //  BETWEEN the rays and are simply not there. At 25 m/s that left 0.4 s of
            //  warning, and rally08 duly plateaued with 54-67 % of episodes ending on an
            //  obstacle once the curriculum reached 2 rocks per 100 m.
            //
            //  Note this replaces an earlier plan to angle fans downward for road-edge
            //  detection. That is no longer needed: the road-relative vector observations
            //  took "left stage" from 73 % to 6 %, so the wide fan's original job is done and
            //  its resolution is better spent forward.
            //
            //  The split is therefore: a narrow, fine fan that must resolve a rock far enough
            //  ahead to steer around it, and a coarse wide one for peripheral awareness.
            // ══════════════════════════════════════════════════════════════
            var detectable = new System.Collections.Generic.List<string> { "Obstacle", "TrackBoundary" };

            // ── Forward: 21 rays over ±25°, so 2.5° apart. At 30 m that is a 1.31 m gap
            //    against a ~2.2 m detection width — a rock cannot hide in it. ±25° still
            //    spans ±14 m at 30 m, which is the road plus both verges.
            var forward = SensorNode(car, "Sensor_Forward");
            var fwdRay = ComponentEnsure.Ensure<RayPerceptionSensorComponent3D>(forward);
            fwdRay.SensorName = "ForwardSensor";
            fwdRay.DetectableTags = detectable;
            fwdRay.RaysPerDirection = 10;         // 21 rays
            fwdRay.MaxRayDegrees = 25f;
            fwdRay.RayLength = 60f;
            // 0.6 rather than 0.35: it widens each ray to ~2.2 m, which is what turns the
            // resolution figures above from marginal into comfortable. It errs toward false
            // positives, and that is the safe direction — a phantom rock costs a swerve, a
            // missed one ends the episode.
            fwdRay.SphereCastRadius = 0.6f;
            fwdRay.StartVerticalOffset = 0f;
            fwdRay.EndVerticalOffset = 0f;
            fwdRay.RayLayerMask = ~0;

            // ── Peripheral: the old wide fan, thinned from 17 rays to 11. It is no longer
            //    responsible for finding the road, only for noticing something out to the
            //    side, so the rays it gives up pay for the forward fan's resolution.
            var wide = SensorNode(car, "Sensor_Wide");
            var wideRay = ComponentEnsure.Ensure<RayPerceptionSensorComponent3D>(wide);
            wideRay.SensorName = "WideSensor";
            wideRay.DetectableTags = detectable;
            wideRay.RaysPerDirection = 5;         // 11 rays
            wideRay.MaxRayDegrees = 80f;
            wideRay.RayLength = 45f;
            wideRay.SphereCastRadius = 0.5f;
            wideRay.StartVerticalOffset = 0f;
            wideRay.EndVerticalOffset = 0f;
            wideRay.RayLayerMask = ~0;

            // The single-fan node this replaces. Left behind by an older build it would keep
            // emitting a third observation the policy was never trained with.
            Transform stale = car.transform.Find("Sensor");
            if (stale != null) Object.DestroyImmediate(stale.gameObject);

            // ── Keep the best lap of every run as a path.
            //
            //    A .onnx is only loadable against the observation vector it was trained
            //    with, so rally08 became unwatchable the moment the road observations were
            //    added and every future change will cost another era the same way. A
            //    recorded path has no such coupling: it replays on any version of the game,
            //    which is what makes "rally08 against rally10, same stage, side by side"
            //    possible at all. It only touches the disk when a run beats its own record.
            var ghost = ComponentEnsure.Ensure<GhostRecorder>(car);
            ghost.sampleHz = 10f;
            ghost.keepEveryFinish = true;

            AttachTrainedPolicy(bp);

            var agentSerialized = new SerializedObject(agent);
            agentSerialized.ApplyModifiedPropertiesWithoutUndo();

            EditorUtility.SetDirty(car);
        }

        /// <summary>
        /// Puts <see cref="TrainedModelPath"/> into Behaviour Parameters, and says so either
        /// way — a missing policy is a perfectly normal state before the first run, but
        /// finding out by watching the car sit still is not.
        ///
        /// Written through SerializedObject rather than the public Model property on
        /// purpose: that setter calls UpdateAgentPolicy, which calls Agent.ReloadPolicy on
        /// an agent that has never been initialised because the scene is not playing.
        /// </summary>
        static void AttachTrainedPolicy(BehaviorParameters bp)
        {
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(TrainedModelPath);
            if (model == null)
            {
                Debug.LogWarning(
                    $"[Rally] No trained policy at {TrainedModelPath} — the car will fall back " +
                    "to keyboard control on Play. Train one, then copy its .onnx there.");
                return;
            }

            var so = new SerializedObject(bp);
            so.FindProperty("m_Model").objectReferenceValue = model;
            so.ApplyModifiedPropertiesWithoutUndo();

            Debug.Log($"[Rally] Trained policy attached: {System.IO.Path.GetFileName(TrainedModelPath)}. " +
                      "Play drives itself; mlagents-learn still overrides it when connected.");
        }

        /// <summary>
        /// A sensor mount on the car, at bonnet height ahead of the axle line. Kept on a child
        /// so the eye point can move without dragging the car's own origin around, and reused
        /// in place across rebuilds so nothing else holding a reference to it breaks.
        /// </summary>
        static GameObject SensorNode(GameObject car, string name)
        {
            Transform t = car.transform.Find(name);
            GameObject node = t != null ? t.gameObject : new GameObject(name);
            node.transform.SetParent(car.transform, false);
            node.transform.localPosition = new Vector3(0f, 0.70f, ImprezaSpec.NoseZ - 0.2f);
            node.transform.localRotation = Quaternion.identity;
            node.transform.localScale = Vector3.one;
            return node;
        }

        static void BuildCamera(GameObject car)
        {
            var go = new GameObject("Main Camera");
            go.tag = "MainCamera";

            var cam = go.AddComponent<Camera>();
            cam.fieldOfView = 62f;
            cam.nearClipPlane = 0.2f;
            cam.farClipPlane = 1200f;
            cam.clearFlags = CameraClearFlags.Skybox;
            go.AddComponent<AudioListener>();

            var director = go.AddComponent<SpectatorDirector>();
            director.currentMode = SpectatorDirector.CameraMode.ThirdPerson;
            director.orbitDistance = 8f;
            director.orbitHeight = 3f;
            director.followSmoothness = 6f;

            if (car != null)
            {
                director.targetCar = car.transform;
                director.carRb = car.GetComponent<Rigidbody>();
                go.transform.position = car.transform.position - car.transform.forward * 8f + Vector3.up * 3f;
                go.transform.LookAt(car.transform.position + Vector3.up);
            }
        }

        // ══════════════════════════════════════════════════════════════
        //  TELEMETRY HUD
        //
        //  TelemetryUI has existed since early on and has never once been on screen: it
        //  needs a Canvas with three Texts and a RectTransform for the friction circle,
        //  and nothing in the project ever built one. A component wired to nothing is
        //  worse than no component — it reads as a working feature.
        //
        //  It earns its place in THIS scene rather than a showcase one because this is the
        //  only scene there is, and because the way a policy gets watched is Rally > Watch >
        //  Latest Checkpoint. Watching a car drive with no idea what gear it is in or
        //  whether the tyres are saturated is most of why the obstacle plateau went
        //  undiagnosed for three runs.
        //
        //  It costs nothing in training: a headless player under -nographics does not
        //  render, and Text.text is only assigned when the value is on screen to change.
        // ══════════════════════════════════════════════════════════════

        static void BuildTelemetryHud(GameObject car)
        {
            if (car == null) return;

            var canvasGo = new GameObject("TelemetryHUD");
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;

            var scaler = canvasGo.AddComponent<UnityEngine.UI.CanvasScaler>();
            scaler.uiScaleMode = UnityEngine.UI.CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1280f, 720f);

            // No GraphicRaycaster and no EventSystem: nothing here is clickable, and a
            // raycaster in a training scene is per-frame work for no purpose.

            var telemetry = canvasGo.AddComponent<TelemetryUI>();
            telemetry.vehicle = car.GetComponent<VehicleController>();
            telemetry.frontLeftTire = FindCornerTyre(car, DynamicSuspension.Corner.FrontLeft);

            telemetry.speedText = HudText(canvasGo, "Speed", new Vector2(24f, 96f), 40, "0 KM/H");
            telemetry.gearText  = HudText(canvasGo, "Gear",  new Vector2(24f, 56f), 26, "GEAR 0");
            telemetry.rpmText   = HudText(canvasGo, "Rpm",   new Vector2(24f, 24f), 26, "0 RPM");

            telemetry.gForceDot = FrictionCircle(canvasGo);
            // maxUIOffset is in the same units as the circle's radius, or the dot leaves it.
            telemetry.maxUIOffset = 60f;
            telemetry.gForceUIMultiplier = 6f;   // ~1 g reaches 60 % of the way out
        }

        static PacejkaTireModel FindCornerTyre(GameObject car, DynamicSuspension.Corner corner)
        {
            foreach (var strut in car.GetComponentsInChildren<DynamicSuspension>())
                if (strut.corner == corner) return strut.GetComponent<PacejkaTireModel>();
            return null;
        }

        /// <summary>
        /// A line of HUD text, anchored to the bottom-left so it does not move with the
        /// window. LegacyRuntime.ttf is the built-in font in Unity 6 — Arial.ttf, which
        /// every older example uses, was removed and returns null.
        /// </summary>
        static UnityEngine.UI.Text HudText(GameObject parent, string name, Vector2 offset,
                                           int size, string placeholder)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform, false);

            var text = go.AddComponent<UnityEngine.UI.Text>();
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = size;
            text.text = placeholder;
            text.color = Color.white;
            text.alignment = TextAnchor.LowerLeft;
            text.horizontalOverflow = HorizontalWrapMode.Overflow;
            text.verticalOverflow = VerticalWrapMode.Overflow;

            var rect = text.rectTransform;
            rect.anchorMin = rect.anchorMax = rect.pivot = Vector2.zero;
            rect.anchoredPosition = offset;
            rect.sizeDelta = new Vector2(360f, size + 8f);
            return text;
        }

        /// <summary>
        /// The friction circle: a ring the dot moves inside, showing which way and how hard
        /// the tyres are being asked to work. This is the one instrument that makes a
        /// Pacejka model legible from the outside — the dot leaving the circle IS the
        /// moment grip ran out.
        /// </summary>
        static RectTransform FrictionCircle(GameObject parent)
        {
            var ring = new GameObject("FrictionCircle");
            ring.transform.SetParent(parent.transform, false);

            var background = ring.AddComponent<UnityEngine.UI.Image>();
            background.color = new Color(1f, 1f, 1f, 0.12f);

            var ringRect = background.rectTransform;
            ringRect.anchorMin = ringRect.anchorMax = ringRect.pivot = new Vector2(1f, 0f);
            ringRect.anchoredPosition = new Vector2(-90f, 90f);
            ringRect.sizeDelta = new Vector2(120f, 120f);

            var dot = new GameObject("Dot");
            dot.transform.SetParent(ring.transform, false);

            var dotImage = dot.AddComponent<UnityEngine.UI.Image>();
            dotImage.color = new Color(1f, 0.85f, 0.2f, 0.95f);

            var dotRect = dotImage.rectTransform;
            dotRect.anchorMin = dotRect.anchorMax = dotRect.pivot = new Vector2(0.5f, 0.5f);
            dotRect.sizeDelta = new Vector2(10f, 10f);
            dotRect.anchoredPosition = Vector2.zero;
            return dotRect;
        }

        // ══════════════════════════════════════════════════════════════
        //  HELPERS
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// A scene that is not in the build settings cannot be used by a standalone
        /// training build, which is how you actually run training at speed.
        /// </summary>
        static void AddSceneToBuildSettings()
        {
            var scenes = new System.Collections.Generic.List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);
            if (scenes.Exists(s => s.path == ScenePath)) return;
            scenes.Insert(0, new EditorBuildSettingsScene(ScenePath, true));
            EditorBuildSettings.scenes = scenes.ToArray();
        }

        static void EnsureFolder(string folder)
        {
            if (AssetDatabase.IsValidFolder(folder)) return;
            string parent = Path.GetDirectoryName(folder).Replace('\\', '/');
            string leaf = Path.GetFileName(folder);
            EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, leaf);
        }
    }
}
