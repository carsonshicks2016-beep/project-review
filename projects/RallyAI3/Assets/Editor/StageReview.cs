using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Core.Environment;
using Core.Physics;

namespace EditorScripts
{
    /// <summary>
    /// Fixed camera shots of a fixed stage, rendered headless, so a change to the look of
    /// the world is judged on pictures side by side rather than on a description.
    ///
    /// It renders the REAL training scene — its camera settings, sun, ambient, fog and sky —
    /// on one pinned seed, so two runs differ only in what was changed in between. The car
    /// is moved into the chase shots for scale; nothing is saved, so the scene on disk is
    /// untouched.
    ///
    /// Run:  Unity -batchmode -quit -projectPath . -executeMethod EditorScripts.StageReview.Execute
    ///       -reviewLabel before [-reviewBare]
    /// Output: .rally/visual-review/stage/{label}-{shot}.png
    /// </summary>
    public static class StageReview
    {
        const string Folder = ".rally/visual-review/stage";
        const int ReviewSeed = 20260727;
        static int Width = 1280, Height = 720;
        static bool dressed;
        static string overrideLabel;
        static bool? overrideLegacy;

        [Serializable] class CourseManifest { public string id, bundle, bundle_hash; public int seed; }

        public static void Execute()
        {
            string label = overrideLabel ?? Argument("-reviewLabel") ?? "current";
            if (int.TryParse(Argument("-reviewWidth"), out int width)) Width = width;
            if (int.TryParse(Argument("-reviewHeight"), out int height)) Height = height;
            if (Width < 640 || Height < 360) throw new ArgumentException("Review resolution is too small");
            bool legacy = overrideLegacy ?? Argument("-reviewLegacyForest") == "1";
            Environment.SetEnvironmentVariable("RALLY_FOREST_V3", legacy ? "0" : "1");
            Directory.CreateDirectory(Folder);

            EditorSceneManager.OpenScene("Assets/Scenes/RallyTraining.unity", OpenSceneMode.Single);
            var track = UnityEngine.Object.FindAnyObjectByType<TrackGenerator>();
            if (track == null) throw new InvalidOperationException("No TrackGenerator in the training scene");
            string manifestPath = Argument("-reviewCourse");
            CourseManifest manifest = null;
            if (!string.IsNullOrEmpty(manifestPath))
            {
                manifest = JsonUtility.FromJson<CourseManifest>(File.ReadAllText(manifestPath));
                using (var hash = System.Security.Cryptography.SHA256.Create())
                using (var input = File.OpenRead(manifest.bundle))
                    if (BitConverter.ToString(hash.ComputeHash(input)).Replace("-", "").ToLowerInvariant() != manifest.bundle_hash)
                        throw new InvalidDataException("Review course bundle hash mismatch");
                var bundle = AssetBundle.LoadFromFile(manifest.bundle);
                if (bundle == null) throw new InvalidDataException("Cannot load review course bundle");
                var frozen = UnityEngine.Object.Instantiate(bundle.LoadAllAssets<GameObject>()[0]);
                UnityEngine.Object.DestroyImmediate(track.gameObject);
                track = frozen.GetComponent<TrackGenerator>();
                bundle.Unload(false);
                if (!track.frozenCourse || track.seed != manifest.seed) throw new InvalidDataException("Review course identity mismatch");
            }
            else
            {
                track.frozenCourse = false;
                track.randomiseSeed = false;
                track.seed = ReviewSeed;
                track.GenerateTrack();
            }
            Physics.SyncTransforms();
            // -reviewBare renders the stage without the dressing, for a like-for-like baseline.
            dressed = Array.IndexOf(Environment.GetCommandLineArgs(), "-reviewBare") < 0;
            if (dressed) StageDressing.Dress(track);

            Camera camera = Camera.main;
            if (camera == null) throw new InvalidOperationException("No main camera in the training scene");
            // Switch off the camera's game behaviours (the chase director would move it), but
            // keep the image effects: the glare and grade are part of what is being judged.
            foreach (var behaviour in camera.GetComponents<MonoBehaviour>())
                if (!(behaviour is SunGlare)) behaviour.enabled = false;
            var car = UnityEngine.Object.FindAnyObjectByType<VehicleController>();

            // Chase shots at the game's own framing — 8 m back, 3 m up — then two that the
            // chase camera never shows but a stage is judged on: the verge from low down,
            // and a long look down the road.
            Shot(track, camera, car, label, "chase-early", 0.12f, Chase);
            Shot(track, camera, car, label, "chase-mid",   0.45f, Chase);
            Shot(track, camera, car, label, "chase-late",  0.75f, Chase);
            Shot(track, camera, car, label, "verge-low",   0.30f, VergeLow);
            Shot(track, camera, car, label, "long-view",   0.60f, LongView);
            Shot(track, camera, car, label, "into-sun",    0.40f, IntoSun);
            Shot(track, camera, car, label, "rear-quarter", 0.30f, RearQuarter);
            Shot(track, camera, car, label, "front-quarter", 0.30f, FrontQuarter);
            Shot(track, camera, car, label, "side", 0.30f, Side);
            Shot(track, camera, car, label, "hood", 0.30f, Hood);
            Shot(track, camera, car, label, "driver", 0.30f, Driver);
            int crest = track.spawnWaypointIndex + 1;
            for (int i = crest + 1; i < track.FinishWaypointIndex; i++)
                if (track.waypoints[i].y > track.waypoints[crest].y) crest = i;
            float walked = 0f, length = 0f;
            for (int i = 1; i < track.waypoints.Count; i++)
            {
                float distance = Vector3.Distance(track.waypoints[i - 1], track.waypoints[i]);
                length += distance;
                if (i <= crest) walked += distance;
            }
            Shot(track, camera, car, label, "crest", walked / Mathf.Max(1f, length), Chase);

            // Two fixed shots of the furniture: the start gantry from behind the start line,
            // and the run into the sharpest corner, where the tape and chevrons are.
            if (track.TryGetSpawn(out Vector3 spawn, out Vector3 ahead))
                Fixed(camera, label, "start", spawn - ahead * 14f + Vector3.up * 3f, spawn + ahead * 20f + Vector3.up * 3f);
            if (StageDressing.TryCornerView(track, 35f, out Vector3 approach, out Vector3 apex))
                Fixed(camera, label, "corner", approach + Vector3.up * 2.4f, apex + Vector3.up * 1f);
            File.WriteAllText($"{Folder}/{label}-identity.json", JsonUtility.ToJson(new ReviewIdentity {
                course = manifest != null ? manifest.id : "generated-review-only", seed = track.seed,
                bundleHash = manifest != null ? manifest.bundle_hash : "", width = Width, height = Height,
                fixedDeltaTime = Time.fixedDeltaTime, sun = RenderSettings.sun != null ? RenderSettings.sun.transform.rotation : Quaternion.identity,
                colliders = track.GetComponentsInChildren<Collider>(true).Length }, true));
            Debug.Log($"[StageReview] {label}: shots of seed {track.seed} in {Folder}");
        }

        public static void Comparison()
        {
            try
            {
                foreach (bool legacy in new[] { true, false })
                foreach (int height in new[] { 720, 1080 })
                {
                    Height = height; Width = height == 720 ? 1280 : 1920;
                    overrideLabel = $"{Argument("-reviewLabel") ?? "v3"}-{(legacy ? "baseline" : "kit")}-final-{height}";
                    overrideLegacy = legacy;
                    Execute();
                }
            }
            finally { overrideLabel = null; overrideLegacy = null; }
        }

        [Serializable] class ReviewIdentity
        {
            public string course, bundleHash;
            public int seed, width, height, colliders;
            public float fixedDeltaTime;
            public Quaternion sun;
        }

        delegate void Frame(Vector3 road, Vector3 forward, Vector3 right, Transform camera);

        static void Chase(Vector3 road, Vector3 forward, Vector3 right, Transform camera)
        {
            camera.position = road - forward * 8f + Vector3.up * 3f;
            camera.LookAt(road + Vector3.up * 1f + forward * 4f);
        }

        static void RearQuarter(Vector3 p, Vector3 f, Vector3 r, Transform c) { c.position = p - f * 3f + r * 4f + Vector3.up * 2f; c.LookAt(p + f * 4f + Vector3.up); }
        static void FrontQuarter(Vector3 p, Vector3 f, Vector3 r, Transform c) { c.position = p + f * 10f + r * 4f + Vector3.up * 2f; c.LookAt(p + f * 4f + Vector3.up); }
        static void Side(Vector3 p, Vector3 f, Vector3 r, Transform c) { c.position = p + f * 4f + r * 7f + Vector3.up * 1.5f; c.LookAt(p + f * 4f + Vector3.up); }
        static void Hood(Vector3 p, Vector3 f, Vector3 r, Transform c) { c.position = p + f * 5.38f + Vector3.up * 1.42f; c.LookAt(c.position + f * 30f); }
        static void Driver(Vector3 p, Vector3 f, Vector3 r, Transform c) { c.position = p + f * 3.88f + r * 0.36f + Vector3.up * 1.04f; c.LookAt(c.position + f * 30f); }

        /// <summary>A shot from one point looking at another, with no car moved into it.</summary>
        static void Fixed(Camera camera, string label, string name, Vector3 from, Vector3 to)
        {
            camera.transform.position = from;
            camera.transform.LookAt(to);
            foreach (var follower in UnityEngine.Object.FindObjectsByType<BackdropFollower>(FindObjectsSortMode.None))
                follower.transform.position = new Vector3(from.x, follower.fixedHeight, from.z);
            Capture(camera, label, name);
        }

        /// <summary>From the chase height, turned toward the sun, so the glare is always judged.</summary>
        static void IntoSun(Vector3 road, Vector3 forward, Vector3 right, Transform camera)
        {
            camera.position = road - forward * 8f + Vector3.up * 3f;
            Light sun = RenderSettings.sun;
            Vector3 toSun = sun != null ? -sun.transform.forward : forward;
            Vector3 flat = new Vector3(toSun.x, 0f, toSun.z).normalized;
            camera.rotation = Quaternion.LookRotation(flat * 0.9f + Vector3.up * 0.18f);
        }

        static void VergeLow(Vector3 road, Vector3 forward, Vector3 right, Transform camera)
        {
            camera.position = road + right * 7.5f + Vector3.up * 0.9f - forward * 3f;
            camera.LookAt(road + forward * 25f + Vector3.up * 0.8f);
        }

        static void LongView(Vector3 road, Vector3 forward, Vector3 right, Transform camera)
        {
            camera.position = road + Vector3.up * 6f;
            camera.LookAt(road + forward * 80f + Vector3.up * 1f);
        }

        static void Shot(TrackGenerator track, Camera camera, VehicleController car,
                         string label, string name, float along, Frame frame)
        {
            RoadPoint(track, along, out Vector3 road, out Vector3 forward);
            Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;

            if (car != null)
            {
                // The car sits a few metres ahead of the camera's road point, on the road.
                RoadPoint(track, along + 4f / Mathf.Max(1f, track.trackLength), out Vector3 carAt, out Vector3 carForward);
                var surface = track.GetComponent<MeshCollider>();
                if (surface != null && surface.Raycast(new Ray(carAt + Vector3.up * 100f, Vector3.down), out RaycastHit hit, 200f))
                    carAt = hit.point;
                car.transform.SetPositionAndRotation(carAt + Vector3.up * 0.05f, Quaternion.LookRotation(carForward));
            }

            if (car != null && dressed) DriveDust(track, car, along + 4f / Mathf.Max(1f, track.trackLength), name == "into-sun" ? 4f : 1.5f);

            frame(road, forward, right, camera.transform);

            // The distant ridge follows the camera in LateUpdate, which never runs between
            // offline shots; left alone it stays wherever the scene was saved and turns up
            // as a giant wall beside the road.
            foreach (var follower in UnityEngine.Object.FindObjectsByType<BackdropFollower>(FindObjectsSortMode.None))
            {
                Vector3 c = camera.transform.position;
                follower.transform.position = new Vector3(c.x, follower.fixedHeight, c.z);
            }

            Capture(camera, label, name);
        }

        static void Capture(Camera camera, string label, string name)
        {
            var target = new RenderTexture(Width, Height, 24);
            camera.targetTexture = target;
            camera.Render();
            RenderTexture.active = target;
            var image = new Texture2D(Width, Height, TextureFormat.RGB24, false);
            image.ReadPixels(new Rect(0, 0, Width, Height), 0, 0);
            image.Apply();
            File.WriteAllBytes($"{Folder}/{label}-{name}.png", image.EncodeToPNG());
            RenderTexture.active = null;
            camera.targetTexture = null;
            UnityEngine.Object.DestroyImmediate(image);
            UnityEngine.Object.DestroyImmediate(target);
        }

        /// <summary>
        /// Lay the dust trail the car would have left arriving here: drive it up the road for
        /// a few seconds at rally pace, emitting through the plume's own per-wheel code and
        /// stepping the particles by hand, then leave it at its shot position.
        /// </summary>
        static void DriveDust(TrackGenerator track, VehicleController car, float carAlong, float slip)
        {
            var plume = car.GetComponent<DustPlume>();
            if (plume == null) plume = car.gameObject.AddComponent<DustPlume>();
            plume.ClearTrail();

            const float speed = 24f, seconds = 3.5f, dt = 1f / 30f;
            float total = 0f;
            for (int i = 1; i < track.waypoints.Count; i++) total += Vector3.Distance(track.waypoints[i - 1], track.waypoints[i]);
            if (total <= 0f) return;

            for (float t = 0f; t < seconds; t += dt)
            {
                float f = carAlong - speed * (seconds - t) / total;
                if (f < 0f) continue;
                RoadPoint(track, f, out Vector3 at, out Vector3 forward);
                Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;
                for (int w = 0; w < 4; w++)
                {
                    bool front = w < 2;
                    Vector3 contact = at + right * (w % 2 == 0 ? -0.75f : 0.75f) + forward * (front ? 1.26f : -1.26f);
                    plume.EmitWheel(w, dt, contact, forward * speed, slip, 0.6f, front ? 0.5f : 1f);
                }
                plume.Advance(dt);
            }
        }

        /// <summary>World point and heading on the centreline, a fraction of the way along by distance.</summary>
        static void RoadPoint(TrackGenerator track, float fraction, out Vector3 point, out Vector3 forward)
        {
            var w = track.waypoints;
            float total = 0f;
            for (int i = 1; i < w.Count; i++) total += Vector3.Distance(w[i - 1], w[i]);
            float goal = Mathf.Clamp01(fraction) * total, walked = 0f;
            for (int i = 1; i < w.Count; i++)
            {
                float step = Vector3.Distance(w[i - 1], w[i]);
                if (walked + step >= goal || i == w.Count - 1)
                {
                    float t = step > 0f ? Mathf.Clamp01((goal - walked) / step) : 0f;
                    point = track.transform.TransformPoint(Vector3.Lerp(w[i - 1], w[i], t));
                    forward = track.transform.TransformDirection(w[i] - w[i - 1]).normalized;
                    return;
                }
                walked += step;
            }
            point = track.transform.position;
            forward = track.transform.forward;
        }

        static string Argument(string name)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == name) return args[i + 1];
            return null;
        }
    }
}
