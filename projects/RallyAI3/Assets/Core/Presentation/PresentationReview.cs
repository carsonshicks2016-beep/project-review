using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;
using Unity.Profiling.LowLevel.Unsafe;
using Core.ML;
using UI;

namespace Core.Presentation
{
    // Opt-in bounded instrumentation. No installation or sampling in normal viewing/training.
    public sealed class PresentationReview : MonoBehaviour
    {
        [Serializable] class Sample
        {
            public float wallSeconds, simulationSeconds;
            public long drawCalls, batches, triangles, unityMemory, processMemory;
            public int particles, objects, renderers, meshes, colliders;
            public int materials, tracksideStation;
            public bool carInSafeFrame;
            public bool viewerPaused;
            public Vector3 carPosition, cameraPosition;
        }
        [Serializable] class Summary
        {
            public string course, checkpoint, unity, device;
            public bool forestV3;
            public string cameraMode;
            public string[] availableCounters;
            public int width, height, fpsCap, frames;
            public float medianMs, p95Ms, p99Ms, averageFps, simulationScale, fixedDeltaTime;
            public Sample[] samples;
            public float[] frameTimesMs;
        }
        readonly List<float> frameTimes = new List<float>(4000);
        readonly List<Sample> samples = new List<Sample>(45);
        ProfilerRecorder draws, batches, triangles, memory;
        string folder;
        float start, nextSample, nextCapture = 3f;
        int shot;
        int sequenceFrame;
        float nextSequence = 4f;
        bool finished;
        bool modeSet, countersListed;
        float resultStart = -1, nextResultFrame;
        int resultFrames;
        SpectatorDirector.CameraMode requestedMode;
        readonly List<string> counterNames = new List<string>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            string path = System.Environment.GetEnvironmentVariable("RALLY_PRESENTATION_REVIEW");
            if (string.IsNullOrEmpty(path) || SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null ||
                !LabRuntime.Enabled || !LabRuntime.Config.viewer) return;
            new GameObject("Presentation Review").AddComponent<PresentationReview>().folder = path;
        }
        void Start()
        {
            Directory.CreateDirectory(folder);
            start = Time.realtimeSinceStartup;
            Enum.TryParse(System.Environment.GetEnvironmentVariable("RALLY_REVIEW_CAMERA"), out requestedMode);
            draws = Counter(ProfilerCategory.Render, "Standard Draw Calls Count");
            batches = Counter(ProfilerCategory.Render, "Batches Count");
            triangles = Counter(ProfilerCategory.Render, "Triangles Count");
            memory = Counter(ProfilerCategory.Memory, "Total Used Memory");
        }
        static ProfilerRecorder Counter(ProfilerCategory category, string name)
        {
            try { return ProfilerRecorder.StartNew(category, name, 1); }
            catch (Exception) { return default; }
        }
        static long Value(ProfilerRecorder recorder) => recorder.Valid && recorder.Count > 0 ? recorder.LastValue : -1;
        void LateUpdate()
        {
            float elapsed = Time.realtimeSinceStartup - start;
            if (!modeSet)
            {
                var director = FindAnyObjectByType<SpectatorDirector>();
                if (director != null) { director.SetMode(requestedMode); modeSet = true; }
            }
            if (!countersListed && elapsed > 3f)
            {
                countersListed = true;
                var handles = new List<ProfilerRecorderHandle>();
                ProfilerRecorderHandle.GetAvailable(handles);
                foreach (var handle in handles)
                {
                    var info = ProfilerRecorderHandle.GetDescription(handle);
                    counterNames.Add(info.Category.Name + "/" + info.Name);
                    if (info.Name == "Standard Draw Calls Count" && Value(draws) < 0)
                    {
                        draws.Dispose();
                        draws = Counter(info.Category, info.Name);
                    }
                    if ((info.Name == "Batches Count" || info.Name == "Batches") && Value(batches) < 0)
                    {
                        batches.Dispose();
                        batches = Counter(info.Category, info.Name);
                    }
                }
                counterNames.Sort();
            }
            if (elapsed > 2f && !LabRuntime.ViewerPaused && frameTimes.Count < 6000)
                frameTimes.Add(Time.unscaledDeltaTime * 1000f);
            if (elapsed >= nextSample && !finished)
            {
                nextSample = elapsed + 1f;
                int count = 0;
                foreach (var ps in FindObjectsByType<ParticleSystem>(FindObjectsSortMode.None)) count += ps.particleCount;
                long working = -1;
                try { working = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64; } catch (Exception) { }
                samples.Add(new Sample { wallSeconds = elapsed, simulationSeconds = Time.time,
                    drawCalls = Value(draws), batches = Value(batches), triangles = Value(triangles), unityMemory = Value(memory),
                    processMemory = working, particles = count,
                    objects = FindObjectsByType<Transform>(FindObjectsSortMode.None).Length,
                    renderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None).Length,
                    meshes = Resources.FindObjectsOfTypeAll<Mesh>().Length,
                    colliders = FindObjectsByType<Collider>(FindObjectsSortMode.None).Length,
                    materials = Resources.FindObjectsOfTypeAll<Material>().Length,
                    viewerPaused=LabRuntime.ViewerPaused,
                    carPosition=FindAnyObjectByType<Core.Physics.VehicleController>()?.transform.position ?? Vector3.zero,
                    cameraPosition=Camera.main!=null?Camera.main.transform.position:Vector3.zero,
                    tracksideStation = FindAnyObjectByType<SpectatorDirector>()?.ActiveTracksideStation ?? -1,
                    carInSafeFrame = FindAnyObjectByType<SpectatorDirector>()?.CarInSafeFrame ?? false });
            }
            if (elapsed >= nextCapture && shot < 6 && System.Environment.GetEnvironmentVariable("RALLY_REVIEW_NO_SCREENSHOTS") != "1")
            {
                ScreenCapture.CaptureScreenshot(Path.Combine(folder, $"drive-{shot:00}.png"));
                shot++; nextCapture += 4f;
            }
            if (elapsed >= nextSequence && sequenceFrame < 100 &&
                System.Environment.GetEnvironmentVariable("RALLY_REVIEW_SEQUENCE") == "1")
            {
                string frames = Path.Combine(folder, "sequence");
                Directory.CreateDirectory(frames);
                ScreenCapture.CaptureScreenshot(Path.Combine(frames, $"frame-{sequenceFrame++:0000}.png"));
                nextSequence = elapsed + .1f;
            }
            bool resultReview=System.Environment.GetEnvironmentVariable("RALLY_RESULT_REVIEW")=="1";
            if(resultReview && LabRuntime.ViewerPaused)
            {
                if(resultStart<0)
                {
                    resultStart=elapsed; nextResultFrame=elapsed+1;
                    var agent=FindAnyObjectByType<RallyAgent>();
                    if(agent!=null && agent.ViewerResult.HasValue)
                        File.WriteAllText(Path.Combine(folder,"viewer-result.json"),JsonUtility.ToJson(agent.ViewerResult.Value,true));
                }
                if(elapsed>=nextResultFrame && resultFrames<100)
                {
                    string frames=Path.Combine(folder,"result-sequence"); Directory.CreateDirectory(frames);
                    ScreenCapture.CaptureScreenshot(Path.Combine(frames,$"frame-{resultFrames++:0000}.png"));
                    nextResultFrame=elapsed+.1f;
                }
            }
            if (!LabRuntime.CircuitInspection && (elapsed > (resultReview?100f:40f) ||
                (LabRuntime.ViewerPaused && (resultReview?elapsed-resultStart>14f:elapsed>4f))))
            {
                Finish();
                Application.Quit();
            }
        }
        void Finish()
        {
            if (finished || string.IsNullOrEmpty(folder)) return;
            finished = true;
            float sum = 0f;
            foreach (float dt in frameTimes) sum += dt;
            float[] chronologicalTimes = frameTimes.ToArray();
            frameTimes.Sort();
            File.WriteAllText(Path.Combine(folder, "performance.json"), JsonUtility.ToJson(new Summary {
                course = LabRuntime.Config.courseId, checkpoint = LabRuntime.Config.checkpointId,
                unity = Application.unityVersion, device = SystemInfo.graphicsDeviceName,
                forestV3 = System.Environment.GetEnvironmentVariable("RALLY_FOREST_V3") != "0",
                cameraMode = requestedMode.ToString(), availableCounters = counterNames.ToArray(),
                width = Screen.width, height = Screen.height, fpsCap = Application.targetFrameRate,
                simulationScale = LabRuntime.Config.timeScale, fixedDeltaTime = Time.fixedDeltaTime,
                frames = frameTimes.Count, medianMs = Percentile(.5f), p95Ms = Percentile(.95f), p99Ms = Percentile(.99f),
                averageFps = sum > 0f ? 1000f * frameTimes.Count / sum : -1f, samples = samples.ToArray(),
                frameTimesMs = chronologicalTimes }, true));
        }
        float Percentile(float q) => frameTimes.Count == 0 ? -1f : frameTimes[Mathf.Min(frameTimes.Count - 1, Mathf.FloorToInt(q * frameTimes.Count))];
        void OnDestroy() { Finish(); draws.Dispose(); batches.Dispose(); triangles.Dispose(); memory.Dispose(); }
        void OnApplicationQuit() => Finish();
    }
}
