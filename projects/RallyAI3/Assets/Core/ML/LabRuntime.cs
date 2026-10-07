using System;
using System.IO;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Demonstrations;
using Unity.InferenceEngine;
using Core.Environment;

namespace Core.ML
{
    [Serializable]
    public class LabLaunch
    {
        public int schema = 1;
        public string mode = "generalist", courseBundle = "", courseId = "", courseName = "", courseFamily = "", policyBundle = "";
        public string output = "", runId = "", checkpointId = "", contract = "";
        public string reward = "baseline-v1";
        public string startingGear = "neutral";
        public string spawnProfile = "fixed";
        public int episodeSeconds;
        public string controlMode = "constant-throttle";
        public int seed = 42, attempts = 20, refresh = 15;
        public float timeScale = 10f, controlProbeSeconds, controlSteer, controlDrive = 0.1f, controlTargetSpeed = 8f;
        public bool evaluation, viewer, deterministic = true, controlProbe, recordDemonstration;
        public int[] excludedSeeds = Array.Empty<int>();
    }

    public static class LabRuntime
    {
        public static LabLaunch Config { get; private set; }
        public static bool ViewerPaused { get; private set; }
        static System.Random stages;
        static AssetBundle courseBundle, policyBundle;
        public static bool Enabled => Config != null;
        public static bool CircuitInspection => !string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_INSPECTION"));
        public static int NextSeed()
        {
            if (stages == null)
            {
                int port = 0;
                var args = System.Environment.GetCommandLineArgs();
                for (int i = 0; i + 1 < args.Length; i++)
                    if (args[i] == "--mlagents-port") int.TryParse(args[i + 1], out port);
                stages = new System.Random(unchecked(Config.seed * 397 ^ port));
            }
            int value;
            do { value = stages.Next(); } while (Array.IndexOf(Config.excludedSeeds, value) >= 0);
            return value;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void Read()
        {
            Config = null; stages = null; ViewerPaused = false;
            string path = System.Environment.GetEnvironmentVariable("RALLY_LAB_LAUNCH");
            if (string.IsNullOrEmpty(path)) return;
            Config = JsonUtility.FromJson<LabLaunch>(File.ReadAllText(path));
            if (Config.schema != 1) throw new InvalidDataException("Unsupported launch schema");
            Directory.CreateDirectory(Config.output);
            System.Environment.SetEnvironmentVariable("RALLY_EPISODE_LOG_DIR", Config.output);
            System.Environment.SetEnvironmentVariable("RALLY_GHOST_LABEL", Config.runId);
            System.Environment.SetEnvironmentVariable("RALLY_GHOST_DIR", Path.Combine(Config.output, "ghosts"));
            Application.runInBackground = true;
            Application.targetFrameRate = Config.viewer ? 60 : -1;
            Time.timeScale = Config.viewer ? 1f : Config.timeScale;
            if (Config.evaluation)
            {
                System.Environment.SetEnvironmentVariable("RALLY_EVAL_EPISODES", Config.attempts.ToString());
                System.Environment.SetEnvironmentVariable("RALLY_EVAL_TIMESCALE", Config.timeScale.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }
            UnityEngine.Random.InitState(Config.seed);
        }

        public static void PauseViewer()
        {
            if (!Enabled || !Config.viewer || ViewerPaused) return;
            ViewerPaused = true;
            Time.timeScale = 0f;
            AudioListener.pause = true;
        }

        public static void RetryViewer()
        {
            if (!Enabled || !Config.viewer || !ViewerPaused) return;
            ViewerPaused = false;
            AudioListener.pause = false;
            Time.timeScale = 1f;
            var agent = UnityEngine.Object.FindAnyObjectByType<RallyAgent>();
            if (agent != null) agent.OnEpisodeBegin();
        }

        public static bool ReplaceCourse(TrackGenerator original)
        {
            if (!Enabled) return false;
            if (string.IsNullOrEmpty(Config.courseBundle))
            {
                original.seed = NextSeed();
                original.randomiseSeed = false;
                return false;
            }
            courseBundle = AssetBundle.LoadFromFile(Config.courseBundle);
            if (courseBundle == null) throw new InvalidDataException("Cannot load frozen course");
            var prefab = courseBundle.LoadAllAssets<GameObject>()[0];
            var replacement = UnityEngine.Object.Instantiate(prefab);
            replacement.name = "FrozenCourse";
            original.gameObject.SetActive(false);
            UnityEngine.Object.Destroy(original.gameObject);
            return true;
        }

        public static TrackGenerator ReloadCourseForInspection()
        {
            if(!CircuitInspection)throw new InvalidOperationException("Diagnostic reload requires inspection mode");
            var old=UnityEngine.Object.FindAnyObjectByType<TrackGenerator>();
            if(old!=null){old.gameObject.SetActive(false);UnityEngine.Object.Destroy(old.gameObject);}
            if(courseBundle!=null)courseBundle.Unload(true);
            courseBundle=AssetBundle.LoadFromFile(Config.courseBundle);
            var replacement=UnityEngine.Object.Instantiate(courseBundle.LoadAllAssets<GameObject>()[0]);replacement.name="FrozenCourse";
            return replacement.GetComponent<TrackGenerator>();
        }

        public static void Configure(RallyAgent agent)
        {
            if (!Enabled) return;
            var track=UnityEngine.Object.FindAnyObjectByType<TrackGenerator>();
            if(track!=null && track.importedCircuit)
            {
                int seconds=Config.episodeSeconds>0?Config.episodeSeconds:track.circuitEpisodeSeconds;
                agent.MaxStep=Mathf.RoundToInt(seconds/Time.fixedDeltaTime);
            }
            agent.stageRefreshEpisodes = Config.mode == "specialist" ? 0 : Config.refresh;
            if (!Config.evaluation && !Config.viewer) return;
            var behavior = agent.GetComponent<BehaviorParameters>();
            if (Config.controlProbe)
            {
                behavior.Model = null;
                behavior.BehaviorType = BehaviorType.HeuristicOnly;
                if (Config.recordDemonstration)
                {
                    var recorder = agent.GetComponent<DemonstrationRecorder>();
                    if (recorder == null) recorder = agent.gameObject.AddComponent<DemonstrationRecorder>();
                    recorder.Record = true;
                    recorder.DemonstrationName = "RallyTeacher";
                    recorder.DemonstrationDirectory = Path.Combine(Config.output, "demonstrations");
                }
                return;
            }
            policyBundle = AssetBundle.LoadFromFile(Config.policyBundle);
            if (policyBundle == null) throw new InvalidDataException("Cannot load checkpoint bundle");
            var models = policyBundle.LoadAllAssets<ModelAsset>();
            if (models.Length != 1) throw new InvalidDataException("Checkpoint must contain one model");
            // Agent.SetModel signals episode completion. During Agent.Initialize the
            // observation sensor and action buffers do not exist yet, so update the
            // behavior configuration directly and let initialization build its policy.
            behavior.Model = models[0];
            behavior.DeterministicInference = Config.deterministic;
            behavior.InferenceDevice = InferenceDevice.Burst;
            behavior.BehaviorType = BehaviorType.InferenceOnly;
        }
    }
}
