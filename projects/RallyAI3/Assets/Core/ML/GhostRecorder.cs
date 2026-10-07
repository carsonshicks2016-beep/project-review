using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Core.Environment;

namespace Core.ML
{
    /// <summary>
    /// Records the car's path and keeps the good ones.
    ///
    /// Recording every episode would be tens of thousands of files and gigabytes of a
    /// training run's disk, almost all of it showing a car crashing within twenty seconds.
    /// What is worth keeping is the best a run ever managed — that is the thing you want
    /// to put next to the NEXT run's best, on the same stage, to see whether the change
    /// helped. So this keeps a finish, or anything that got further than the run has got
    /// before, and nothing else.
    ///
    /// Attach it to the car alongside <see cref="RallyAgent"/>. It costs one Vector3 and
    /// one Quaternion per tenth of a second while an episode is running, and touches the
    /// disk only when a record is beaten.
    /// </summary>
    [RequireComponent(typeof(RallyAgent))]
    public class GhostRecorder : MonoBehaviour
    {
        [Tooltip("Samples per second. 10 Hz is smooth once interpolated and keeps a full " +
                 "120 s lap under a couple of hundred kilobytes.")]
        public float sampleHz = 10f;

        [Tooltip("Which run this is, written into every ghost so two eras can be told " +
                 "apart on screen. The RALLY_GHOST_LABEL environment variable overrides it, " +
                 "which is how a headless run labels itself without editing the scene.")]
        public string label = "";

        [Tooltip("Also keep any episode that finishes the stage, even if it is slower than " +
                 "a previous finish. A finish is rare enough to be worth the file.")]
        public bool keepEveryFinish = true;

        [Tooltip("Where ghosts are written, relative to the project root.")]
        public string outputDirectory = "results/ghosts";

        private RallyAgent agent;
        private TrackGenerator trackForSeed;

        private readonly List<Vector3> positions = new List<Vector3>();
        private readonly List<Quaternion> rotations = new List<Quaternion>();
        private float nextSampleTime;

        /// <summary>Furthest this process has got. Only a better episode earns a file.</summary>
        private int bestWaypoints = -1;

        private bool broken;

        void Awake()
        {
            agent = GetComponent<RallyAgent>();
            trackForSeed = FindAnyObjectByType<TrackGenerator>();

            string fromEnvironment = null;
            try { fromEnvironment = System.Environment.GetEnvironmentVariable("RALLY_GHOST_LABEL"); }
            catch { /* sandboxed player; the inspector value stands */ }
            if (!string.IsNullOrEmpty(fromEnvironment)) label = fromEnvironment;
            string managedDirectory = System.Environment.GetEnvironmentVariable("RALLY_GHOST_DIR");
            if (!string.IsNullOrEmpty(managedDirectory)) outputDirectory = managedDirectory;

            agent.EpisodeEnded += OnEpisodeEnded;
        }

        void OnDestroy()
        {
            if (agent != null) agent.EpisodeEnded -= OnEpisodeEnded;
        }

        void FixedUpdate()
        {
            if (Time.time < nextSampleTime) return;
            nextSampleTime = Time.time + 1f / Mathf.Max(1f, sampleHz);

            positions.Add(transform.position);
            rotations.Add(transform.rotation);
        }

        private void OnEpisodeEnded(EpisodeOutcome outcome, int waypointsReached)
        {
            bool finished = outcome == EpisodeOutcome.Finished;
            bool record = waypointsReached > bestWaypoints || (finished && keepEveryFinish);

            if (record && positions.Count > 1)
            {
                bestWaypoints = Mathf.Max(bestWaypoints, waypointsReached);
                // Length from the sample count rather than a wall clock: the samples ARE
                // the recording, so this is exactly the duration the ghost will replay for.
                Write(outcome, waypointsReached, positions.Count / Mathf.Max(1f, sampleHz));
            }

            // Cleared whether or not it was kept — the next episode is a different lap.
            positions.Clear();
            rotations.Clear();
            nextSampleTime = 0f;
        }

        private void Write(EpisodeOutcome outcome, int waypointsReached, float seconds)
        {
            if (broken) return;

            try
            {
                var ghost = new GhostPath
                {
                    seed = trackForSeed != null ? trackForSeed.CurrentSeed : 0,
                    label = string.IsNullOrEmpty(label) ? "unlabelled" : label,
                    outcome = outcome.ToString(),
                    waypoints = waypointsReached,
                    target = agent.WaypointTarget,
                    seconds = seconds,
                    hz = sampleHz,
                    positions = positions.ToArray(),
                    rotations = rotations.ToArray()
                };

                string directory = Path.Combine(ProjectRoot(), outputDirectory);
                Directory.CreateDirectory(directory);

                // Seed first in the name: the only ghosts worth putting side by side are
                // ones from the same stage, so the filename sorts them together.
                string file = $"ghost-{ghost.seed}-{ghost.label}-{waypointsReached:00}wp.json";
                File.WriteAllText(Path.Combine(directory, file), JsonUtility.ToJson(ghost));

                Debug.Log($"[Ghost] Kept {file} — {waypointsReached}/{ghost.target} waypoints " +
                          $"in {seconds:0.0} s.", this);
            }
            catch (System.Exception e)
            {
                broken = true;
                Debug.LogWarning($"[Ghost] Disabled after a write failure: {e.Message}", this);
            }
        }

        /// <summary>
        /// Same walk-up as <see cref="EpisodeLog"/>, and for the same reason: a built player
        /// runs with its working directory set to the folder holding the .app, so writing
        /// to a relative path puts the output in Builds/ where nobody looks.
        /// </summary>
        static string ProjectRoot()
        {
            var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
            for (int up = 0; up < 4 && dir != null; up++, dir = dir.Parent)
            {
                if (Directory.Exists(Path.Combine(dir.FullName, "Assets")) &&
                    Directory.Exists(Path.Combine(dir.FullName, "results")))
                    return dir.FullName;
            }
            return Directory.GetCurrentDirectory();
        }
    }
}
