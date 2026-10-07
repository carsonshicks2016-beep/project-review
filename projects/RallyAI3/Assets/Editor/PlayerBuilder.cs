using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace EditorScripts
{
    /// <summary>
    /// Builds the standalone player that headless training runs against.
    ///
    /// Training in the Editor works and is the right place to watch the car and debug,
    /// but it is roughly an order of magnitude slower than a player, and it can only ever
    /// run ONE environment. mlagents-learn can drive several copies of a built player at
    /// once (--num-envs), and on an 8-performance-core machine that is the difference
    /// between a twelve-hour run and a one-hour one.
    /// </summary>
    public static class PlayerBuilder
    {
        const string ScenePath  = "Assets/Scenes/RallyTraining.unity";
        const string BuildDir   = "Builds";
        const string PlayerName = "RallyTraining";

        [MenuItem("Rally/Training/Build Player (for headless training)", false, 3)]
        public static void Build()
        {
            RallyAudioAssets.Prepare();
            if (!File.Exists(ScenePath))
            {
                Debug.LogError($"[Rally] No scene at {ScenePath}. Run Build Training Scene first.");
                return;
            }

            Directory.CreateDirectory(BuildDir);
            string target = Path.Combine(BuildDir, PlayerName + ".app");

            var options = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = target,
                target = BuildTarget.StandaloneOSX,
                // No Development flag: the profiler hooks cost throughput, and this player
                // exists only to be run thousands of times as fast as possible.
                options = BuildOptions.None
            };

            BuildReport report = BuildPipeline.BuildPlayer(options);
            BuildSummary summary = report.summary;

            if (summary.result != BuildResult.Succeeded)
            {
                Debug.LogError($"[Rally] Player build {summary.result} — {summary.totalErrors} errors. " +
                               "Check the Console above for the first one.");
                return;
            }

            Debug.Log(
                $"[Rally] Player built to {target} ({summary.totalSize / (1024 * 1024)} MB, " +
                $"{summary.totalTime.TotalSeconds:0} s).\n" +
                $"  Train headless with:\n" +
                $"    .venv/bin/mlagents-learn config/rally_ppo.yaml --run-id=<id> \\\n" +
                $"        --env={target} --num-envs=6 --no-graphics\n" +
                $"  Rebuild this whenever the scene or any script changes — the player is a " +
                $"snapshot, it does not follow the project.");
        }
    }
}
