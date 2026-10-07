using System;
using System.IO;
using System.Linq;
using Unity.InferenceEngine;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEngine;
using Core.ML;

namespace EditorScripts
{
    /// <summary>
    /// Watch a training run drive, without interrupting it.
    ///
    /// A headless run is six standalone players with no windows, so there is nothing to look
    /// at while the interesting part happens. But it drops a checkpoint every
    /// checkpoint_interval steps — roughly every seven minutes at the rate this project
    /// trains — and those are ordinary .onnx files. Loading the newest one into the editor
    /// scene and pressing Play shows you what the policy could do a few minutes ago.
    ///
    /// This does not disturb the run. The trainer talks to its own players on their own
    /// ports; the editor fails to find a trainer on 5004, says so, and falls back to running
    /// the assigned model. The only real cost is CPU contention with the six workers, which
    /// is why <see cref="WatchFrameRate"/> exists.
    ///
    /// Re-run the menu item while play mode is running and it hot-swaps to whatever checkpoint
    /// has appeared since, so you can sit and watch the policy improve over a long run.
    /// </summary>
    public static class RallyPolicyViewer
    {
        /// <summary>Where the builder expects to find the policy it assigns. A slot, not a run.</summary>
        const string CurrentSlot = "Assets/ML-Agents/Models/RallyDriver-current.onnx";

        const string ResultsFolder = "results";

        /// <summary>
        /// Editor frame rate while watching. Capped because the whole point is to watch a run
        /// that is still going: an uncapped editor will happily take a core off the six
        /// workers, and the run is worth more than the frame rate is.
        /// </summary>
        const int WatchFrameRate = 60;

        // ══════════════════════════════════════════════════════════════
        //  MENU
        // ══════════════════════════════════════════════════════════════

        [MenuItem("Rally/Watch/Latest Checkpoint %#w", false, 0)]
        public static void WatchLatest()
        {
            // Checkpoint is a struct, so FirstOrDefault would hand back a zeroed one rather
            // than null. Ask the array how many it found instead.
            var found = FindCheckpoints();
            if (found.Length == 0)
            {
                Debug.LogWarning($"[Watch] No checkpoints under {ResultsFolder}/. Train something first.");
                return;
            }
            Load(found[0]);
        }

        [MenuItem("Rally/Watch/List Checkpoints", false, 1)]
        public static void ListCheckpoints()
        {
            var all = FindCheckpoints().Take(12).ToList();
            if (all.Count == 0)
            {
                Debug.LogWarning($"[Watch] No checkpoints under {ResultsFolder}/.");
                return;
            }

            var lines = all.Select(c =>
                $"  {c.Run,-10} step {c.Step,12:n0}   {c.Written:HH:mm:ss}");
            Debug.Log($"[Watch] {all.Count} most recent checkpoints:\n{string.Join("\n", lines)}\n" +
                      "Rally > Watch > Latest Checkpoint (Cmd+Shift+W) loads the top one.");
        }

        // ══════════════════════════════════════════════════════════════
        //  LOAD
        // ══════════════════════════════════════════════════════════════

        static void Load(Checkpoint checkpoint)
        {
            // Copied into a fixed slot rather than imported where it lies, because results/ is
            // outside Assets/ and Unity cannot reference an asset it has not imported. The
            // fixed name is also what TrainingSceneBuilder assigns, so a later scene rebuild
            // keeps whatever was last watched.
            Directory.CreateDirectory(Path.GetDirectoryName(CurrentSlot));
            File.Copy(checkpoint.Path, CurrentSlot, true);
            AssetDatabase.ImportAsset(CurrentSlot, ImportAssetOptions.ForceUpdate);

            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(CurrentSlot);
            if (model == null)
            {
                Debug.LogError($"[Watch] {CurrentSlot} did not import as a model.");
                return;
            }

            var bp = UnityEngine.Object.FindAnyObjectByType<BehaviorParameters>();
            if (bp == null)
            {
                Debug.LogError("[Watch] No BehaviorParameters in the open scene. " +
                               "Open the training scene first.");
                return;
            }

            if (Application.isPlaying) AssignLive(bp, model);
            else AssignInEditor(bp, model);

            Debug.Log($"[Watch] {checkpoint.Run} @ step {checkpoint.Step:n0} " +
                      $"(written {checkpoint.Written:HH:mm:ss}).\n" +
                      (Application.isPlaying
                          ? "Swapped in live — run this again to pull the next checkpoint."
                          : "Press Play. Cmd+Shift+W re-runs this, including during play."));
        }

        /// <summary>
        /// Hot-swap while the scene is running.
        ///
        /// The null first is not superstition. Agent.SetModel returns early when the behaviour
        /// name, model reference and device are all unchanged — and because every checkpoint
        /// is copied over the SAME asset path, the ModelAsset reference does not change even
        /// though its contents just did. Setting null in between makes the second call differ,
        /// so the policy is genuinely rebuilt.
        /// </summary>
        static void AssignLive(BehaviorParameters bp, ModelAsset model)
        {
            var agent = bp.GetComponent<RallyAgent>();
            if (agent == null)
            {
                Debug.LogError("[Watch] No RallyAgent alongside BehaviorParameters.");
                return;
            }
            agent.SetModel(bp.BehaviorName, null);
            agent.SetModel(bp.BehaviorName, model);
            Application.targetFrameRate = WatchFrameRate;
        }

        /// <summary>
        /// Assign while stopped. Written through SerializedObject rather than the public Model
        /// property, whose setter calls ReloadPolicy on an agent that has never been
        /// initialised because the scene is not playing.
        /// </summary>
        static void AssignInEditor(BehaviorParameters bp, ModelAsset model)
        {
            var so = new SerializedObject(bp);
            so.FindProperty("m_Model").objectReferenceValue = model;
            so.ApplyModifiedPropertiesWithoutUndo();

            // Watching is exactly when you want to be told why each run ended.
            var agent = bp.GetComponent<RallyAgent>();
            if (agent != null && !agent.logEpisodeEnds)
            {
                var aso = new SerializedObject(agent);
                aso.FindProperty("logEpisodeEnds").boolValue = true;
                aso.ApplyModifiedPropertiesWithoutUndo();
            }

            EditorUtility.SetDirty(bp.gameObject);
        }

        // ══════════════════════════════════════════════════════════════
        //  DISCOVERY
        // ══════════════════════════════════════════════════════════════

        struct Checkpoint
        {
            public string Path, Run;
            public long Step;
            public DateTime Written;
        }

        /// <summary>
        /// Every checkpoint under results/, newest first. Sorted by write time rather than by
        /// step, so the top entry is the latest thing the RUNNING trainer produced even when
        /// several runs are on disk with overlapping step numbers.
        /// </summary>
        static Checkpoint[] FindCheckpoints()
        {
            string root = Path.Combine(Directory.GetCurrentDirectory(), ResultsFolder);
            if (!Directory.Exists(root)) return Array.Empty<Checkpoint>();

            return Directory.GetFiles(root, "RallyDriver-*.onnx", SearchOption.AllDirectories)
                .Select(p => new Checkpoint
                {
                    Path = p,
                    Run = RunNameOf(p),
                    Step = StepOf(p),
                    Written = File.GetLastWriteTime(p)
                })
                .OrderByDescending(c => c.Written)
                .ToArray();
        }

        /// <summary>results/&lt;run&gt;/RallyDriver/RallyDriver-1234.onnx -> "&lt;run&gt;".</summary>
        static string RunNameOf(string path)
        {
            var dir = new DirectoryInfo(Path.GetDirectoryName(path));
            return dir?.Parent?.Name ?? "?";
        }

        /// <summary>The trailing number in RallyDriver-1234567.onnx, or 0 if there isn't one.</summary>
        static long StepOf(string path)
        {
            string name = Path.GetFileNameWithoutExtension(path);
            int dash = name.LastIndexOf('-');
            return dash >= 0 && long.TryParse(name.Substring(dash + 1), out long step) ? step : 0;
        }
    }
}
