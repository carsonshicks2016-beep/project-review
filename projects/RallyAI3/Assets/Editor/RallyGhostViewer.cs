using System.Collections.Generic;
using System.IO;
using System.Linq;
using Core.Environment;
using Core.ML;
using UnityEditor;
using UnityEngine;

namespace EditorScripts
{
    /// <summary>
    /// Races recorded runs against each other on the stage they were driven on.
    ///
    /// The problem this solves: a policy is only loadable against the observation vector it
    /// was trained with, so every time the agent gains a sense the previous era becomes
    /// unwatchable. rally08 is already gone that way. Four runs of work exist on disk and
    /// most of them can never be seen again, which means "did the change help?" can only
    /// ever be answered with summary statistics.
    ///
    /// A ghost is just a list of poses, so it survives anything. This finds the ghosts for
    /// a stage, sets the scene's generator to that stage, and puts them all on it at once.
    ///
    ///     Rally > Watch > Race Ghosts on a Stage
    /// </summary>
    public static class RallyGhostViewer
    {
        const string GhostFolder = "results/ghosts";

        /// <summary>Colours handed out in order, so two ghosts are never the same colour.</summary>
        static readonly Color[] Palette =
        {
            new Color(0.25f, 0.75f, 1.00f, 0.45f),   // blue
            new Color(1.00f, 0.55f, 0.20f, 0.45f),   // orange
            new Color(0.45f, 1.00f, 0.45f, 0.45f),   // green
            new Color(1.00f, 0.35f, 0.70f, 0.45f),   // pink
        };

        [MenuItem("Rally/Watch/Race Ghosts on a Stage", false, 2)]
        public static void RaceGhosts()
        {
            var ghosts = LoadAll();
            if (ghosts.Count == 0)
            {
                Debug.LogWarning(
                    $"[Ghost] No ghosts under {GhostFolder}/. They are written by GhostRecorder " +
                    "during a run — add one to the car in the training scene, then train or " +
                    "evaluate something.");
                return;
            }

            // The stage with the most ghosts on it: that is the one where a comparison has
            // the most to say. A stage with one ghost is a recording, not a race.
            var byStage = ghosts.GroupBy(g => g.path.seed)
                                .OrderByDescending(g => g.Count())
                                .ThenByDescending(g => g.Max(x => x.written.Ticks))
                                .First();

            int seed = byStage.Key;
            var chosen = byStage.OrderByDescending(g => g.path.waypoints).Take(Palette.Length).ToList();

            var track = Object.FindAnyObjectByType<TrackGenerator>();
            if (track == null)
            {
                Debug.LogError("[Ghost] No TrackGenerator in the open scene. " +
                               "Open Assets/Scenes/RallyTraining.unity first.");
                return;
            }

            // The stage must match or the ghosts drive through scenery. Setting the seed
            // here rather than asking the user to is the whole point of the menu item.
            track.seed = seed;
            track.randomiseSeed = false;
            track.GenerateTrack();

            ClearExistingGhosts();

            var host = new GameObject("Ghosts");
            for (int i = 0; i < chosen.Count; i++)
            {
                var node = new GameObject($"Ghost_{chosen[i].path.label}");
                node.transform.SetParent(host.transform, false);

                var player = node.AddComponent<GhostPlayer>();
                player.ghostFile = chosen[i].file;
                player.tint = Palette[i % Palette.Length];
            }

            var summary = string.Join("\n", chosen.Select((g, i) =>
                $"  {ColourName(i),-7} {g.path.label,-12} {g.path.waypoints}/{g.path.target} waypoints" +
                $"   {g.path.outcome}   {g.path.seconds:0.0} s"));

            Debug.Log($"[Ghost] Stage {seed} loaded with {chosen.Count} ghost(s):\n{summary}\n" +
                      "Press Play. They all restart whenever the live car's episode does.");

            Selection.activeGameObject = host;
        }

        [MenuItem("Rally/Watch/List Ghosts", false, 3)]
        public static void ListGhosts()
        {
            var ghosts = LoadAll();
            if (ghosts.Count == 0)
            {
                Debug.LogWarning($"[Ghost] No ghosts under {GhostFolder}/.");
                return;
            }

            var lines = ghosts
                .OrderBy(g => g.path.seed)
                .ThenByDescending(g => g.path.waypoints)
                .Select(g => $"  stage {g.path.seed,12}  {g.path.label,-12} " +
                             $"{g.path.waypoints,2}/{g.path.target,-2} waypoints  " +
                             $"{g.path.outcome,-11} {g.path.seconds,5:0.0} s");

            Debug.Log($"[Ghost] {ghosts.Count} ghost(s):\n{string.Join("\n", lines)}");
        }

        [MenuItem("Rally/Watch/Remove Ghosts from Scene", false, 4)]
        public static void ClearExistingGhosts()
        {
            foreach (var player in Object.FindObjectsByType<GhostPlayer>(FindObjectsSortMode.None))
                if (player != null) Object.DestroyImmediate(player.gameObject);

            // And the node that held them, so repeated use does not leave a row of empties.
            var host = GameObject.Find("Ghosts");
            if (host != null) Object.DestroyImmediate(host);
        }

        struct Entry
        {
            public string file;
            public GhostPath path;
            public System.DateTime written;
        }

        static List<Entry> LoadAll()
        {
            var found = new List<Entry>();
            if (!Directory.Exists(GhostFolder)) return found;

            foreach (string file in Directory.GetFiles(GhostFolder, "*.json"))
            {
                GhostPath path;
                try { path = JsonUtility.FromJson<GhostPath>(File.ReadAllText(file)); }
                catch (System.Exception e)
                {
                    Debug.LogWarning($"[Ghost] Skipping {Path.GetFileName(file)}: {e.Message}");
                    continue;
                }
                if (path == null || path.positions == null || path.positions.Length < 2) continue;

                found.Add(new Entry { file = file, path = path, written = File.GetLastWriteTime(file) });
            }
            return found;
        }

        static string ColourName(int index) => index switch
        {
            0 => "blue",
            1 => "orange",
            2 => "green",
            _ => "pink"
        };
    }
}
