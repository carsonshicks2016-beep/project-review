using UnityEditor;
using UnityEngine;
using Core.ML;
using Core.Physics;

namespace EditorScripts
{
    /// <summary>
    /// Live training dashboard: what the run is doing right now, and why episodes end.
    ///
    /// Deliberately not a reward-curve viewer — TensorBoard already does that properly,
    /// and duplicating it badly would just be a second thing to distrust. This answers
    /// the question TensorBoard cannot: a flat reward curve looks identical whether the
    /// car is rolling over on turn-in, timing out halfway down the stage, or driving
    /// into the scenery, and those need three different fixes.
    /// </summary>
    public class RallyTrainingMonitor : EditorWindow
    {
        [MenuItem("Rally/Training/Training Monitor", false, 2)]
        public static void Open()
        {
            var w = GetWindow<RallyTrainingMonitor>("Rally Training");
            w.minSize = new Vector2(320, 460);
            w.Show();
        }

        static readonly Color Ink       = new Color(0.85f, 0.85f, 0.84f);
        static readonly Color Muted     = new Color(0.55f, 0.55f, 0.53f);
        static readonly Color Good      = new Color(0.11f, 0.69f, 0.48f);
        static readonly Color Warn      = new Color(0.92f, 0.63f, 0.00f);
        static readonly Color Bad       = new Color(0.89f, 0.29f, 0.28f);
        static readonly Color Accent    = new Color(0.17f, 0.47f, 0.84f);
        static readonly Color GridLine  = new Color(1f, 1f, 1f, 0.08f);

        RallyAgent agent;
        VehicleController vehicle;
        Vector2 scroll;

        void OnEnable()  { EditorApplication.update += Tick; }
        void OnDisable() { EditorApplication.update -= Tick; }

        double lastRepaint;
        void Tick()
        {
            // 10 Hz is plenty and keeps the editor responsive during training.
            if (EditorApplication.timeSinceStartup - lastRepaint < 0.1) return;
            lastRepaint = EditorApplication.timeSinceStartup;
            Repaint();
        }

        void OnGUI()
        {
            if (agent == null)   agent   = Object.FindAnyObjectByType<RallyAgent>();
            if (vehicle == null) vehicle = Object.FindAnyObjectByType<VehicleController>();

            scroll = EditorGUILayout.BeginScrollView(scroll);

            if (!Application.isPlaying)
            {
                EditorGUILayout.HelpBox(
                    "Not playing.\n\n" +
                    "1.  .venv/bin/mlagents-learn config/rally_ppo.yaml --run-id=<id>\n" +
                    "2.  Press Play when it says it is listening\n\n" +
                    "Obstacle density is set by the obstacle_density curriculum in the YAML, " +
                    "not by hand — leave Spawn Obstacles alone.",
                    MessageType.Info);

                // Worth saying plainly, because this window looks like it monitors "training"
                // and it does not: TrainingStats is in-memory, in-editor, single-environment.
                // The way training is actually run — a built player plus --num-envs — is
                // invisible here, and that is what the on-disk episode log is for.
                EditorGUILayout.HelpBox(
                    "This window only sees training running INSIDE the editor. A headless run " +
                    "(--env with --num-envs) reports nothing here; read results/episodes/*.jsonl " +
                    "or TensorBoard for those.",
                    MessageType.None);
            }
            else if (agent == null)
            {
                EditorGUILayout.HelpBox("No RallyAgent in the scene.", MessageType.Warning);
            }

            DrawSession();
            DrawOutcomes();
            DrawRewardSparkline();
            DrawLive();

            EditorGUILayout.Space(6);
            using (new EditorGUI.DisabledScope(TrainingStats.EpisodeCount == 0))
                if (GUILayout.Button("Reset counters"))
                    TrainingStats.ResetAll();

            EditorGUILayout.EndScrollView();
        }

        // ── sections ─────────────────────────────────────────────────────────────

        void DrawSession()
        {
            Header("Session");
            int n = TrainingStats.EpisodeCount;
            float mins = TrainingStats.SessionStartTime < 0f ? 0f
                       : (Time.realtimeSinceStartup - TrainingStats.SessionStartTime) / 60f;

            Row("Episodes", n.ToString());
            Row("Running for", mins < 1f ? "under a minute" : $"{mins:0} min");
            Row($"Mean reward (last {TrainingStats.Window})", n == 0 ? "—" : $"{TrainingStats.MeanReward:0.0}");
            Row($"Mean length (last {TrainingStats.Window})", n == 0 ? "—" : $"{TrainingStats.MeanSeconds:0.0} s");
            Row("Mean waypoints", n == 0 ? "—" : $"{TrainingStats.MeanWaypoints:0.0}");
            Row("Best waypoints", n == 0 ? "—" : TrainingStats.BestWaypoints.ToString());
        }

        void DrawOutcomes()
        {
            Header("Why episodes end");
            if (TrainingStats.EpisodeCount == 0)
            {
                EditorGUILayout.LabelField("No episodes yet.", Style(Muted));
                return;
            }

            DrawOutcomeBar(EpisodeOutcome.Finished,    "Finished stage", Good);
            DrawOutcomeBar(EpisodeOutcome.Stalled,     "Stalled (parked)", Warn);
            DrawOutcomeBar(EpisodeOutcome.TimedOut,    "Timed out",      Warn);
            DrawOutcomeBar(EpisodeOutcome.RolledOver,  "Rolled over",    Bad);
            DrawOutcomeBar(EpisodeOutcome.HitObstacle, "Hit a rock",     Bad);
            DrawOutcomeBar(EpisodeOutcome.HardImpact,  "Hard impact",    Bad);
            DrawOutcomeBar(EpisodeOutcome.FellOff,     "Fell off stage", Bad);

            EditorGUILayout.Space(2);
            EditorGUILayout.LabelField(Diagnose(), Style(Muted, wrap: true));
        }

        void DrawOutcomeBar(EpisodeOutcome outcome, string label, Color color)
        {
            float share = TrainingStats.OutcomeShare(outcome);
            int count = TrainingStats.OutcomeCounts[(int)outcome];

            Rect line = EditorGUILayout.GetControlRect(false, 18f);
            var labelRect = new Rect(line.x, line.y, 108f, line.height);
            var barRect   = new Rect(line.x + 112f, line.y + 4f, line.width - 112f - 52f, 10f);
            var pctRect   = new Rect(line.xMax - 48f, line.y, 48f, line.height);

            EditorGUI.LabelField(labelRect, label, Style(Ink));
            EditorGUI.DrawRect(barRect, GridLine);
            if (share > 0f)
                EditorGUI.DrawRect(new Rect(barRect.x, barRect.y, barRect.width * share, barRect.height), color);
            EditorGUI.LabelField(pctRect, $"{share * 100f:0}%  {count}", Style(Muted, right: true));
        }

        /// <summary>Turns the outcome mix into the one sentence worth acting on.</summary>
        string Diagnose()
        {
            int n = TrainingStats.EpisodeCount;
            if (n < 10) return "Too few episodes to read anything into yet.";

            float rolled = TrainingStats.OutcomeShare(EpisodeOutcome.RolledOver);
            float rock   = TrainingStats.OutcomeShare(EpisodeOutcome.HitObstacle);
            float off    = TrainingStats.OutcomeShare(EpisodeOutcome.FellOff);
            float timed  = TrainingStats.OutcomeShare(EpisodeOutcome.TimedOut);
            float done   = TrainingStats.OutcomeShare(EpisodeOutcome.Finished);
            float stall  = TrainingStats.OutcomeShare(EpisodeOutcome.Stalled);

            // The advice here used to be "turn rocks on" and "turn randomiseSeed on", both of
            // which are now automatic — the stage is redrawn every RallyAgent.stageRefreshEpisodes
            // and rock density comes from the obstacle_density curriculum. Telling someone to
            // reach for switches the trainer is already driving is worse than saying nothing.
            if (done > 0.5f)   return "Completing the stage more often than not, and on a stage that changes — this is real driving, not a memorised line. Let the curriculum raise the rock density.";
            if (stall > 0.6f)  return "Mostly parking. Early on this is just an untrained policy sitting on the brake and it should fade within a few hundred thousand steps. If it persists, the drive axis is not finding throttle — raise beta so it explores harder.";
            if (rock > 0.5f)   return "Mostly dying on rocks. The curriculum has moved faster than the policy; loosen the obstacle_density thresholds in the YAML rather than touching Spawn Obstacles.";
            if (rolled > 0.4f) return "Mostly rolling over. It is carrying too much speed into corners — expect this to fade as it learns to brake.";
            if (off > 0.4f)    return "Mostly driving off the stage. Early on this is normal. If it persists past a couple of million steps, the road observations are not enough and the ray sensor needs to see the surface.";
            if (timed > 0.6f)  return "Mostly timing out — surviving but not making progress. Watch that mean waypoints is climbing.";
            return "Mixed outcomes, which is what healthy early training looks like.";
        }

        void DrawRewardSparkline()
        {
            Header("Reward per episode");
            Rect r = EditorGUILayout.GetControlRect(false, 70f);
            EditorGUI.DrawRect(r, new Color(0f, 0f, 0f, 0.18f));

            var hist = TrainingStats.RewardHistory;
            if (hist.Count < 2)
            {
                EditorGUI.LabelField(r, "  collecting…", Style(Muted));
                return;
            }

            float lo = float.MaxValue, hi = float.MinValue;
            foreach (float v in hist) { lo = Mathf.Min(lo, v); hi = Mathf.Max(hi, v); }
            if (Mathf.Approximately(lo, hi)) { hi = lo + 1f; }

            // Zero line, so a negative mean reward is obvious at a glance.
            float zeroY = Mathf.Lerp(r.yMax - 4f, r.y + 4f, Mathf.InverseLerp(lo, hi, 0f));
            if (lo < 0f && hi > 0f) EditorGUI.DrawRect(new Rect(r.x, zeroY, r.width, 1f), GridLine);

            Handles.BeginGUI();
            Handles.color = Accent;
            int n = hist.Count;
            Vector3 prev = Vector3.zero;
            for (int i = 0; i < n; i++)
            {
                float x = Mathf.Lerp(r.x + 2f, r.xMax - 2f, n == 1 ? 1f : i / (float)(n - 1));
                float y = Mathf.Lerp(r.yMax - 4f, r.y + 4f, Mathf.InverseLerp(lo, hi, hist[i]));
                var p = new Vector3(x, y, 0f);
                if (i > 0) Handles.DrawAAPolyLine(2f, prev, p);
                prev = p;
            }
            Handles.EndGUI();

            EditorGUI.LabelField(new Rect(r.x + 4f, r.y + 2f, 120f, 14f), $"{hi:0.0}", Style(Muted));
            EditorGUI.LabelField(new Rect(r.x + 4f, r.yMax - 16f, 120f, 14f), $"{lo:0.0}", Style(Muted));
        }

        void DrawLive()
        {
            Header("Car, right now");
            if (!Application.isPlaying || vehicle == null)
            {
                EditorGUILayout.LabelField("—", Style(Muted));
                return;
            }

            Row("Speed", $"{vehicle.currentSpeedKmh:0} km/h");
            Row("Gear / rpm", $"{(vehicle.currentGear == -1 ? "R" : vehicle.currentGear == 0 ? "N" : vehicle.currentGear.ToString())}   {vehicle.currentRpm:0} rpm");
            Row("Wheels on the ground", $"{vehicle.groundedWheels} / 4");
            Row("Drift angle", $"{vehicle.driftAngleDeg:0.0}°");
            Row("Throttle / brake", $"{vehicle.throttleInput:0.00} / {vehicle.brakeInput:0.00}");
            Row("Steering", $"{vehicle.steeringInput:+0.00;-0.00}");

            if (agent != null)
            {
                int reached = agent.WaypointsReached, target = agent.WaypointTarget;
                Rect r = EditorGUILayout.GetControlRect(false, 16f);
                var lab = new Rect(r.x, r.y, 108f, r.height);
                var bar = new Rect(r.x + 112f, r.y + 3f, r.width - 112f - 52f, 10f);
                var val = new Rect(r.xMax - 48f, r.y, 48f, r.height);
                EditorGUI.LabelField(lab, "Stage progress", Style(Ink));
                EditorGUI.DrawRect(bar, GridLine);
                EditorGUI.DrawRect(new Rect(bar.x, bar.y, bar.width * Mathf.Clamp01(reached / (float)target), bar.height), Accent);
                EditorGUI.LabelField(val, $"{reached}/{target}", Style(Muted, right: true));
            }
        }

        // ── chrome ───────────────────────────────────────────────────────────────

        void Header(string text)
        {
            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField(text, Style(Muted, bold: true));
            Rect r = EditorGUILayout.GetControlRect(false, 1f);
            EditorGUI.DrawRect(r, GridLine);
            EditorGUILayout.Space(2);
        }

        void Row(string label, string value)
        {
            Rect r = EditorGUILayout.GetControlRect(false, 16f);
            EditorGUI.LabelField(new Rect(r.x, r.y, r.width * 0.55f, r.height), label, Style(Ink));
            EditorGUI.LabelField(new Rect(r.x + r.width * 0.55f, r.y, r.width * 0.45f, r.height), value,
                                 Style(Muted, right: true));
        }

        static GUIStyle Style(Color c, bool bold = false, bool right = false, bool wrap = false)
        {
            var s = new GUIStyle(EditorStyles.label)
            {
                wordWrap = wrap,
                fontStyle = bold ? FontStyle.Bold : FontStyle.Normal,
                alignment = right ? TextAnchor.MiddleRight : TextAnchor.MiddleLeft
            };
            s.normal.textColor = c;
            return s;
        }
    }
}
