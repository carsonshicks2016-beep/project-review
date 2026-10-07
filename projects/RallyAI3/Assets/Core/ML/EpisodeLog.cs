using System;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.MLAgents;
using UnityEngine;

namespace Core.ML
{
    /// <summary>
    /// Appends one JSON line per episode to disk, so a finished run can be examined
    /// afterwards rather than only watched while it happens.
    ///
    /// WHY THIS EXISTS
    ///
    /// A headless run is how training is actually done — a standalone player plus
    /// --num-envs — and a headless run publishes almost nothing per episode. The
    /// player logs under results/&lt;run&gt;/run_logs/Player-N.log are about forty lines
    /// each and carry only the startup banner, because logEpisodeEnds is off (six
    /// processes writing unstructured prose is worse than silence). The only other
    /// channel is TensorBoard at summary_freq, which for a 1.18M-step run came to
    /// sixty data points for the whole thing.
    ///
    /// So the interesting question — "which stage seeds is it failing on, and how?" —
    /// was unanswerable after the fact. <see cref="TrainingStats"/> holds it, but only
    /// in memory, only in the editor, and only for one environment.
    ///
    /// One line per episode is small (a 1M-step run is a few thousand lines), append-only,
    /// and trivially readable by anything: jq, pandas, or a dashboard later.
    ///
    /// WORKER IDENTITY
    ///
    /// Six players run at once and must not share a file. ML-Agents gives each one
    /// --mlagents-port (base_port + worker_id), which is the same argument Academy itself
    /// parses to find its trainer, so it is a reliable per-worker identity. The editor has
    /// no such argument and is labelled "editor".
    /// </summary>
    public static class EpisodeLog
    {
        /// <summary>Override the output directory. Set this if the working directory is not the project root.</summary>
        public const string DirectoryEnvVar = "RALLY_EPISODE_LOG_DIR";

        /// <summary>Relative to the process working directory, which mlagents-learn inherits from the shell.</summary>
        const string DefaultDirectory = "results/episodes";

        /// <summary>The same argument Academy reads to find its trainer.</summary>
        const string PortFlag = "--mlagents-port";

        static readonly UTF8Encoding NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

        static string filePath;
        static string session;
        static int episodeIndex;
        static bool resolved;
        static bool broken;

        /// <summary>Where lines are being written, or null if the sink is unavailable.</summary>
        public static string FilePath => filePath;

        /// <summary>Episodes written by this process so far.</summary>
        public static int Count => episodeIndex;

        /// <summary>
        /// Records one finished episode. Never throws: a training run must not die because
        /// a disk is full or a directory is read-only, so a failure disables the sink and
        /// says so once.
        /// </summary>
        public static void Append(in EpisodeRecord record)
        {
            if (broken) return;

            try
            {
                if (!resolved) Resolve();
                if (broken) return;

                episodeIndex++;

                // Built by hand rather than through a serialiser: the only dependency
                // available everywhere is JsonUtility, which cannot emit a flat object with
                // mixed types cleanly, and one line of text is not worth a package.
                //
                // Every number goes through InvariantCulture. Under a locale that uses a
                // decimal comma, the default ToString would emit 3,42 and produce a file
                // that is not JSON at all — and it would do it silently, on someone else's
                // machine, months from now.
                var sb = new StringBuilder(420);
                sb.Append('{');
                if (record.managed)
                {
                    Int(sb, "schema", 1); sb.Append(',');
                    Str(sb, "runId", LabRuntime.Config.runId); sb.Append(',');
                    Str(sb, "courseId", LabRuntime.Config.courseId); sb.Append(',');
                    Str(sb, "checkpointId", LabRuntime.Config.checkpointId); sb.Append(',');
                    Int(sb, "ticks", record.ticks); sb.Append(',');
                    Int(sb, "attempt", record.attempt); sb.Append(',');
                    sb.Append("\"valid\":").Append(record.valid ? "true" : "false").Append(',');
                    if(!string.IsNullOrEmpty(record.circuitRevision))
                    {
                        Str(sb,"circuitRevision",record.circuitRevision);sb.Append(',');
                        Str(sb,"spawnProfile",record.spawnProfile);sb.Append(',');
                        Str(sb,"invalidationReason",record.invalidationReason);sb.Append(',');
                        Num(sb,"startStation",record.circuitStart,"0.###");sb.Append(',');
                        Num(sb,"endStation",record.circuitEnd,"0.###");sb.Append(',');
                        Num(sb,"episodeSeconds",record.episodeSeconds,"0.###");sb.Append(',');
                        if(record.spawnStation>=0){Num(sb,"spawnStation",record.spawnStation,"0.#");sb.Append(',');}
                    }
                    sb.Append("\"splits\":[");
                    for (int i = 0; record.splits != null && i < record.splits.Length; i++)
                    {
                        if (i > 0) sb.Append(',');
                        sb.Append(record.splits[i].ToString("0.###", CultureInfo.InvariantCulture));
                    }
                    sb.Append("],");
                }
                Num(sb, "t", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0, "0.###");
                sb.Append(',');
                Str(sb, "session", session);
                sb.Append(',');
                Str(sb, "worker", WorkerId());
                sb.Append(',');
                Int(sb, "ep", episodeIndex);
                sb.Append(',');
                // Aligns each episode to the training curve, so an outcome can be read
                // against the reward at the step it happened rather than by wall clock.
                Int(sb, "step", Academy.IsInitialized ? Academy.Instance.TotalStepCount : 0);
                sb.Append(',');
                Str(sb, "outcome", record.outcome.ToString());
                sb.Append(',');
                Num(sb, "reward", record.reward, "0.####");
                sb.Append(',');
                Num(sb, "seconds", record.seconds, "0.###");
                sb.Append(',');
                Int(sb, "waypoints", record.waypointsReached);
                sb.Append(',');
                Int(sb, "target", record.waypointTarget);
                sb.Append(',');
                // The stage this episode actually ran on. With the stage rebuilt every
                // stageRefreshEpisodes and generation deterministic from the seed, this is
                // what makes a bad run reproducible: the seed can be put straight back into
                // TrackGenerator and driven by hand.
                Int(sb, "seed", record.stageSeed);
                sb.Append(',');
                Num(sb, "rocks", record.obstacleDensity, "0.###");

                // ── Failure forensics. Everything above this line was in the original
                //    format and keeps its name, so a reader written for the old logs still
                //    works on the new ones and the two can be compared directly.
                sb.Append(',');
                Num(sb, "station", record.station, "0.#");
                sb.Append(',');
                Num(sb, "off", record.lateralOffset, "0.##");
                sb.Append(',');
                Num(sb, "speed", record.speed, "0.##");
                sb.Append(',');
                Num(sb, "peakSpeed", record.peakSpeed, "0.##");
                sb.Append(',');
                Num(sb, "curve", record.curvature, "0.####");
                sb.Append(',');
                Num(sb, "heading", record.headingError, "0.####");
                sb.Append(',');
                Int(sb, "wheels", record.wheelsGrounded);
                sb.Append(',');
                Num(sb, "air", record.airborneSeconds, "0.###");
                sb.Append(',');
                Num(sb, "flight", record.longestFlight, "0.###");
                sb.Append(',');
                Num(sb, "upright", record.uprightDot, "0.###");
                sb.Append(',');
                Num(sb, "steer", record.steer, "0.###");
                sb.Append(',');
                Num(sb, "drive", record.drive, "0.###");
                sb.Append(',');
                Str(sb, "hit", record.hitWhat ?? "");

                // Only meaningful for a road-rock strike, and writing them anyway would
                // invite an analysis that averages a gap width over episodes where no gap
                // was ever measured.
                if (record.hitRoadRock)
                {
                    sb.Append(',');
                    Num(sb, "rockLat", record.rockLateral, "0.##");
                    sb.Append(',');
                    Num(sb, "gap", record.gapWidth, "0.##");
                    sb.Append(',');
                    Num(sb, "gapCentre", record.gapCentre, "0.##");
                }

                if (!string.IsNullOrEmpty(record.hitWhat))
                {
                    sb.Append(',');
                    Num(sb, "warning", record.warningSeconds, "0.###");
                    sb.Append(',');
                    Num(sb, "hitLat", record.struckLateral, "0.##");
                    sb.Append(',');
                    Num(sb, "hitDist", record.struckDistance, "0.##");
                    sb.Append(',');
                    Num(sb, "hitStation", record.struckStation, "0.#");
                }

                sb.Append('}');
                sb.Append('\n');

                // UTF8Encoding(false), not Encoding.UTF8. The latter emits a byte-order mark
                // when it CREATES the file, which puts three invisible bytes in front of the
                // first '{' and makes line 1 — and only line 1 — invalid JSON. Python's
                // json.loads rejects it outright ("Unexpected UTF-8 BOM"); every subsequent
                // line parses, so the corruption looks like a one-off bad record rather than
                // an encoding choice.
                File.AppendAllText(filePath, sb.ToString(), NoBom);
            }
            catch (Exception e)
            {
                broken = true;
                Debug.LogWarning($"[EpisodeLog] Disabled after a write failure: {e.Message}");
            }
        }

        /// <summary>
        /// Picks the output path once. One file per process per run, so six concurrent
        /// workers never interleave and two runs never mix.
        /// </summary>
        static void Resolve()
        {
            resolved = true;

            // Fully qualified: this project has its own Core.Environment namespace, which from
            // inside Core.ML shadows the System one and resolves first.
            string dir = System.Environment.GetEnvironmentVariable(DirectoryEnvVar);
            if (string.IsNullOrEmpty(dir))
                dir = Path.Combine(ProjectRoot(), DefaultDirectory);

            Directory.CreateDirectory(dir);

            session = DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
            filePath = Path.Combine(dir, $"episodes-{WorkerId()}-{session}.jsonl");

            Debug.Log($"[EpisodeLog] Writing episodes to {filePath}");
        }

        /// <summary>
        /// Walks up from the working directory looking for the project root.
        ///
        /// The working directory alone is not enough, and this cost a run's worth of data. In
        /// the editor it IS the project root, so writing to "./results/episodes" was correct.
        /// But mlagents-learn launches a built player with the working directory set to the
        /// folder holding the .app, so six headless workers wrote to Builds/results/episodes/ —
        /// which is gitignored, and which nobody thought to look in.
        ///
        /// A directory containing both Assets and results is the project root from either
        /// starting point: the editor matches immediately, a player one level down in Builds/
        /// matches after one step up. Falls back to the working directory if neither is found,
        /// which is what <see cref="DirectoryEnvVar"/> exists to override.
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

        /// <summary>
        /// This process's worker identity: its ML-Agents port, or "editor" when there is no
        /// port argument. Sanitised because it ends up in a filename and in JSON.
        /// </summary>
        static string WorkerId()
        {
            string[] args;
            try { args = System.Environment.GetCommandLineArgs(); }   // see note in Resolve()
            catch { return "unknown"; }

            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] != PortFlag) continue;
                string raw = args[i + 1];
                var clean = new StringBuilder(raw.Length);
                foreach (char c in raw)
                    if (char.IsLetterOrDigit(c)) clean.Append(c);
                if (clean.Length > 0) return clean.ToString();
            }
            return "editor";
        }

        /// <summary>
        /// A JSON string field. Escaped, because one of these is now a GameObject name and
        /// a name is only conventionally well behaved — a stray quote would break the line
        /// and, being append-only, every line after it in the same file.
        /// </summary>
        static void Str(StringBuilder sb, string key, string value)
        {
            sb.Append('"').Append(key).Append("\":\"");
            foreach (char c in value ?? "")
            {
                switch (c)
                {
                    case '"':  sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n");  break;
                    case '\r': sb.Append("\\r");  break;
                    case '\t': sb.Append("\\t");  break;
                    default:
                        if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }

        static void Int(StringBuilder sb, string key, int value)
        {
            sb.Append('"').Append(key).Append("\":")
              .Append(value.ToString(CultureInfo.InvariantCulture));
        }

        static void Num(StringBuilder sb, string key, double value, string format)
        {
            if (double.IsNaN(value) || double.IsInfinity(value)) value = 0.0;   // JSON has no NaN
            sb.Append('"').Append(key).Append("\":")
              .Append(value.ToString(format, CultureInfo.InvariantCulture));
        }
    }
}
