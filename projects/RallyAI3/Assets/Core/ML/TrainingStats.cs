using System.Collections.Generic;

namespace Core.ML
{
    /// <summary>How an episode came to an end.</summary>
    public enum EpisodeOutcome
    {
        Finished,       // reached the end of the stage
        RolledOver,
        HitObstacle,
        HardImpact,     // crashed into something that was not a tagged rock
        FellOff,        // left the mesh
        TimedOut,       // ran out of MaxStep without finishing or dying
        /// <summary>
        /// Sat still long enough that the rest of the episode could not tell us anything.
        /// Appended rather than inserted: these values are serialised into scenes and
        /// indexed into <see cref="TrainingStats.OutcomeCounts"/>, so renumbering the
        /// existing ones would silently relabel every historical count.
        /// </summary>
        Stalled
    }

    /// <summary>
    /// Live training counters, held in memory for the editor monitor to draw.
    ///
    /// This deliberately does NOT reproduce reward and loss curves — TensorBoard already
    /// owns those and does them better. What TensorBoard cannot tell you is WHY episodes
    /// are ending, and early on that is the number you steer by: a policy dying on
    /// rollovers needs completely different work from one timing out halfway down the
    /// stage, and both look like "reward is flat" on a graph.
    ///
    /// Everything here is static and cleared on domain reload, which is what you want —
    /// counters should not survive a script recompile and quietly mislead you.
    /// </summary>
    public static class TrainingStats
    {
        /// <summary>How many recent episodes the rolling means average over.</summary>
        public const int Window = 100;

        public static int EpisodeCount { get; private set; }
        public static readonly int[] OutcomeCounts =
            new int[System.Enum.GetValues(typeof(EpisodeOutcome)).Length];

        static readonly Queue<float> rewardWindow = new Queue<float>();
        static readonly Queue<float> lengthWindow = new Queue<float>();
        static readonly Queue<float> progressWindow = new Queue<float>();

        /// <summary>Every episode's reward, for the sparkline. Capped so it cannot grow forever.</summary>
        public static readonly List<float> RewardHistory = new List<float>();
        const int HistoryCap = 2000;

        public static float LastReward { get; private set; }
        public static float LastSeconds { get; private set; }
        public static int LastWaypoints { get; private set; }
        public static int BestWaypoints { get; private set; }
        public static EpisodeOutcome LastOutcome { get; private set; }
        public static float SessionStartTime { get; private set; } = -1f;

        public static float MeanReward => Mean(rewardWindow);
        public static float MeanSeconds => Mean(lengthWindow);
        public static float MeanWaypoints => Mean(progressWindow);

        /// <summary>
        /// Share of ALL episodes this session that ended this way, 0..1.
        ///
        /// Not the last <see cref="Window"/> — that window applies to the reward, length and
        /// progress means only. <see cref="OutcomeCounts"/> is a running total, so late in a
        /// session this lags a genuine change in behaviour rather than showing it. Use "Reset
        /// counters" after changing something to get a clean read.
        /// </summary>
        public static float OutcomeShare(EpisodeOutcome outcome)
            => EpisodeCount == 0 ? 0f : OutcomeCounts[(int)outcome] / (float)EpisodeCount;

        public static void Record(EpisodeOutcome outcome, float reward, float seconds, int waypointsReached)
        {
            if (SessionStartTime < 0f) SessionStartTime = UnityEngine.Time.realtimeSinceStartup;

            EpisodeCount++;
            OutcomeCounts[(int)outcome]++;
            LastOutcome = outcome;
            LastReward = reward;
            LastSeconds = seconds;
            LastWaypoints = waypointsReached;
            if (waypointsReached > BestWaypoints) BestWaypoints = waypointsReached;

            Push(rewardWindow, reward);
            Push(lengthWindow, seconds);
            Push(progressWindow, waypointsReached);

            RewardHistory.Add(reward);
            if (RewardHistory.Count > HistoryCap) RewardHistory.RemoveRange(0, RewardHistory.Count - HistoryCap);
        }

        public static void ResetAll()
        {
            EpisodeCount = 0;
            for (int i = 0; i < OutcomeCounts.Length; i++) OutcomeCounts[i] = 0;
            rewardWindow.Clear(); lengthWindow.Clear(); progressWindow.Clear();
            RewardHistory.Clear();
            LastReward = LastSeconds = 0f;
            LastWaypoints = BestWaypoints = 0;
            SessionStartTime = -1f;
        }

        static void Push(Queue<float> q, float v)
        {
            q.Enqueue(v);
            while (q.Count > Window) q.Dequeue();
        }

        static float Mean(Queue<float> q)
        {
            if (q.Count == 0) return 0f;
            float s = 0f;
            foreach (float v in q) s += v;
            return s / q.Count;
        }
    }
}
