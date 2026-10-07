namespace Core.ML
{
    /// <summary>
    /// Everything known about one finished episode, as one value.
    ///
    /// WHY THIS EXISTS
    ///
    /// The log used to carry seven numbers: outcome, reward, seconds, waypoints, target,
    /// seed, rock density. That is enough to say WHAT the rate of each failure was and
    /// nothing at all about why, which is exactly where three runs of work stalled:
    ///
    ///   * obstacle strikes sat at 45-53 % across every configuration and a sensor upgrade
    ///     that tripled detection range, and nothing on disk could say whether the rock was
    ///     avoidable, whether it was even a road rock, or how much warning it gave;
    ///   * rollovers sat at 15-20 % across every run and every difficulty, entirely
    ///     undiagnosed — too fast for the corner, a bad landing off the racing line and a
    ///     structural problem with the car all produce the same one word, "RolledOver".
    ///
    /// So each of these fields exists to separate two hypotheses that the old log conflated.
    /// They are gathered at the instant the episode ends and never averaged: a mean over a
    /// summary window is what TensorBoard is for, and it is precisely what cannot answer
    /// "was there a gap".
    ///
    /// Passed as a struct rather than as arguments because Append had already reached seven
    /// positional floats and this doubles it.
    /// </summary>
    public struct EpisodeRecord
    {
        public bool managed, valid;
        public string circuitRevision, spawnProfile, invalidationReason;
        public float circuitStart, circuitEnd, episodeSeconds, spawnStation;
        public int ticks, attempt;
        public float[] splits;
        // ── What happened ────────────────────────────────────────────────
        public EpisodeOutcome outcome;
        public float reward;
        public float seconds;
        public int waypointsReached;
        public int waypointTarget;

        /// <summary>The stage this episode ran on. Put it back into TrackGenerator.seed to replay it.</summary>
        public int stageSeed;

        /// <summary>Rocks per 100 m the curriculum had reached, so outcomes read against difficulty.</summary>
        public float obstacleDensity;

        // ── Where it ended ───────────────────────────────────────────────
        /// <summary>Metres along the stage.</summary>
        public float station;

        /// <summary>Signed metres from the centreline, positive to the RIGHT.</summary>
        public float lateralOffset;

        // ── What the car was doing ───────────────────────────────────────
        //
        //  Sampled at the last moment the car was still LEVEL, not at the instant the
        //  episode ended. For every outcome but a rollover those are the same frame. For a
        //  rollover they are not, and the difference is the whole question: by the time
        //  up-dot falls below 0.1 the car is already inverted and has scrubbed off much of
        //  the speed that put it there. The speed that mattered is the one it carried in.

        /// <summary>Forward speed, m/s.</summary>
        public float speed;

        /// <summary>Fastest the car went this episode, m/s.</summary>
        public float peakSpeed;

        /// <summary>Radians the road turns over the next 25 m. Positive turns right.</summary>
        public float curvature;

        /// <summary>Heading error against the road tangent, radians, positive pointed right.</summary>
        public float headingError;

        /// <summary>Wheels touching the ground, 0-4.</summary>
        public int wheelsGrounded;

        /// <summary>Seconds the car had been off all four wheels. Zero when it was on the ground.</summary>
        public float airborneSeconds;

        /// <summary>Longest single flight this episode, seconds.</summary>
        public float longestFlight;

        /// <summary>Dot of the car's up against world up. 1 is level, 0 on its side.</summary>
        public float uprightDot;

        /// <summary>Last commanded steering, -1..1.</summary>
        public float steer;

        /// <summary>Last commanded drive, positive throttle, negative brake.</summary>
        public float drive;

        // ── What it hit ──────────────────────────────────────────────────
        /// <summary>
        /// Name of the object struck, or empty. "RoadRock" is a rock in the racing line;
        /// "Tree" and "Boulder" are scenery, which the car can only meet after running wide.
        /// </summary>
        public string hitWhat;

        /// <summary>True when <see cref="hitWhat"/> was a road rock and the fields below mean something.</summary>
        public bool hitRoadRock;

        /// <summary>The struck rock's signed offset from the centreline, metres.</summary>
        public float rockLateral;

        /// <summary>
        /// Signed offset of whatever was struck, metres — rock, tree, boulder or terrain.
        /// Read against the car's own offset this says whether the car was where it thought
        /// it was: a tree at 13 m hit by a car reading 3 m is an instrument problem, not a
        /// driving one.
        /// </summary>
        public float struckLateral;

        /// <summary>Unsigned horizontal distance from the centreline to whatever was struck, metres.</summary>
        public float struckDistance;

        /// <summary>Metres along the stage of whatever was struck.</summary>
        public float struckStation;

        /// <summary>Widest rock-free corridor across the road where the strike happened, metres.</summary>
        public float gapWidth;

        /// <summary>Signed offset of that corridor's centre, metres. Where the car should have been.</summary>
        public float gapCentre;

        /// <summary>
        /// Seconds the struck object spent inside the forward ray fan before impact — the
        /// warning the policy actually got. Negative means it was never in the fan at all,
        /// which is the signature of hitting something while sideways or off the road.
        /// </summary>
        public float warningSeconds;
    }
}
