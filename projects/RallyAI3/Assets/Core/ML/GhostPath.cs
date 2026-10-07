using System;
using UnityEngine;

namespace Core.ML
{
    /// <summary>
    /// One recorded lap: where the car went, not the brain that drove it.
    ///
    /// WHY A PATH AND NOT A POLICY
    ///
    /// A trained policy is only loadable against the exact observation vector it was
    /// trained with. Every time the agent gains a sense, every previous era becomes
    /// unwatchable — rally08 is already unloadable, because the six road-relative
    /// observations that came after it changed the input width, and a .onnx with the wrong
    /// input width does not load at all. Four runs of history are on disk and none of the
    /// early ones can ever be seen again.
    ///
    /// A path has no such coupling. It is a list of places the car was, so it replays
    /// against any future version of the game, and it makes the comparison that actually
    /// settles arguments possible: rally08 and rally10 on the same stage, side by side, at
    /// the same moment.
    ///
    /// It is also honest about what it is. A ghost is not a re-simulation — it will not
    /// react to a stage that has changed under it, and if the terrain generator changes
    /// shape the ghost will drive through the new scenery. That is a feature for comparing
    /// two drivers on one stage and a trap for anything else, so the seed is recorded here
    /// and the viewer refuses to replay a ghost onto a different one.
    /// </summary>
    [Serializable]
    public class GhostPath
    {
        /// <summary>Stage this was driven on. The viewer sets the generator to it before replaying.</summary>
        public int seed;

        /// <summary>Which run produced it, e.g. "rally10". Free text, shown next to the ghost.</summary>
        public string label = "";

        public string outcome = "";
        public int waypoints;
        public int target;
        public float seconds;

        /// <summary>Samples per second. Playback interpolates between them.</summary>
        public float hz = 10f;

        public Vector3[] positions = Array.Empty<Vector3>();
        public Quaternion[] rotations = Array.Empty<Quaternion>();

        /// <summary>Seconds of driving this path covers.</summary>
        public float Duration => hz > 0f && positions != null ? positions.Length / hz : 0f;

        /// <summary>
        /// Where the car was at a time within the path, interpolated. Times past the end
        /// hold the final pose rather than wrapping — a ghost that finished should sit at
        /// the finish, not teleport back to the start.
        /// </summary>
        public void Sample(float time, out Vector3 position, out Quaternion rotation)
        {
            position = Vector3.zero;
            rotation = Quaternion.identity;
            if (positions == null || positions.Length == 0) return;

            float exact = Mathf.Max(0f, time) * hz;
            int index = Mathf.FloorToInt(exact);

            if (index >= positions.Length - 1)
            {
                position = positions[positions.Length - 1];
                rotation = rotations != null && rotations.Length == positions.Length
                    ? rotations[rotations.Length - 1] : Quaternion.identity;
                return;
            }

            float t = exact - index;
            position = Vector3.Lerp(positions[index], positions[index + 1], t);
            if (rotations != null && rotations.Length == positions.Length)
                rotation = Quaternion.Slerp(rotations[index], rotations[index + 1], t);
        }
    }
}
