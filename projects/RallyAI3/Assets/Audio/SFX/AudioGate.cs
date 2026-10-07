using UnityEngine;
using Unity.MLAgents;

namespace Audio
{
    /// <summary>
    /// Whether the procedural sources should be making any sound at all.
    ///
    /// Training is the case this exists for. ML-Agents runs the simulation at whatever
    /// --time-scale the trainer was launched with, and the audio thread does not care:
    /// it asks for its next buffer every few milliseconds of WALL time regardless. At 20x
    /// the engine's RPM lurches by a second's worth of revs between one buffer and the
    /// next, so what comes out is not an engine, it is tearing — and it costs CPU on every
    /// one of the millions of steps a run takes.
    ///
    /// Headless runs are already safe, because -batchmode has no audio device at all. The
    /// exposure is specifically training from inside the editor, which is how this project
    /// is usually driven.
    /// </summary>
    public static class AudioGate
    {
        public static bool ShouldRun
        {
            get
            {
                if (Application.isBatchMode) return false;

                // Deliberately a window rather than Approximately(). A trainer running at
                // 1.5x is still a trainer, and an editor paused at timeScale 0 has to fall
                // silent rather than hold one buffer's worth of engine as a drone.
                if (Time.timeScale < 0.99f || Time.timeScale > 1.01f) return false;

                // IsInitialized has to come first: reading Academy.Instance CREATES the
                // singleton, so asking the question the other way round would spin up an
                // Academy in scenes that never wanted one.
                if (Academy.IsInitialized && Academy.Instance.IsCommunicatorOn) return false;

                return true;
            }
        }
    }
}
