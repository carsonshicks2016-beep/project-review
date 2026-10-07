using UnityEngine;

namespace Core.Physics
{
    /// <summary>
    /// Drop this on any collider the car can drive on to tell the tyres what they
    /// are standing on. Without it a surface is treated as dry gravel (1.0).
    /// </summary>
    public class SurfaceProperties : MonoBehaviour
    {
        public enum Surface { Tarmac, Gravel, Dirt, Snow, Ice, Grass }

        [Tooltip("Preset that fills in the friction and rolling drag below.")]
        public Surface surface = Surface.Gravel;

        [Tooltip("Multiplier on the tyre's peak friction coefficient.")]
        [Range(0.05f, 2f)] public float gripMultiplier = 1.0f;

        [Tooltip("Extra rolling resistance from a soft or loose surface.")]
        [Range(0f, 0.2f)] public float rollingResistance = 0.022f;

        [Tooltip("How much loose material the tyres kick up. Drives the particle rate.")]
        [Range(0f, 1f)] public float looseness = 0.6f;

        void OnValidate() => ApplyPreset();
        void Reset() => ApplyPreset();

        public void ApplyPreset()
        {
            switch (surface)
            {
                case Surface.Tarmac: gripMultiplier = 1.25f; rollingResistance = 0.014f; looseness = 0.00f; break;
                case Surface.Gravel: gripMultiplier = 1.00f; rollingResistance = 0.022f; looseness = 0.60f; break;
                case Surface.Dirt:   gripMultiplier = 0.88f; rollingResistance = 0.030f; looseness = 0.80f; break;
                case Surface.Snow:   gripMultiplier = 0.45f; rollingResistance = 0.045f; looseness = 0.90f; break;
                case Surface.Ice:    gripMultiplier = 0.22f; rollingResistance = 0.010f; looseness = 0.10f; break;
                case Surface.Grass:  gripMultiplier = 0.62f; rollingResistance = 0.055f; looseness = 0.45f; break;
            }
        }

        /// <summary>
        /// Reads the surface under a contact. Falls back to dry gravel defaults when
        /// the collider has no SurfaceProperties on it or anywhere above it.
        /// </summary>
        public static void Sample(Collider col, out float grip, out float rollingDrag, out float looseness)
        {
            grip = 1f;
            rollingDrag = ImprezaSpec.RollingResistance;
            looseness = 0.6f;

            if (col == null) return;

            SurfaceProperties sp = col.GetComponentInParent<SurfaceProperties>();
            if (sp == null) return;

            grip = sp.gripMultiplier;
            rollingDrag = sp.rollingResistance;
            looseness = sp.looseness;
        }
    }
}
