using System;
using System.IO;
using System.Reflection;
using Core.Environment;
using Core.ML;
using Core.Physics;
using Core.Presentation;
using UnityEditor;
using UnityEngine;

namespace EditorScripts
{
    public static class EffectsReview
    {
        public static void Validate()
        {
            var root = new GameObject("Effects reset validation");
            try
            {
                var vehicle = root.AddComponent<VehicleController>();
                var legacy = root.AddComponent<DustPlume>();
                var dust = Child<ParticleSystem>(root, "Dust");
                var gravel = Child<ParticleSystem>(root, "Gravel");
                var trails = new TrailRenderer[RallyDustController.TrackSegmentsPerWheel];
                for (int i = 0; i < trails.Length; i++)
                    trails[i] = Child<TrailRenderer>(root, "Track" + i);
                var controller = root.AddComponent<RallyDustController>();
                controller.Configure(vehicle, new PacejkaTireModel[] { null },
                    new[] { dust }, new[] { gravel }, trails, null);
                if (legacy.enabled) throw new Exception("Legacy dust still enabled");
                var reset = typeof(RallyDustController).GetMethod("OnEpisodeEnded",
                    BindingFlags.Instance | BindingFlags.NonPublic);
                for (int repeat = 0; repeat < 8; repeat++)
                {
                    dust.Emit(12); gravel.Emit(7);
                    foreach (var trail in trails)
                    {
                        trail.emitting = true;
                        trail.AddPosition(Vector3.zero); trail.AddPosition(Vector3.forward);
                    }
                    if (dust.particleCount == 0 || trails[0].positionCount == 0)
                        throw new Exception("Reset fixture did not populate effects");
                    reset.Invoke(controller, new object[] { EpisodeOutcome.TimedOut, 0 });
                    if (dust.particleCount != 0 || gravel.particleCount != 0)
                        throw new Exception("Particles survived episode reset");
                    foreach (var trail in trails)
                        if (trail.positionCount != 0 || trail.emitting)
                            throw new Exception("Track survived episode reset");
                }
                string folder = Path.Combine(Directory.GetParent(Application.dataPath).FullName,
                    ".rally/visual-review/v3");
                Directory.CreateDirectory(folder);
                File.WriteAllText(Path.Combine(folder, "effects-validation.json"),
                    "{\"repeatedResets\":8,\"legacyDustDisabled\":true,\"particlesCleared\":true,\"tracksCleared\":true}");
                Debug.Log("[EffectsReview] Eight populated episode-reset checks passed");
            }
            finally { UnityEngine.Object.DestroyImmediate(root); }
        }

        static T Child<T>(GameObject root, string name) where T : Component
        {
            var child = new GameObject(name);
            child.transform.SetParent(root.transform, false);
            return child.AddComponent<T>();
        }
    }
}
