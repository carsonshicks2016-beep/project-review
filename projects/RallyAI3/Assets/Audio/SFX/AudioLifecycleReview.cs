using System;
using System.Collections;
using System.IO;
using UnityEngine;
using Core.ML;

namespace Audio
{
    // Bounded native lifetime check, never installed during ordinary watching or training.
    public sealed class AudioLifecycleReview : MonoBehaviour
    {
        [Serializable] class Inventory
        {
            public int sources, listeners, clips, synths, controllers;
            public bool Matches(Inventory other) => sources == other.sources && listeners == other.listeners &&
                clips == other.clips && synths == other.synths && controllers == other.controllers;
        }
        [Serializable] class Report { public bool passed; public Inventory baseline; public Inventory[] reloads; }
        string folder;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            string path = Environment.GetEnvironmentVariable("RALLY_AUDIO_LIFECYCLE");
            if (string.IsNullOrEmpty(path) || Application.isBatchMode || !LabRuntime.Enabled || !LabRuntime.Config.viewer) return;
            new GameObject("Audio Lifetime Review").AddComponent<AudioLifecycleReview>().folder = path;
        }
        static Inventory Count()
        {
            var value = new Inventory {
                sources = FindObjectsByType<AudioSource>(FindObjectsSortMode.None).Length,
                synths = FindObjectsByType<ProceduralAudio>(FindObjectsSortMode.None).Length,
                controllers = FindObjectsByType<AudioPerspectiveController>(FindObjectsSortMode.None).Length };
            foreach (var listener in FindObjectsByType<AudioListener>(FindObjectsSortMode.None))
                if (listener.isActiveAndEnabled) value.listeners++;
            foreach (var clip in Resources.FindObjectsOfTypeAll<AudioClip>())
                if (clip.name == "EngineSynth" || clip.name == "TyreSynth" || clip.name == "WindSynth" ||
                    clip.name == "ImpactSynth" || clip.name == "ForestAmbience") value.clips++;
            return value;
        }
        IEnumerator Start()
        {
            yield return new WaitForSecondsRealtime(3);
            var report = new Report { passed = true, baseline = Count(), reloads = new Inventory[8] };
            for (int cycle = 0; cycle < 8; cycle++)
            {
                var root = new GameObject("Audio Reload Fixture");
                foreach (var type in new[] { typeof(EngineSynth), typeof(TyreSynth), typeof(WindSynth), typeof(ImpactSynth), typeof(ForestAmbience) })
                {
                    var node = new GameObject(type.Name); node.transform.SetParent(root.transform, false);
                    var voice = (ProceduralAudio)node.AddComponent(type); voice.volume = 0;
                    voice.ResetPlayback();
                }
                yield return null; yield return null;
                root.SetActive(false); Destroy(root);
                yield return null; yield return null;
                report.reloads[cycle] = Count();
                report.passed &= report.baseline.Matches(report.reloads[cycle]);
            }
            Directory.CreateDirectory(folder);
            File.WriteAllText(Path.Combine(folder, "audio-lifecycle.json"), JsonUtility.ToJson(report, true));
            Debug.Log("[AudioLifecycleReview] " + JsonUtility.ToJson(report));
            Destroy(gameObject);
        }
    }
}
