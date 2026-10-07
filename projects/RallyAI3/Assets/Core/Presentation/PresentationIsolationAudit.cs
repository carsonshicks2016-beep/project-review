using System;
using System.Collections;
using System.IO;
using Core.Environment;
using Core.ML;
using UnityEngine;
using UnityEngine.Rendering;

namespace Core.Presentation
{
    public sealed class PresentationIsolationAudit : MonoBehaviour
    {
        [Serializable] class Report
        {
            public bool headless, managed, viewer;
            public int dressing, renovation, legacyDust, wheelEffects, forestLods;
            public int audioControllers, liveSynths, ambience, impactVoices, generatedAudioClips;
            public int cockpits;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            if (string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("RALLY_PRESENTATION_AUDIT"))) return;
            new GameObject("Presentation Isolation Audit").AddComponent<PresentationIsolationAudit>();
        }

        IEnumerator Start()
        {
            yield return new WaitForSecondsRealtime(2);
            var report = new Report {
                headless = SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null,
                managed = LabRuntime.Enabled, viewer = LabRuntime.Enabled && LabRuntime.Config.viewer,
                dressing = FindObjectsByType<StageDressing>(FindObjectsSortMode.None).Length,
                renovation = FindObjectsByType<RallyVisualRenovation>(FindObjectsSortMode.None).Length,
                legacyDust = FindObjectsByType<DustPlume>(FindObjectsSortMode.None).Length,
                wheelEffects = FindObjectsByType<RallyDustController>(FindObjectsSortMode.None).Length
            };
            report.audioControllers = FindObjectsByType<Audio.AudioPerspectiveController>(FindObjectsSortMode.None).Length;
            report.cockpits = FindObjectsByType<RallyCockpit>(FindObjectsSortMode.None).Length;
            report.ambience = FindObjectsByType<Audio.ForestAmbience>(FindObjectsSortMode.None).Length;
            report.impactVoices = FindObjectsByType<Audio.ImpactSynth>(FindObjectsSortMode.None).Length;
            foreach (var synth in FindObjectsByType<Audio.ProceduralAudio>(FindObjectsSortMode.None))
                if (synth.IsLive) report.liveSynths++;
            foreach (var clip in Resources.FindObjectsOfTypeAll<AudioClip>())
                if (clip.name == "EngineSynth" || clip.name == "TyreSynth" || clip.name == "WindSynth" ||
                    clip.name == "ImpactSynth" || clip.name == "ForestAmbience") report.generatedAudioClips++;
            foreach (var lod in FindObjectsByType<LODGroup>(FindObjectsSortMode.None))
                if (lod.name.StartsWith("ForestV3_")) report.forestLods++;
            string folder = System.Environment.GetEnvironmentVariable("RALLY_PRESENTATION_AUDIT");
            Directory.CreateDirectory(folder);
            File.WriteAllText(Path.Combine(folder, "isolation.json"), JsonUtility.ToJson(report, true));
            Debug.Log("[PresentationIsolationAudit] " + JsonUtility.ToJson(report));
            Destroy(gameObject);
        }
    }
}
