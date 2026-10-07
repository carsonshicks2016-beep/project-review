using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.Audio;

namespace EditorScripts
{
    public static class RallyAudioAssets
    {
        const string PathName = "Assets/Resources/Audio/RallyMix.mixer";
        const BindingFlags Flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;

        [MenuItem("Rally/Audio/Prepare Mixer")]
        public static void Prepare()
        {
            var existing = AssetDatabase.LoadAssetAtPath<AudioMixer>(PathName);
            if (existing != null) { Validate(existing); return; }
            Directory.CreateDirectory("Assets/Resources/Audio");
            AssetDatabase.Refresh();
            // Unity exposes mixer authoring to its editor only; isolate the version-specific bridge here.
            var type = typeof(Editor).Assembly.GetType("UnityEditor.Audio.AudioMixerController", true);
            var mixer = (AudioMixer)type.GetMethod("CreateMixerControllerAtPath", Flags)
                .Invoke(null, new object[] { PathName });
            var master = type.GetProperty("masterGroup", Flags).GetValue(mixer);
            var create = type.GetMethod("CreateNewGroup", Flags);
            var parent = type.GetMethod("AddChildToParent", Flags);
            foreach (string name in new[] { "Engine", "Tyres", "Wind", "Impacts", "Ambience" })
            {
                var group = create.Invoke(mixer, new object[] { name, false });
                parent.Invoke(mixer, new[] { group, master });
            }
            var snapshot = type.GetProperty("TargetSnapshot", Flags).GetValue(mixer);
            master.GetType().GetMethod("SetValueForVolume", Flags).Invoke(master, new[] { mixer, snapshot, (object)(-3f) });
            EditorUtility.SetDirty(mixer);
            AssetDatabase.SaveAssets();
            Validate(mixer);
        }

        static void Validate(AudioMixer mixer)
        {
            foreach (string name in new[] { "Engine", "Tyres", "Wind", "Impacts", "Ambience" })
                if (mixer.FindMatchingGroups(name).Length != 1)
                    throw new InvalidOperationException("Audio mixer group missing: " + name);
            Debug.Log("Rally audio mixer: five independent groups, master headroom -3 dB.");
        }
    }
}
