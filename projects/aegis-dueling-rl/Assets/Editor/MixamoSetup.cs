using UnityEditor;
using UnityEngine;

namespace Aegis.EditorTools
{
    /// <summary>
    /// Configures every FBX under Assets/Mixamo as a **Humanoid** rig (so their AnimationClips
    /// retarget correctly and the imitation reward can read bones via Animator.GetBoneTransform).
    /// Run from the menu **Aegis ▸ Configure Mixamo Imports (Humanoid)** after adding new clips,
    /// or via -executeMethod Aegis.EditorTools.MixamoSetup.ConfigureAndImport in batch mode.
    /// </summary>
    public static class MixamoSetup
    {
        [MenuItem("Aegis/Configure Mixamo Imports (Humanoid)")]
        public static void ConfigureAndImport()
        {
            var guids = AssetDatabase.FindAssets("t:Model", new[] { "Assets/Mixamo" });
            int n = 0;
            foreach (var g in guids)
            {
                var path = AssetDatabase.GUIDToAssetPath(g);
                if (AssetImporter.GetAtPath(path) is ModelImporter mi)
                {
                    mi.animationType = ModelImporterAnimationType.Human;
                    mi.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
                    mi.SaveAndReimport();
                    n++;
                    Debug.Log($"[Aegis] Humanoid: {path}");
                }
            }
            Debug.Log($"[Aegis] Configured {n} Mixamo model(s) as Humanoid.");
        }
    }
}
