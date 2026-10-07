using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Aegis.EditorTools
{
    /// <summary>
    /// Builds a standalone player of the Arena scene so the trainer can run headless
    /// (mlagents-learn --env=Build/AegisArena.app). Run: -executeMethod
    /// Aegis.EditorTools.BuildPlayerTool.BuildMac
    /// </summary>
    public static class BuildPlayerTool
    {
        public static void BuildMac()
        {
            var opts = new BuildPlayerOptions
            {
                scenes = new[] { "Assets/Scenes/Arena.unity" },
                locationPathName = "Build/AegisArena.app",
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.Development
            };
            var report = BuildPipeline.BuildPlayer(opts);
            Debug.Log($"[Aegis] Player build: {report.summary.result}, {report.summary.totalSize} bytes, " +
                      $"errors {report.summary.totalErrors}");
        }
    }
}
