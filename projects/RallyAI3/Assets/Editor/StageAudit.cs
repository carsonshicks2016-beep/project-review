using System.Collections.Generic;
using System.Linq;
using Core.Environment;
using UnityEditor;
using UnityEngine;

namespace EditorScripts
{
    /// <summary>
    /// Checks that the stage the generator produced is the stage it thinks it produced.
    ///
    /// WHY THIS EXISTS
    ///
    /// Procedural terrain fails quietly, and the failures that matter here are the ones
    /// that make a stage unfair to drive rather than the ones that look wrong. An
    /// evaluation of rally09 recorded forty-five episodes ending against a tree with the
    /// car level, all four wheels on the ground, and one to three metres from the
    /// centreline — on a road twelve metres wide, with scenery placed no closer than
    /// thirteen metres. Both of those cannot be true, and no amount of reading the code
    /// settles which one is wrong.
    ///
    /// So this measures the stage rather than describing it:
    ///
    ///   * how far the scenery really is from the driving line, as the AGENT measures it;
    ///   * how far the agent's own centreline is from the one the road mesh is built on;
    ///   * whether the road ever passes close enough to itself that scenery placed beside
    ///     one part of it lands beside another.
    ///
    ///     Rally > Diagnose > Audit Stage Geometry
    /// </summary>
    public static class StageAudit
    {
        /// <summary>Stages to draw and measure. Enough that a rare failure shows up at all.</summary>
        const int Stages = 25;

        [MenuItem("Rally/Diagnose/Audit Stage Geometry", false, 0)]
        public static void Audit()
        {
            // Batch mode starts on an empty scene, so open the training one rather than
            // reporting "no TrackGenerator" at a caller who cannot open a scene by hand.
            const string scenePath = "Assets/Scenes/RallyTraining.unity";
            if (Object.FindAnyObjectByType<TrackGenerator>() == null &&
                System.IO.File.Exists(scenePath))
                UnityEditor.SceneManagement.EditorSceneManager.OpenScene(scenePath);

            var track = Object.FindAnyObjectByType<TrackGenerator>();
            if (track == null)
            {
                Debug.LogError($"[Audit] No TrackGenerator, and no scene at {scenePath}. " +
                               "Run Rally > Training > Build Training Scene first.");
                return;
            }

            int originalSeed = track.seed;
            var propDistances = new List<float>();
            var splineGaps = new List<float>();
            var selfApproaches = new List<float>();
            int stagesWithPropsOnRoad = 0;

            for (int s = 0; s < Stages; s++)
            {
                track.seed = 20260727 + s * 7919;      // a prime stride, so no seed repeats
                track.GenerateTrack();

                float nearestProp = MeasureProps(track, propDistances);
                if (nearestProp <= track.roadWidth * 0.5f) stagesWithPropsOnRoad++;

                splineGaps.Add(MeasureCentrelineDisagreement(track));
                selfApproaches.Add(MeasureSelfApproach(track));
            }

            track.seed = originalSeed;
            track.GenerateTrack();

            Report("scenery to the agent's centreline", propDistances, "m");
            Report("agent centreline vs road mesh centreline", splineGaps, "m");
            Report("closest the road comes to itself", selfApproaches, "m");

            Debug.Log(
                $"[Audit] {Stages} stages.\n" +
                $"  {stagesWithPropsOnRoad} of {Stages} had scenery within the road's own width " +
                $"({track.roadWidth:0} m) of the centreline the agent steers by.\n" +
                $"  propClearance is {track.propClearance:0} m, so anything under that is the " +
                "generator and the agent disagreeing about where the road is.");
        }

        /// <summary>
        /// Distance from every tree and boulder to the centreline AS THE AGENT MEASURES IT
        /// — LateralOffset, the same call that decides "left the stage". Returns the
        /// nearest one on this stage.
        /// </summary>
        static float MeasureProps(TrackGenerator track, List<float> into)
        {
            if (track.props == null) return float.MaxValue;

            float nearest = float.MaxValue;
            foreach (Transform child in track.props.transform)
            {
                if (child.name != "Tree" && child.name != "Boulder") continue;

                Vector3 world = track.transform.TransformPoint(child.localPosition);
                // Hint of -1 would clamp to the first segments; ask for the true minimum by
                // scanning, since this is an offline audit and cost does not matter.
                float best = float.MaxValue;
                for (int w = 0; w < track.waypoints.Count; w += 2)
                    best = Mathf.Min(best, track.LateralOffset(world, w));

                into.Add(best);
                nearest = Mathf.Min(nearest, best);
            }
            return nearest;
        }

        /// <summary>
        /// How far apart the two centrelines are.
        ///
        /// The agent's road-relative observations project onto the POLYLINE through the
        /// waypoints. The road mesh, and every prop placed beside it, is built on the
        /// Catmull-Rom SPLINE through those same waypoints. On a straight they coincide;
        /// through a corner the spline bows away from the chord, and the gap is a straight
        /// error in the number the policy steers by.
        /// </summary>
        static float MeasureCentrelineDisagreement(TrackGenerator track)
        {
            if (track.waypoints.Count < 4) return 0f;

            float worst = 0f;
            for (int i = 0; i < track.waypoints.Count - 3; i++)
            {
                Vector3 p0 = track.waypoints[i], p1 = track.waypoints[i + 1];
                Vector3 p2 = track.waypoints[i + 2], p3 = track.waypoints[i + 3];

                for (int step = 1; step < 8; step++)
                {
                    float t = step / 8f;
                    Vector3 onSpline = SplineMath.GetCatmullRomPosition(t, p0, p1, p2, p3);
                    Vector3 world = track.transform.TransformPoint(onSpline);

                    float best = float.MaxValue;
                    for (int w = 0; w < track.waypoints.Count; w += 2)
                        best = Mathf.Min(best, track.LateralOffset(world, w));
                    worst = Mathf.Max(worst, best);
                }
            }
            return worst;
        }

        /// <summary>
        /// Closest approach between two non-adjacent parts of the stage. Where this drops
        /// below the road width plus the prop clearance, scenery placed beside one part of
        /// the road is standing beside another part of it.
        /// </summary>
        static float MeasureSelfApproach(TrackGenerator track)
        {
            float closest = float.MaxValue;
            var w = track.waypoints;
            for (int a = 0; a < w.Count; a++)
                for (int b = a + 3; b < w.Count; b++)
                    closest = Mathf.Min(closest, Vector2.Distance(
                        new Vector2(w[a].x, w[a].z), new Vector2(w[b].x, w[b].z)));
            return closest;
        }

        static void Report(string what, List<float> values, string unit)
        {
            if (values.Count == 0) { Debug.Log($"[Audit] {what}: nothing measured."); return; }

            var sorted = values.Where(v => v < float.MaxValue).OrderBy(v => v).ToList();
            if (sorted.Count == 0) { Debug.Log($"[Audit] {what}: nothing measured."); return; }

            float At(float f) => sorted[Mathf.Min(sorted.Count - 1, (int)(f * sorted.Count))];
            Debug.Log($"[Audit] {what}: n={sorted.Count}  " +
                      $"min {sorted[0]:0.00}{unit}  1% {At(0.01f):0.00}{unit}  " +
                      $"median {At(0.5f):0.00}{unit}  max {sorted[sorted.Count - 1]:0.00}{unit}");
        }
    }
}
