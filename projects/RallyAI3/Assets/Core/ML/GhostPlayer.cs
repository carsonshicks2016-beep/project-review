using System.IO;
using UnityEngine;

namespace Core.ML
{
    /// <summary>
    /// Replays a recorded path as a translucent car running alongside the live one.
    ///
    /// The ghost is driven by nothing — it is a playback head on a list of poses, so it
    /// has no physics, no collider and no effect on the race. It restarts whenever the
    /// live agent's episode does, which is what keeps the two synchronised: both start at
    /// the same instant on the same stage, and the gap between them at any moment is the
    /// difference between the two drivers.
    ///
    /// See <see cref="GhostPath"/> for why this exists at all rather than simply loading
    /// the old policy.
    /// </summary>
    public class GhostPlayer : MonoBehaviour
    {
        [Tooltip("Ghost file to replay, relative to the project root or absolute.")]
        public string ghostFile = "";

        [Tooltip("Agent to stay in step with. Found automatically if left empty.")]
        public RallyAgent syncTo;

        [Tooltip("Colour of the ghost body.")]
        public Color tint = new Color(0.25f, 0.75f, 1f, 0.45f);

        [Tooltip("Mesh to wear. Left empty, the ghost copies the live car's mesh, which is " +
                 "the honest thing to show — the same car, driven differently.")]
        public Mesh bodyMesh;

        private GhostPath path;
        private Transform body;
        private float startTime;

        /// <summary>The loaded path, or null if the file was missing or unreadable.</summary>
        public GhostPath Path => path;

        void Start()
        {
            if (syncTo == null) syncTo = FindAnyObjectByType<RallyAgent>();
            if (syncTo != null) syncTo.EpisodeEnded += Restart;

            if (!Load()) { enabled = false; return; }

            BuildBody();
            startTime = Time.time;
        }

        void OnDestroy()
        {
            if (syncTo != null) syncTo.EpisodeEnded -= Restart;
        }

        private void Restart(EpisodeOutcome outcome, int waypoints) => startTime = Time.time;

        void Update()
        {
            if (path == null || body == null) return;
            path.Sample(Time.time - startTime, out Vector3 position, out Quaternion rotation);
            body.SetPositionAndRotation(position, rotation);
        }

        private bool Load()
        {
            if (string.IsNullOrEmpty(ghostFile))
            {
                Debug.LogWarning("[Ghost] No ghost file set.", this);
                return false;
            }

            string full = System.IO.Path.IsPathRooted(ghostFile)
                ? ghostFile
                : System.IO.Path.Combine(Directory.GetCurrentDirectory(), ghostFile);

            if (!File.Exists(full))
            {
                Debug.LogWarning($"[Ghost] No ghost at {full}.", this);
                return false;
            }

            path = JsonUtility.FromJson<GhostPath>(File.ReadAllText(full));
            if (path == null || path.positions == null || path.positions.Length < 2)
            {
                Debug.LogWarning($"[Ghost] {full} holds no usable path.", this);
                return false;
            }

            // ── The stage has to match, and this is not a formality.
            //
            //    A ghost is a list of world positions, not a re-simulation. Replayed onto a
            //    different seed it drives serenely through trees and hangs in mid-air over
            //    the terrain, and it looks enough like a bug in the ghost system to waste an
            //    afternoon. Refuse instead of showing a lie.
            var track = FindAnyObjectByType<Core.Environment.TrackGenerator>();
            if (track != null && track.CurrentSeed != path.seed)
            {
                Debug.LogError(
                    $"[Ghost] {System.IO.Path.GetFileName(full)} was driven on stage {path.seed}, " +
                    $"but this scene is stage {track.CurrentSeed}. A ghost only means anything " +
                    "on the stage it was recorded on — set TrackGenerator.seed to " +
                    $"{path.seed} and regenerate.", this);
                return false;
            }

            Debug.Log($"[Ghost] {path.label}: {path.waypoints}/{path.target} waypoints, " +
                      $"{path.outcome}, {path.Duration:0.0} s on stage {path.seed}.", this);
            return true;
        }

        /// <summary>
        /// Builds the visible ghost. No Rigidbody and no Collider anywhere on it: a ghost
        /// that could be hit would change the race it exists to observe.
        /// </summary>
        private void BuildBody()
        {
            var go = new GameObject($"Ghost_{path.label}");
            go.transform.SetParent(transform, false);
            body = go.transform;

            Mesh mesh = bodyMesh;
            if (mesh == null)
            {
                // The live car's own mesh, so the comparison is like for like.
                var live = syncTo != null ? syncTo.GetComponentInChildren<MeshFilter>() : null;
                if (live != null) mesh = live.sharedMesh;
            }

            if (mesh != null)
            {
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
            }
            else
            {
                // Nothing to copy — a car-sized box still shows where the old run was.
                var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
                Object.Destroy(box.GetComponent<Collider>());
                box.transform.SetParent(go.transform, false);
                box.transform.localScale = new Vector3(1.73f, 1.35f, 4.34f);
                box.GetComponent<MeshRenderer>().sharedMaterial = GhostMaterial();
                return;
            }

            go.AddComponent<MeshRenderer>().sharedMaterial = GhostMaterial();
        }

        private Material GhostMaterial()
        {
            // Built here rather than referenced as an asset so a ghost can be dropped into
            // any scene without dragging a material dependency behind it.
            var shader = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Unlit/Color");
            var material = new Material(shader) { name = "GhostMaterial" };
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", tint);
            if (material.HasProperty("_Color")) material.SetColor("_Color", tint);
            return material;
        }
    }
}
