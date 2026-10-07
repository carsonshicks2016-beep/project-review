using System.Collections.Generic;
using UnityEngine;

namespace Core.Environment
{
    /// <summary>
    /// Rally furniture: the things that make a forest road into a special stage.
    ///
    ///   corners    course tape on stakes along the outside, red-and-white chevron boards
    ///              facing the cars as they arrive and pointing into the turn, and straw
    ///              bales stacked on the outside of the sharpest ones
    ///   straights  the occasional banner along the verge
    ///   ends       a banner gantry over the road at the start and at the finish
    ///
    /// Corners are found from the road itself, by how far it turns over about 24 m, so a
    /// frozen course and a fresh one are both furnished without any extra data. Look only,
    /// like the rest of the dressing: no colliders, nothing the AI can sense.
    /// </summary>
    public partial class StageDressing
    {
        const string FurniturePath = "StageDressing/Furniture";

        // Atlas regions (u0, v0, u1, v1). See FurnitureTextureBuilder for the layout.
        static readonly Rect White = new Rect(0.01f, 0.01f, 0.23f, 0.23f);
        static readonly Rect Tape = new Rect(0.26f, 0.01f, 0.23f, 0.23f);
        static readonly Rect Straw = new Rect(0.51f, 0.01f, 0.23f, 0.23f);
        static readonly Rect Wood = new Rect(0.76f, 0.01f, 0.23f, 0.23f);
        static readonly Rect Chevron = new Rect(0.0f, 0.25f, 0.5f, 0.25f);
        static readonly Rect Red = new Rect(0.52f, 0.27f, 0.46f, 0.21f);
        static readonly Rect StartBanner = new Rect(0f, 0.5f, 1f, 0.125f);
        static readonly Rect FinishBanner = new Rect(0f, 0.625f, 1f, 0.125f);
        static readonly Rect RallyBanner = new Rect(0f, 0.75f, 1f, 0.125f);
        static readonly Rect Blue = new Rect(0.02f, 0.88f, 0.96f, 0.10f);

        // Stages here bend in long sweeps rather than tight hairpins, so the turn is measured
        // over 40 m: over a shorter window even the sharpest bend reads as nearly straight.
        const float CornerWindow = 20f;   // metres either side used to measure the turn
        const float CornerAngle = 12f;    // degrees of turn over the window that makes a corner

        struct Corner { public float start, end, apex, turn; }

        static void BuildFurniture(TrackGenerator track, List<CentrePoint> centre, MeshCollider ground, Transform parent)
        {
            var material = Resources.Load<Material>(FurniturePath);
            if (material == null)
            {
                Debug.LogWarning($"[StageDressing] Resources/{FurniturePath} missing — no rally furniture.");
                return;
            }

            var mesh = new FurnitureMesh();
            var road = track.GetComponent<MeshCollider>();
            float roadHalf = track.roadWidth * 0.5f;
            float total = centre[centre.Count - 1].distance;
            var rng = new System.Random(track.seed * 3571 + 11);

            // Where the stage starts and ends, as distances along the centreline.
            float startAt = 0f, finishAt = total;
            if (track.waypoints.Count > track.spawnWaypointIndex)
                startAt = DistanceAt(centre, track.waypoints[track.spawnWaypointIndex]);
            if (track.waypoints.Count > track.FinishWaypointIndex)
                finishAt = DistanceAt(centre, track.waypoints[track.FinishWaypointIndex]);

            var corners = FindCorners(centre, total);
            int tapes = 0, boards = 0, bales = 0;
            foreach (var corner in corners)
            {
                if (corner.apex < startAt + 25f || corner.apex > finishAt - 10f) continue;
                int outside = corner.turn > 0f ? -1 : 1;   // a right turn's outside is the left

                // Tape along the outside, from before the corner to after it.
                Vector3? previous = null;
                for (float s = corner.start - 8f; s <= corner.end + 8f; s += 3.5f)
                {
                    Vector3 foot = Ground(track, ground, road, centre, s, outside * (roadHalf + 2.6f));
                    mesh.Box(foot + Vector3.up * 0.55f, new Vector3(0.06f, 1.1f, 0.06f), Quaternion.identity, Wood, Color.white);
                    if (previous.HasValue) mesh.Ribbon(previous.Value + Vector3.up * 0.95f, foot + Vector3.up * 0.95f, 0.07f, Tape);
                    previous = foot;
                }
                tapes++;

                // Chevrons around the apex, facing the arriving car, pointing into the turn.
                int count = Mathf.Abs(corner.turn) > CornerAngle * 1.3f ? 3 : 2;
                for (int c = 0; c < count; c++)
                {
                    float s = corner.apex + (c - (count - 1) * 0.5f) * 7f;
                    Sample(centre, s, out _, out Vector3 forward, out _, out _);
                    Vector3 foot = Ground(track, ground, road, centre, s, outside * (roadHalf + 1.7f));
                    mesh.Box(foot + Vector3.up * 0.7f, new Vector3(0.09f, 1.4f, 0.09f), Quaternion.LookRotation(forward), Wood, Color.white);
                    // The board faces back down the road, turned a little toward it.
                    Quaternion facing = Quaternion.LookRotation(-forward) * Quaternion.Euler(0f, outside * 20f, 0f);
                    mesh.Board(foot + Vector3.up * 1.15f, 1.3f, 0.72f, facing, Chevron, mirror: corner.turn < 0f);
                    boards++;
                }

                // Bales on the outside of the sharp ones, where a car that runs wide lands.
                if (Mathf.Abs(corner.turn) > CornerAngle * 1.3f)
                {
                    int n = 3 + rng.Next(3);
                    for (int b = 0; b < n; b++)
                    {
                        float s = corner.apex + (b - n * 0.5f) * 1.25f;
                        Sample(centre, s, out _, out Vector3 forward, out _, out _);
                        Vector3 foot = Ground(track, ground, road, centre, s, outside * (roadHalf + 1.1f));
                        var rot = Quaternion.LookRotation(forward) * Quaternion.Euler(0f, (float)rng.NextDouble() * 8f - 4f, 0f);
                        mesh.Box(foot + Vector3.up * 0.28f, new Vector3(0.55f, 0.5f, 1.15f), rot, Straw, Color.white);
                        if (b % 2 == 1 && b < n - 1)
                            mesh.Box(foot + Vector3.up * 0.78f + forward * 0.6f, new Vector3(0.55f, 0.5f, 1.15f), rot, Straw, new Color(0.92f, 0.9f, 0.85f));
                        bales++;
                    }
                }
            }

            // Banners on some of the straights, angled toward the arriving cars.
            int banners = 0;
            for (float s = startAt + 80f; s < finishAt - 40f; s += 140f + (float)rng.NextDouble() * 120f)
            {
                if (InCorner(corners, s, 15f)) continue;
                int side = rng.Next(2) == 0 ? -1 : 1;
                Sample(centre, s, out _, out Vector3 forward, out Vector3 right, out _);
                Vector3 foot = Ground(track, ground, road, centre, s, side * (roadHalf + 3.2f));
                Quaternion facing = Quaternion.LookRotation(-right * side) * Quaternion.Euler(0f, -side * 30f, 0f);
                Vector3 across = facing * Vector3.right;
                mesh.Box(foot + across * 2.1f + Vector3.up * 0.7f, new Vector3(0.1f, 1.4f, 0.1f), facing, Wood, Color.white);
                mesh.Box(foot - across * 2.1f + Vector3.up * 0.7f, new Vector3(0.1f, 1.4f, 0.1f), facing, Wood, Color.white);
                mesh.Board(foot + Vector3.up * 0.95f, 4.2f, 0.8f, facing, RallyBanner, mirror: false);
                banners++;
            }

            Gantry(mesh, track, ground, road, centre, Mathf.Min(startAt + 6f, total), roadHalf, StartBanner);
            Gantry(mesh, track, ground, road, centre, finishAt, roadHalf, FinishBanner);

            var root = new GameObject("Furniture") { layer = 2 };
            root.transform.SetParent(parent, false);
            mesh.Attach(root.transform, "Furniture", material, true);
            float sharpest = 0f;
            foreach (var c in corners) sharpest = Mathf.Max(sharpest, Mathf.Abs(c.turn));
            Debug.Log($"[StageDressing] furniture: {corners.Count} corners (sharpest {sharpest:0} deg), {tapes} taped, {boards} chevrons, {bales} bales, {banners} banners, 2 gantries");
        }

        /// <summary>A banner across the road on two pillars, reading correctly from both sides.</summary>
        static void Gantry(FurnitureMesh mesh, TrackGenerator track, MeshCollider ground, MeshCollider road,
                           List<CentrePoint> centre, float s, float roadHalf, Rect banner)
        {
            Sample(centre, s, out Vector3 at, out Vector3 forward, out _, out _);
            float span = roadHalf + 1.3f;
            Quaternion facing = Quaternion.LookRotation(-forward);
            float top = at.y;
            for (int side = -1; side <= 1; side += 2)
            {
                Vector3 foot = Ground(track, ground, road, centre, s, side * span);
                top = Mathf.Max(top, foot.y);
                mesh.Box(foot + Vector3.up * 2.9f, new Vector3(0.45f, 5.8f, 0.45f), facing, Blue, Color.white);
            }
            Vector3 middle = new Vector3(at.x, top + 4.9f, at.z);
            mesh.Board(middle - forward * 0.04f, span * 2f, 1.1f, facing, banner, mirror: false);
            mesh.Board(middle + forward * 0.04f, span * 2f, 1.1f, Quaternion.LookRotation(forward), banner, mirror: false);
        }

        /// <summary>
        /// The sharpest corner on the stage, as two world points on the centreline: one some
        /// way before the apex and the apex itself. For the offline review's corner shot.
        /// </summary>
        public static bool TryCornerView(TrackGenerator track, float before, out Vector3 approach, out Vector3 apex)
        {
            approach = apex = Vector3.zero;
            if (track.waypoints.Count < 4) return false;
            var centre = Centreline(track);
            float total = centre[centre.Count - 1].distance;
            Corner best = default; bool found = false;
            foreach (var c in FindCorners(centre, total))
                if (c.apex - before > 0f && (!found || Mathf.Abs(c.turn) > Mathf.Abs(best.turn))) { best = c; found = true; }
            if (!found) return false;
            Sample(centre, best.apex - before, out Vector3 a, out _, out _, out _);
            Sample(centre, best.apex, out Vector3 b, out _, out _, out _);
            approach = track.transform.TransformPoint(a);
            apex = track.transform.TransformPoint(b);
            return true;
        }

        /// <summary>Runs of the road where it turns by more than CornerAngle over the window.</summary>
        static List<Corner> FindCorners(List<CentrePoint> centre, float total)
        {
            var corners = new List<Corner>();
            bool open = false;
            var current = new Corner();
            float peak = 0f;
            for (float s = CornerWindow; s < total - CornerWindow; s += 2f)
            {
                Sample(centre, s - CornerWindow, out Vector3 a, out _, out _, out _);
                Sample(centre, s, out Vector3 b, out _, out _, out _);
                Sample(centre, s + CornerWindow, out Vector3 c, out _, out _, out _);
                Vector3 d1 = b - a, d2 = c - b; d1.y = 0f; d2.y = 0f;
                float turn = Vector3.SignedAngle(d1, d2, Vector3.up);

                bool inCorner = Mathf.Abs(turn) > CornerAngle;
                if (inCorner && open && Mathf.Sign(turn) != Mathf.Sign(current.turn)) { corners.Add(current); open = false; }
                if (inCorner && !open) { current = new Corner { start = s, end = s, apex = s, turn = turn }; peak = Mathf.Abs(turn); open = true; }
                else if (inCorner)
                {
                    current.end = s;
                    if (Mathf.Abs(turn) > peak) { peak = Mathf.Abs(turn); current.apex = s; current.turn = turn; }
                }
                else if (open) { corners.Add(current); open = false; }
            }
            if (open) corners.Add(current);
            return corners;
        }

        static bool InCorner(List<Corner> corners, float s, float margin)
        {
            foreach (var c in corners) if (s > c.start - margin && s < c.end + margin) return true;
            return false;
        }

        static float DistanceAt(List<CentrePoint> centre, Vector3 local)
        {
            float best = float.MaxValue, at = 0f;
            foreach (var c in centre)
            {
                float d = (c.position - local).sqrMagnitude;
                if (d < best) { best = d; at = c.distance; }
            }
            return at;
        }

        /// <summary>The ground under a point beside the road, in track space: terrain, else road, else the centreline height.</summary>
        static Vector3 Ground(TrackGenerator track, MeshCollider terrain, MeshCollider road, List<CentrePoint> centre, float s, float lateral)
        {
            Sample(centre, s, out Vector3 at, out _, out Vector3 right, out _);
            Vector3 local = at + right * lateral;
            Vector3 world = track.transform.TransformPoint(local);
            var ray = new Ray(world + Vector3.up * 300f, Vector3.down);
            if (terrain != null && terrain.Raycast(ray, out RaycastHit hit, 600f)) return track.transform.InverseTransformPoint(hit.point);
            if (road != null && road.Raycast(ray, out hit, 600f)) return track.transform.InverseTransformPoint(hit.point);
            return local;
        }

        /// <summary>Boxes, boards and ribbons, all textured from the furniture atlas.</summary>
        class FurnitureMesh : Batch
        {
            static Vector2 UV(Rect r, float u, float v) => new Vector2(r.x + r.width * u, r.y + r.height * v);

            public void Box(Vector3 centre, Vector3 size, Quaternion rotation, Rect region, Color tint)
            {
                Vector3 h = size * 0.5f;
                Vector3[] axes = { Vector3.right, Vector3.left, Vector3.up, Vector3.down, Vector3.forward, Vector3.back };
                foreach (var n in axes)
                {
                    // Two axes spanning this face.
                    Vector3 u = Mathf.Abs(n.y) > 0.5f ? Vector3.right : Vector3.Cross(Vector3.up, n);
                    Vector3 v = Vector3.Cross(n, u);
                    Vector3 c = Vector3.Scale(n, h);
                    Vector3 du = Vector3.Scale(u, h), dv = Vector3.Scale(v, h);
                    float shade = n.y > 0.5f ? 1f : n.y < -0.5f ? 0.5f : 0.85f;
                    int b = verts.Count;
                    Vector3 wn = rotation * n;
                    Add(centre + rotation * (c - du - dv), wn, UV(region, 0f, 0f), tint * shade);
                    Add(centre + rotation * (c + du - dv), wn, UV(region, 1f, 0f), tint * shade);
                    Add(centre + rotation * (c + du + dv), wn, UV(region, 1f, 1f), tint * shade);
                    Add(centre + rotation * (c - du + dv), wn, UV(region, 0f, 1f), tint * shade);
                    Quad(b, b + 1, b + 2, b + 3);
                }
            }

            /// <summary>A flat board, facing along rotation's forward, readable from the front.</summary>
            public void Board(Vector3 centre, float width, float height, Quaternion rotation, Rect region, bool mirror)
            {
                // Seen from the front, the viewer's right is the board's -right.
                Vector3 right = rotation * Vector3.left * (width * 0.5f);
                Vector3 up = Vector3.up * (height * 0.5f);
                Vector3 n = rotation * Vector3.forward;
                float u0 = mirror ? 1f : 0f, u1 = mirror ? 0f : 1f;
                int b = verts.Count;
                Add(centre - right - up, n, UV(region, u0, 0f), Color.white);
                Add(centre + right - up, n, UV(region, u1, 0f), Color.white);
                Add(centre + right + up, n, UV(region, u1, 1f), Color.white);
                Add(centre - right + up, n, UV(region, u0, 1f), Color.white);
                Quad(b, b + 1, b + 2, b + 3);
            }

            /// <summary>Course tape from one stake to the next, sagging a little in the middle.</summary>
            public void Ribbon(Vector3 a, Vector3 b, float height, Rect region)
            {
                const int Segments = 4;
                Vector3 up = Vector3.up * (height * 0.5f);
                Vector3 n = Vector3.Cross(b - a, Vector3.up).normalized;
                int start = verts.Count;
                for (int i = 0; i <= Segments; i++)
                {
                    float t = i / (float)Segments;
                    Vector3 p = Vector3.Lerp(a, b, t) + Vector3.down * (0.08f * 4f * t * (1f - t));
                    Add(p - up, n, UV(region, t, 0.3f), Color.white);
                    Add(p + up, n, UV(region, t, 0.7f), Color.white);
                }
                for (int i = 0; i < Segments; i++)
                {
                    int s = start + i * 2;
                    Quad(s, s + 2, s + 3, s + 1);
                }
            }
        }
    }
}
