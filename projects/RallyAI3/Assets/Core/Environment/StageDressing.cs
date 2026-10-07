using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Core.Environment
{
    /// <summary>
    /// What a person watching sees and the car never touches: the forest, the road and ground
    /// surfaces, and (through StageAtmosphere) the sun, sky and glare.
    ///
    /// LOOK ONLY, ON PURPOSE. The generator's own trees and boulders are part of training —
    /// they carry colliders, the ray sensor reports them, hitting one ends the episode, and
    /// frozen courses and checkpoints depend on them staying exactly where a seed puts them.
    /// Everything built here has no collider, no tag and sits on the Ignore Raycast layer,
    /// so the AI's world is the same with or without it. Two more rules keep it that way:
    ///
    ///  * It never draws from UnityEngine.Random. The generator places its scenery from that
    ///    shared stream, so one extra draw here would move every collidable tree and road
    ///    rock for that seed. The forest has its own System.Random, seeded from the stage.
    ///  * It never builds without a graphics device. Training players run with -nographics,
    ///    so the six workers pay nothing for it at all.
    ///
    /// WHY A SEPARATE COMPONENT rather than more of TrackGenerator. Frozen courses are baked
    /// prefabs loaded from an asset bundle; their generator never runs. This reads only what
    /// a frozen course keeps — the waypoints, the terrain mesh, the props — so the same forest
    /// stands on a frozen course, a freshly generated one, and the offline review renders.
    ///
    /// THE LOOK is a Colin McRae 04 forest stage: a wall of tall, dark spruce standing a few
    /// metres back from the road, dense enough that you cannot see through it to where the
    /// ground stops. The treeline starts at the generator's own prop clearance, so the visual
    /// trees are never nearer the road than the collidable ones and the car can never be seen
    /// driving through a trunk.
    /// </summary>
    public partial class StageDressing : MonoBehaviour
    {
        const string ForestName = "Forest";
        const string MaterialPath = "StageDressing/ForestTrees";
        const string RoadMaterialPath = "StageDressing/Road";
        const string GroundMaterialPath = "StageDressing/Ground";
        const string VergeFoliagePath = "StageDressing/VergeFoliage";
        const string VergeStonesPath = "StageDressing/VergeStones";

        // ── Forest layout, in metres ──
        const float StationStep = 2.3f;       // along the road, between columns of trees
        const float TreelineWobble = 5f;      // how far the front edge wanders back from the clearance
        const float RowSpacingFront = 2.9f;   // lateral spacing at the front of the forest
        const float RowSpacingBack = 4.6f;    // and at the terrain edge
        const float EdgeMargin = 1.5f;        // stay this far inside the terrain's outer edge
        const float PropGap = 2.5f;           // and this far from any collidable tree or boulder
        const float ChunkLength = 100f;       // one mesh per this much stage, for culling

        TrackGenerator dressed;
        GameObject dressedTerrain;
        float nextCheck;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null) return;
            if (Core.ML.LabRuntime.Enabled && !Core.ML.LabRuntime.Config.viewer) return;
            new GameObject("StageDressing").AddComponent<StageDressing>();
        }

        void LateUpdate()
        {
            // Polled, not evented: the course can be swapped for a frozen one before this
            // exists, and regenerated between episodes, and both just change which terrain
            // object is live.
            if (Time.unscaledTime < nextCheck) return;
            nextCheck = Time.unscaledTime + 0.5f;

            // Every car gets a dust plume, once. Cars can be spawned after the stage is.
            foreach (var car in FindObjectsByType<Core.Physics.VehicleController>(FindObjectsSortMode.None))
                if (car.GetComponent<DustPlume>() == null &&
                    car.GetComponent<Core.Presentation.RallyDustController>() == null)
                    car.gameObject.AddComponent<DustPlume>();

            var track = FindAnyObjectByType<TrackGenerator>();
            if (track == null || track.terrain == null) return;
            if (track.importedCircuit) { DressCircuitNearby(track); return; }
            if (track == dressed && track.terrain == dressedTerrain) return;

            Dress(track);
            dressed = track;
            dressedTerrain = track.terrain;
        }

        /// <summary>Build (or rebuild) the forest for this stage. Safe to call outside play mode.</summary>
        public static void Dress(TrackGenerator track)
        {
            if (track.importedCircuit) { DressCircuitNearby(track, true); return; }
            Transform old = track.transform.Find(ForestName);
            if (old != null)
            {
                old.gameObject.SetActive(false);
                if (Application.isPlaying) Destroy(old.gameObject);
                else DestroyImmediate(old.gameObject);
            }

            StageAtmosphere.Apply(track);

            // The road surface. Swapped rather than edited, because a frozen course brings
            // its own copy of the old material inside its bundle.
            var roadMaterial = Resources.Load<Material>(RoadMaterialPath);
            var roadRenderer = track.GetComponent<MeshRenderer>();
            if (roadMaterial != null && roadRenderer != null) roadRenderer.sharedMaterial = roadMaterial;
            var groundMaterial = Resources.Load<Material>(GroundMaterialPath);
            var groundRenderer = track.terrain != null ? track.terrain.GetComponent<MeshRenderer>() : null;
            if (groundMaterial != null && groundRenderer != null)
            {
                groundMaterial.SetFloat("_UVMetres", track.terrainTileMetres);
                groundMaterial.SetFloat("_RoadHalfWidth", track.roadWidth * .5f);
                groundMaterial.SetFloat("_Treeline", track.propClearance);
                groundRenderer.sharedMaterial = groundMaterial;
            }
            if (roadMaterial != null) roadMaterial.SetFloat("_UVMetres", track.roadTileMetres);

            var material = Resources.Load<Material>(MaterialPath);
            if (material == null)
            {
                Debug.LogWarning($"[StageDressing] Resources/{MaterialPath} missing — no forest.");
                return;
            }
            var ground = track.terrain != null ? track.terrain.GetComponent<MeshCollider>() : null;
            if (ground == null || track.waypoints.Count < 4) return;

            var centre = Centreline(track);
            if (centre.Count < 2) return;

            var root = new GameObject(ForestName);
            root.layer = 2;   // Ignore Raycast
            root.transform.SetParent(track.transform, false);
            root.AddComponent<DressingMeshOwner>();
            var reviewForest = UsesReviewKit(track) ? new ReviewForest(root.transform, material) : null;
            if (track.props != null)
            {
                foreach (var renderer in track.props.GetComponentsInChildren<MeshRenderer>())
                    if (renderer.name == "Trees") renderer.enabled = reviewForest == null;
                if (reviewForest != null)
                    foreach (var trunk in track.props.GetComponentsInChildren<CapsuleCollider>())
                    {
                        if (trunk.name != "Tree") continue;
                        Vector3 foot = track.transform.InverseTransformPoint(trunk.transform.position);
                        float best = float.MaxValue, station = 0f;
                        foreach (var point in centre)
                        {
                            float distance = (point.position - foot).sqrMagnitude;
                            if (distance >= best) continue;
                            best = distance; station = point.distance;
                        }
                        float yaw = Mathf.Repeat(foot.x * 13.73f + foot.z * 7.19f, 360f);
                        reviewForest.Add(foot, trunk.height, yaw, .5f, .5f, false, .5f, station, true, trunk.radius);
                    }
            }

            var props = PropGrid(track);
            var rng = new System.Random(track.seed * 7919 + 104729);
            var chunks = new Dictionary<int, TreeMesh>();
            int planted = 0;

            float clearance = track.propClearance + 0.5f;
            float outer = track.terrainHalfWidth - EdgeMargin;
            float total = centre[centre.Count - 1].distance;

            for (float s = 0f; s < total; s += StationStep)
            {
                Sample(centre, s, out Vector3 at, out Vector3 forward, out Vector3 right, out int hint);

                for (int side = -1; side <= 1; side += 2)
                {
                    // The front edge wanders, so the treeline is a wall with bays in it, not
                    // a hedge trimmed to a constant distance.
                    float wobble = Mathf.PerlinNoise(s * 0.021f + (side > 0 ? 31.7f : 7.3f), track.seed % 997 * 0.13f);
                    float lateral = clearance + wobble * TreelineWobble + (float)rng.NextDouble() * 1.2f;

                    while (lateral < outer)
                    {
                        // Every draw for this tree happens whether or not it is kept, so a
                        // rejected spot never shifts the trees after it.
                        float along = ((float)rng.NextDouble() - 0.5f) * StationStep * 1.4f;
                        float sway = ((float)rng.NextDouble() - 0.5f) * 1.6f;
                        float size = (float)rng.NextDouble();
                        float lean = (float)rng.NextDouble();
                        float shade = (float)rng.NextDouble();
                        float yaw = (float)rng.NextDouble() * 360f;

                        Vector3 local = at + right * (side * (lateral + sway)) + forward * along;
                        float depth = Mathf.InverseLerp(clearance, outer, lateral);
                        bool edge = depth < 0.12f;
                        lateral += Mathf.Lerp(RowSpacingFront, RowSpacingBack, depth) * (0.75f + 0.5f * (float)rng.NextDouble());

                        // Mostly mature spruce, with young ones at the edge where the light
                        // reaches — they are what fills the gaps between the big trunks.
                        float height = edge && size < 0.3f
                            ? Mathf.Lerp(4.5f, 9f, size / 0.3f)
                            : Mathf.Lerp(11f, 24f, size * size) * Mathf.Lerp(1f, 1.1f, depth);

                        if (NearestRoad(centre, local, hint) < clearance) continue;
                        if (NearProp(props, local)) continue;

                        Vector3 world = track.transform.TransformPoint(local);
                        var ray = new Ray(world + Vector3.up * 300f, Vector3.down);
                        if (!ground.Raycast(ray, out RaycastHit hit, 600f)) continue;
                        Vector3 foot = track.transform.InverseTransformPoint(hit.point);

                        if (reviewForest != null && reviewForest.Contains(s))
                        {
                            reviewForest.Add(foot, height, yaw, lean, shade, edge, depth, s);
                            continue;
                        }

                        int key = Mathf.FloorToInt(s / ChunkLength);
                        if (!chunks.TryGetValue(key, out TreeMesh mesh)) chunks[key] = mesh = new TreeMesh();
                        mesh.AddSpruce(foot, height, yaw, lean, shade, edge);
                        planted++;
                    }
                }
            }

            foreach (var pair in chunks)
                pair.Value.Attach(root.transform, $"Forest_{pair.Key:00}", material);
            reviewForest?.Attach();

            BuildBacking(track, centre, ground, outer, material, root.transform);
            BuildVerge(track, centre, ground, props, root.transform);
            BuildFurniture(track, centre, ground, root.transform);
            var meshOwner = root.GetComponent<DressingMeshOwner>();
            foreach (var filter in root.GetComponentsInChildren<MeshFilter>()) meshOwner.Register(filter.sharedMesh);

            Debug.Log($"[StageDressing] forest: {planted} trees in {chunks.Count} chunks (seed {track.seed})");
        }

        /// <summary>
        /// A dark band standing along the terrain's outer edge, behind the last row of trees.
        ///
        /// The ground stops 45 m out. Where it falls away from the road, the gaps between the
        /// trunks looked straight past that edge into the sky, and a forest you can see sky
        /// through at eye level reads as a single row of trees. This is the inside of the
        /// forest instead: near-black green, from well below the ground (so a dip never
        /// shows its foot) to the height of the undergrowth. Fog takes it the rest of the way.
        /// </summary>
        static void BuildBacking(TrackGenerator track, List<CentrePoint> centre, MeshCollider ground,
                                 float outer, Material material, Transform parent)
        {
            var verts = new List<Vector3>(); var normals = new List<Vector3>();
            var uvs = new List<Vector2>(); var colors = new List<Color32>(); var tris = new List<int>();
            var dark = new Color32(34, 46, 38, 255);
            float total = centre[centre.Count - 1].distance;

            for (int side = -1; side <= 1; side += 2)
            {
                int start = verts.Count, columns = 0;
                for (float s = 0f; s <= total; s += 4f)
                {
                    Sample(centre, s, out Vector3 at, out _, out Vector3 right, out _);
                    Vector3 local = at + right * (side * (outer + 1f));
                    Vector3 world = track.transform.TransformPoint(local);
                    float y = ground.Raycast(new Ray(world + Vector3.up * 300f, Vector3.down), out RaycastHit hit, 600f)
                        ? track.transform.InverseTransformPoint(hit.point).y : at.y;
                    Vector3 facing = -right * side;
                    // Top at undergrowth height, but never below the road's eye line: where the
                    // ground drops away, 7 m above it can still be under the camera.
                    float top = Mathf.Max(y + 7f, at.y + 10f);
                    verts.Add(new Vector3(local.x, y - 40f, local.z)); verts.Add(new Vector3(local.x, top, local.z));
                    normals.Add(facing); normals.Add(facing);
                    uvs.Add(new Vector2(0.75f, 0.1f)); uvs.Add(new Vector2(0.75f, 0.9f));
                    colors.Add(dark); colors.Add(dark);
                    columns++;
                }
                for (int c = 0; c < columns - 1; c++)
                {
                    int a = start + c * 2, b = a + 2;
                    // Both windings. The band is seen from one side in practice, but which side
                    // is front depends on which way the road happens to curve here, and a
                    // culled band is an invisible one.
                    tris.Add(a); tris.Add(a + 1); tris.Add(b); tris.Add(b); tris.Add(a + 1); tris.Add(b + 1);
                    tris.Add(a); tris.Add(b); tris.Add(a + 1); tris.Add(b); tris.Add(b + 1); tris.Add(a + 1);
                }
            }

            var mesh = new Mesh { name = "ForestBacking", indexFormat = IndexFormat.UInt32 };
            mesh.SetVertices(verts); mesh.SetNormals(normals); mesh.SetUVs(0, uvs);
            mesh.SetColors(colors); mesh.SetTriangles(tris, 0); mesh.RecalculateBounds();
            var go = new GameObject("ForestBacking") { layer = 2 };
            go.transform.SetParent(parent, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
        }

        /// <summary>
        /// The ground between the road and the trees: grass, ferns and stones.
        ///
        /// Placed by distance from the road, the way a real verge sorts itself out:
        ///   road edge   stones thrown off by every car, and little grass
        ///   open verge  grass tufts, in clumps rather than an even carpet
        ///   treeline    taller grass and ferns, in the shade where ferns grow
        /// Nothing lands on the road surface, including on the inside of a corner, and
        /// nothing grows out of a collidable tree or boulder.
        /// </summary>
        static void BuildVerge(TrackGenerator track, List<CentrePoint> centre, MeshCollider ground,
                               Dictionary<long, List<Vector3>> props, Transform parent)
        {
            var plantMaterial = Resources.Load<Material>(VergeFoliagePath);
            var stoneMaterial = Resources.Load<Material>(VergeStonesPath);
            if (plantMaterial == null || stoneMaterial == null)
            {
                Debug.LogWarning("[StageDressing] verge materials missing — no verge dressing.");
                return;
            }

            var root = new GameObject("Verge") { layer = 2 };
            root.transform.SetParent(parent, false);

            var rng = new System.Random(track.seed * 6151 + 97);
            var plants = new Dictionary<int, PlantMesh>();
            var stones = new Dictionary<int, StoneMesh>();
            int tufts = 0, ferns = 0, rocks = 0;

            float roadHalf = track.roadWidth * 0.5f;
            float near = roadHalf + 0.25f;
            float treeline = track.propClearance;
            float far = treeline + 5f;
            float total = centre[centre.Count - 1].distance;
            const float Step = 0.7f;
            const int PerSide = 14;

            for (float s = 0f; s < total; s += Step)
            {
                Sample(centre, s, out Vector3 at, out Vector3 forward, out Vector3 right, out int hint);
                for (int side = -1; side <= 1; side += 2)
                {
                    for (int k = 0; k < PerSide; k++)
                    {
                        // Fixed draws per candidate, kept or not, so the layout is stable.
                        float u = (float)rng.NextDouble();
                        float roll = (float)rng.NextDouble();
                        float along = ((float)rng.NextDouble() - 0.5f) * Step;
                        float size = (float)rng.NextDouble();
                        float yaw = (float)rng.NextDouble() * 360f;
                        float hue = (float)rng.NextDouble();

                        float lateral = Mathf.Lerp(near, far, u);
                        Vector3 local = at + right * (side * lateral) + forward * along;

                        // Clumps: a slow field along and across the verge.
                        float clump = Mathf.PerlinNoise(s * 0.09f + side * 17.3f, lateral * 0.21f + track.seed % 101);
                        float edgeZone = 1f - Mathf.InverseLerp(roadHalf, roadHalf + 2.5f, lateral);
                        float shade = Mathf.InverseLerp(treeline - 2f, treeline + 2f, lateral);

                        float pStone = 0.03f + 0.35f * edgeZone;
                        float pFern = shade * 0.28f * Mathf.InverseLerp(0.35f, 0.7f, clump);
                        float pTuft = (1f - edgeZone * 0.85f) * Mathf.Lerp(0.015f, 0.38f, Mathf.SmoothStep(0.42f, 0.76f, clump));

                        int kind = roll < pStone ? 0 : roll < pStone + pFern ? 1 : roll < pStone + pFern + pTuft ? 2 : -1;
                        if (kind < 0) continue;
                        if (NearestRoad(centre, local, hint) < near) continue;
                        if (NearProp(props, local)) continue;

                        Vector3 world = track.transform.TransformPoint(local);
                        if (!ground.Raycast(new Ray(world + Vector3.up * 300f, Vector3.down), out RaycastHit hit, 600f)) continue;
                        Vector3 foot = track.transform.InverseTransformPoint(hit.point);
                        int key = Mathf.FloorToInt(s / ChunkLength);

                        if (kind == 0)
                        {
                            if (!stones.TryGetValue(key, out var batch)) stones[key] = batch = new StoneMesh();
                            float sz = Mathf.Lerp(0.09f, 0.45f, size * size * size);
                            Color tone = Color.Lerp(new Color(0.85f, 0.82f, 0.76f), new Color(1.05f, 0.95f, 0.80f), hue);
                            batch.AddStone(foot, sz, yaw, s + k * 0.37f, tone);
                            rocks++;
                        }
                        else
                        {
                            if (!plants.TryGetValue(key, out var batch)) plants[key] = batch = new PlantMesh();
                            if (kind == 1)
                            {
                                Color tint = Color.Lerp(new Color(0.40f, 0.57f, 0.48f), new Color(0.65f, 0.70f, 0.51f), hue);
                                batch.AddFern(foot, Mathf.Lerp(0.6f, 1.15f, size), yaw, tint);
                                ferns++;
                            }
                            else
                            {
                                // Short near the road, where it is driven over and kicked; long
                                // toward the trees. Some of it dry and pale.
                                float height = Mathf.Lerp(0.14f, 0.62f, size * size) * Mathf.Lerp(1f, 1.4f, shade);
                                Color tint = hue < 0.8f
                                    ? Color.Lerp(new Color(0.40f, 0.52f, 0.43f), new Color(0.68f, 0.70f, 0.55f), hue / 0.8f)
                                    : new Color(0.72f, 0.66f, 0.49f);
                                if (size < .23f) batch.AddFern(foot, height * 1.15f, yaw, tint);
                                else batch.AddTuft(foot, height * (hue < .4f ? .7f : 1f), yaw, tint * .83f);
                                tufts++;
                            }
                        }
                    }
                }
            }

            // No shadows from either: thousands of small casters for marks a few centimetres
            // long, which the shadow map is too coarse to draw anyway.
            foreach (var pair in plants) pair.Value.Attach(root.transform, $"VergePlants_{pair.Key:00}", plantMaterial, false);
            foreach (var pair in stones) pair.Value.Attach(root.transform, $"VergeStones_{pair.Key:00}", stoneMaterial, false);
            Debug.Log($"[StageDressing] verge: {tufts} tufts, {ferns} ferns, {rocks} stones");
        }

        // ══════════════════════════════════════════════════════════════
        //  ROAD
        // ══════════════════════════════════════════════════════════════

        struct CentrePoint { public Vector3 position; public float distance; }

        /// <summary>
        /// The road's centreline in the track's local space, rebuilt the same way the
        /// generator builds the road: Catmull-Rom through the waypoints. The waypoints alone
        /// are 50 m apart, and a straight line between two of them cuts the inside of a
        /// corner by metres — enough to plant trees on the road.
        /// </summary>
        static List<CentrePoint> Centreline(TrackGenerator track)
        {
            var w = track.waypoints;
            int steps = Mathf.Max(4, track.resolutionPerSegment);
            var points = new List<CentrePoint>();
            float distance = 0f;
            for (int i = 0; i < w.Count - 3; i++)
            {
                for (int j = (i == 0 ? 0 : 1); j <= steps; j++)
                {
                    Vector3 p = SplineMath.GetCatmullRomPosition(j / (float)steps, w[i], w[i + 1], w[i + 2], w[i + 3]);
                    if (points.Count > 0) distance += Vector3.Distance(points[points.Count - 1].position, p);
                    points.Add(new CentrePoint { position = p, distance = distance });
                }
            }
            return points;
        }

        static void Sample(List<CentrePoint> c, float s, out Vector3 at, out Vector3 forward, out Vector3 right, out int index)
        {
            int lo = 0, hi = c.Count - 1;
            while (hi - lo > 1)
            {
                int mid = (lo + hi) / 2;
                if (c[mid].distance <= s) lo = mid; else hi = mid;
            }
            float span = c[hi].distance - c[lo].distance;
            float t = span > 0f ? (s - c[lo].distance) / span : 0f;
            at = Vector3.Lerp(c[lo].position, c[hi].position, t);
            forward = c[hi].position - c[lo].position;
            forward.y = 0f;
            forward = forward.sqrMagnitude > 1e-6f ? forward.normalized : Vector3.forward;
            right = Vector3.Cross(Vector3.up, forward);
            index = lo;
        }

        /// <summary>
        /// Horizontal distance to the nearest centreline point. A tree that is 14 m to the
        /// side of THIS station can be 4 m from the road a little further on, on the inside
        /// of a hairpin, so the check has to look along the road and not just across it.
        /// </summary>
        static float NearestRoad(List<CentrePoint> c, Vector3 p, int hint)
        {
            float best = float.MaxValue;
            int from = Mathf.Max(0, hint - 120), to = Mathf.Min(c.Count - 1, hint + 120);
            for (int i = from; i <= to; i++)
            {
                float dx = c[i].position.x - p.x, dz = c[i].position.z - p.z;
                float d = dx * dx + dz * dz;
                if (d < best) best = d;
            }
            return Mathf.Sqrt(best);
        }

        // ══════════════════════════════════════════════════════════════
        //  COLLIDABLE PROPS
        // ══════════════════════════════════════════════════════════════

        /// <summary>The generator's own trees and boulders, bucketed on a 4 m grid in track space.</summary>
        static Dictionary<long, List<Vector3>> PropGrid(TrackGenerator track)
        {
            var grid = new Dictionary<long, List<Vector3>>();
            if (track.props == null) return grid;
            foreach (Transform child in track.props.transform)
            {
                if (child.GetComponent<Collider>() == null) continue;
                Vector3 p = track.transform.InverseTransformPoint(child.position);
                long key = Cell(p);
                if (!grid.TryGetValue(key, out var list)) grid[key] = list = new List<Vector3>();
                list.Add(p);
            }
            return grid;
        }

        static bool NearProp(Dictionary<long, List<Vector3>> grid, Vector3 p)
        {
            int cx = Mathf.FloorToInt(p.x / 4f), cz = Mathf.FloorToInt(p.z / 4f);
            for (int x = cx - 1; x <= cx + 1; x++)
            for (int z = cz - 1; z <= cz + 1; z++)
            {
                if (!grid.TryGetValue(((long)x << 32) ^ (uint)z, out var list)) continue;
                foreach (var q in list)
                {
                    float dx = q.x - p.x, dz = q.z - p.z;
                    if (dx * dx + dz * dz < PropGap * PropGap) return true;
                }
            }
            return false;
        }

        static long Cell(Vector3 p) => ((long)Mathf.FloorToInt(p.x / 4f) << 32) ^ (uint)Mathf.FloorToInt(p.z / 4f);

        // ══════════════════════════════════════════════════════════════
        //  SPRUCE
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// Low-poly spruce merged into one mesh per stretch of stage. About forty vertices a
        /// tree: a five-sided trunk and four stacked five-sided cones. Shape comes from the
        /// silhouette and the colour comes from vertex tint, which is how a stage of several
        /// thousand of them stays affordable and still does not look like one tree copied.
        /// </summary>
        /// <summary>
        /// Geometry for one kind of dressing over one stretch of stage, merged into a single
        /// mesh: a few draw calls for thousands of items, and still culled by stretch.
        /// </summary>
        class Batch
        {
            protected readonly List<Vector3> verts = new List<Vector3>();
            protected readonly List<Vector3> normals = new List<Vector3>();
            protected readonly List<Vector2> uvs = new List<Vector2>();
            protected readonly List<Color32> colors = new List<Color32>();
            protected readonly List<int> tris = new List<int>();

            protected void Add(Vector3 p, Vector3 n, Vector2 uv, Color c)
            {
                verts.Add(p); normals.Add(n); uvs.Add(uv);
                colors.Add((Color32)new Color(Mathf.Clamp01(c.r), Mathf.Clamp01(c.g), Mathf.Clamp01(c.b), 1f));
            }

            protected void Quad(int a, int b, int c, int d)
            {
                tris.Add(a); tris.Add(b); tris.Add(c);
                tris.Add(a); tris.Add(c); tris.Add(d);
            }

            protected static float Fraction(float x) => x - Mathf.Floor(x);

            public void Attach(Transform parent, string name, Material material, bool castShadows = true)
            {
                if (verts.Count == 0) return;
                var mesh = new Mesh { name = name };
                mesh.indexFormat = IndexFormat.UInt32;
                mesh.SetVertices(verts);
                mesh.SetNormals(normals);
                mesh.SetUVs(0, uvs);
                mesh.SetColors(colors);
                mesh.SetTriangles(tris, 0);
                mesh.RecalculateBounds();

                var go = new GameObject(name);
                go.layer = 2;
                go.transform.SetParent(parent, false);
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                var owner = parent.GetComponentInParent<DressingMeshOwner>();
                if (owner != null) owner.Register(mesh);
                var renderer = go.AddComponent<MeshRenderer>();
                renderer.sharedMaterial = material;
                renderer.shadowCastingMode = castShadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
                renderer.receiveShadows = true;
            }
        }

        class TreeMesh : Batch
        {
            const int Sides = 8;

            // The foliage texture is bark on the left half and needles on the right; the
            // insets keep fetches clear of the seam. Same atlas the generator's trees use.
            const float BarkU0 = 0.02f, BarkU1 = 0.47f;
            const float NeedleU0 = 0.53f, NeedleU1 = 0.98f;

            public void AddSpruce(Vector3 foot, float height, float yawDegrees, float lean, float shade, bool edge)
            {
                Quaternion yaw = Quaternion.Euler(0f, yawDegrees, 0f);
                // A slight lean, so the forest is not a parade ground.
                Vector3 top = yaw * new Vector3((lean - 0.5f) * height * 0.03f, 0f, 0f);

                // Tint: from a cold blue-green to a warmer yellow-green, and darker overall
                // for some — a forest edge is many shades, never one.
                Color needle = Color.Lerp(new Color(0.60f, 0.76f, 0.80f), new Color(0.92f, 0.94f, 0.74f), shade * shade)
                             * Mathf.Lerp(0.58f, 0.92f, Fraction(shade * 7.31f));
                Color bark = new Color(0.95f, 0.90f, 0.85f) * Mathf.Lerp(0.75f, 1.0f, shade);

                // ── Trunk, sunk a little so it never floats on a slope ──
                float radius = height * 0.016f + 0.08f;
                float trunkTop = height * 0.92f;
                int b = verts.Count;
                for (int s = 0; s <= Sides; s++)
                {
                    float a = s / (float)Sides * Mathf.PI * 2f;
                    Vector3 o = yaw * new Vector3(Mathf.Cos(a), 0f, Mathf.Sin(a));
                    float u = Mathf.Lerp(BarkU0, BarkU1, s / (float)Sides);
                    Add(foot + o * radius + Vector3.down * 0.4f, o, new Vector2(u, 0f), bark * 0.55f);
                    Add(foot + o * radius * 0.35f + top + Vector3.up * trunkTop, o, new Vector2(u, 1f), bark);
                }
                for (int s = 0; s < Sides; s++)
                {
                    int c = b + s * 2, n = c + 2;
                    tris.Add(c); tris.Add(c + 1); tris.Add(n);
                    tris.Add(n); tris.Add(c + 1); tris.Add(n + 1);
                }

                // ── Canopy: stacked cones, widest at the bottom. Trees on the forest edge get
                //    light all the way down and keep their branches nearly to the ground;
                //    the ones behind lose their lower branches and show bare trunk. That
                //    difference is what makes the edge read as a wall and the inside as
                //    depth. The lower tiers are darker, because the inside of a spruce
                //    never sees the sun.
                float canopyBase = height * (edge ? Mathf.Lerp(0.03f, 0.10f, lean) : Mathf.Lerp(0.20f, 0.34f, lean));
                float canopySpan = height - canopyBase;
                int tiers = 6 + (int)(Fraction(shade * 3.7f + lean * 5.1f) * 3f);
                for (int tier = 0; tier < tiers; tier++)
                {
                    float f = tier / (float)tiers;
                    float y0 = canopyBase + canopySpan * f * 0.86f;
                    float y1 = Mathf.Min(height, y0 + canopySpan * (0.25f - f * 0.06f));
                    float rad = height * 0.145f * (1f - f * 0.78f) * Mathf.Lerp(0.68f, 1.22f, lean);
                    float skirt = canopySpan * 0.06f;   // tips droop below the tier's base
                    Color tint = needle * Mathf.Lerp(0.55f, 1.0f, f + 0.25f);
                    float k = Mathf.Lerp(0f, 1f, f + 0.25f);

                    int apex = verts.Count;
                    Vector3 branchOffset = yaw * new Vector3(
                        Mathf.Sin(tier * 2.17f + shade * 9f), 0f,
                        Mathf.Cos(tier * 1.73f + lean * 8f)) * rad * 0.22f;
                    Vector3 apexPos = foot + top * (y1 / height) + Vector3.up * y1 + branchOffset;
                    Add(apexPos, Vector3.up, new Vector2((NeedleU0 + NeedleU1) * 0.5f, 1f), tint * 1.08f);
                    for (int s = 0; s <= Sides; s++)
                    {
                        float a = (s / (float)Sides + tier * 0.5f / Sides) * Mathf.PI * 2f;
                        Vector3 o = yaw * new Vector3(Mathf.Cos(a), 0f, Mathf.Sin(a));
                        // Ragged: each branch tip a different length and droop, from a hash of
                        // the tree and the tip, so a tier is never a clean cone.
                        float r = Fraction(Mathf.Sin(foot.x * 12.9898f + foot.z * 78.233f + tier * 37.7f + (s % Sides) * 4.13f) * 43758.55f);
                        Vector3 p = foot + top * (y0 / height) + o * rad * Mathf.Lerp(0.48f, 1.22f, r)
                                  + Vector3.up * (y0 - skirt * Mathf.Lerp(0.4f, 1.6f, r));
                        Vector3 n = (o * (y1 - y0) + Vector3.up * rad).normalized;
                        Add(p, n, new Vector2(Mathf.Lerp(NeedleU0, NeedleU1, s / (float)Sides), 0f), tint * Mathf.Lerp(0.8f, 1f, k));
                    }
                    for (int s = 0; s < Sides; s++)
                    {
                        tris.Add(apex); tris.Add(apex + 2 + s); tris.Add(apex + 1 + s);
                    }

                    // Underside. The chase camera rides 3 m up and the lowest tiers start
                    // around there, so it looks up into them; an open cone is see-through
                    // from inside. Dark, because nothing lights the underside of a spruce.
                    int hub = verts.Count;
                    Add(apexPos + Vector3.down * (y1 - y0 + skirt * 0.4f), Vector3.down,
                        new Vector2((NeedleU0 + NeedleU1) * 0.5f, 0.2f), tint * 0.45f);
                    for (int s = 0; s < Sides; s++)
                    {
                        tris.Add(hub); tris.Add(apex + 1 + s); tris.Add(apex + 2 + s);
                    }
                }
            }
        }

        // ══════════════════════════════════════════════════════════════
        //  VERGE
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// Grass tufts and ferns on cut-out cards. A tuft is three cards crossed like an
        /// asterisk; a fern is six fronds radiating from a crown and arching outward.
        /// </summary>
        class PlantMesh : Batch
        {
            // Atlas: grass on the left half, a fern frond on the right.
            const float GrassU0 = 0.01f, GrassU1 = 0.49f, FernU0 = 0.51f, FernU1 = 0.99f;

            public void AddTuft(Vector3 foot, float height, float yawDegrees, Color tint)
            {
                float width = height * 1.6f;
                for (int c = 0; c < 3; c++)
                {
                    Quaternion yaw = Quaternion.Euler(0f, yawDegrees + c * 60f, 0f);
                    Vector3 across = yaw * Vector3.right * (width * 0.5f);
                    Vector3 lean = yaw * Vector3.forward * (height * 0.12f * (c - 1));
                    // Normals lean up: grass should light like the ground it grows from, not
                    // like a wall facing whichever way the card happens to.
                    Vector3 n = (Vector3.up * 2f + yaw * Vector3.forward).normalized;
                    int b = verts.Count;
                    Add(foot - across + Vector3.down * 0.05f, n, new Vector2(GrassU0, 0f), tint * 0.55f);
                    Add(foot + across + Vector3.down * 0.05f, n, new Vector2(GrassU1, 0f), tint * 0.55f);
                    Add(foot + across + lean + Vector3.up * height, n, new Vector2(GrassU1, 1f), tint);
                    Add(foot - across + lean + Vector3.up * height, n, new Vector2(GrassU0, 1f), tint);
                    Quad(b, b + 1, b + 2, b + 3);
                }
            }

            public void AddFern(Vector3 foot, float length, float yawDegrees, Color tint)
            {
                const int Fronds = 6;
                for (int f = 0; f < Fronds; f++)
                {
                    float angle = yawDegrees + f * (360f / Fronds) + Mathf.Sin(f * 2.3f + yawDegrees) * 18f;
                    Quaternion yaw = Quaternion.Euler(0f, angle, 0f);
                    Vector3 outward = yaw * Vector3.forward, side = yaw * Vector3.right;
                    float half = length * 0.2f;
                    // Two segments: up and out from the crown, then arching back down.
                    Vector3 p0 = foot + Vector3.up * 0.05f;
                    Vector3 p1 = p0 + outward * (length * 0.55f) + Vector3.up * (length * 0.45f);
                    Vector3 p2 = p0 + outward * length + Vector3.up * (length * 0.18f);
                    Vector3 n = Vector3.up;
                    int b = verts.Count;
                    Add(p0 - side * half * 0.4f, n, new Vector2(FernU0, 0f), tint * 0.6f);
                    Add(p0 + side * half * 0.4f, n, new Vector2(FernU1, 0f), tint * 0.6f);
                    Add(p1 + side * half, n, new Vector2(FernU1, 0.55f), tint);
                    Add(p1 - side * half, n, new Vector2(FernU0, 0.55f), tint);
                    Add(p2 + side * half * 0.5f, n, new Vector2(FernU1, 1f), tint * 1.05f);
                    Add(p2 - side * half * 0.5f, n, new Vector2(FernU0, 1f), tint * 1.05f);
                    Quad(b, b + 1, b + 2, b + 3);
                    Quad(b + 3, b + 2, b + 4, b + 5);
                }
            }
        }

        /// <summary>Small stones: a jittered octahedron each, half sunk into the ground.</summary>
        class StoneMesh : Batch
        {
            static readonly Vector3[] Corners =
            {
                Vector3.right, Vector3.left, Vector3.up, Vector3.down, Vector3.forward, Vector3.back
            };
            static readonly int[] Faces =
            {
                2, 4, 0,  2, 0, 5,  2, 5, 1,  2, 1, 4,
                3, 0, 4,  3, 5, 0,  3, 1, 5,  3, 4, 1
            };

            public void AddStone(Vector3 foot, float size, float yawDegrees, float seed, Color tint)
            {
                Quaternion yaw = Quaternion.Euler(0f, yawDegrees, 0f);
                // Flat-shaded: three vertices per face, so each facet catches the light.
                for (int f = 0; f < Faces.Length; f += 3)
                {
                    int b = verts.Count;
                    var p = new Vector3[3];
                    for (int k = 0; k < 3; k++)
                    {
                        int c = Faces[f + k];
                        float j = 0.7f + 0.6f * Fraction(Mathf.Sin(seed * 91.7f + c * 13.3f) * 437.5f);
                        Vector3 v = Corners[c] * j;
                        v.y *= 0.55f;   // stones lie flat
                        p[k] = foot + yaw * (v * size * 0.5f) + Vector3.down * size * 0.12f;
                    }
                    Vector3 n = Vector3.Cross(p[1] - p[0], p[2] - p[0]).normalized;
                    for (int k = 0; k < 3; k++)
                        Add(p[k], n, new Vector2(p[k].x, p[k].z) * 1.6f, tint);
                    tris.Add(b); tris.Add(b + 1); tris.Add(b + 2);
                }
            }
        }
    }
}
