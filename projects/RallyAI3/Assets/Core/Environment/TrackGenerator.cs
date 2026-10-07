using UnityEngine;
using System.Collections.Generic;
using Core.Physics;   // SurfaceProperties — tells the tyres what they are standing on

namespace Core.Environment
{
    [DefaultExecutionOrder(-200)]
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public partial class TrackGenerator : MonoBehaviour
    {
        [HideInInspector] public bool frozenCourse;
        [Header("Determinism")]
        [Tooltip("Seeds the stage shape, the camber and the rock layout together. The same " +
                 "seed gives byte-identical terrain every run, which is what makes a training " +
                 "result reproducible and a bug reproducible.")]
        public int seed = 20260727;
        [Tooltip("Draw a fresh seed each time the track is built. On = every run is a different " +
                 "stage, which is what you want for a policy that has to generalise. Off = one " +
                 "fixed stage, which is what you want while debugging.")]
        public bool randomiseSeed = false;

        /// <summary>Seed the current terrain was actually built from.</summary>
        public int CurrentSeed { get; private set; }

        [Header("Generation Settings")]
        public int controlPointsCount = 20;
        public float trackLength = 1000f;
        public bool longStageRhythm;
        public float roadWidth = 12f;

        // ══════════════════════════════════════════════════════════════
        //  TESSELLATION, and why it is this fine
        //
        //  PS1_Lit maps textures AFFINELY — its UV is declared `noperspective`, so the
        //  rasterizer interpolates it linearly in screen space instead of dividing through
        //  by w. That is the effect we want, but its error grows with how much DEPTH varies
        //  across a single triangle, and it is therefore a function of polygon size, not of
        //  the shader.
        //
        //  At the old 10 samples per 50 m segment the road was one strip 12 m wide stepping
        //  5 m at a time, so the quad under the camera spanned a 2:1 range of depth and the
        //  gravel visibly bent and swam across the bottom of the frame. Real PS1 titles hit
        //  exactly this and answered it exactly this way: keep polygons small on screen.
        //
        //  40 steps over a 50 m segment is 1.25 m longitudinally, and 8 lanes across a 12 m
        //  road is 1.5 m laterally — roughly square quads, and about a sixteenth of the old
        //  area each. The cost is a few thousand vertices on a mesh that had a few hundred,
        //  which is nothing; the road is drawn in one draw call either way.
        // ══════════════════════════════════════════════════════════════
        [Tooltip("Spline samples per control-point segment. Drives how finely the road AND " +
                 "the terrain are tessellated along the stage. Raising it reduces affine " +
                 "texture warping; it no longer affects how many rocks spawn.")]
        [Range(4, 80)] public int resolutionPerSegment = 40;

        [Tooltip("Lanes the road mesh is split into across its width. Purely tessellation — " +
                 "it does not change the road's shape, only its triangle size, which is what " +
                 "controls how far the affine mapping can swim before a vertex corrects it.")]
        [Range(1, 16)] public int roadLanes = 8;
        
        // ══════════════════════════════════════════════════════════════
        //  TERRAIN NOISE
        //
        //  Every frequency here is in CYCLES PER METRE along the stage, so
        //  frequency * trackLength is literally "how many features you get". Keep an
        //  eye on that product: the previous version keyed shape and elevation off the
        //  control-point INDEX instead of distance, which meant the whole 1000 m stage
        //  sampled 1.1 and 0.11 of a Perlin cell respectively — one lazy bend and six
        //  centimetres of elevation change across the entire course.
        // ══════════════════════════════════════════════════════════════
        [Header("Stage Shape")]
        [Tooltip("How far the heading may swing either side of the stage's overall " +
                 "direction. Held below 90 on purpose: cos(heading) then stays positive, " +
                 "so every segment advances down +Z and the stage CANNOT cross itself.")]
        [Range(0f, 85f)] public float maxHeadingDeviation = 70f;
        [Tooltip("Corner features per metre. 0.004 puts a direction change every ~250 m. " +
                 "Corner sharpness is roughly maxHeadingDeviation * shapeFrequency — raise " +
                 "either for a twistier stage.")]
        public float shapeFrequency = 0.004f;

        [Header("Elevation & Camber")]
        [Tooltip("Peak-to-peak elevation change over the stage, in metres.")]
        public float elevationScale = 12f;
        [Tooltip("Crests and dips per metre. 0.004 gives one every ~250 m.")]
        public float elevationFrequency = 0.004f;
        [Tooltip("Peak banking angle, in degrees, either way.")]
        public float camberScale = 15f;
        [Tooltip("Camber changes per metre. 0.008 gives a new attitude every ~125 m.")]
        public float camberFrequency = 0.008f;

        // ══════════════════════════════════════════════════════════════
        //  TERRAIN
        //
        //  The stage is CUT INTO a landscape rather than laid on top of one. Terrain is
        //  swept outward from the road edge and blended to meet it exactly, so the road
        //  always sits in the ground however the ground moves.
        //
        //  The character is deliberately ASYMMETRIC and varies along the stage: one
        //  section has a bank rising on the left and the ground falling away on the
        //  right, the next opens out into flat country. That single property is most of
        //  what makes terrain read as a rally stage instead of a racetrack — and it is
        //  what gives the ray sensor something meaningful to tell apart, because "bank
        //  on the left" and "drop on the left" are different problems.
        //
        //  This replaced a pair of symmetric 1.2 m banks down both sides. Those solved
        //  the mechanics — the rays had something to hit, and the car stopped falling
        //  into empty space — but a matched wall either side is a bobsleigh run.
        // ══════════════════════════════════════════════════════════════
        [Header("Terrain")]
        public bool buildTerrain = true;
        [Tooltip("How far the ground extends either side of the CENTRELINE, in metres.")]
        public float terrainHalfWidth = 45f;
        [Tooltip("Lateral samples per side. Spacing is finer near the road, where it matters.")]
        [Range(4, 40)] public int terrainResolution = 16;
        [Tooltip("Biggest rise or fall the landform reaches at its outer edge, in metres.")]
        public float terrainRelief = 14f;
        [Tooltip("How often the landform changes character, per metre. 0.003 is a new " +
                 "bank or drop roughly every 300 m.")]
        public float terrainFrequency = 0.003f;
        [Tooltip("How often the stage swings between dramatic and open, per metre.")]
        public float dramaFrequency = 0.0015f;
        [Tooltip("Depth of the drainage ditch immediately beside the road. This is what " +
                 "catches a car that puts one wheel wide.")]
        public float ditchDepth = 0.7f;
        [Tooltip("How far out the ditch and shoulder extend before the landform takes over.")]
        public float shoulderWidth = 3.5f;
        [Tooltip("Small-scale bumpiness of the ground away from the road.")]
        public float terrainRoughness = 1.6f;

        // ══════════════════════════════════════════════════════════════
        //  TEXTURING
        //
        //  Ground texture coordinates are laid out in REAL METRES, not normalised per
        //  strip. The strips are wildly different widths — the innermost terrain step is
        //  under a metre across and the outermost is over thirty — so a 0..1 coordinate
        //  per strip would fit one texture repeat across each of them and the ground
        //  would visibly change scale as it recedes from the road.
        //
        //  The tile sizes below are therefore the real thing they claim to be: how much
        //  ground one repeat of the texture covers. Road and terrain are kept separate
        //  because gravel is a finer material than scrub and wants a shorter repeat.
        // ══════════════════════════════════════════════════════════════
        [Header("Texturing")]
        [Tooltip("Metres of road covered by one repeat of the gravel texture.")]
        public float roadTileMetres = 6f;
        [Tooltip("Metres of ground covered by one repeat of the terrain texture.")]
        public float terrainTileMetres = 12f;
        [Tooltip("Metres of boulder covered by one repeat of the stone texture. Much " +
                 "shorter than the ground tiles — a 2 m rock showing a twelfth of the " +
                 "texture is a rock with no visible surface at all.")]
        public float rockTileMetres = 1.4f;

        // ══════════════════════════════════════════════════════════════
        //  BACKDROP
        //
        //  The stage is a kilometre long but the ground stops at terrainHalfWidth — 45 m
        //  — while the camera sees out to 1200. Everything between was void, which is
        //  what made the stage read as a ribbon of road in an empty room.
        //
        //  This fills it with a ring of low hills built from the SAME noise field as the
        //  near terrain, so the horizon is continuous with the ground you are driving on
        //  rather than pasted in behind it. No collider and no Obstacle tag: it is out of
        //  reach of both the car and the ray sensor, so it cannot change the RL task.
        // ══════════════════════════════════════════════════════════════
        [Header("Backdrop")]
        public bool buildBackdrop = true;
        [Tooltip("Distance from the viewer to the ridge line, in metres. Wants to sit " +
                 "inside the camera's far clip and beyond the fog's end distance.")]
        public float backdropRadius = 400f;
        [Tooltip("Height of the ridge above its base, in metres, before noise.")]
        public float backdropHeight = 70f;
        [Tooltip("How far the hillside continues BELOW the horizon. Not decoration — this " +
                 "is what covers the gap wherever the near terrain falls away into a drop " +
                 "and you find yourself looking down past its outer edge.")]
        public float backdropDepth = 300f;
        [Tooltip("Segments around the ring. Low on purpose — these are silhouettes.")]
        [Range(12, 128)] public int backdropSegments = 48;
        [Tooltip("Material for the distant hills. Falls back to the terrain material.")]
        public Material backdropMaterial;

        // ══════════════════════════════════════════════════════════════
        //  SCENERY
        //
        //  Trees and boulders standing in the terrain. These are NOT decoration. They
        //  carry colliders, they are tagged Obstacle so the ray sensor reports them and
        //  hitting one ends the episode, and they are what makes leaving the road cost
        //  something beyond time. A drop you can sometimes drive out of; a tree you
        //  cannot.
        //
        //  Visuals are merged into one mesh per kind, because ~370 separate renderers
        //  makes the editor crawl. Colliders stay individual — they have to be — but a
        //  capsule or a box is close to free. Headless training skips rendering entirely,
        //  so the merge is purely so this is workable to look at.
        // ══════════════════════════════════════════════════════════════
        [Header("Scenery")]
        public bool scatterProps = true;
        [Tooltip("Nothing is placed closer to the centreline than this, leaving a clear " +
                 "corridor either side before the treeline starts. The road is 12 m wide.")]
        public float propClearance = 13f;
        [Tooltip("Trees per 100 m of stage.")]
        public float treesPer100m = 26f;
        [Tooltip("Boulders per 100 m of stage.")]
        public float rocksPer100m = 8f;
        public Vector2 treeHeightRange = new Vector2(4.5f, 9f);
        public Vector2 rockSizeRange = new Vector2(0.6f, 2.2f);
        public Material treeMaterial, rockMaterial;

        [Header("Obstacles")]
        public bool spawnObstacles = true;

        /// <summary>
        /// Rocks per 100 m, in the same units as <see cref="treesPer100m"/> and
        /// <see cref="rocksPer100m"/>.
        ///
        /// This used to be a per-SAMPLE probability, which quietly tied the difficulty of the
        /// stage to its tessellation: raising resolutionPerSegment to kill the affine warping
        /// would have multiplied the rock count by the same factor and made the curriculum's
        /// first rung four times harder for a reason that has nothing to do with driving.
        /// Density is the thing that was always meant; samples were just where the loop was.
        /// </summary>
        [Tooltip("Rocks per 100 m of stage. Independent of tessellation.")]
        [Range(0f, 20f)] public float obstaclesPer100m = 2f;
        [Tooltip("Nothing is placed within this radius of the car's spawn point. Without it a " +
                 "rock can be generated inside the car, and the episode starts with the chassis " +
                 "being depenetrated out of a box collider — which throws it straight onto its roof.")]
        public float obstacleSpawnClearance = 40f;

        [Header("Spawn")]
        [Tooltip("Waypoint the car starts from. Far enough in that the rear wheels are on the " +
                 "mesh — the road only exists between waypoints[1] and waypoints[count-2].")]
        public int spawnWaypointIndex = 3;

        [Tooltip("Material for the landscape. Falls back to the road material if unset.")]
        public Material terrainMaterial;

        public List<Vector3> waypoints = new List<Vector3>();
        private Mesh roadMesh;
        static Mesh[] roadRockMeshes;
        [SerializeField] private List<GameObject> obstacles = new List<GameObject>();

        // ══════════════════════════════════════════════════════════════
        //  ROAD ROCKS, AS DATA
        //
        //  The generator used to place a cube and forget everything about it. That made
        //  the one question worth asking after a crash unanswerable: was there anywhere
        //  else to go? Half of all failures across rally08-10 were an obstacle strike, at
        //  45-53 % regardless of the sensor upgrade that tripled detection range — which
        //  either means the policy cannot use what it sees, or means the stage sometimes
        //  offers no line through. Those call for opposite fixes and nothing on disk could
        //  tell them apart.
        //
        //  Keeping the station and the lateral offset of every rock costs a few hundred
        //  bytes per stage and turns that argument into <see cref="WidestGapAt"/>.
        // ══════════════════════════════════════════════════════════════

        /// <summary>One road rock, in road-relative terms rather than as a transform.</summary>
        public struct RoadRock
        {
            /// <summary>Metres along the stage, same convention as <see cref="RoadFrame.distanceAlong"/>.</summary>
            public float station;
            /// <summary>Metres from the centreline, positive to the RIGHT of the direction of travel.</summary>
            public float lateral;
            /// <summary>Half-width of the rock across the road, metres.</summary>
            public float halfWidth;
        }

        private readonly List<RoadRock> roadRocks = new List<RoadRock>();

        /// <summary>Every rock on the driveable surface of the current stage.</summary>
        public IReadOnlyList<RoadRock> RoadRocks => roadRocks;

        const string TerrainName = "Terrain";
        const string PropsName = "Scenery";
        const string BackdropName = "Backdrop";
        /// <summary>Trees and boulders. Tagged Obstacle — hitting one ends the episode.</summary>
        [field: SerializeField] public GameObject props { get; private set; }
        /// <summary>The landscape the road is cut into, tagged TrackBoundary.</summary>
        [field: SerializeField] public GameObject terrain { get; private set; }
        /// <summary>Distant hills. No collider, no tag — invisible to the agent.</summary>
        [field: SerializeField] public GameObject backdrop { get; private set; }

        // Per-sample spline frame, kept so the terrain pass can sweep outward from it.
        [SerializeField] private List<Vector3> samplePos = new List<Vector3>();
        [SerializeField] private List<Vector3> sampleRight = new List<Vector3>();
        [SerializeField] private List<float> sampleDistance = new List<float>();
        float noiseOriginCut, noiseOriginDrama, noiseOriginRough;

        void Awake()
        {
            if (frozenCourse) { CurrentSeed = seed; return; }
            if (Core.ML.LabRuntime.ReplaceCourse(this)) return;
            GenerateTrack();
        }

        public void GenerateTrack()
        {
            if (frozenCourse) { CurrentSeed = seed; return; }
            if (importedCircuit) { GenerateCircuit(); return; }
            if (randomiseSeed) seed = Random.Range(int.MinValue, int.MaxValue);
            UnityEngine.Random.State callerRandomState = Random.state;
            CurrentSeed = seed;

            // Perlin noise has no seed of its own, so the seed has to move the sampling
            // ORIGIN instead. Without this every stage came out the same shape however
            // the run was seeded — only the rocks moved.
            // Perlin repeats on a period of 256, so an origin inside that range reaches
            // every distinct piece of the field. Large offsets buy nothing and cost
            // float precision — a 10000-ish origin was flattening whole stretches of the
            // camber field to a single repeated value.
            Random.InitState(seed);
            noiseOriginShape  = Random.Range(0f, 256f);
            noiseOriginHeight = Random.Range(0f, 256f);
            noiseOriginCamber = Random.Range(0f, 256f);
            noiseOriginCut    = Random.Range(0f, 256f);
            noiseOriginDrama  = Random.Range(0f, 256f);
            noiseOriginRough  = Random.Range(0f, 256f);

            ClearObstacles();
            GenerateControlPoints();
            BuildRoadMesh();
            LogStageReport();
            Random.state = callerRandomState;
        }

        /// <summary>
        /// Prints what the generator actually produced, rather than what it was asked for.
        /// Procedural terrain fails quietly — the parameters look reasonable and the stage
        /// comes out flat, or straight, or folded through itself — and none of that is
        /// visible from the road at eye level.
        /// </summary>
        private void LogStageReport()
        {
            if (waypoints.Count < 4) { Debug.LogWarning("[Track] Too few waypoints to report on."); return; }

            float minHdg = float.MaxValue, maxHdg = float.MinValue, sharpest = 0f, minRadius = float.MaxValue;
            float prevHdg = 0f;
            float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
            float segmentLength = trackLength / controlPointsCount;

            for (int i = 0; i < waypoints.Count; i++)
            {
                minX = Mathf.Min(minX, waypoints[i].x); maxX = Mathf.Max(maxX, waypoints[i].x);
                minY = Mathf.Min(minY, waypoints[i].y); maxY = Mathf.Max(maxY, waypoints[i].y);

                if (i == 0) continue;
                Vector3 d = waypoints[i] - waypoints[i - 1];
                float hdg = Mathf.Atan2(d.x, d.z) * Mathf.Rad2Deg;
                minHdg = Mathf.Min(minHdg, hdg); maxHdg = Mathf.Max(maxHdg, hdg);

                if (i > 1)
                {
                    float turn = Mathf.DeltaAngle(prevHdg, hdg);
                    if (Mathf.Abs(turn) > Mathf.Abs(sharpest)) sharpest = turn;
                    // Radius of the arc that turns through `turn` over one segment.
                    float half = Mathf.Abs(turn) * 0.5f * Mathf.Deg2Rad;
                    if (half > 1e-4f) minRadius = Mathf.Min(minRadius, segmentLength / (2f * Mathf.Sin(half)));
                }
                prevHdg = hdg;
            }

            // Any pair of non-adjacent centreline points closer than the road width means
            // two pieces of road are sitting on top of each other.
            float closest = float.MaxValue; int ca = -1, cb = -1;
            for (int a = 0; a < waypoints.Count; a++)
                for (int b = a + 3; b < waypoints.Count; b++)
                {
                    float d = Vector2.Distance(new Vector2(waypoints[a].x, waypoints[a].z),
                                               new Vector2(waypoints[b].x, waypoints[b].z));
                    if (d < closest) { closest = d; ca = a; cb = b; }
                }

            string verdict = closest < roadWidth
                ? $"  ** OVERLAP: waypoints {ca} and {cb} are {closest:0.0} m apart on a {roadWidth:0} m road **"
                : "  no overlap";

            string land = terrain != null
                ? $"{terrainHalfWidth:0} m either side, relief +/-{terrainRelief:0} m, tagged {terrain.tag}"
                : "** NONE — the road is a ribbon in empty space and the rays see nothing **";

            // Scenery count AND how close it came to the driving line. The count alone said
            // nothing about whether the stage was fair: an evaluation found the car ending
            // episodes against trees 0.3-4 m from the centreline of a 12 m road, on a
            // generator configured never to place one closer than propClearance metres.
            // A number that is supposed to be a floor is worth printing next to the floor.
            int treeN = 0, rockN = 0;
            float nearestProp = float.MaxValue;
            if (props != null)
                foreach (Transform child in props.transform)
                {
                    if (child.name == "Tree") treeN++;
                    else if (child.name == "Boulder") rockN++;
                    else continue;

                    float d = LateralOffset(transform.TransformPoint(child.localPosition), 0);
                    for (int w = 2; w < waypoints.Count; w += 2)
                        d = Mathf.Min(d, LateralOffset(transform.TransformPoint(child.localPosition), w));
                    nearestProp = Mathf.Min(nearestProp, d);
                }

            // ── Scenery roots, plural.
            //
            //    ScatterProps replaces the previous stage's scenery with Destroy(), which in
            //    PLAY MODE is deferred to the end of the frame while the replacement is
            //    created immediately. If anything regenerates the stage twice before that
            //    frame ends, or the destroy does not land, the old trees stay in the world —
            //    standing wherever the OLD road went, which after a regeneration is often
            //    the middle of the new one.
            //
            //    That is not hypothetical. An evaluation ended episodes against trees 1.7 m
            //    from the centreline on stages this very report measured as 14.4 m clear.
            //    A count is the cheapest possible test for it, so it is printed every time.
            int sceneryRoots = 0;
            foreach (Transform child in transform)
                if (child.name == PropsName) sceneryRoots++;

            string leak = sceneryRoots > 1
                ? $"  ** {sceneryRoots} scenery roots — the previous stage's trees are still in the world **"
                : "";

            string clearance = nearestProp == float.MaxValue
                ? "no scenery"
                : nearestProp < propClearance - 1f
                    ? $"{nearestProp:0.0} m  ** BELOW the {propClearance:0} m clearance — scenery is on the road **"
                    : $"{nearestProp:0.0} m (clearance {propClearance:0} m)";

            Debug.Log(
                $"[Track] seed {CurrentSeed} — {trackLength:0} m, {roadWidth:0} m wide, {obstacles.Count} road rocks\n" +
                $"  scenery    {treeN} trees, {rockN} boulders, tagged Obstacle (collidable, ends the episode)\n" +
                $"  nearest    {clearance}{leak}\n" +
                $"  terrain    {land}\n" +
                $"  heading    {minHdg:+0.0;-0.0} .. {maxHdg:+0.0;-0.0} deg (span {maxHdg - minHdg:0.0})\n" +
                $"  sharpest   {sharpest:+0.0;-0.0} deg per {segmentLength:0} m" +
                (minRadius < float.MaxValue ? $"  ->  tightest radius {minRadius:0} m" : "") + "\n" +
                $"  lateral    {maxX - minX:0.0} m     elevation {maxY - minY:0.00} m\n" +
                $"  clearance  closest non-adjacent waypoints {closest:0.0} m{verdict}", this);
        }

        float noiseOriginShape, noiseOriginHeight, noiseOriginCamber;

        // ══════════════════════════════════════════════════════════════
        //  REPLACING A STAGE
        //
        //  Every generated child — scenery, terrain, backdrop, rocks — is replaced by
        //  destroying the old one and building a new one. In the editor that is exact,
        //  because DestroyImmediate is immediate. In PLAY MODE it was not, and the
        //  difference is the whole bug:
        //
        //    * Destroy() is deferred to the end of the frame. Until then the old object is
        //      still in the world, still has live colliders, and is still findable by name.
        //    * The replacement is created in the SAME frame, appended after it.
        //    * Transform.Find returns the FIRST match — which is the oldest object, the one
        //      already queued for destruction — so the next regeneration destroyed a corpse
        //      and left the live one alone.
        //
        //  The result was a steady state of THREE scenery roots: every stage carried two
        //  previous stages' trees, standing wherever those roads used to go, which after a
        //  regeneration is frequently the middle of the new one. Measured directly — stages
        //  whose own report said the nearest tree was 14.4 m away, ending episodes against
        //  trees 1.7 m from the centreline.
        //
        //  The same pattern applied to the terrain, which carries a MeshCollider tagged
        //  TrackBoundary, so the car was also hitting landforms from stages it had already
        //  left behind.
        //
        //  Renaming and deactivating before the Destroy fixes both halves: the object stops
        //  colliding immediately, and it can never be found by name again.
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// Removes EVERY generated child with this name, immediately in effect if not in
        /// memory. Every one, not the first — the leak above hid behind "there is only ever
        /// one of these".
        /// </summary>
        private void RemoveGenerated(string childName)
        {
            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                Transform child = transform.GetChild(i);
                if (child.name != childName) continue;
                RetireImmediately(child.gameObject);
            }
        }

        /// <summary>
        /// Takes an object out of the world now and out of memory whenever Unity gets to it.
        /// </summary>
        private void RetireImmediately(GameObject go)
        {
            if (go == null) return;

            if (!Application.isPlaying) { DestroyImmediate(go); return; }

            // Deactivating is what actually stops it colliding this frame; renaming is what
            // stops Transform.Find handing it back as if it were still the live one.
            go.name = go.name + " (retired)";
            go.SetActive(false);
            Destroy(go);
        }

        /// <summary>
        /// Regenerating the track rebuilds the road mesh in place but would leave the
        /// previous run's rocks behind, stacked on top of the new ones.
        /// </summary>
        private void ClearObstacles()
        {
            foreach (var obstacle in obstacles) RetireImmediately(obstacle);
            // The training scene has historically been saved with generated road rocks
            // as children. A new generator instance cannot track those in this runtime
            // list, so they survive a course regeneration unless the hierarchy is also
            // swept. This is especially visible when building a zero-rock course from a
            // rock-populated scene template.
            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                Transform child = transform.GetChild(i);
                if (child.name == "RoadRock") RetireImmediately(child.gameObject);
            }
            obstacles.Clear();
            roadRocks.Clear();
        }

        // ══════════════════════════════════════════════════════════════
        //  WHICH CENTRELINE
        //
        //  There are two, and for most of this project's life the agent used the wrong one.
        //
        //  The waypoints are twenty control points about fifty metres apart. The road mesh,
        //  its edges, the terrain sweep and every tree, boulder and rock are built on the
        //  Catmull-Rom SPLINE through them. The polyline through them is a different curve:
        //  through a corner the spline bows away from the chord, and measured over
        //  twenty-five generated stages the two disagree by 2.7 to 4.4 m.
        //
        //  The road's half-width is 6 m. So a car sitting exactly on the surface the mesh
        //  defines could read up to 4.4 m of lateral offset — most of the way to the edge —
        //  and a car reading a comfortable 2 m could be on the verge. That number is the
        //  policy's primary lane-keeping signal and the test that ends an episode for
        //  leaving the stage, and it was wrong by most of a lane.
        //
        //  It also explained an impossible measurement: forty-five evaluation episodes
        //  ended against a tree with the car level, all four wheels down and reading 1-3 m
        //  of offset, on a stage whose scenery is never placed closer than 13 m. Audited
        //  against this polyline, that same scenery came out as close as 9.5 m.
        //
        //  The fix is to project onto the same curve everything else is built on.
        //  samplePos already holds it — the mesh builder fills it every 1.25 m — so this
        //  is a change of which array to search, not new geometry.
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// Samples either side of the hint that are searched. 1.25 m apart at the default
        /// tessellation, so this is about 150 m of road in each direction — more than a car
        /// can cross between the waypoint hint being updated.
        /// </summary>
        const int SampleSearchRadius = 120;

        /// <summary>
        /// Turns a waypoint hint into an index into <see cref="samplePos"/>. Segment i of
        /// the mesh spans waypoints[i+1]..[i+2], which is the same convention
        /// <see cref="RoadFrame.distanceAlong"/> uses.
        /// </summary>
        int SampleIndexForWaypoint(int hintWaypoint)
        {
            int perSegment = Mathf.Max(1, resolutionPerSegment);
            return Mathf.Clamp((hintWaypoint - 1) * perSegment, 0, Mathf.Max(0, samplePos.Count - 1));
        }

        /// <summary>
        /// Nearest sample on the road centreline to a point, searching near the hint.
        /// Returns false when the mesh has not been built and there are no samples.
        /// </summary>
        bool NearestSample(Vector3 localPoint, int hintWaypoint, out int index, out float distance)
        {
            index = 0;
            distance = 0f;
            if (samplePos.Count == 0) return false;

            int centre = SampleIndexForWaypoint(hintWaypoint);
            int lo = Mathf.Max(0, centre - SampleSearchRadius);
            int hi = Mathf.Min(samplePos.Count - 1, centre + SampleSearchRadius);

            float best = float.MaxValue;
            for (int i = lo; i <= hi; i++)
            {
                // Horizontal only. A crest under the car is not the car being off-road.
                float dx = localPoint.x - samplePos[i].x;
                float dz = localPoint.z - samplePos[i].z;
                float d2 = dx * dx + dz * dz;
                if (d2 < best) { best = d2; index = i; }
            }

            distance = Mathf.Sqrt(best);
            return true;
        }

        /// <summary>
        /// How far a world point is from the road centreline, measured horizontally.
        ///
        /// With terrain under everything, "left the stage" can no longer mean "fell off
        /// the mesh" — there is nothing to fall off. This is the honest test instead, and
        /// it also separates "ran wide onto the shoulder" from "is in the scenery".
        ///
        /// Only samples near the hint are checked: the caller knows roughly where the car
        /// is, and a full scan would be O(samples) every physics step for no benefit.
        /// </summary>
        public float LateralOffset(Vector3 worldPos, int hintWaypoint)
        {
            if (importedCircuit) return Mathf.Abs(CircuitFrame(worldPos, hintWaypoint).signedOffset);
            Vector3 local = transform.InverseTransformPoint(worldPos);

            if (NearestSample(local, hintWaypoint, out _, out float distance)) return distance;

            // No mesh yet. Fall back to the control-point polyline, which is coarse but is
            // better than reporting zero — that would read as "perfectly on the centreline".
            if (waypoints.Count < 2) return 0f;

            int lo = Mathf.Clamp(hintWaypoint - 3, 0, waypoints.Count - 2);
            int hi = Mathf.Clamp(hintWaypoint + 2, 0, waypoints.Count - 2);

            float best = float.MaxValue;
            for (int i = lo; i <= hi; i++)
            {
                Vector3 a = waypoints[i], ab = waypoints[i + 1] - waypoints[i];
                float t = Mathf.Clamp01(Vector3.Dot(local - a, ab) / Mathf.Max(1e-4f, ab.sqrMagnitude));
                Vector3 p = a + ab * t;
                float d = new Vector2(local.x - p.x, local.z - p.z).magnitude;
                if (d < best) best = d;
            }
            return best;
        }

        // ══════════════════════════════════════════════════════════════
        //  ROAD-RELATIVE STATE
        //
        //  Where the car sits on the road, and what the road does next. This exists
        //  because the ray sensor cannot answer either question: its rays lie in a
        //  horizontal plane at bonnet height, and a road edge is a change of SURFACE, not
        //  a wall, so there is nothing there for a level ray to hit. Measured on this
        //  stage, the nearest thing the sensor can detect is the terrain rising into the
        //  ray plane 8.7-32.7 m PAST the road edge, and only where the landform happens to
        //  rise at all — on a falling side there is nothing within the full 60 m.
        //
        //  So a policy driving on observations alone had no way to know where the road was.
        //  It could only memorise one stage, which is exactly what it did.
        // ══════════════════════════════════════════════════════════════

        /// <summary>Where a point sits relative to the road centreline.</summary>
        public struct RoadFrame
        {
            public bool valid;
            /// <summary>Metres from the centreline. Positive is to the RIGHT of the direction of travel.</summary>
            public float signedOffset;
            /// <summary>Centreline direction at the nearest point, track-local, normalised.</summary>
            public Vector3 tangent;
            /// <summary>Approximate metres travelled along the stage, in the same units the mesh uses.</summary>
            public float distanceAlong;
        }

        /// <summary>Metres of stage per control-point segment. The mesh uses this same convention.</summary>
        float SegmentLength => trackLength / Mathf.Max(1, controlPointsCount);

        /// <summary>
        /// Projects a world position onto the centreline and reports which side of it the
        /// point is on, which way the road is heading there, and how far along it is.
        ///
        /// Searches only near <paramref name="hintWaypoint"/>, like
        /// <see cref="LateralOffset"/> — the caller always knows roughly where the car is.
        /// </summary>
        public RoadFrame SampleRoadFrame(Vector3 worldPos, int hintWaypoint)
        {
            if (importedCircuit) return CircuitFrame(worldPos, hintWaypoint);
            var frame = new RoadFrame { tangent = Vector3.forward };

            Vector3 local = transform.InverseTransformPoint(worldPos);
            if (!NearestSample(local, hintWaypoint, out int i, out _)) return frame;

            // Tangent from the neighbouring samples, so it describes the curve rather than
            // the chord — and so it is continuous, which a per-segment tangent was not:
            // crossing a waypoint used to step the reported heading by the whole corner.
            int back = Mathf.Max(0, i - 1);
            int forward = Mathf.Min(samplePos.Count - 1, i + 1);
            Vector3 span = samplePos[forward] - samplePos[back];
            Vector3 tangent = new Vector3(span.x, 0f, span.z);
            if (tangent.sqrMagnitude < 1e-8f) tangent = Vector3.forward;
            tangent.Normalize();

            Vector3 toPoint = local - samplePos[i];

            // Cross(up, tangent) is the tangent's right-hand side in Unity's left-handed
            // frame, so a positive dot means the point is to the right of the road.
            Vector3 rightOfRoad = Vector3.Cross(Vector3.up, tangent);

            frame.valid = true;
            frame.tangent = tangent;
            frame.signedOffset = Vector3.Dot(new Vector3(toPoint.x, 0f, toPoint.z), rightOfRoad);
            // Straight off the sample, which is the mesh builder's own distance convention
            // rather than a reconstruction of it.
            frame.distanceAlong = sampleDistance[i];
            return frame;
        }

        /// <summary>
        /// How far the road turns between <paramref name="distanceAlong"/> and
        /// <paramref name="lookahead"/> metres further on, in radians. Positive turns RIGHT,
        /// matching the sign of <c>VehicleController.steeringInput</c>.
        ///
        /// Read off the same Catmull-Rom spline the road mesh is built from, so it describes
        /// the road the car is actually driving on rather than the coarse waypoint polyline.
        /// This is the observation that lets a policy slow down for a corner it has not
        /// reached yet — without it, gamma is being asked to infer the corner from nothing.
        /// </summary>
        public float RoadTurnAhead(float distanceAlong, float lookahead)
        {
            if (importedCircuit) return SignedHorizontalAngle(CircuitTangent(distanceAlong), CircuitTangent(distanceAlong + lookahead));
            if (waypoints.Count < 4) return 0f;
            Vector3 here = SplineTangentAt(distanceAlong);
            Vector3 there = SplineTangentAt(distanceAlong + lookahead);
            return SignedHorizontalAngle(here, there);
        }

        /// <summary>
        /// The widest rock-free corridor across the road at a station, in metres, and where
        /// its centre sits relative to the centreline.
        ///
        /// This is the measurement that decides whether an obstacle strike was the policy's
        /// fault. "It hit a rock" says nothing on its own — the interesting split is between
        /// a stage that offered a 9 m gap the car drove straight past, and one where three
        /// rocks landed abreast and there was no line through at any speed. Those need
        /// completely different fixes, and until now nothing recorded which had happened.
        ///
        /// <paramref name="window"/> is how much stage either side counts as "here". It wants
        /// to be about the length of the car plus the distance it covers while committed to a
        /// line — a rock 20 m further on is not part of this gap.
        ///
        /// Returns false when the stage has no rocks at all near the station, in which case
        /// the whole road is clear and there is nothing to report.
        /// </summary>
        public bool WidestGapAt(float station, float window, out float gapWidth, out float gapCentre)
        {
            gapWidth = roadWidth;
            gapCentre = 0f;
            if (roadRocks.Count == 0) return false;

            // Rocks near this station, as blocked lateral intervals.
            var blocked = new List<Vector2>();       // x = left edge, y = right edge
            foreach (RoadRock rock in roadRocks)
            {
                if (Mathf.Abs(rock.station - station) > window) continue;
                blocked.Add(new Vector2(rock.lateral - rock.halfWidth, rock.lateral + rock.halfWidth));
            }
            if (blocked.Count == 0) return false;

            blocked.Sort((a, b) => a.x.CompareTo(b.x));

            // Sweep left to right, keeping the largest run of road nothing is standing in.
            float half = roadWidth * 0.5f;
            float cursor = -half;
            float best = 0f, bestCentre = 0f;

            foreach (Vector2 span in blocked)
            {
                float free = span.x - cursor;
                if (free > best) { best = free; bestCentre = cursor + free * 0.5f; }
                cursor = Mathf.Max(cursor, span.y);
            }

            float tail = half - cursor;              // the run past the last rock
            if (tail > best) { best = tail; bestCentre = cursor + tail * 0.5f; }

            gapWidth = Mathf.Max(0f, best);
            gapCentre = bestCentre;
            return true;
        }

        /// <summary>Spline tangent at a distance along the stage, track-local and normalised.</summary>
        Vector3 SplineTangentAt(float distanceAlong)
        {
            // Inverse of the mesh builder's mapping: distanceAlong = (i + 1 + t) * SegmentLength.
            float u = distanceAlong / Mathf.Max(0.01f, SegmentLength) - 1f;
            int i = Mathf.Clamp(Mathf.FloorToInt(u), 0, waypoints.Count - 4);
            float t = Mathf.Clamp01(u - i);

            Vector3 tan = SplineMath.GetTangent(t, waypoints[i], waypoints[i + 1],
                                                   waypoints[i + 2], waypoints[i + 3]);
            Vector3 flat = new Vector3(tan.x, 0f, tan.z);
            return flat.sqrMagnitude < 1e-8f ? Vector3.forward : flat.normalized;
        }

        /// <summary>
        /// Signed angle from a to b about world up, in radians. Positive is a turn to the
        /// RIGHT, so it agrees with the steering convention rather than with the maths one.
        /// </summary>
        static float SignedHorizontalAngle(Vector3 a, Vector3 b)
        {
            float dot = Mathf.Clamp(a.x * b.x + a.z * b.z, -1f, 1f);
            float cross = a.x * b.z - a.z * b.x;      // positive when b is LEFT of a
            return -Mathf.Atan2(cross, dot);
        }

        /// <summary>
        /// Last waypoint with road under it. The mesh only spans waypoints[1]..[count-2],
        /// so anything past this is off the end of the world.
        /// </summary>
        public int FinishWaypointIndex => Mathf.Max(0, waypoints.Count - 2);

        /// <summary>
        /// Where the car starts and which way it points, in world space. False until the
        /// track has been generated and is long enough to have a start line.
        /// </summary>
        public bool TryGetSpawn(out Vector3 position, out Vector3 forward)
        {
            if (importedCircuit)
            {
                position = transform.TransformPoint(CircuitSurfacePoint(circuitStart,0));
                forward = transform.TransformDirection(CircuitTangent(circuitStart));
                return true;
            }
            position = Vector3.zero;
            forward = Vector3.forward;
            if (waypoints.Count <= spawnWaypointIndex + 1) return false;

            position = transform.TransformPoint(waypoints[spawnWaypointIndex]);
            forward = (transform.TransformPoint(waypoints[spawnWaypointIndex + 1]) - position).normalized;
            return true;
        }

        private void GenerateControlPoints()
        {
            waypoints.Clear();
            float segmentLength = trackLength / controlPointsCount;
            Vector3 currentPos = Vector3.zero;
            
            // The stage is walked out one segment at a time: read a heading, take a step.
            //
            // Two things this is deliberately NOT doing.
            //
            // It does not offset x from a z that marches straight down the axis — that
            // caps heading at atan(dx / segmentLength) and cannot make a corner at all.
            //
            // And it does not integrate a turn rate. That is a random walk: it wanders
            // arbitrarily far from the stage direction and eventually curls the road back
            // through itself, and damping it back toward straight to prevent that decays
            // corners as fast as the noise builds them — which flattened the sharpest
            // corner on the stage from 22 deg per segment to 11.
            //
            // Reading an ABSOLUTE heading, bounded to +/-maxHeadingDeviation, fixes both.
            // Because that bound is under 90 deg, cos(heading) never goes negative, every
            // segment advances down +Z, and self-intersection is impossible by
            // construction rather than by tuning.
            Vector3 pos = Vector3.zero;

            for (int i = 0; i < controlPointsCount + 3; i++) // Extra for Catmull-Rom
            {
                // Sample by DISTANCE ALONG THE STAGE, not by control-point index. The
                // index only ever runs 0..22, so any frequency small enough to look like
                // a frequency left the noise essentially flat across the whole course.
                float distance = i * segmentLength;

                float yOffset = (Mathf.PerlinNoise(noiseOriginHeight + distance * elevationFrequency, 100f) - 0.5f)
                                * elevationScale;

                waypoints.Add(new Vector3(pos.x, yOffset, pos.z));

                // Perlin's practical range is about 0.25..0.75, not 0..1, so the raw
                // (n - 0.5) only ever asks for half the deviation configured. Expanding
                // by 4 and clamping uses the whole range, and the clamp's flat top is
                // what holds a long constant-radius corner instead of a brief peak.
                float n = Mathf.PerlinNoise(noiseOriginShape + distance * shapeFrequency, 0f);
                float heading = Mathf.Clamp((n - 0.5f) * 4f, -1f, 1f)
                                * maxHeadingDeviation * Mathf.Deg2Rad;
                if (longStageRhythm)
                {
                    // Independent broad pacing envelopes join fast and technical sections smoothly.
                    float rhythm = .5f + .5f * Mathf.Sin(distance * Mathf.PI * 2 / 700f);
                    float fastHeading = (Mathf.PerlinNoise(noiseOriginShape + distance * .0015f, 17f)-.5f)*.6f;
                    heading = Mathf.Lerp(fastHeading, heading, Mathf.SmoothStep(.2f,1f,rhythm));
                }

                pos += new Vector3(Mathf.Sin(heading), 0f, Mathf.Cos(heading)) * segmentLength;
            }
        }

        private void BuildRoadMesh()
        {
            // Four longitudinal lines are swept down the stage and then stitched into
            // strips: outer-verge / road-edge / road-edge / outer-verge. Keeping them as
            // separate lines rather than emitting triangles inline is what lets the road
            // and the verges become two different objects with two different tags.
            var lineLeft  = new List<Vector3>();
            var lineRight = new List<Vector3>();
            samplePos.Clear(); sampleRight.Clear(); sampleDistance.Clear();

            float segmentLength = trackLength / controlPointsCount;

            // Rock density is specified per 100 m, so convert it to a per-sample probability
            // using this tessellation's actual sample spacing. That keeps the expected number
            // of rocks fixed while resolutionPerSegment is free to change for image quality.
            float sampleSpacing = segmentLength / Mathf.Max(1, resolutionPerSegment);
            float obstacleChancePerSample = Mathf.Clamp01(obstaclesPer100m * sampleSpacing / 100f);

            for (int i = 0; i < waypoints.Count - 3; i++)
            {
                Vector3 p0 = waypoints[i];
                Vector3 p1 = waypoints[i + 1];
                Vector3 p2 = waypoints[i + 2];
                Vector3 p3 = waypoints[i + 3];

                // Every segment after the first starts where the last one ended, so its
                // j = 0 sample would be a duplicate of the previous j = resolution.
                for (int j = (i == 0 ? 0 : 1); j <= resolutionPerSegment; j++)
                {
                    float t = j / (float)resolutionPerSegment;

                    Vector3 pos = SplineMath.GetCatmullRomPosition(t, p0, p1, p2, p3);
                    Vector3 tangent = SplineMath.GetTangent(t, p0, p1, p2, p3);

                    // Camber, sampled by distance along the stage. Segment i interpolates
                    // between waypoints[i+1] and [i+2], so this sample sits at (i+1+t)
                    // segments in. Using pos.z instead only works while the road runs
                    // straight down +Z — the moment it turns, world z stops tracking
                    // distance and the banking starts repeating on itself.
                    float distanceAlong = (i + 1 + t) * segmentLength;
                    float camber = (Mathf.PerlinNoise(noiseOriginCamber + distanceAlong * camberFrequency, 50f) - 0.5f)
                                   * camberScale;
                    Vector3 normal = Quaternion.AngleAxis(camber, tangent) * Vector3.up;
                    Vector3 right = Vector3.Cross(normal, tangent).normalized;

                    Vector3 leftEdge  = pos - right * roadWidth * 0.5f;
                    Vector3 rightEdge = pos + right * roadWidth * 0.5f;

                    lineLeft.Add(leftEdge);
                    lineRight.Add(rightEdge);

                    samplePos.Add(pos);
                    sampleRight.Add(right);
                    sampleDistance.Add(distanceAlong);

                    if (spawnObstacles && Random.value < obstacleChancePerSample && !IsInSpawnZone(pos))
                    {
                        GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
                        // Named, not left as "Cube": trees, boulders and road rocks all carry
                        // the Obstacle tag, so the name is the only thing that says WHICH was
                        // hit. "Obstacle strikes are half of all failures" is a useless
                        // sentence if a tree met after leaving the road counts the same as a
                        // rock in the racing line — they are different problems.
                        obstacle.name = "RoadRock";
                        try { obstacle.tag = "Obstacle"; } catch { }
                        obstacle.transform.SetParent(transform, false);
                        // Everything here is in the track's local space — the mesh is too —
                        // so this is a LOCAL position. Writing it to .position would put the
                        // rocks somewhere else the moment the track object is moved.
                        float lateralOffset = Random.Range(-roadWidth * 0.4f, roadWidth * 0.4f);
                        obstacle.transform.localPosition = pos + right * lateralOffset + normal * 0.5f;
                        ApplyRoadRockVisual(obstacle, CurrentSeed ^ (roadRocks.Count * 397));
                        obstacles.Add(obstacle);

                        // The same rock in road-relative terms, so a crash can be judged
                        // against the line that was available. A default cube primitive is
                        // 1 m on a side, so half a metre either side of its centre.
                        roadRocks.Add(new RoadRock
                        {
                            station   = distanceAlong,
                            lateral   = lateralOffset,
                            halfWidth = 0.5f
                        });
                    }
                }
            }

            // ── The driveable surface ────────────────────────────────────────────
            //  Split across its width into lanes. This does not change the road's SHAPE at
            //  all — every lane boundary is sampled from the same per-sample spline frame the
            //  edges come from, so the surface it describes is identical — it only shrinks the
            //  triangles the affine mapping has to cross before a vertex pins the UV back to
            //  where perspective says it belongs. See the tessellation note on roadLanes.
            //
            //  The outermost two lines are the very same List objects the terrain sweeps
            //  outward from, so the seam at the road edge is bit-identical and cannot crack.
            float roadHalfWidth = roadWidth * 0.5f;
            int lanes = Mathf.Max(1, roadLanes);

            var laneLines = new List<Vector3>[lanes + 1];
            var laneLateral = new float[lanes + 1];
            for (int m = 0; m <= lanes; m++)
            {
                laneLateral[m] = -roadHalfWidth + roadWidth * m / lanes;
                laneLines[m] = m == 0     ? lineLeft
                             : m == lanes ? lineRight
                                          : RoadLine(laneLateral[m]);
            }

            var roadStrips = new Strip[lanes];
            for (int m = 0; m < lanes; m++)
                roadStrips[m] = new Strip(laneLines[m], laneLines[m + 1],
                                          laneLateral[m], laneLateral[m + 1]);

            roadMesh = BuildStrips("RoadMesh", roadStrips, roadTileMetres);
            GetComponent<MeshFilter>().sharedMesh = roadMesh;

            MeshRenderer renderer = GetComponent<MeshRenderer>();
            if (renderer.sharedMaterial == null)
                renderer.sharedMaterial = new Material(Shader.Find("Standard"));

            MeshCollider collider = gameObject.GetComponent<MeshCollider>();
            if (collider == null) collider = gameObject.AddComponent<MeshCollider>();
            collider.sharedMesh = null;      // force a re-cook; assigning the same Mesh is a no-op
            collider.sharedMesh = roadMesh;

            var roadSurface = gameObject.GetComponent<SurfaceProperties>();
            if (roadSurface == null) roadSurface = gameObject.AddComponent<SurfaceProperties>();
            roadSurface.surface = SurfaceProperties.Surface.Gravel;
            roadSurface.ApplyPreset();

            BuildTerrain(lineLeft, lineRight);
            ScatterProps();
            BuildBackdrop();
        }

        void ApplyRoadRockVisual(GameObject obstacle, int visualSeed)
        {
            MeshFilter filter = obstacle.GetComponent<MeshFilter>();
            if (filter != null) filter.sharedMesh = RoadRockMesh(visualSeed);

            MeshRenderer renderer = obstacle.GetComponent<MeshRenderer>();
            if (renderer == null) return;

            MeshRenderer trackRenderer = GetComponent<MeshRenderer>();
            renderer.sharedMaterial = rockMaterial != null ? rockMaterial :
                trackRenderer != null ? trackRenderer.sharedMaterial : null;
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
            renderer.receiveShadows = true;
        }

        public void RefreshRoadRockVisuals(Material presentationMaterial)
        {
            int index = 0;
            for (int i = 0; i < transform.childCount; i++)
            {
                Transform child = transform.GetChild(i);
                if (child.name != "RoadRock") continue;

                ApplyRoadRockVisual(child.gameObject, CurrentSeed ^ (index * 397));
                MeshRenderer renderer = child.GetComponent<MeshRenderer>();
                if (renderer != null && presentationMaterial != null)
                    renderer.sharedMaterial = presentationMaterial;
                index++;
            }
        }

        static Mesh RoadRockMesh(int seed)
        {
            const int variants = 4;
            if (roadRockMeshes == null || roadRockMeshes.Length != variants)
            {
                roadRockMeshes = new Mesh[variants];
                for (int i = 0; i < variants; i++)
                    roadRockMeshes[i] = BuildRoadRockMesh(0x51f15e + i * 7919);
            }

            int index = (seed & 0x7fffffff) % variants;
            return roadRockMeshes[index];
        }

        static Mesh BuildRoadRockMesh(int seed)
        {
            const int sides = 9;
            var rng = new System.Random(seed);
            var vertices = new List<Vector3>(sides * 2 + 2);
            var uvs = new List<Vector2>(sides * 2 + 2);
            var triangles = new List<int>(sides * 12);
            float phase = (float)rng.NextDouble() * Mathf.PI * 2f;
            Vector3 bottom = new Vector3(RandomSigned(rng) * 0.045f, -0.49f, RandomSigned(rng) * 0.045f);
            Vector3 crown = new Vector3(RandomSigned(rng) * 0.08f, 0.40f + (float)rng.NextDouble() * 0.09f,
                                        RandomSigned(rng) * 0.08f);

            vertices.Add(bottom);
            uvs.Add(new Vector2(bottom.x + 0.5f, bottom.z + 0.5f));
            int lowerStart = vertices.Count;
            for (int i = 0; i < sides; i++)
            {
                float angle = phase + i * Mathf.PI * 2f / sides;
                float radius = Mathf.Lerp(0.39f, 0.49f, (float)rng.NextDouble());
                Vector3 point = new Vector3(Mathf.Cos(angle) * radius, -0.45f + (float)rng.NextDouble() * 0.08f,
                                            Mathf.Sin(angle) * radius);
                vertices.Add(point);
                uvs.Add(new Vector2(point.x + 0.5f, point.z + 0.5f));
            }

            int upperStart = vertices.Count;
            for (int i = 0; i < sides; i++)
            {
                float angle = phase + i * Mathf.PI * 2f / sides + 0.11f;
                float radius = Mathf.Lerp(0.27f, 0.39f, (float)rng.NextDouble());
                Vector3 point = new Vector3(Mathf.Cos(angle) * radius,
                                            -0.10f + (float)rng.NextDouble() * 0.27f,
                                            Mathf.Sin(angle) * radius);
                vertices.Add(point);
                uvs.Add(new Vector2(point.x + 0.5f, point.z + 0.5f));
            }

            int crownIndex = vertices.Count;
            vertices.Add(crown);
            uvs.Add(new Vector2(crown.x + 0.5f, crown.z + 0.5f));

            for (int i = 0; i < sides; i++)
            {
                int next = (i + 1) % sides;
                int lower = lowerStart + i, lowerNext = lowerStart + next;
                int upper = upperStart + i, upperNext = upperStart + next;
                triangles.Add(lower); triangles.Add(upper); triangles.Add(upperNext);
                triangles.Add(lower); triangles.Add(upperNext); triangles.Add(lowerNext);
                triangles.Add(0); triangles.Add(lower); triangles.Add(lowerNext);
                triangles.Add(crownIndex); triangles.Add(upperNext); triangles.Add(upper);
            }

            var mesh = new Mesh { name = "Rally Road Boulder", hideFlags = HideFlags.DontSave };
            mesh.SetVertices(vertices);
            mesh.SetUVs(0, uvs);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        static float RandomSigned(System.Random rng) => (float)rng.NextDouble() * 2f - 1f;

        /// <summary>
        /// Sweeps the landscape outward from both road edges and stitches it into one
        /// mesh. Three things are layered on top of each other, outward from the road:
        ///
        ///   1. a drainage ditch in the shoulder — shallow, always there, and the thing
        ///      that actually catches a car that puts one wheel wide;
        ///   2. the landform, which either RISES into a cut bank or FALLS away into a
        ///      drop. Left and right are sampled from independent noise, so a section
        ///      can have a bank one side and nothing the other — which is the whole
        ///      point, and what the ray sensor has to learn to tell apart;
        ///   3. small-scale roughness, faded in with distance so it never disturbs the
        ///      road edge.
        ///
        /// The relief of (2) is scaled by a slow "drama" field, so the stage moves
        /// between mountain cut and open country along its length instead of being one
        /// texture the whole way.
        /// </summary>
        private void BuildTerrain(List<Vector3> left, List<Vector3> right)
        {
            RemoveGenerated(TerrainName);
            terrain = null;

            if (!buildTerrain || terrainHalfWidth <= roadWidth * 0.5f + 1f) return;

            int n = samplePos.Count;
            if (n < 2 || left.Count != n || right.Count != n) return;

            int steps = Mathf.Max(2, terrainResolution);
            float roadHalf = roadWidth * 0.5f;

            // Lateral offset of each step, precomputed so the mesh loop and the
            // texture coordinates cannot disagree about where a line actually is.
            // Finer spacing near the road, where the car interacts with the ground,
            // and coarser out in the scenery.
            var outward = new float[steps + 1];
            for (int k = 0; k <= steps; k++)
                outward[k] = Mathf.Pow(k / (float)steps, 1.6f) * (terrainHalfWidth - roadHalf);

            // One line of vertices per lateral step, per side. Index 0 is the road edge
            // itself, so the terrain shares that seam exactly and cannot crack open.
            var linesL = new List<Vector3>[steps + 1];
            var linesR = new List<Vector3>[steps + 1];
            for (int k = 0; k <= steps; k++)
            {
                linesL[k] = new List<Vector3>(n);
                linesR[k] = new List<Vector3>(n);
            }

            for (int i = 0; i < n; i++)
            {
                Vector3 pos = samplePos[i];
                Vector3 rightDir = sampleRight[i];
                float d = sampleDistance[i];

                // Landform character for this station. Left and right read different
                // slices of the field so they vary independently.
                float slopeL = SlopeAt(d, -1f);
                float slopeR = SlopeAt(d, +1f);

                linesL[0].Add(left[i]);
                linesR[0].Add(right[i]);

                for (int k = 1; k <= steps; k++)
                {
                    float o = outward[k];
                    linesL[k].Add(left[i]  - rightDir * o + Vector3.up * Profile(o, slopeL, pos, -1f));
                    linesR[k].Add(right[i] + rightDir * o + Vector3.up * Profile(o, slopeR, pos, +1f));
                }
            }

            var strips = new List<Strip>();
            // Right side sweeps outward along +right, so B is already the outer line.
            // The left side sweeps the other way, so the pair has to be reversed or the
            // faces come out pointing into the ground.
            for (int k = 0; k < steps; k++)
            {
                // Distance from the CENTRELINE, signed, so the texture runs continuously
                // out from the road instead of restarting at every lateral step.
                float inner = roadHalf + outward[k], outer = roadHalf + outward[k + 1];
                strips.Add(new Strip(linesR[k], linesR[k + 1],  inner,  outer));
                strips.Add(new Strip(linesL[k + 1], linesL[k], -outer, -inner));
            }

            var mesh = BuildStrips("TerrainMesh", strips.ToArray(), terrainTileMetres);

            var go = new GameObject(TerrainName);
            try { go.tag = "TrackBoundary"; } catch { }
            go.transform.SetParent(transform, false);

            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = terrainMaterial != null ? terrainMaterial
                                                        : GetComponent<MeshRenderer>().sharedMaterial;

            go.AddComponent<MeshCollider>().sharedMesh = mesh;

            var sp = go.AddComponent<SurfaceProperties>();
            sp.surface = SurfaceProperties.Surface.Grass;
            sp.ApplyPreset();

            terrain = go;
        }

        /// <summary>
        /// A ring of distant hills, standing where the ground used to simply stop.
        ///
        /// The wall runs from far BELOW the horizon up to a noisy ridge line, rather than
        /// standing on a base. That is what makes it hold up: the terrain either side of
        /// the road rises into banks and falls into drops, and wherever it falls you are
        /// looking down past its outer edge at whatever lies behind. A ring that began at
        /// ground level would show sky through that gap. This one has a few hundred metres
        /// of hillside there instead, fogged down to a flat haze — which is exactly what
        /// distant ground looks like anyway.
        ///
        /// Built once at the stage's mean elevation and carried along by BackdropFollower,
        /// which writes a WORLD position, so this object assumes the track it hangs under
        /// is unrotated and unscaled. Nothing here has a collider and nothing is tagged:
        /// the ray sensor cannot see it and the car cannot reach it, so the stage is
        /// visually a different place and physically the same one.
        /// </summary>
        private void BuildBackdrop()
        {
            RemoveGenerated(BackdropName);
            backdrop = null;

            int n = samplePos.Count;
            if (!buildBackdrop || n < 2 || backdropRadius <= 1f) return;

            // The stage's mean height, not its first sample. The follower holds Y fixed,
            // so a ridge levelled with the start line would be metres out by the finish.
            float baseY = 0f;
            for (int i = 0; i < n; i++) baseY += samplePos[i].y;
            baseY /= n;

            int seg = Mathf.Max(12, backdropSegments);
            var verts = new List<Vector3>((seg + 1) * 2);
            var uvs   = new List<Vector2>((seg + 1) * 2);
            var tris  = new List<int>(seg * 6);

            for (int s = 0; s <= seg; s++)
            {
                float a = s / (float)seg * Mathf.PI * 2f;
                float cos = Mathf.Cos(a), sin = Mathf.Sin(a);

                // Sampled ON A CIRCLE through the noise field rather than along a line.
                // That is what lets the ridge close up seamlessly: s = 0 and s = seg are
                // literally the same point in the field, not two points that have to be
                // trusted to sit a whole number of periods apart.
                float coarse = Mathf.PerlinNoise(noiseOriginCut   + 4f + cos * 1.3f,
                                                 noiseOriginCut   + 4f + sin * 1.3f);
                float fine   = Mathf.PerlinNoise(noiseOriginRough + 9f + cos * 3.7f,
                                                 noiseOriginRough + 9f + sin * 3.7f);
                float h = Mathf.Max(1f, backdropHeight * (0.34f + 0.52f * coarse + 0.28f * fine));

                Vector3 dir = new Vector3(cos, 0f, sin) * backdropRadius;
                float u = s / (float)seg;

                // Height is relative to zero here, NOT to baseY: the follower puts baseY
                // into the transform, and carrying it in the vertices too would place the
                // ridge at twice the stage's elevation.
                //
                // v = 0 at the horizon and 1 at the ridge top, so the texture's pale base
                // lands where the haze is thickest. Everything below the horizon runs
                // negative and the clamped sampler holds it at that same pale colour.
                verts.Add(dir + Vector3.down * backdropDepth);
                uvs.Add(new Vector2(u, -backdropDepth / h));
                verts.Add(dir + Vector3.up * h);
                uvs.Add(new Vector2(u, 1f));
            }

            // Wound to face INWARD. The viewer is inside the ring, and a wall whose faces
            // point away from you is one you can see straight through.
            for (int s = 0; s < seg; s++)
            {
                int b0 = s * 2, t0 = b0 + 1, b1 = b0 + 2, t1 = b1 + 1;
                tris.Add(b0); tris.Add(b1); tris.Add(t0);
                tris.Add(t0); tris.Add(b1); tris.Add(t1);
            }

            var mesh = new Mesh { name = "BackdropMesh" };
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.SetUVs(0, uvs);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            var go = new GameObject(BackdropName);
            go.transform.SetParent(transform, false);
            go.transform.localPosition = new Vector3(0f, baseY, 0f);

            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = backdropMaterial != null ? backdropMaterial
                              : terrainMaterial != null ? terrainMaterial
                              : GetComponent<MeshRenderer>().sharedMaterial;

            // A wall this size casting into the sun's shadow map would spend the entire
            // budget on geometry the camera can barely resolve through the fog.
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows = false;

            go.AddComponent<BackdropFollower>().fixedHeight = baseY;
            backdrop = go;
        }

        /// <summary>
        /// Scatters trees and boulders across the terrain, biased outward so there is a
        /// clear corridor beside the road and a thickening treeline beyond it.
        /// </summary>
        private void ScatterProps()
        {
            RemoveGenerated(PropsName);
            props = null;

            int n = samplePos.Count;
            if (!scatterProps || n < 2 || terrainHalfWidth <= propClearance + 2f) return;

            var root = new GameObject(PropsName);
            root.transform.SetParent(transform, false);

            var treeVerts = new List<Vector3>(); var treeTris = new List<int>();
            var treeUVs  = new List<Vector2>();
            var rockVerts = new List<Vector3>(); var rockTris = new List<int>();
            var rockUVs  = new List<Vector2>();

            float stageLength = sampleDistance[n - 1];
            int treeCount = Mathf.RoundToInt(stageLength / 100f * treesPer100m);
            int rockCount = Mathf.RoundToInt(stageLength / 100f * rocksPer100m);

            for (int k = 0; k < treeCount + rockCount; k++)
            {
                bool isTree = k < treeCount;

                int i = Random.Range(0, n);
                if (IsInSpawnZone(samplePos[i])) continue;

                // Area-weighted outward: sqrt makes distant ground as likely per square
                // metre as near ground, which reads as a treeline rather than a hedge.
                float lateral = Mathf.Lerp(propClearance, terrainHalfWidth, Mathf.Sqrt(Random.value))
                                * (Random.value < 0.5f ? -1f : 1f);
                Vector3 at = GroundPoint(i, lateral);
                Quaternion yaw = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

                var go = new GameObject(isTree ? "Tree" : "Boulder");
                try { go.tag = "Obstacle"; } catch { }
                go.transform.SetParent(root.transform, false);
                go.transform.localPosition = at;

                if (isTree)
                {
                    float h = Random.Range(treeHeightRange.x, treeHeightRange.y);
                    float r = h * Random.Range(0.045f, 0.075f);
                    AppendTree(treeVerts, treeTris, treeUVs, at, yaw, h, r);

                    var cap = go.AddComponent<CapsuleCollider>();
                    cap.radius = Mathf.Max(0.25f, r * 1.6f);
                    cap.height = h;
                    cap.center = new Vector3(0f, h * 0.5f, 0f);
                }
                else
                {
                    float sz = Random.Range(rockSizeRange.x, rockSizeRange.y);
                    AppendRock(rockVerts, rockTris, rockUVs, at, yaw, sz, rockTileMetres);

                    var box = go.AddComponent<BoxCollider>();
                    box.size = Vector3.one * sz;
                    box.center = new Vector3(0f, sz * 0.35f, 0f);
                }
            }

            AttachMerged(root.transform, "Trees", treeVerts, treeTris, treeUVs, treeMaterial);
            AttachMerged(root.transform, "Boulders", rockVerts, rockTris, rockUVs, rockMaterial);
            props = root;
        }

        private void AttachMerged(Transform parent, string name,
                                  List<Vector3> verts, List<int> tris, List<Vector2> uvs, Material mat)
        {
            if (verts.Count == 0) return;
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);

            var mesh = new Mesh { name = name + "Mesh" };
            if (verts.Count > 65000) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.SetUVs(0, uvs);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = mat != null ? mat : GetComponent<MeshRenderer>().sharedMaterial;
        }

        // Bark occupies the left half of the foliage texture and needles the right, because
        // every tree in the stage is merged into ONE mesh with ONE material and a trunk and
        // a canopy therefore have to come out of the same image. The insets keep both
        // halves clear of the middle: point sampling still lands on the far side of the
        // divide if a coordinate sits exactly on it, which puts needles on the trunk.
        const float BarkU0 = 0.02f, BarkU1 = 0.47f;
        const float NeedleU0 = 0.53f, NeedleU1 = 0.98f;

        /// <summary>Seed-stable, varied conifer silhouettes; collision stays on the original capsule.</summary>
        private static void AppendTree(List<Vector3> v, List<int> t, List<Vector2> uv,
                                       Vector3 at, Quaternion yaw, float height, float radius)
        {
            const int sides = 9;
            float shape = Mathf.PerlinNoise(at.x * 0.173f + 17.4f, at.z * 0.173f + 83.2f);
            float asymmetry = Mathf.PerlinNoise(at.x * 0.317f - 11.8f, at.z * 0.317f + 29.1f);
            float trunkH = height * Mathf.Lerp(0.34f, 0.42f, shape);
            int b = v.Count;

            // Note the <= : one extra column of vertices, in the same place as the first
            // but carrying the far end of the texture range. Without it the closing face
            // has to run from u=0.47 back to u=0.02 and the whole bark texture is squeezed
            // into it, mirrored — one seam per trunk, all of them facing the same way.
            for (int s = 0; s <= sides; s++)
            {
                float a = s / (float)sides * Mathf.PI * 2f;
                Vector3 o = yaw * new Vector3(Mathf.Cos(a) * radius, 0f, Mathf.Sin(a) * radius);
                float u = Mathf.Lerp(BarkU0, BarkU1, s / (float)sides);

                v.Add(at + o);                               uv.Add(new Vector2(u, 0f));
                v.Add(at + o * 0.7f + Vector3.up * trunkH);  uv.Add(new Vector2(u, 1f));
            }
            for (int s = 0; s < sides; s++)
            {
                int c = b + s * 2, nx = b + (s + 1) * 2;
                t.Add(c); t.Add(c + 1); t.Add(nx);
                t.Add(nx); t.Add(c + 1); t.Add(nx + 1);
            }

            int tiers = shape > 0.52f ? 4 : 3;
            for (int tier = 0; tier < tiers; tier++)
            {
                float fraction = tier / (float)tiers;
                float y0 = trunkH + height * (0.035f + fraction * 0.39f);
                float y1 = Mathf.Min(height * 0.99f, y0 + height * (0.31f - fraction * 0.06f));
                float rad = height * (0.275f - fraction * 0.066f) * Mathf.Lerp(0.90f, 1.08f, shape);
                float tipRadius = rad * Mathf.Lerp(0.07f, 0.18f, asymmetry);
                Vector3 tierOffset = yaw * new Vector3(
                    Mathf.Sin(shape * 8f + tier * 1.9f) * height * 0.018f,
                    0f,
                    Mathf.Cos(asymmetry * 7f + tier * 2.3f) * height * 0.018f);

                int apex = v.Count;
                v.Add(at + tierOffset * 0.55f + Vector3.up * y1);
                // The apex is one vertex shared by every face of the cone, so it gets one
                // coordinate: the middle of the needle range. The faces either side of it
                // stretch to reach it, which on a clumped noise texture is invisible.
                uv.Add(new Vector2((NeedleU0 + NeedleU1) * 0.5f, 1f));

                for (int s = 0; s <= sides; s++)
                {
                    float a = s / (float)sides * Mathf.PI * 2f;
                    v.Add(at + tierOffset + yaw * new Vector3(Mathf.Cos(a) * rad, 0f, Mathf.Sin(a) * rad) + Vector3.up * y0);
                    uv.Add(new Vector2(Mathf.Lerp(NeedleU0, NeedleU1, s / (float)sides), 0f));
                }

                int topRingStart = v.Count;
                for (int s = 0; s <= sides; s++)
                {
                    float a = s / (float)sides * Mathf.PI * 2f;
                    v.Add(at + tierOffset * 0.58f + yaw * new Vector3(Mathf.Cos(a) * tipRadius, 0f, Mathf.Sin(a) * tipRadius)
                        + Vector3.up * (y1 - height * 0.018f));
                    uv.Add(new Vector2(Mathf.Lerp(NeedleU0, NeedleU1, s / (float)sides), 0.72f));
                }
                for (int s = 0; s < sides; s++)
                {
                    int lowerA = apex + 1 + s, lowerB = lowerA + 1;
                    int upperA = topRingStart + s, upperB = upperA + 1;
                    t.Add(lowerA); t.Add(upperA); t.Add(upperB);
                    t.Add(lowerA); t.Add(upperB); t.Add(lowerB);
                    t.Add(apex); t.Add(upperB); t.Add(upperA);
                }
            }
        }

        /// <summary>A jittered octahedron. Cheap, and never twice the same shape.</summary>
        private static void AppendRock(List<Vector3> v, List<int> t, List<Vector2> uv,
                                       Vector3 at, Quaternion yaw, float size, float tileMetres)
        {
            Vector3[] o = {
                new Vector3( 0f,  0.9f,  0f), new Vector3( 1f,  0.15f,  0f),
                new Vector3( 0f,  0.1f,  1f), new Vector3(-1f,  0.15f,  0f),
                new Vector3( 0f,  0.1f, -1f), new Vector3( 0f, -0.35f,  0f)
            };

            var p = new Vector3[o.Length];
            for (int i = 0; i < o.Length; i++)
            {
                Vector3 j = new Vector3(Random.Range(-0.18f, 0.18f), Random.Range(-0.12f, 0.12f),
                                        Random.Range(-0.18f, 0.18f));
                p[i] = yaw * ((o[i] + j) * size * 0.5f);
            }

            int[] f = { 0,1,2, 0,2,3, 0,3,4, 0,4,1, 5,2,1, 5,3,2, 5,4,3, 5,1,4 };
            float inv = 1f / Mathf.Max(0.05f, tileMetres);

            // Where this boulder reads from in the stone texture. Derived from where it
            // stands rather than drawn from Random, deliberately: the RNG sequence here
            // also places every prop after this one, so taking two more numbers out of it
            // would shift the whole scatter and quietly change the stage for a given seed.
            var origin = new Vector2(at.x * 0.137f, at.z * 0.191f);

            // A vertex per FACE rather than per corner. Two reasons, and both matter:
            // the faces need independent texture coordinates, and RecalculateNormals
            // averages across whatever vertices it finds shared — which rounds an
            // eight-sided boulder off into a soft blob instead of leaving it faceted.
            for (int i = 0; i < f.Length; i += 3)
            {
                Vector3 a = p[f[i]], bb = p[f[i + 1]], c = p[f[i + 2]];

                // Project each face onto its own plane. Constant texel density whatever
                // the boulder's size, and no face is ever edge-on to the projection the
                // way a straight top-down cast leaves the vertical ones.
                Vector3 nrm = Vector3.Cross(bb - a, c - a).normalized;
                Vector3 tan = Vector3.Cross(nrm, Mathf.Abs(nrm.y) > 0.9f ? Vector3.forward : Vector3.up).normalized;
                Vector3 bit = Vector3.Cross(nrm, tan);

                int b = v.Count;
                v.Add(at + a);  uv.Add(origin + new Vector2(Vector3.Dot(a, tan), Vector3.Dot(a, bit)) * inv);
                v.Add(at + bb); uv.Add(origin + new Vector2(Vector3.Dot(bb, tan), Vector3.Dot(bb, bit)) * inv);
                v.Add(at + c);  uv.Add(origin + new Vector2(Vector3.Dot(c, tan), Vector3.Dot(c, bit)) * inv);
                t.Add(b); t.Add(b + 1); t.Add(b + 2);
            }
        }

        /// <summary>
        /// The landform's rise (positive, a cut bank) or fall (negative, a drop) at this
        /// station on this side. One source of truth: the terrain mesh and the scenery
        /// scatter both read it, so a tree cannot end up floating above the hillside it
        /// is supposed to be standing on.
        /// </summary>
        private float SlopeAt(float distanceAlong, float side)
        {
            float drama = Mathf.PerlinNoise(noiseOriginDrama + distanceAlong * dramaFrequency, 7f);
            float relief = terrainRelief * drama;
            float key = side < 0f ? 11f : 29f;
            return ((Mathf.PerlinNoise(noiseOriginCut + distanceAlong * terrainFrequency, key) - 0.5f) * 2f) * relief;
        }

        /// <summary>
        /// A point on the generated ground, at <paramref name="lateral"/> metres from the
        /// centreline (signed: negative is left) at spline sample <paramref name="i"/>.
        /// </summary>
        private Vector3 GroundPoint(int i, float lateral)
        {
            Vector3 pos = samplePos[i], rightDir = sampleRight[i];
            float roadHalf = roadWidth * 0.5f;
            float side = Mathf.Sign(lateral);
            float outward = Mathf.Abs(lateral) - roadHalf;
            if (outward <= 0f) return pos + rightDir * lateral;

            float h = Profile(outward, SlopeAt(sampleDistance[i], side), pos, side);
            return pos + rightDir * lateral + Vector3.up * h;
        }

        /// <summary>
        /// Height of the ground at <paramref name="outward"/> metres beyond the road
        /// edge, relative to that edge.
        /// </summary>
        private float Profile(float outward, float slope, Vector3 pos, float side)
        {
            // Ditch: a parabola across the shoulder, zero at both ends so it meets the
            // road flush and hands over cleanly to the landform.
            float e = Mathf.Clamp01(outward / Mathf.Max(0.5f, shoulderWidth));
            float h = -ditchDepth * 4f * e * (1f - e);

            // Landform: quadratic so it leaves the road edge flat and steepens outward,
            // rather than starting as a wall.
            float far = Mathf.Clamp01(outward / Mathf.Max(1f, terrainHalfWidth - roadWidth * 0.5f));
            h += slope * far * far;

            // Roughness, faded in so the road edge stays clean.
            Vector3 sample = pos + Vector3.right * (outward * side);
            h += (Mathf.PerlinNoise(noiseOriginRough + sample.x * 0.05f,
                                    noiseOriginRough + sample.z * 0.05f) - 0.5f) * terrainRoughness * far;

            return h;
        }

        /// <summary>One quad strip: two parallel lines, B on the more positive side.</summary>
        private struct Strip
        {
            public readonly List<Vector3> a, b;

            /// <summary>
            /// Where each line sits across the stage, in metres from the centreline.
            /// Carried rather than measured because texture coordinates have to run
            /// continuously from one strip into the next: computing u from a strip's
            /// own width would restart it at zero on every step out from the road and
            /// put a seam down each one.
            /// </summary>
            public readonly float lateralA, lateralB;

            public Strip(List<Vector3> a, List<Vector3> b, float lateralA, float lateralB)
            {
                this.a = a; this.b = b;
                this.lateralA = lateralA; this.lateralB = lateralB;
            }
        }

        /// <summary>
        /// Stitches pairs of longitudinal lines into a triangle strip. Winding is chosen
        /// so the faces point up: for a triangle (p0,p1,p2) Unity's face normal is
        /// cross(p1-p0, p2-p0), which for (prevA, curA, prevB) comes out +Y as long as
        /// B is the more positive side along the road's right vector.
        ///
        /// <paramref name="tileMetres"/> is how much ground one repeat of the texture
        /// covers. Coordinates are laid out in real metres — lateral offset across,
        /// distance along the stage up — rather than normalised 0..1 per strip. That
        /// matters because the strips are wildly different widths: the innermost
        /// terrain step is under a metre across and the outermost is thirty, so
        /// normalised coordinates would stretch one texture repeat over each of them
        /// and the ground would visibly change scale as it recedes.
        /// </summary>
        /// <summary>
        /// A line running the whole length of the stage at a fixed lateral offset from the
        /// centreline, lifted straight off the per-sample spline frame so it carries exactly
        /// the same camber and elevation as the road edges do. Used to tessellate the road
        /// across its width without inventing any new geometry.
        /// </summary>
        private List<Vector3> RoadLine(float lateral)
        {
            var line = new List<Vector3>(samplePos.Count);
            for (int k = 0; k < samplePos.Count; k++)
                line.Add(samplePos[k] + sampleRight[k] * lateral);
            return line;
        }

        private Mesh BuildStrips(string name, Strip[] strips, float tileMetres)
        {
            var verts = new List<Vector3>();
            var tris  = new List<int>();
            var uvs   = new List<Vector2>();

            float invTile = 1f / Mathf.Max(0.01f, tileMetres);

            foreach (Strip strip in strips)
            {
                List<Vector3> a = strip.a, b = strip.b;
                int baseIndex = verts.Count;
                int n = Mathf.Min(a.Count, b.Count);
                float uA = strip.lateralA * invTile, uB = strip.lateralB * invTile;

                for (int k = 0; k < n; k++)
                {
                    // Distance along the stage, not sample index: the samples are not
                    // evenly spaced, and keying off the index would compress the
                    // texture through corners where they bunch up.
                    float v = (k < sampleDistance.Count ? sampleDistance[k] : k) * invTile;

                    verts.Add(a[k]); verts.Add(b[k]);
                    uvs.Add(new Vector2(uA, v));
                    uvs.Add(new Vector2(uB, v));

                    if (k == 0) continue;
                    int prevA = baseIndex + (k - 1) * 2, prevB = prevA + 1;
                    int curA  = baseIndex + k * 2,       curB  = curA + 1;
                    tris.Add(prevA); tris.Add(curA); tris.Add(prevB);
                    tris.Add(prevB); tris.Add(curA); tris.Add(curB);
                }
            }

            var mesh = new Mesh { name = name };
            if (verts.Count > 65000) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.SetUVs(0, uvs);
            mesh.RecalculateNormals();   // faceted for PS1
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>
        /// True if this point is close enough to the start line that a rock there could be
        /// touching the car when the episode begins.
        /// </summary>
        private bool IsInSpawnZone(Vector3 localPos)
        {
            if (waypoints.Count <= spawnWaypointIndex) return false;
            return Vector3.Distance(localPos, waypoints[spawnWaypointIndex]) < obstacleSpawnClearance;
        }
    }
}
