using System.Collections.Generic;
using UnityEngine;
using Core.Physics;

namespace EditorScripts
{
    /// <summary>
    /// Builds the 1998 Subaru Impreza WRC — body, wheels and livery — as pure data.
    /// No scene access, no AssetDatabase: hand it nothing, get a Mesh back. The
    /// scene wiring and asset saving live in <see cref="RallyCarBuilder"/>.
    ///
    /// The body is lofted through eighteen cross-sections taken from the real car's
    /// side profile, in the same coordinate frame the physics uses: y = 0 is the
    /// ground, +Z is forward, wheelbase centred on the origin. That is the whole
    /// point of sharing <see cref="ImprezaSpec"/> — the arches end up exactly where
    /// the suspension puts the wheels, because both read the same numbers.
    ///
    /// Panel faces keep independent UVs and material seams. A position-aware normal
    /// pass softens the long body transitions while preserving crisp panel breaks.
    /// </summary>
    public static class ImprezaMeshBuilder
    {
        // ─── Sub-mesh slots ──────────────────────────────────────────
        public const int BODY  = 0;   // painted panels, livery-mapped
        public const int GLASS = 1;   // blacked-out rally glass
        public const int TRIM  = 2;   // underbody, arch liners, struts, splitter
        public const int LAMP  = 3;   // headlights
        public const int TAIL  = 4;   // tail lights
        public const int SUBMESH_COUNT = 5;

        public const int WHEEL_RIM  = 0;
        public const int WHEEL_TYRE = 1;
        public const int WHEEL_SUBMESH_COUNT = 2;

        // ─── Livery atlas layout ─────────────────────────────────────
        //  v 0.00 – 0.28 : eight flat colour swatches, side by side
        //  v 0.31 – 0.585: plan view of the car (u = nose↔tail, v = left↔right)
        //  v 0.60 – 1.00 : side elevation  (u = nose↔tail, v = ground↔roof)
        const float SwatchV = 0.14f;
        const float PlanV0 = 0.31f, PlanV1 = 0.585f;
        const float SideV0 = 0.60f, SideV1 = 1.00f;

        /// <summary>Tallest point the side elevation covers, in metres.</summary>
        const float SideHeight = 1.50f;
        /// <summary>Half-width the plan view covers, in metres.</summary>
        const float PlanHalfWidth = 0.95f;

        public enum Paint { Blue = 0, Dark = 1, Yellow = 2, White = 3, Gold = 4, Rubber = 5, Amber = 6, Red = 7 }

        /// <summary>How a face picks its texture coordinates.</summary>
        enum UvMode { Side, Plan, Swatch }

        // ══════════════════════════════════════════════════════════════
        //  CROSS-SECTIONS
        // ══════════════════════════════════════════════════════════════
        //
        //         P0 ────────── P1     yTop    (roof, bonnet or boot lid)
        //        /                \
        //      P5                  P2  yBelt   (belt line / fender crown)
        //       |                  |
        //      P4 ────────────── P3   yFloor  (sill, or the roof of a wheel arch)
        //
        //  Where a section has no greenhouse, yTop == yBelt and the P0/P5 and
        //  P1/P2 pairs collapse together — the upper side quad degenerates to
        //  nothing and the top quad becomes the bonnet or the boot lid.
        //
        struct Section
        {
            public float z;
            public float yFloor, yBelt, yTop;
            public float hwFloor, hwBody, hwTop;

            public Vector3 P0 => new Vector3(-hwTop,   yTop,   z);
            public Vector3 P1 => new Vector3( hwTop,   yTop,   z);
            public Vector3 P2 => new Vector3( hwBody,  yBelt,  z);
            public Vector3 P3 => new Vector3( hwFloor, yFloor, z);
            public Vector3 P4 => new Vector3(-hwFloor, yFloor, z);
            public Vector3 P5 => new Vector3(-hwBody,  yBelt,  z);
        }

        static Section Sec(float z, float yF, float yB, float yT, float hwF, float hwB, float hwT)
            => new Section { z = z, yFloor = yF, yBelt = yB, yTop = yT,
                             hwFloor = hwF, hwBody = hwB, hwTop = hwT };

        //  z         yFloor yBelt  yTop   hwFloor hwBody hwTop
        //  Arch sections trace the wheel opening instead of cutting a rectangular notch.
        //  Their crown clears the tyre at full bump; the lower points follow the visible
        //  fender radius so the body reads as a sedan rather than a slab over four wheels.
        const float ArchRoof = 0.84f;

        //  The GC8 is a boxy little sedan, not a wedge: both ends are close to
        //  vertical and the bonnet is flat, but its glasshouse is narrower than the
        //  fenders and both screens rake into a distinct boot. The cross-sections below
        //  keep the homologated overall envelope while giving the car a real shoulder.
        static readonly Section[] Sections =
        {
            Sec( 2.13f, 0.30f, 0.80f, 0.80f, 0.70f, 0.74f, 0.74f),  //  0 bumper face — blunt
            Sec( 2.02f, 0.22f, 0.87f, 0.87f, 0.74f, 0.82f, 0.82f),  //  1 headlight line
            Sec( 1.86f, 0.20f, 0.90f, 0.90f, 0.76f, 0.85f, 0.85f),  //  2 front of the bonnet
            Sec( 1.81f, 0.29f, 0.92f, 0.92f, 0.78f, 0.85f, 0.85f),  //  3 front arch leading shoulder
            Sec( 1.74f, 0.54f, 0.94f, 0.94f, 0.82f, 0.87f, 0.87f),  //  4 front arch rising
            Sec( 1.64f, 0.69f, 0.95f, 0.95f, 0.87f, 0.90f, 0.90f),  //  5 flare begins
            Sec( 1.45f, 0.81f, 0.96f, 0.96f, 0.90f, 0.90f, 0.90f),  //  6 arch crown approach
            Sec( 1.26f, ArchRoof, 0.96f, 0.96f, 0.905f, 0.905f, 0.905f), // 7 FRONT AXLE
            Sec( 1.07f, 0.81f, 0.95f, 0.95f, 0.90f, 0.90f, 0.90f),  //  8 arch crown exit
            Sec( 0.88f, 0.69f, 0.94f, 0.94f, 0.87f, 0.89f, 0.89f),  //  9 rear flare
            Sec( 0.77f, 0.54f, 0.94f, 0.94f, 0.82f, 0.87f, 0.87f),  // 10 arch falling
            Sec( 0.71f, 0.29f, 0.94f, 0.94f, 0.78f, 0.85f, 0.85f),  // 11 arch trailing shoulder
            Sec( 0.62f, 0.20f, 0.94f, 0.94f, 0.76f, 0.85f, 0.85f),  // 12 sill / cowl
            Sec( 0.20f, 0.20f, 0.98f, 1.34f, 0.76f, 0.84f, 0.64f),  // 13 windscreen crown
            Sec(-0.62f, 0.20f, 0.98f, 1.32f, 0.76f, 0.84f, 0.64f),  // 14 roof rear
            Sec(-0.71f, 0.29f, 0.99f, 0.99f, 0.78f, 0.85f, 0.85f),  // 15 rear arch leading shoulder
            Sec(-0.77f, 0.54f, 0.99f, 0.99f, 0.82f, 0.87f, 0.87f),  // 16 arch rising
            Sec(-0.88f, 0.69f, 1.00f, 1.00f, 0.87f, 0.89f, 0.89f),  // 17 rear flare
            Sec(-1.07f, 0.81f, 1.01f, 1.01f, 0.90f, 0.90f, 0.90f),  // 18 arch crown approach
            Sec(-1.26f, ArchRoof, 1.01f, 1.01f, 0.905f, 0.905f, 0.905f), // 19 REAR AXLE
            Sec(-1.45f, 0.81f, 1.00f, 1.00f, 0.90f, 0.90f, 0.90f),  // 20 arch crown exit
            Sec(-1.64f, 0.69f, 0.99f, 0.99f, 0.87f, 0.89f, 0.89f),  // 21 rear flare
            Sec(-1.75f, 0.54f, 0.99f, 0.99f, 0.82f, 0.87f, 0.87f),  // 22 arch falling
            Sec(-1.81f, 0.29f, 0.99f, 0.99f, 0.78f, 0.85f, 0.85f),  // 23 arch trailing shoulder
            Sec(-1.95f, 0.24f, 0.99f, 0.99f, 0.72f, 0.83f, 0.83f),  // 24 rear bumper
            Sec(-2.21f, 0.32f, 0.96f, 0.96f, 0.66f, 0.78f, 0.78f),  // 25 tail face — blunt
        };

        // Glazing lives on specific joins between sections.
        const int PairWindscreen = 12;  // section 12 → 13
        const int PairRoof       = 13;  // section 13 → 14, side glass on the flanks

        // ─── Mesh accumulators ───────────────────────────────────────
        static List<Vector3> s_verts;
        static List<Vector2> s_uvs;
        static List<int>[]   s_tris;

        // ══════════════════════════════════════════════════════════════
        //  BODY
        // ══════════════════════════════════════════════════════════════
        public static Mesh BuildBody()
        {
            Begin(SUBMESH_COUNT);

            SkinSections();
            RearWindow();
            EndCaps();
            WheelArchLiners();
            Underbody();
            BonnetScoop();
            RearWing();
            Splitter();
            SideSkirtsAndFlaps();
            DoorDetails();
            Mirrors();
            Lights();
            RoofVent();

            Mesh mesh = new Mesh { name = "Impreza_WRC_Body" };
            mesh.SetVertices(s_verts);
            mesh.SetUVs(0, s_uvs);
            mesh.subMeshCount = SUBMESH_COUNT;
            for (int i = 0; i < SUBMESH_COUNT; i++)
                mesh.SetTriangles(s_tris[i], i);

            mesh.RecalculateNormals();
            SmoothBodyNormals(mesh);
            mesh.RecalculateBounds();
            End();
            return mesh;
        }

        static void SmoothBodyNormals(Mesh mesh)
        {
            Vector3[] vertices = mesh.vertices;
            Vector3[] normals = mesh.normals;
            int[] bodyTriangles = mesh.GetTriangles(BODY);
            var adjacentNormals = new Dictionary<Vector3Int, List<Vector3>>();

            for (int i = 0; i < bodyTriangles.Length; i += 3)
            {
                int a = bodyTriangles[i], b = bodyTriangles[i + 1], c = bodyTriangles[i + 2];
                Vector3 weightedNormal = Vector3.Cross(vertices[b] - vertices[a], vertices[c] - vertices[a]);
                if (weightedNormal.sqrMagnitude < 1e-10f) continue;
                AddNormal(adjacentNormals, PositionKey(vertices[a]), weightedNormal);
                AddNormal(adjacentNormals, PositionKey(vertices[b]), weightedNormal);
                AddNormal(adjacentNormals, PositionKey(vertices[c]), weightedNormal);
            }

            const float smoothingLimit = 0.57f;
            var processed = new bool[vertices.Length];
            for (int i = 0; i < bodyTriangles.Length; i++)
            {
                int vertex = bodyTriangles[i];
                if (processed[vertex]) continue;
                processed[vertex] = true;
                if (!adjacentNormals.TryGetValue(PositionKey(vertices[vertex]), out List<Vector3> candidates)) continue;
                Vector3 reference = normals[vertex].normalized;
                Vector3 sum = Vector3.zero;
                for (int j = 0; j < candidates.Count; j++)
                {
                    Vector3 candidate = candidates[j];
                    if (Vector3.Dot(reference, candidate.normalized) >= smoothingLimit) sum += candidate;
                }
                if (sum.sqrMagnitude > 1e-10f) normals[vertex] = sum.normalized;
            }

            mesh.normals = normals;
        }

        static void AddNormal(Dictionary<Vector3Int, List<Vector3>> groups, Vector3Int key, Vector3 normal)
        {
            if (!groups.TryGetValue(key, out List<Vector3> values))
            {
                values = new List<Vector3>(4);
                groups.Add(key, values);
            }
            values.Add(normal);
        }

        static Vector3Int PositionKey(Vector3 point)
            => new Vector3Int(Mathf.RoundToInt(point.x * 10000f), Mathf.RoundToInt(point.y * 10000f),
                              Mathf.RoundToInt(point.z * 10000f));

        /// <summary>Lofts the skin between every adjacent pair of cross-sections.</summary>
        static void SkinSections()
        {
            for (int i = 0; i < Sections.Length - 1; i++)
            {
                Section a = Sections[i];      // the one further forward
                Section b = Sections[i + 1];

                int topSub   = i == PairWindscreen ? GLASS : BODY;
                int upperSub = (i == PairRoof) ? GLASS : BODY;

                // Top: bonnet, screen, roof, boot lid.
                Quad(a.P0, a.P1, b.P1, b.P0, topSub,
                     topSub == GLASS ? UvMode.Swatch : UvMode.Plan,
                     topSub == GLASS ? Paint.Dark : Paint.Blue);

                // Right flank, above and below the belt line.
                if (upperSub == GLASS) SplitSideGlass(a.P2, b.P2, b.P1, a.P1);
                else Quad(a.P2, b.P2, b.P1, a.P1, upperSub, UvMode.Side, Paint.Dark);
                Quad(a.P2, a.P3, b.P3, b.P2, BODY, UvMode.Side);

                // Underside — or, across an arch, the roof of the wheel well.
                Quad(a.P3, a.P4, b.P4, b.P3, TRIM, UvMode.Swatch, Paint.Dark);

                // Left flank.
                Quad(a.P4, a.P5, b.P5, b.P4, BODY, UvMode.Side);
                if (upperSub == GLASS) SplitSideGlass(a.P0, b.P0, b.P5, a.P5);
                else Quad(a.P0, b.P0, b.P5, a.P5, upperSub, UvMode.Side, Paint.Dark);
            }
        }

        static void RearWindow()
        {
            Section roofRear = Sections[14];
            Vector3 bootLeft = new Vector3(-0.74f, 1.015f, -1.02f);
            Vector3 bootRight = new Vector3(0.74f, 1.015f, -1.02f);
            Quad(roofRear.P0, roofRear.P1, bootRight, bootLeft, GLASS, UvMode.Swatch, Paint.Dark);
        }

        static void SplitSideGlass(Vector3 frontBottom, Vector3 rearBottom,
                                   Vector3 rearTop, Vector3 frontTop)
        {
            Vector3 bottom(float t) => Vector3.Lerp(frontBottom, rearBottom, t);
            Vector3 top(float t) => Vector3.Lerp(frontTop, rearTop, t);

            // Leave painted A/C pillars at the ends and a broad, body-coloured B-pillar
            // between the front and rear panes. The GC8's four-door silhouette reads
            // immediately, even at chase-camera distance.
            Pane(0.055f, 0.43f, GLASS);
            Pane(0.43f, 0.485f, BODY);
            Pane(0.485f, 0.945f, GLASS);

            void Pane(float start, float end, int material)
            {
                Quad(bottom(start), bottom(end), top(end), top(start), material,
                     material == GLASS ? UvMode.Swatch : UvMode.Side, Paint.Dark);
            }
        }

        static void EndCaps()
        {
            Section nose = Sections[0];
            Quad(nose.P4, nose.P3, nose.P1, nose.P0, BODY, UvMode.Swatch, Paint.Blue);

            Section tail = Sections[Sections.Length - 1];
            Quad(tail.P0, tail.P1, tail.P3, tail.P4, BODY, UvMode.Swatch, Paint.Blue);
        }

        /// <summary>
        /// The arch sections cut a notch straight through the car. These two boxes
        /// plug the middle of the notch, which both closes the bodywork and forms
        /// the inner arch walls the wheels tuck up against.
        /// </summary>
        static void WheelArchLiners()
        {
            const float innerX = 0.62f;
            const float floorY = 0.20f;
            float roofY = ArchRoof + 0.01f;

            AddBox(-innerX, floorY,  0.62f, innerX, roofY,  1.86f, TRIM, Paint.Dark);
            AddBox(-innerX, floorY, -1.86f, innerX, roofY, -0.62f, TRIM, Paint.Dark);
        }

        /// <summary>Flat floor pan under the whole car — a rally skid plate.</summary>
        static void Underbody()
        {
            AddBox(-0.70f, ImprezaSpec.GroundClearance - 0.02f, -2.05f,
                    0.70f, ImprezaSpec.GroundClearance + 0.03f, 1.98f, TRIM, Paint.Dark);
        }

        /// <summary>The bonnet intake every Impreza is recognised by.</summary>
        static void BonnetScoop()
        {
            AddBox(-0.27f, 0.93f, 1.00f, 0.27f, 1.03f, 1.58f, BODY, Paint.Blue);
            // The mouth of the intake, sunk into the front face of the scoop.
            Quad(new Vector3(-0.22f, 1.00f, 1.585f), new Vector3(-0.22f, 0.955f, 1.585f),
                 new Vector3( 0.22f, 0.955f, 1.585f), new Vector3( 0.22f, 1.00f, 1.585f),
                 TRIM, UvMode.Swatch, Paint.Dark);
        }

        /// <summary>WRC-spec rear wing: two uprights, a plane and a pair of end plates.</summary>
        static void RearWing()
        {
            AddBox(-0.55f, 1.015f, -1.92f, -0.49f, 1.29f, -1.82f, TRIM, Paint.Dark);
            AddBox( 0.49f, 1.015f, -1.92f,  0.55f, 1.29f, -1.82f, TRIM, Paint.Dark);

            AddBox(-0.84f, 1.285f, -2.045f, 0.84f, 1.345f, -1.72f, BODY, Paint.Blue);

            AddBox(-0.88f, 1.245f, -2.08f, -0.83f, 1.395f, -1.67f, BODY, Paint.Yellow);
            AddBox( 0.83f, 1.245f, -2.08f,  0.88f, 1.395f, -1.67f, BODY, Paint.Yellow);
        }

        static void DoorDetails()
        {
            // Small recessed handles sit on the side panels, below the side glazing.
            AddBox(-0.872f, 0.735f, -0.30f, -0.858f, 0.765f, -0.08f, TRIM, Paint.Dark);
            AddBox( 0.858f, 0.735f, -0.30f,  0.872f, 0.765f, -0.08f, TRIM, Paint.Dark);

            // A narrow sill accent separates the blue door skin from the gravel-black
            // rally skirt without enlarging the collision envelope.
            AddBox(-0.91f, 0.285f, -0.92f, -0.895f, 0.315f, 0.55f, BODY, Paint.Yellow);
            AddBox( 0.895f, 0.285f, -0.92f,  0.91f, 0.315f, 0.55f, BODY, Paint.Yellow);
        }

        static void Splitter()
        {
            AddBox(-0.84f, 0.155f, 1.90f, 0.84f, 0.205f, 2.17f, TRIM, Paint.Dark);
        }

        static void SideSkirtsAndFlaps()
        {
            // Sills, running between the two arches.
            AddBox(-0.91f, 0.20f, -1.00f, -0.80f, 0.31f, 0.86f, TRIM, Paint.Dark);
            AddBox( 0.80f, 0.20f, -1.00f,  0.91f, 0.31f, 0.86f, TRIM, Paint.Dark);

            // Mud flaps, hung just behind all four wheels.
            AddBox(-0.90f, 0.06f,  0.820f, -0.62f, 0.42f,  0.850f, TRIM, Paint.Dark);
            AddBox( 0.62f, 0.06f,  0.820f,  0.90f, 0.42f,  0.850f, TRIM, Paint.Dark);
            AddBox(-0.90f, 0.06f, -1.605f, -0.62f, 0.42f, -1.575f, TRIM, Paint.Dark);
            AddBox( 0.62f, 0.06f, -1.605f,  0.90f, 0.42f, -1.575f, TRIM, Paint.Dark);
        }

        static void Mirrors()
        {
            // Stalk, then the housing, on both sides.
            AddBox(-0.90f, 0.95f, 0.50f, -0.84f, 1.00f, 0.58f, TRIM, Paint.Dark);
            AddBox(-1.03f, 0.96f, 0.44f, -0.88f, 1.09f, 0.57f, BODY, Paint.Blue);

            AddBox( 0.84f, 0.95f, 0.50f,  0.90f, 1.00f, 0.58f, TRIM, Paint.Dark);
            AddBox( 0.88f, 0.96f, 0.44f,  1.03f, 1.09f, 0.57f, BODY, Paint.Blue);
        }

        /// <summary>
        /// Head lights and grille sit on the sloping nose panel, tail lights on the
        /// flat tail. Both are lifted a centimetre off the skin so they read as
        /// separate lenses rather than z-fighting with the paint.
        /// </summary>
        static void Lights()
        {
            // Head lights, grille and bumper vent all sit on the blunt nose panel,
            // a centimetre proud of it so they read as separate lenses.
            const float nz = 2.142f;
            FrontQuad(nz, -0.64f, -0.21f, 0.55f, 0.77f, TRIM, Paint.Dark);
            FrontQuad(nz,  0.21f,  0.64f, 0.55f, 0.77f, TRIM, Paint.Dark);
            FrontQuad(nz + 0.006f, -0.60f, -0.25f, 0.59f, 0.74f, LAMP, Paint.White);
            FrontQuad(nz + 0.006f,  0.25f,  0.60f, 0.59f, 0.74f, LAMP, Paint.White);
            FrontQuad(nz + 0.012f, -0.27f, -0.22f, 0.60f, 0.70f, LAMP, Paint.Amber);
            FrontQuad(nz + 0.012f,  0.22f,  0.27f, 0.60f, 0.70f, LAMP, Paint.Amber);
            FrontQuad(nz, -0.18f,  0.18f, 0.56f, 0.75f, TRIM, Paint.Dark);   // grille
            FrontQuad(nz, -0.50f,  0.50f, 0.34f, 0.50f, TRIM, Paint.Dark);   // bumper vent

            // Three low-relief grille slats and a small central badge.
            for (int i = 0; i < 3; i++)
            {
                float y = 0.59f + i * 0.045f;
                FrontQuad(nz + 0.008f, -0.145f, 0.145f, y, y + 0.012f, BODY, Paint.White);
            }
            FrontQuad(nz + 0.014f, -0.035f, 0.035f, 0.655f, 0.69f, BODY, Paint.Yellow);

            // Tail lights on the rear face, standing 1 cm proud of it.
            const float tz = -2.222f;
            RearQuad(-0.66f, -0.19f, 0.56f, 0.87f, TRIM, Paint.Dark);
            RearQuad( 0.19f,  0.66f, 0.56f, 0.87f, TRIM, Paint.Dark);
            RearQuad(-0.62f, -0.24f, 0.62f, 0.82f, TAIL, Paint.Red, -2.235f);
            RearQuad( 0.24f,  0.62f, 0.62f, 0.82f, TAIL, Paint.Red, -2.235f);
            RearQuad(-0.40f, -0.27f, 0.64f, 0.70f, LAMP, Paint.Amber, -2.242f);
            RearQuad( 0.27f,  0.40f, 0.64f, 0.70f, LAMP, Paint.Amber, -2.242f);
            RearQuad(-0.59f, -0.27f, 0.565f, 0.595f, BODY, Paint.White, -2.242f);
            RearQuad( 0.27f,  0.59f, 0.565f, 0.595f, BODY, Paint.White, -2.242f);

            // Period-accurate yellow tail plate and contrasting rear details.
            RearQuad(-0.255f, 0.255f, 0.45f, 0.61f, BODY, Paint.Yellow, -2.235f);
            for (int digit = 0; digit < 3; digit++)
            {
                float center = (digit - 1) * 0.12f;
                const float halfWidth = 0.025f, thickness = 0.009f;
                RearQuad(center - halfWidth, center + halfWidth, 0.586f, 0.595f, TRIM, Paint.Dark, -2.242f);
                RearQuad(center - halfWidth, center + halfWidth, 0.531f, 0.540f, TRIM, Paint.Dark, -2.242f);
                RearQuad(center - halfWidth, center + halfWidth, 0.476f, 0.485f, TRIM, Paint.Dark, -2.242f);
                RearQuad(center - halfWidth, center - halfWidth + thickness, 0.540f, 0.586f, TRIM, Paint.Dark, -2.242f);
                RearQuad(center + halfWidth - thickness, center + halfWidth, 0.485f, 0.531f, TRIM, Paint.Dark, -2.242f);
            }
        }

        static void RearQuad(float x0, float x1, float y0, float y1, int sub, Paint paint, float z = -2.228f)
        {
            Quad(new Vector3(x0, y1, z), new Vector3(x1, y1, z),
                 new Vector3(x1, y0, z), new Vector3(x0, y0, z),
                 sub, UvMode.Swatch, paint);
        }

        static void RoofVent()
        {
            AddBox(-0.22f, 1.345f, -0.30f, 0.22f, 1.385f, 0.02f, TRIM, Paint.Dark);
        }

        /// <summary>A flat panel facing straight ahead, for lenses on the nose.</summary>
        static void FrontQuad(float z, float x0, float x1, float y0, float y1, int sub, Paint paint)
        {
            Quad(new Vector3(x0, y1, z), new Vector3(x0, y0, z),
                 new Vector3(x1, y0, z), new Vector3(x1, y1, z),
                 sub, UvMode.Swatch, paint);
        }

        // ══════════════════════════════════════════════════════════════
        //  WHEEL
        // ══════════════════════════════════════════════════════════════
        //
        //  Authored around the X axis: the wheel spins about local X, so the
        //  suspension can apply steer as a Y rotation and camber as a Z rotation
        //  without any of the 90-degree fudging the old rig needed.
        //
        //  Twelve segments, gold and dark alternating on the face, which at this
        //  resolution reads exactly like the six-spoke Speedline the works cars ran.
        //
        public static Mesh BuildWheel()
        {
            Begin(WHEEL_SUBMESH_COUNT);

            const int   segs = 18;
            float R  = ImprezaSpec.WheelRadius;          // tread
            float Rr = ImprezaSpec.WheelRadius * 0.62f;  // rim lip
            float W  = ImprezaSpec.TyreWidth * 0.5f;     // half width
            float faceX = W * 0.55f;                     // dished wheel face
            float hubX  = W * 0.20f;

            float[] cy = new float[segs + 1];
            float[] cz = new float[segs + 1];
            for (int i = 0; i <= segs; i++)
            {
                float a = i * (2f * Mathf.PI / segs);
                cy[i] = Mathf.Cos(a);
                cz[i] = Mathf.Sin(a);
            }

            Vector2 gold   = SwatchUv(Paint.Gold);
            Vector2 rubber = SwatchUv(Paint.Rubber);
            Vector2 dark   = SwatchUv(Paint.Dark);

            for (int i = 0; i < segs; i++)
            {
                int n = i + 1;

                Vector3 outerI = new Vector3( W, cy[i] * R, cz[i] * R);
                Vector3 outerN = new Vector3( W, cy[n] * R, cz[n] * R);
                Vector3 innerI = new Vector3(-W, cy[i] * R, cz[i] * R);
                Vector3 innerN = new Vector3(-W, cy[n] * R, cz[n] * R);

                // Tread band.
                RawQuad(outerI, innerI, innerN, outerN, WHEEL_TYRE, rubber);

                // Sidewalls, from the rim lip out to the tread.
                Vector3 lipOutI = new Vector3( W, cy[i] * Rr, cz[i] * Rr);
                Vector3 lipOutN = new Vector3( W, cy[n] * Rr, cz[n] * Rr);
                Vector3 lipInI  = new Vector3(-W, cy[i] * Rr, cz[i] * Rr);
                Vector3 lipInN  = new Vector3(-W, cy[n] * Rr, cz[n] * Rr);

                RawQuad(lipOutI, outerI, outerN, lipOutN, WHEEL_TYRE, rubber);
                RawQuad(lipInI,  lipInN, innerN, innerI,  WHEEL_TYRE, rubber);

                // Outer face: spokes and the gaps between them.
                Vector3 faceI = new Vector3(faceX, cy[i] * Rr, cz[i] * Rr);
                Vector3 faceN = new Vector3(faceX, cy[n] * Rr, cz[n] * Rr);
                Vector3 hub   = new Vector3(hubX, 0f, 0f);
                bool spoke = (i % 3) == 0;
                RawTri(hub, faceI, faceN, spoke ? WHEEL_RIM : WHEEL_TYRE, spoke ? gold : dark);

                // The rim lip itself, joining the dished face to the sidewall.
                RawQuad(faceI, faceN, lipOutN, lipOutI, WHEEL_RIM, gold);

                // Inner face, always dark — you barely see it.
                Vector3 backI = new Vector3(-W * 0.7f, cy[i] * Rr, cz[i] * Rr);
                Vector3 backN = new Vector3(-W * 0.7f, cy[n] * Rr, cz[n] * Rr);
                RawTri(new Vector3(-hubX, 0f, 0f), backN, backI, WHEEL_TYRE, dark);
                RawQuad(lipInI, lipInN, backN, backI, WHEEL_TYRE, dark);
            }

            Mesh mesh = new Mesh { name = "Impreza_WRC_Wheel" };
            mesh.SetVertices(s_verts);
            mesh.SetUVs(0, s_uvs);
            mesh.subMeshCount = WHEEL_SUBMESH_COUNT;
            for (int i = 0; i < WHEEL_SUBMESH_COUNT; i++)
                mesh.SetTriangles(s_tris[i], i);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            End();
            return mesh;
        }

        // ══════════════════════════════════════════════════════════════
        //  STEERING WHEEL
        // ══════════════════════════════════════════════════════════════
        //
        //  Authored around the Z axis — it turns about local Z — because that is what
        //  RobotDriverIK writes to. The tilt of the steering column is put on a PARENT
        //  node instead of baked in here, so the script can keep writing a plain
        //  localRotation without having to know anything about the column angle.
        //
        // Three spokes and a finer rim for the close-up spectator cockpit.
        //
        public static Mesh BuildSteeringWheel()
        {
            Begin(WHEEL_SUBMESH_COUNT);

            const int segs = 48;
            const float R = 0.175f;        // 350 mm across the rim, a rally wheel
            const float tube = 0.017f;     // rim thickness
            const float hubR = 0.032f;

            Vector2 dark = SwatchUv(Paint.Dark);
            Vector2 gold = SwatchUv(Paint.Gold);

            for (int i = 0; i < segs; i++)
            {
                int n = (i + 1) % segs;
                float ai = i * (2f * Mathf.PI / segs);
                float an = n * (2f * Mathf.PI / segs);

                // The rim as a squared-off ring: a front face, a back face and an
                // outer edge. Cheaper than a swept torus and identical at this size.
                Vector3 OuterFront(float a) => new Vector3(Mathf.Cos(a) * (R + tube), Mathf.Sin(a) * (R + tube), tube);
                Vector3 InnerFront(float a) => new Vector3(Mathf.Cos(a) * (R - tube), Mathf.Sin(a) * (R - tube), tube);
                Vector3 OuterBack(float a)  => new Vector3(Mathf.Cos(a) * (R + tube), Mathf.Sin(a) * (R + tube), -tube);
                Vector3 InnerBack(float a)  => new Vector3(Mathf.Cos(a) * (R - tube), Mathf.Sin(a) * (R - tube), -tube);

                RawQuad(InnerFront(ai), OuterFront(ai), OuterFront(an), InnerFront(an), WHEEL_TYRE, dark);
                RawQuad(InnerBack(an), OuterBack(an), OuterBack(ai), InnerBack(ai), WHEEL_TYRE, dark);
                RawQuad(OuterFront(ai), OuterBack(ai), OuterBack(an), OuterFront(an), WHEEL_TYRE, dark);
                RawQuad(InnerBack(ai), InnerBack(an), InnerFront(an), InnerFront(ai), WHEEL_TYRE, dark);
            }

            // Three spokes at 90, 210 and 330 degrees — the classic rally layout, with
            // the top of the rim left clear so the driver can see the road over it.
            float[] spokeAngles = { Mathf.PI * 0.5f, Mathf.PI * 7f / 6f, Mathf.PI * 11f / 6f };
            const float spokeHalf = 0.014f;

            foreach (float a in spokeAngles)
            {
                Vector3 along = new Vector3(Mathf.Cos(a), Mathf.Sin(a), 0f);
                Vector3 across = new Vector3(-Mathf.Sin(a), Mathf.Cos(a), 0f) * spokeHalf;

                Vector3 innerL = along * hubR - across;
                Vector3 innerR = along * hubR + across;
                Vector3 outerL = along * R - across;
                Vector3 outerR = along * R + across;

                Vector3 front = Vector3.forward * (tube * 0.5f);
                RawQuad(innerL + front, outerL + front, outerR + front, innerR + front, WHEEL_RIM, gold);
                RawQuad(innerR - front, outerR - front, outerL - front, innerL - front, WHEEL_RIM, gold);
            }

            // The boss, so the spokes meet something rather than converging on a point.
            for (int i = 0; i < segs; i++)
            {
                int n = (i + 1) % segs;
                float ai = i * (2f * Mathf.PI / segs);
                float an = n * (2f * Mathf.PI / segs);
                Vector3 hubI = new Vector3(Mathf.Cos(ai) * hubR, Mathf.Sin(ai) * hubR, tube);
                Vector3 hubN = new Vector3(Mathf.Cos(an) * hubR, Mathf.Sin(an) * hubR, tube);
                RawTri(new Vector3(0f, 0f, tube), hubI, hubN, WHEEL_RIM, gold);
            }

            Mesh mesh = new Mesh { name = "Impreza_WRC_SteeringWheel" };
            mesh.SetVertices(s_verts);
            mesh.SetUVs(0, s_uvs);
            mesh.subMeshCount = WHEEL_SUBMESH_COUNT;
            for (int i = 0; i < WHEEL_SUBMESH_COUNT; i++)
                mesh.SetTriangles(s_tris[i], i);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            End();
            return mesh;
        }

        // ══════════════════════════════════════════════════════════════
        //  LIVERY
        // ══════════════════════════════════════════════════════════════
        //
        //  One atlas holds the whole car: a strip of flat colours along the bottom,
        //  the plan view in the middle, the side elevation on top. Painting in car
        //  metres rather than pixels means the swooshes land on the doors no matter
        //  what resolution the texture is generated at.
        //
        public static Texture2D BuildLivery(int size = 256)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, false)
            {
                name = "Impreza_555_Livery",
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 4
            };

            var px = new Color32[size * size];

            for (int y = 0; y < size; y++)
            {
                float v = (y + 0.5f) / size;
                for (int x = 0; x < size; x++)
                {
                    float u = (x + 0.5f) / size;
                    px[y * size + x] = SampleLivery(u, v);
                }
            }

            tex.SetPixels32(px);
            tex.Apply(true, false);
            return tex;
        }

        static Color32 SampleLivery(float u, float v)
        {
            if (v < 0.28f)
            {
                int slot = Mathf.Clamp(Mathf.FloorToInt(u * 8f), 0, 7);
                return PaintColour((Paint)slot);
            }

            if (v >= PlanV0 && v <= PlanV1)
            {
                float t = Mathf.InverseLerp(PlanV0, PlanV1, v);
                float xCar = (t * 2f - 1f) * PlanHalfWidth;
                float zCar = Mathf.Lerp(ImprezaSpec.TailZ, ImprezaSpec.NoseZ, u);
                return PaintPlan(zCar, xCar);
            }

            if (v >= SideV0)
            {
                float t = Mathf.InverseLerp(SideV0, SideV1, v);
                float yCar = t * SideHeight;
                float zCar = Mathf.Lerp(ImprezaSpec.TailZ, ImprezaSpec.NoseZ, u);
                return PaintSide(zCar, yCar);
            }

            return PaintColour(Paint.Blue);
        }

        /// <summary>
        /// The side of the works 555 car: rally blue, a white door panel, and the
        /// nested yellow arcs sweeping over each wheel arch.
        /// </summary>
        static Color32 PaintSide(float z, float y)
        {
            Color32 c = PaintColour(Paint.Blue);

            // Shadow line under the doors.
            if (y < 0.31f) c = PaintColour(Paint.Dark);

            // The bold band running the length of the flanks.
            if (InRect(z, y, -1.32f, 1.32f, 0.32f, 0.46f)) c = PaintColour(Paint.Yellow);

            // Two nested arcs sweeping around each arch — the 555 swoosh. Centred
            // on the wheel so they follow the arch line exactly, and floored so
            // they never dive down into the wheel opening.
            foreach (float cz in ArchCentresZ)
            {
                if (Ring(z, y, cz, ImprezaSpec.WheelRadius, 0.41f, 0.57f, 0.33f)) c = PaintColour(Paint.Yellow);
                if (Ring(z, y, cz, ImprezaSpec.WheelRadius, 0.63f, 0.73f, 0.62f)) c = PaintColour(Paint.Yellow);
            }

            // White door panel with a yellow surround — where the 555 went.
            if (InRect(z, y, -0.50f, 0.54f, 0.56f, 0.92f))
            {
                bool edge = !InRect(z, y, -0.46f, 0.50f, 0.60f, 0.88f);
                c = PaintColour(edge ? Paint.Yellow : Paint.White);
            }

            // Panel shut lines, kept clear of the door panel itself.
            if (y > 0.32f && y < 0.98f &&
                (Mathf.Abs(z - 0.62f) < 0.010f || Mathf.Abs(z + 0.62f) < 0.010f))
                c = PaintColour(Paint.Dark);

            return c;
        }

        static readonly float[] ArchCentresZ = { ImprezaSpec.AxleFrontZ, ImprezaSpec.AxleRearZ };

        /// <summary>Looking down on the car: yellow leading edge, roof panel, flank stripes.</summary>
        static Color32 PaintPlan(float z, float x)
        {
            Color32 c = PaintColour(Paint.Blue);
            float ax = Mathf.Abs(x);

            // Yellow band wrapped over the leading edge of the bonnet.
            if (z > 1.74f && z < 2.06f) c = PaintColour(Paint.Yellow);

            // Twin stripes running back down the bonnet, either side of the scoop.
            // A curve here would mirror into a bullseye across the centreline.
            if (ax > 0.32f && ax < 0.50f && z > 0.96f && z < 1.76f) c = PaintColour(Paint.Yellow);

            // Stripes down the shoulders, either side of the roof.
            if (ax > 0.62f && ax < 0.78f && z > -1.10f && z < 0.72f) c = PaintColour(Paint.Yellow);

            // Roof panel.
            if (z > -0.58f && z < 0.14f && ax < 0.44f) c = PaintColour(Paint.White);

            // Boot lid band.
            if (z < -1.78f && z > -2.12f) c = PaintColour(Paint.Yellow);

            return c;
        }

        /// <summary>
        /// An annulus between two radii, clipped to everything above yMin so it
        /// reads as an arc sweeping over the wheel rather than a closed ring.
        /// </summary>
        static bool Ring(float z, float y, float cz, float cy, float r0, float r1, float yMin)
        {
            if (y < yMin) return false;
            float dz = z - cz, dy = y - cy;
            float d = Mathf.Sqrt(dz * dz + dy * dy);
            return d >= r0 && d <= r1;
        }

        static bool InRect(float a, float b, float a0, float a1, float b0, float b1)
            => a >= a0 && a <= a1 && b >= b0 && b <= b1;

        public static Color32 PaintColour(Paint p)
        {
            switch (p)
            {
                case Paint.Blue:   return ImprezaSpec.RallyBlue;
                case Paint.Yellow: return ImprezaSpec.RallyYellow;
                case Paint.White:  return ImprezaSpec.RallyWhite;
                case Paint.Gold:   return ImprezaSpec.RimGold;
                case Paint.Rubber: return ImprezaSpec.TyreBlack;
                case Paint.Amber:  return ImprezaSpec.LampAmber;
                case Paint.Red:    return ImprezaSpec.LampRed;
                default:           return ImprezaSpec.TrimBlack;
            }
        }

        static Vector2 SwatchUv(Paint p) => new Vector2(((int)p + 0.5f) / 8f, SwatchV);

        // ══════════════════════════════════════════════════════════════
        //  MESH PLUMBING
        // ══════════════════════════════════════════════════════════════
        static void Begin(int submeshes)
        {
            s_verts = new List<Vector3>(1024);
            s_uvs   = new List<Vector2>(1024);
            s_tris  = new List<int>[submeshes];
            for (int i = 0; i < submeshes; i++) s_tris[i] = new List<int>(256);
        }

        static void End()
        {
            s_verts = null;
            s_uvs = null;
            s_tris = null;
        }

        /// <summary>
        /// Emits a quad as two triangles with four fresh vertices. Winding is
        /// (a,b,c) then (a,c,d), so the caller must supply the corners in the order
        /// that puts the face normal outward.
        /// </summary>
        static void Quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d,
                         int sub, UvMode mode, Paint paint = Paint.Blue)
        {
            Vector2 ua, ub, uc, ud;
            switch (mode)
            {
                case UvMode.Side:
                    ua = UvSide(a); ub = UvSide(b); uc = UvSide(c); ud = UvSide(d);
                    break;
                case UvMode.Plan:
                    ua = UvPlan(a); ub = UvPlan(b); uc = UvPlan(c); ud = UvPlan(d);
                    break;
                default:
                    ua = ub = uc = ud = SwatchUv(paint);
                    break;
            }
            RawQuad(a, b, c, d, sub, ua, ub, uc, ud);
        }

        static void RawQuad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, int sub, Vector2 uv)
            => RawQuad(a, b, c, d, sub, uv, uv, uv, uv);

        static void RawQuad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, int sub,
                            Vector2 ua, Vector2 ub, Vector2 uc, Vector2 ud)
        {
            // Sections without a greenhouse collapse their upper-side quads to a
            // line. Dropping them here keeps the vertex count honest and stops
            // RecalculateNormals having to invent a normal for a zero-area face.
            Vector3 n = Vector3.Cross(b - a, c - a) + Vector3.Cross(c - a, d - a);
            if (n.sqrMagnitude < 1e-10f) return;

            int i = s_verts.Count;
            s_verts.Add(a); s_verts.Add(b); s_verts.Add(c); s_verts.Add(d);
            s_uvs.Add(ua);  s_uvs.Add(ub);  s_uvs.Add(uc);  s_uvs.Add(ud);

            var t = s_tris[sub];
            t.Add(i); t.Add(i + 1); t.Add(i + 2);
            t.Add(i); t.Add(i + 2); t.Add(i + 3);
        }

        static void RawTri(Vector3 a, Vector3 b, Vector3 c, int sub, Vector2 uv)
        {
            int i = s_verts.Count;
            s_verts.Add(a); s_verts.Add(b); s_verts.Add(c);
            s_uvs.Add(uv);  s_uvs.Add(uv);  s_uvs.Add(uv);
            var t = s_tris[sub];
            t.Add(i); t.Add(i + 1); t.Add(i + 2);
        }

        /// <summary>Axis-aligned box, six faces, all normals pointing out.</summary>
        static void AddBox(float x0, float y0, float z0,
                           float x1, float y1, float z1, int sub, Paint paint)
        {
            UvMode m = sub == BODY ? UvMode.Side : UvMode.Swatch;

            Quad(V(x0,y1,z1), V(x1,y1,z1), V(x1,y1,z0), V(x0,y1,z0), sub, sub == BODY ? UvMode.Plan : m, paint); // +Y
            Quad(V(x0,y0,z0), V(x1,y0,z0), V(x1,y0,z1), V(x0,y0,z1), sub, UvMode.Swatch, Paint.Dark);            // -Y
            Quad(V(x0,y1,z1), V(x0,y0,z1), V(x1,y0,z1), V(x1,y1,z1), sub, m, paint);                             // +Z
            Quad(V(x1,y1,z0), V(x1,y0,z0), V(x0,y0,z0), V(x0,y1,z0), sub, m, paint);                             // -Z
            Quad(V(x1,y1,z1), V(x1,y0,z1), V(x1,y0,z0), V(x1,y1,z0), sub, m, paint);                             // +X
            Quad(V(x0,y1,z0), V(x0,y0,z0), V(x0,y0,z1), V(x0,y1,z1), sub, m, paint);                             // -X
        }

        static Vector3 V(float x, float y, float z) => new Vector3(x, y, z);

        static Vector2 UvSide(Vector3 p)
        {
            float u = Mathf.InverseLerp(ImprezaSpec.TailZ, ImprezaSpec.NoseZ, p.z);
            float t = Mathf.Clamp01(p.y / SideHeight);
            return new Vector2(u, Mathf.Lerp(SideV0, SideV1, t));
        }

        static Vector2 UvPlan(Vector3 p)
        {
            float u = Mathf.InverseLerp(ImprezaSpec.TailZ, ImprezaSpec.NoseZ, p.z);
            float t = Mathf.Clamp01((p.x / PlanHalfWidth + 1f) * 0.5f);
            return new Vector2(u, Mathf.Lerp(PlanV0, PlanV1, t));
        }
    }
}
