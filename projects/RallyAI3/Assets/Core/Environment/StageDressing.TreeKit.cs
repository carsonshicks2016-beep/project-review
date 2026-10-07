using System;
using System.Collections.Generic;
using UnityEngine;

namespace Core.Environment
{
    public partial class StageDressing
    {
        [Serializable] public class TreeSpecies
        {
            public string name;
            public int atlasTile, tiers, boughs;
            public float crownStart, crownRadius, droop, trunkRadius;
        }
        [Serializable] public class TreeKitDefinition
        {
            public int schema, reviewSeed;
            public bool productionCoverage;
            public string name;
            public float sectionStart, sectionEnd, cellLength;
            public TreeSpecies[] species;
        }
        static TreeKitDefinition treeKit;
        static bool kitChecked;
        public static int[] LastReviewSpeciesCounts { get; private set; } = new int[4];

        public static TreeKitDefinition ReviewKit
        {
            get
            {
                if (!kitChecked)
                {
                    kitChecked = true;
                    var source = Resources.Load<TextAsset>("StageDressing/TreeKitV3");
                    if (source != null) treeKit = JsonUtility.FromJson<TreeKitDefinition>(source.text);
                    if (treeKit != null && (treeKit.schema != 1 || treeKit.species == null || treeKit.species.Length != 4 ||
                        treeKit.cellLength < 12 || treeKit.sectionEnd <= treeKit.sectionStart))
                        throw new InvalidOperationException("Invalid Forest V3 review kit");
                }
                return treeKit;
            }
        }

        public static bool UsesReviewKit(TrackGenerator track) => track != null && ReviewKit != null &&
            (ReviewKit.productionCoverage || track.seed == ReviewKit.reviewSeed) &&
            System.Environment.GetEnvironmentVariable("RALLY_FOREST_V3") != "0";

        sealed class ReviewForest
        {
            sealed class Cell
            {
                public readonly BranchTreeMesh[] levels = { new BranchTreeMesh(), new BranchTreeMesh(), new BranchTreeMesh() };
            }
            readonly Dictionary<int, Cell> cells = new Dictionary<int, Cell>();
            readonly TreeKitDefinition kit;
            readonly Transform parent;
            readonly Material bark, foliage;
            int trees;
            readonly int[] speciesCounts = new int[4];

            public ReviewForest(Transform root, Material barkMaterial)
            {
                kit = ReviewKit; parent = root; bark = barkMaterial;
                foliage = Resources.Load<Material>("StageDressing/TreeFoliageV3");
                if (foliage == null) throw new InvalidOperationException("Forest V3 foliage material is missing");
            }
            public bool Contains(float distance) => kit.productionCoverage ||
                (distance >= kit.sectionStart && distance < kit.sectionEnd);

            public void Add(Vector3 foot, float height, float yaw, float lean, float shade, bool edge, float depth, float station, bool anchored = false, float anchorRadius = 0f)
            {
                // A coordinate hash does not consume placement RNG or alter any later tree.
                float select = BranchTreeMesh.Noise(foot, 911);
                if (!anchored && BranchTreeMesh.Noise(foot, 977) > Mathf.Lerp(.92f, .65f, depth)) return;
                int species = height < 9f ? 2 : select < .59f ? 0 : select < .80f ? 1 : select < .91f ? 2 : 3;
                if (!anchored && species == 2) height *= .65f;
                if (!anchored && species == 3) height *= .80f;
                var definition = kit.species[species];
                if (anchored)
                    definition = new TreeSpecies { name = definition.name, atlasTile = definition.atlasTile,
                        tiers = definition.tiers, boughs = definition.boughs, crownStart = definition.crownStart,
                        crownRadius = definition.crownRadius, droop = definition.droop,
                        trunkRadius = Mathf.Max(definition.trunkRadius, anchorRadius / Mathf.Max(height, .1f)) };
                int key = Mathf.FloorToInt(station / kit.cellLength);
                if (!cells.TryGetValue(key, out Cell cell)) cells[key] = cell = new Cell();
                for (int lod = 0; lod < 3; lod++)
                    cell.levels[lod].AddTree(foot, height, yaw, lean, shade, definition, lod);
                trees++;
                speciesCounts[species]++;
            }
            public void Attach()
            {
                LastReviewSpeciesCounts = (int[])speciesCounts.Clone();
                foreach (var item in cells)
                {
                    var group = new GameObject($"ForestV3_{item.Key:00}") { layer = 2 };
                    group.transform.SetParent(parent, false);
                    var lods = new LOD[3];
                    for (int i = 0; i < 3; i++)
                    {
                        var level = new GameObject("LOD" + i) { layer = 2 };
                        level.transform.SetParent(group.transform, false);
                        item.Value.levels[i].trunks.Attach(level.transform, "Trunks", bark);
                        item.Value.levels[i].Attach(level.transform, "Boughs", foliage);
                        lods[i] = new LOD(i == 0 ? .32f : i == 1 ? .12f : .015f,
                            level.GetComponentsInChildren<Renderer>()) { fadeTransitionWidth = .18f };
                    }
                    var controller = group.AddComponent<LODGroup>();
                    controller.fadeMode = LODFadeMode.CrossFade;
                    controller.animateCrossFading = false;
                    controller.SetLODs(lods);
                    controller.RecalculateBounds();
                }
                Debug.Log($"[ForestV3] {trees} branch-based trees in {cells.Count} LOD cells; {(kit.productionCoverage ? "full course" : $"section {kit.sectionStart}-{kit.sectionEnd}m")}");
            }
        }

        sealed class BranchTreeMesh : Batch
        {
            public readonly BranchTrunks trunks = new BranchTrunks();
            public static float Noise(Vector3 p, int salt)
            {
                unchecked
                {
                    uint h = (uint)Mathf.RoundToInt(p.x * 100f) * 374761393u ^
                        (uint)Mathf.RoundToInt(p.z * 100f) * 668265263u ^ (uint)salt * 2246822519u;
                    h = (h ^ (h >> 13)) * 1274126177u;
                    return (h ^ (h >> 16)) / (float)uint.MaxValue;
                }
            }
            public void AddTree(Vector3 foot, float height, float yaw, float lean, float shade, TreeSpecies species, int lod)
            {
                Quaternion rotation = Quaternion.Euler(0f, yaw, 0f);
                Vector3 bend = rotation * new Vector3((lean - .5f) * height * .055f, 0f, height * .012f);
                Color needles = Color.Lerp(new Color(.48f, .64f, .60f), new Color(.72f, .79f, .66f), shade);
                Color barkColor = species.atlasTile == 3 ? new Color(.93f, .93f, .88f) : new Color(.76f, .73f, .66f);
                trunks.Trunk(foot, bend, height, height * species.trunkRadius, barkColor, lod);
                int tiers = lod == 2 ? Mathf.Max(4, species.tiers / 2) : species.tiers;
                int boughs = lod == 0 ? species.boughs : Mathf.Max(3, species.boughs / 2);
                for (int tier = 0; tier < tiers; tier++)
                {
                    float f = tier / Mathf.Max(1f, tiers - 1f);
                    float y = height * Mathf.Lerp(species.crownStart, .96f, f);
                    Vector3 hub = foot + Vector3.up * y + bend * (y / height);
                    float envelope = species.atlasTile == 3 ? Mathf.Sin((.12f + f * .80f) * Mathf.PI) : 1f - f * .85f;
                    float radius = height * species.crownRadius * envelope;
                    for (int bough = 0; bough < boughs; bough++)
                    {
                        int salt = tier * 31 + bough * 11;
                        float jitter = Noise(foot, salt + 101);
                        float angle = yaw + bough * 360f / boughs + tier * 137.5f + (jitter - .5f) * 35f;
                        Vector3 direction = Quaternion.Euler(0f, angle, 0f) * Vector3.forward;
                        float length = radius * Mathf.Lerp(.64f, 1.22f, jitter);
                        Vector3 tip = hub + direction * length - Vector3.up * (length * species.droop);
                        if (lod < 2) trunks.Branch(hub, tip, height * .0035f * (1f - f * .7f), barkColor, lod);
                        Card(hub + direction * .08f, tip, direction, length * (lod == 0 ? .48f : .52f), species.atlasTile,
                            needles * Mathf.Lerp(.73f, 1f, f), lod > 0);
                        if (lod == 0)
                            Card(hub + direction * length * .12f, tip + Vector3.up * length * .12f, direction,
                                length * .38f, species.atlasTile, needles, true);
                    }
                }
            }

            void Card(Vector3 root, Vector3 tip, Vector3 direction, float halfWidth, int tile, Color tint, bool upright)
            {
                Vector3 across = upright ? Vector3.up : Vector3.Cross(Vector3.up, direction).normalized;
                across *= halfWidth;
                Vector3 normal = upright ? Vector3.Cross(direction, Vector3.up).normalized : (Vector3.up + direction * .25f).normalized;
                Vector3 middle = Vector3.Lerp(root, tip, .54f) + Vector3.up * halfWidth * .20f;
                float x = tile % 2 * .5f, y = tile < 2 ? .5f : 0f;
                int start = verts.Count;
                Vector3[] centers = { root, middle, tip };
                for (int row = 0; row < 3; row++)
                {
                    Vector3 width = across * (row == 0 ? .52f : row == 1 ? 1f : .60f);
                    float u = x + .015f + row * .235f;
                    Add(centers[row] - width, normal, new Vector2(u, y + .015f), tint);
                    Add(centers[row] + width, normal, new Vector2(u, y + .485f), tint);
                }
                Quad(start, start + 2, start + 3, start + 1);
                Quad(start + 2, start + 4, start + 5, start + 3);
            }
        }

        sealed class BranchTrunks : Batch
        {
            public void Trunk(Vector3 foot, Vector3 bend, float height, float radius, Color color, int lod)
            {
                int segments = lod == 0 ? 4 : 2;
                for (int i = 0; i < segments; i++)
                {
                    float a = i / (float)segments, b = (i + 1f) / segments;
                    Cylinder(foot + Vector3.up * (height * a - .15f) + bend * a * a,
                        foot + Vector3.up * height * b + bend * b * b,
                        radius * Mathf.Lerp(1f, .10f, a), radius * Mathf.Lerp(1f, .10f, b), color, lod == 0 ? 7 : 5);
                }
            }
            public void Branch(Vector3 root, Vector3 tip, float radius, Color color, int lod) =>
                Cylinder(root, Vector3.Lerp(root, tip, .80f), radius, radius * .22f, color, lod == 0 ? 4 : 3);

            void Cylinder(Vector3 a, Vector3 b, float r0, float r1, Color tint, int sides)
            {
                Vector3 axis = (b - a).normalized;
                Vector3 right = Vector3.Cross(axis, Mathf.Abs(axis.y) > .9f ? Vector3.right : Vector3.up).normalized;
                Vector3 up = Vector3.Cross(axis, right).normalized;
                int start = verts.Count;
                for (int i = 0; i <= sides; i++)
                {
                    float angle = i * Mathf.PI * 2f / sides;
                    Vector3 n = right * Mathf.Cos(angle) + up * Mathf.Sin(angle);
                    float u = Mathf.Lerp(.025f, .47f, i / (float)sides);
                    Add(a + n * r0, n, new Vector2(u, 0f), tint);
                    Add(b + n * r1, n, new Vector2(u, 1f), tint);
                }
                for (int i = 0; i < sides; i++) Quad(start + i * 2, start + i * 2 + 1, start + i * 2 + 3, start + i * 2 + 2);
            }
        }
    }
}
