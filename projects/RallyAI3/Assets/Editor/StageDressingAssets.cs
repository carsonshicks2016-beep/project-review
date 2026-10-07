using System.IO;
using UnityEditor;
using UnityEngine;

namespace EditorScripts
{
    /// <summary>
    /// The material the runtime forest loads from Resources. It lives in Resources because
    /// StageDressing installs itself at runtime with no serialized references, and a shader
    /// only reaches a player build if some included material uses it.
    /// </summary>
    public static class StageDressingAssets
    {
        const string Folder = "Assets/Resources/StageDressing";
        const string ForestPath = Folder + "/ForestTrees.mat";
        const string RoadPath = Folder + "/Road.mat";
        const string GroundPath = Folder + "/Ground.mat";
        const string SkyPath = Folder + "/Sky.mat";
        const string GlarePath = Folder + "/SunGlare.mat";
        const string DustPath = Folder + "/Dust.mat";
        const string VergeFoliagePath = Folder + "/VergeFoliage.mat";
        const string VergeStonesPath = Folder + "/VergeStones.mat";
        const string FurniturePath = Folder + "/Furniture.mat";
        const string TextureFolder = "Assets/Art/Generated";

        [MenuItem("Rally/Stage/Rebuild Dressing Materials")]
        public static void Ensure()
        {
            Directory.CreateDirectory(Folder);
            var shader = Shader.Find("Rally/World_Lit");
            if (shader == null) throw new FileNotFoundException("Rally/World_Lit shader not found");

            var material = AssetDatabase.LoadAssetAtPath<Material>(ForestPath);
            bool created = material == null;
            if (created) material = new Material(shader);
            material.shader = shader;
            material.mainTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Art/Generated/World_Foliage.asset");
            material.SetFloat("_Wrap", 0.5f);
            if (created) AssetDatabase.CreateAsset(material, ForestPath);
            EditorUtility.SetDirty(material);

            var road = MaterialAt(RoadPath, "Rally/Road_Lit");
            road.SetTexture("_PackedTex", TextureAsset("World_RoadPacked", RoadTextureBuilder.BuildPacked));
            road.SetTexture("_LooseTex", TextureAsset("World_RoadLoose", RoadTextureBuilder.BuildLoose));
            EditorUtility.SetDirty(road);

            var ground = MaterialAt(GroundPath, "Rally/Ground_Lit");
            ground.SetTexture("_VergeTex", TextureAsset("World_Verge", RoadTextureBuilder.BuildVerge));
            ground.SetTexture("_FloorTex", TextureAsset("World_ForestFloor", RoadTextureBuilder.BuildForestFloor));
            ground.SetTexture("_LooseTex", road.GetTexture("_LooseTex"));
            ground.SetTexture("_FieldTex", road.GetTexture("_PackedTex"));
            EditorUtility.SetDirty(ground);

            EditorUtility.SetDirty(MaterialAt(SkyPath, "Rally/Sky"));
            EditorUtility.SetDirty(MaterialAt(GlarePath, "Hidden/Rally/SunGlare"));

            var dust = MaterialAt(DustPath, "Rally/Dust");
            dust.mainTexture = TextureAsset("World_DustPuff", BuildDustPuff);
            EditorUtility.SetDirty(dust);

            var plants = MaterialAt(VergeFoliagePath, "Rally/Foliage_Cutout");
            plants.mainTexture = TextureAsset("World_VergeFoliage", RoadTextureBuilder.BuildVergeFoliage);
            plants.SetFloat("_Cutoff", 0.45f);
            plants.SetFloat("_Wrap", 0.6f);
            EditorUtility.SetDirty(plants);

            var stones = MaterialAt(VergeStonesPath, "Rally/World_Lit");
            stones.mainTexture = AssetDatabase.LoadAssetAtPath<Texture2D>($"{TextureFolder}/World_Stone.asset");
            EditorUtility.SetDirty(stones);

            // Furniture on the cut-out shader for its Cull Off: tape and boards are seen
            // from both sides. The atlas is opaque, so nothing is actually cut.
            var furniture = MaterialAt(FurniturePath, "Rally/Foliage_Cutout");
            furniture.mainTexture = TextureAsset("World_Furniture", FurnitureTextureBuilder.Build);
            furniture.SetFloat("_Cutoff", 0.05f);
            furniture.SetFloat("_Wrap", 0.3f);
            EditorUtility.SetDirty(furniture);

            AssetDatabase.SaveAssets();
        }

        /// <summary>Regenerate every dressing texture, for after a texture builder changes.</summary>
        [MenuItem("Rally/Stage/Regenerate Dressing Textures")]
        public static void Regenerate()
        {
            AssetDatabase.DeleteAsset($"{TextureFolder}/World_RoadPacked.asset");
            AssetDatabase.DeleteAsset($"{TextureFolder}/World_RoadLoose.asset");
            AssetDatabase.DeleteAsset($"{TextureFolder}/World_Verge.asset");
            AssetDatabase.DeleteAsset($"{TextureFolder}/World_ForestFloor.asset");
            AssetDatabase.DeleteAsset($"{TextureFolder}/World_DustPuff.asset");
            AssetDatabase.DeleteAsset($"{TextureFolder}/World_VergeFoliage.asset");
            AssetDatabase.DeleteAsset($"{TextureFolder}/World_Furniture.asset");
            Ensure();
        }

        /// <summary>
        /// One dust puff: a soft round falloff broken up by noise, so the edge is billowy and
        /// no two overlapping puffs line up into a visible disc. White — the colour comes from
        /// the particle — with the shape all in alpha.
        /// </summary>
        static Texture2D BuildDustPuff()
        {
            const int size = 128;
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, true)
            {
                name = "World_DustPuff", wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Trilinear
            };
            var px = new Color32[size * size];
            for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                float u = (x + 0.5f) / size, v = (y + 0.5f) / size;
                float dx = u - 0.5f, dy = v - 0.5f;
                float r = Mathf.Sqrt(dx * dx + dy * dy) * 2f;
                float billow = WorldTextureBuilder.Fbm(u, v, 4, 4, 1601);
                float a = Mathf.Clamp01(1f - r - (billow - 0.5f) * 0.7f);
                a = a * a * (3f - 2f * a);
                a *= 0.65f + 0.7f * WorldTextureBuilder.Fbm(u, v, 9, 3, 1607);
                px[y * size + x] = new Color32(255, 255, 255, (byte)Mathf.Round(Mathf.Clamp01(a) * 255f));
            }
            tex.SetPixels32(px);
            tex.Apply(true, false);
            return tex;
        }

        static Material MaterialAt(string path, string shaderName)
        {
            var shader = Shader.Find(shaderName);
            if (shader == null) throw new FileNotFoundException($"{shaderName} shader not found");
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, path);
            }
            material.shader = shader;
            return material;
        }

        /// <summary>Load a generated texture, building and saving it first if it is not there yet.</summary>
        static Texture2D TextureAsset(string name, System.Func<Texture2D> build)
        {
            string path = $"{TextureFolder}/{name}.asset";
            var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            if (texture != null) return texture;
            texture = build();
            AssetDatabase.CreateAsset(texture, path);
            return texture;
        }
    }
}
