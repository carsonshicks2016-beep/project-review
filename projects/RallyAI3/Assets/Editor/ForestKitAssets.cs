using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Core.Environment;

namespace EditorScripts
{
    public static class ForestKitAssets
    {
        [MenuItem("Rally/Stage/Prepare Forest V3 Review Kit")]
        public static void Prepare()
        {
            const string imagePath = "Assets/Art/ForestV3/Branches-v3.png";
            AssetDatabase.Refresh();
            var importer = (TextureImporter)AssetImporter.GetAtPath(imagePath);
            if (importer == null) throw new FileNotFoundException(imagePath);
            importer.textureType = TextureImporterType.Default;
            importer.alphaSource = TextureImporterAlphaSource.FromInput;
            importer.alphaIsTransparency = true;
            importer.mipmapEnabled = true;
            importer.mipMapsPreserveCoverage = true;
            importer.alphaTestReferenceValue = .36f;
            importer.wrapMode = TextureWrapMode.Clamp;
            importer.filterMode = FilterMode.Trilinear;
            importer.anisoLevel = 4;
            importer.maxTextureSize = 2048;
            importer.textureCompression = TextureImporterCompression.CompressedHQ;
            importer.SaveAndReimport();
            var shader = Shader.Find("Rally/TreeFoliageV3");
            if (shader == null || ShaderUtil.ShaderHasError(shader)) throw new InvalidOperationException("Tree foliage shader failed");
            const string materialPath = "Assets/Resources/StageDressing/TreeFoliageV3.mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, materialPath);
            }
            material.shader = shader;
            material.mainTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(imagePath);
            material.SetFloat("_Cutoff", .36f);
            material.enableInstancing = true;
            EditorUtility.SetDirty(material);
            AssetDatabase.SaveAssets();
            Validate();
        }

        [Serializable] class Manifest { public string bundle, bundle_hash; }
        [Serializable] class Report
        {
            public string courseBundleHash, physicsHash;
            public int colliders, renderers, lodCells, treeMeshes, repeatReloads;
            public int materials, objects;
            public int[] speciesCounts;
            public bool noDecorativeColliders, rngUnchanged, repeatedPhysicsUnchanged, stableMeshCount;
        }
        public static void Validate()
        {
            EditorSceneManager.OpenScene("Assets/Scenes/RallyTraining.unity", OpenSceneMode.Single);
            string[] args = System.Environment.GetCommandLineArgs();
            int courseArg = Array.IndexOf(args, "-reviewCourse");
            string manifestPath = courseArg >= 0 && courseArg + 1 < args.Length ? args[courseArg + 1] :
                ".rally/courses/f187eaf562a3681aba54681939a54345c84d484556cef1b97baa2048acb5f8f4/manifest.json";
            var manifest = JsonUtility.FromJson<Manifest>(File.ReadAllText(manifestPath));
            var original = UnityEngine.Object.FindAnyObjectByType<TrackGenerator>();
            var bundle = AssetBundle.LoadFromFile(manifest.bundle);
            if (bundle == null) throw new InvalidDataException("Cannot load frozen review course");
            var track = UnityEngine.Object.Instantiate(bundle.LoadAllAssets<GameObject>()[0]).GetComponent<TrackGenerator>();
            UnityEngine.Object.DestroyImmediate(original.gameObject);
            bundle.Unload(false);
            string before = PhysicsHash(track);
            var rng = UnityEngine.Random.state;
            int firstMeshes = 0, lastMeshes = 0;
            int firstMaterials = 0, firstObjects = 0;
            int lastMaterials = 0, lastObjects = 0;
            for (int i = 0; i < 8; i++)
            {
                StageDressing.Dress(track);
                if (PhysicsHash(track) != before) throw new InvalidOperationException("Dressing changed collision geometry");
                lastMeshes = Resources.FindObjectsOfTypeAll<Mesh>().Count(m => !AssetDatabase.Contains(m));
                lastMaterials = Resources.FindObjectsOfTypeAll<Material>().Count(m => !AssetDatabase.Contains(m));
                lastObjects = track.GetComponentsInChildren<Transform>(true).Length;
                if (i == 0) { firstMeshes = lastMeshes; firstMaterials = lastMaterials; firstObjects = lastObjects; }
                else if (firstMaterials != lastMaterials || firstObjects != lastObjects)
                    throw new InvalidOperationException("Dressing material/object count grew across reloads");
            }
            bool unchangedRng = rng.Equals(UnityEngine.Random.state);
            Transform forest = track.transform.Find("Forest");
            if (forest == null || forest.GetComponentsInChildren<Collider>(true).Length != 0 || !unchangedRng || firstMeshes != lastMeshes)
                throw new InvalidOperationException($"Forest isolation/reload failed: RNG {unchangedRng}, meshes {firstMeshes}->{lastMeshes}");
            foreach (Transform item in forest.GetComponentsInChildren<Transform>())
                if (item.gameObject.layer != 2) throw new InvalidOperationException("Dressing escaped Ignore Raycast layer");
            int lods = forest.GetComponentsInChildren<LODGroup>().Length;
            if (lods < 1) throw new InvalidOperationException("Review kit has no LOD cells");
            if (StageDressing.LastReviewSpeciesCounts.Any(count => count < 1)) throw new InvalidOperationException("An archetype is missing from the review section");
            Directory.CreateDirectory(".rally/visual-review/v3");
            File.WriteAllText(".rally/visual-review/v3/forest-validation.json", JsonUtility.ToJson(new Report {
                courseBundleHash = manifest.bundle_hash, physicsHash = before,
                colliders = track.GetComponentsInChildren<Collider>().Length,
                renderers = forest.GetComponentsInChildren<Renderer>().Length,
                lodCells = lods, treeMeshes = lastMeshes, repeatReloads = 8,
                materials = lastMaterials, objects = lastObjects,
                speciesCounts = StageDressing.LastReviewSpeciesCounts,
                noDecorativeColliders = true, rngUnchanged = unchangedRng,
                repeatedPhysicsUnchanged = true, stableMeshCount = firstMeshes == lastMeshes }, true));
        }

        static string PhysicsHash(TrackGenerator track)
        {
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream))
            using (var hash = SHA256.Create())
            {
                foreach (Collider collider in track.GetComponentsInChildren<Collider>(true))
                {
                    writer.Write(collider.GetType().Name);
                    writer.Write(collider.isTrigger);
                    Matrix4x4 matrix = collider.transform.localToWorldMatrix;
                    for (int i = 0; i < 16; i++) writer.Write(matrix[i]);
                    if (collider is MeshCollider mesh && mesh.sharedMesh != null)
                    {
                        foreach (Vector3 p in mesh.sharedMesh.vertices) { writer.Write(p.x); writer.Write(p.y); writer.Write(p.z); }
                        foreach (int index in mesh.sharedMesh.triangles) writer.Write(index);
                    }
                }
                writer.Flush();
                return BitConverter.ToString(hash.ComputeHash(stream.ToArray())).Replace("-", "").ToLowerInvariant();
            }
        }
    }
}
