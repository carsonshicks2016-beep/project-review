using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using System.Globalization;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Build.Reporting;
using UnityEngine;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Core.Environment;

namespace EditorScripts
{
    [InitializeOnLoad]
    public static class LabBridge
    {
        [Serializable] class Request
        {
            public string id, action, output, input, family;
            public int seed;
            public int firstSector, sectorCount = 16;
            public float rocks;
            public float length;
            public bool hasScenery, scenery;
        }
        [Serializable] class Response { public string id, error; public bool ok; }
        [Serializable] class Definition
        {
            public int schema = 1, seed, spawnIndex, finishIndex, obstacleObjects;
            public float width, length, rocks;
            public bool scenery;
            public string family, resolvedHash;
            public Vector3[] gates;
            public string generatorVersion,terrainHash;
            public string topology, surface, circuitRevision, sourceHash, profileHash, confidence, startReference;
            public float circuitLength, startStation, endStation;
            public int episodeSeconds, firstSector, sectorCount;
            public Vector3[] mapPoints;
            public TrackGenerator.CircuitLandmark[] landmarks;
        }
        [Serializable] class SensorContract
        {
            public string name, tags;
            public float length, maxDegrees;
            public int stacks;
        }
        [Serializable] class Contract
        {
            public int schema = 1, vectors, stacks, continuous;
            public string behavior;
            public SensorContract[] sensors;
            public int[] observationWidths;
            public float timestep;
        }
        static string HashFile(string path){using(var sha=SHA256.Create())return BitConverter.ToString(sha.ComputeHash(File.ReadAllBytes(path))).Replace("-", "").ToLowerInvariant();}
        static readonly string Root = Path.GetFullPath(".");
        static readonly string Bridge = Path.Combine(Root, ".rally/bridge");
        static bool busy;
        static double next;
        static double heartbeat;
        static LabBridge() { EditorApplication.update += Poll; }
        static void Poll()
        {
            if (Application.isBatchMode || EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling || busy) return;
            if (EditorApplication.timeSinceStartup < next) return;
            next = EditorApplication.timeSinceStartup + 1;
            if (EditorApplication.timeSinceStartup >= heartbeat)
            {
                Directory.CreateDirectory(Bridge);
                File.WriteAllText(Path.Combine(Bridge, "editor.json"), "{\"time\":" + DateTimeOffset.UtcNow.ToUnixTimeSeconds() + "}");
                heartbeat = EditorApplication.timeSinceStartup + 5;
            }
            if (File.Exists(Path.Combine(Bridge, "request.json"))) Execute();
        }

        public static void Execute()
        {
            string requestPath = Path.Combine(Bridge, "request.json");
            if (!File.Exists(requestPath)) return;
            var request = JsonUtility.FromJson<Request>(File.ReadAllText(requestPath));
            File.Delete(requestPath);
            busy = true;
            var response = new Response { id = request.id };
            try
            {
                if (request.action == "build") Build(request);
                else if (request.action == "course") Course(request);
                else if (request.action == "prepare") Prepare(request);
                else if (request.action == "refresh-visuals") RefreshVisuals();
                else throw new InvalidOperationException("Unknown Editor request");
                response.ok = true;
            }
            catch (Exception ex) { response.error = ex.ToString(); Debug.LogException(ex); }
            finally
            {
                File.WriteAllText(Path.Combine(Bridge, "response.json"), JsonUtility.ToJson(response));
                busy = false;
            }
        }

        static void Build(Request request)
        {
            RallyAudioAssets.Prepare();
            var scene = EditorSceneManager.OpenScene("Assets/Scenes/RallyTraining.unity", OpenSceneMode.Additive);
            try
            {
                var behavior = scene.GetRootGameObjects().SelectMany(g => g.GetComponentsInChildren<BehaviorParameters>()).First();
                var contract = new Contract {
                    vectors = behavior.BrainParameters.VectorObservationSize,
                    stacks = behavior.BrainParameters.NumStackedVectorObservations,
                    continuous = behavior.BrainParameters.ActionSpec.NumContinuousActions,
                    behavior = behavior.BehaviorName, timestep = Time.fixedDeltaTime,
                    sensors = behavior.GetComponentsInChildren<RayPerceptionSensorComponent3D>()
                        .Select(s => new SensorContract { name = s.SensorName, length = s.RayLength,
                            maxDegrees = s.MaxRayDegrees, stacks = s.ObservationStacks,
                            tags = string.Join(",", s.DetectableTags) }).ToArray()
                };
                contract.observationWidths = new[] { contract.vectors * contract.stacks }.Concat(
                    contract.sensors.Select(s => (int)(2 * behavior.GetComponentsInChildren<RayPerceptionSensorComponent3D>()
                        .First(sensor => sensor.SensorName == s.name).RaysPerDirection + 1) * (s.tags.Split(',').Length + 2) * s.stacks)).ToArray();
                File.WriteAllText(Path.Combine(Root, ".rally/contract.json"), JsonUtility.ToJson(contract, true));
            }
            finally { EditorSceneManager.CloseScene(scene, true); }
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
                scenes = new[] { "Assets/Scenes/RallyTraining.unity" }, locationPathName = request.output,
                target = BuildTarget.StandaloneOSX, options = BuildOptions.None
            });
            if (report.summary.result != BuildResult.Succeeded) throw new Exception("Player build failed");
        }

        static void RefreshVisuals()
        {
            var scene = EditorSceneManager.OpenScene("Assets/Scenes/RallyTraining.unity", OpenSceneMode.Additive);
            try
            {
                var car = scene.GetRootGameObjects()
                    .SelectMany(root => root.GetComponentsInChildren<Core.Physics.VehicleController>(true))
                    .FirstOrDefault();
                if (car == null) throw new Exception("No vehicle found in the training scene");
                Selection.activeGameObject = car.gameObject;
                RallyCarBuilder.RebuildVisuals();
                AssetDatabase.SaveAssets();
            }
            finally
            {
                Selection.activeGameObject = null;
                EditorSceneManager.CloseScene(scene, true);
            }
        }

        static string AssetDirectory(string id)
        {
            string directory = "Assets/RallyLabGenerated/" + id;
            Directory.CreateDirectory(directory);
            AssetDatabase.Refresh();
            return directory;
        }

        static void Prepare(Request request)
        {
            string directory = AssetDirectory(request.id);
            string asset = directory + "/policy.onnx";
            File.Copy(request.input, asset, true);
            AssetDatabase.ImportAsset(asset, ImportAssetOptions.ForceSynchronousImport);
            if (AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(asset) == null)
                throw new Exception("Checkpoint did not import as a model");
            Bundle(request.output, "policy", asset);
        }

        static void Bundle(string output, string name, string asset)
        {
            Directory.CreateDirectory(output);
            var result = BuildPipeline.BuildAssetBundles(output,
                new[] { new AssetBundleBuild { assetBundleName = name, assetNames = new[] { asset } } },
                BuildAssetBundleOptions.ChunkBasedCompression, BuildTarget.StandaloneOSX);
            if (result == null) throw new Exception("Asset bundle build failed");
        }

        static void Course(Request request)
        {
            string directory = AssetDirectory(request.id);
            var scene = EditorSceneManager.OpenScene("Assets/Scenes/RallyTraining.unity", OpenSceneMode.Additive);
            GameObject copy = null;
            try
            {
                var original = scene.GetRootGameObjects().SelectMany(g => g.GetComponentsInChildren<TrackGenerator>()).First();
                copy = UnityEngine.Object.Instantiate(original.gameObject);
                copy.name = "Course";
                var track = copy.GetComponent<TrackGenerator>();
                track.frozenCourse = false; track.randomiseSeed = false;
                track.seed = request.seed;
                track.obstaclesPer100m = request.rocks;
                track.spawnObstacles = request.rocks > 0f;
                if (request.hasScenery) track.scatterProps = request.scenery;
                if (request.family == "nordschleife")
                {
                    string pinned = Path.Combine(Root,"Assets/Resources/Circuits/Nordschleife.json");
                    var data=JsonUtility.FromJson<TrackGenerator.CircuitData>(File.ReadAllText(pinned));
                    track.InstallCircuit(data,request.firstSector,request.sectorCount);
                }
                else if (request.family == "gentle") { track.maxHeadingDeviation = 35; track.elevationScale = 6; }
                else if (request.family == "technical") { track.maxHeadingDeviation = 70; track.shapeFrequency = .007f; }
                else if (request.family == "crests") { track.elevationScale = 18; track.elevationFrequency = .007f; }
                else throw new Exception("Unknown course family");
                if (request.length > 0 && !track.importedCircuit)
                {
                    if (request.length < 1000 || request.length > 5000)
                        throw new Exception("Course length must be 1000-5000 metres");
                    track.trackLength = request.length;
                    track.controlPointsCount = Mathf.CeilToInt(request.length / 50f);
                    track.longStageRhythm = request.length > 1000;
                    if (track.longStageRhythm)
                    {
                        if (request.family == "technical") track.elevationScale = 18;
                        track.elevationFrequency = .005f;
                    }
                }
                track.GenerateTrack();
                if (track.waypoints.Count < 4) throw new Exception("Invalid course geometry");
                track.frozenCourse = true;
                AssetDatabase.StartAssetEditing();
                try
                {
                var meshes = new Dictionary<Mesh, Mesh>();
                int count = 0;
                foreach (var filter in copy.GetComponentsInChildren<MeshFilter>())
                {
                    if (filter.sharedMesh == null) continue;
                    var source = filter.sharedMesh;
                    if (!meshes.TryGetValue(source, out var saved))
                    {
                        saved = UnityEngine.Object.Instantiate(source);
                        AssetDatabase.CreateAsset(saved, directory + "/mesh-" + count++ + ".asset");
                        meshes[source] = saved;
                    }
                    filter.sharedMesh = saved;
                }
                foreach (var collider in copy.GetComponentsInChildren<MeshCollider>())
                {
                    if (collider.sharedMesh != null && meshes.TryGetValue(collider.sharedMesh, out var saved)) collider.sharedMesh = saved;
                }
                var materials = new Dictionary<Material, Material>();
                foreach (var renderer in copy.GetComponentsInChildren<Renderer>())
                {
                    var values = renderer.sharedMaterials;
                    for (int i = 0; i < values.Length; i++)
                    {
                        if (values[i] == null || AssetDatabase.Contains(values[i])) continue;
                        if (!materials.TryGetValue(values[i], out var saved))
                        {
                            saved = new Material(values[i]);
                            if(saved.mainTexture!=null&&!AssetDatabase.Contains(saved.mainTexture))
                            {
                                var texture=UnityEngine.Object.Instantiate(saved.mainTexture);
                                AssetDatabase.CreateAsset(texture,directory+"/texture-"+count+++".asset");
                                saved.mainTexture=texture;
                            }
                            AssetDatabase.CreateAsset(saved, directory + "/material-" + count++ + ".mat");
                            materials[values[i]] = saved;
                        }
                        values[i] = saved;
                    }
                    renderer.sharedMaterials = values;
                }
                }
                finally { AssetDatabase.StopAssetEditing(); }
                string prefab = directory + "/course.prefab";
                PrefabUtility.SaveAsPrefabAsset(copy, prefab);
                AssetDatabase.SaveAssets();
                Bundle(request.output, "course", prefab);
                int obstacleObjects = copy.GetComponentsInChildren<Collider>(true)
                    .Where(c => c.CompareTag("Obstacle") || c.name == "RoadRock" || c.name == "Tree" || c.name == "Boulder")
                    .Select(c => c.gameObject.GetInstanceID()).Distinct().Count();
                var definition = new Definition { seed = request.seed, family = request.family, rocks = request.rocks,
                    scenery = track.scatterProps, obstacleObjects = obstacleObjects,
                    width = track.roadWidth, length = track.trackLength, gates = track.waypoints.ToArray(),
                    spawnIndex = track.spawnWaypointIndex, finishIndex = track.FinishWaypointIndex,
                    resolvedHash = GeometryHash(copy) };
                if(track.importedCircuit)
                {
                    var data=JsonUtility.FromJson<TrackGenerator.CircuitData>(File.ReadAllText(Path.Combine(Root,"Assets/Resources/Circuits/Nordschleife.json")));
                    definition.schema=2;definition.topology="circuit";definition.surface="tarmac";
                    definition.circuitRevision=track.circuitRevision;definition.sourceHash=track.circuitSourceHash;
                    definition.terrainHash=HashFile(Path.Combine(Root,"Assets/Resources/Circuits/NordschleifeTerrain.json"));definition.generatorVersion=track.circuitGeneratorVersion;definition.profileHash=track.circuitProfileHash;definition.confidence=data.confidence;
                    definition.startReference=data.startReference;definition.circuitLength=track.circuitLength;
                    definition.startStation=track.circuitStart;definition.endStation=track.circuitEnd;
                    definition.firstSector=request.firstSector;definition.sectorCount=request.sectorCount;
                    definition.episodeSeconds=track.circuitEpisodeSeconds;definition.landmarks=data.landmarks;
                    definition.mapPoints=data.points.Where((p,i)=>i%12==0).Concat(new[]{data.points[0]}).ToArray();
                }
                File.WriteAllText(Path.Combine(request.output, "definition.json"), JsonUtility.ToJson(definition, true));
            }
            finally
            {
                if (copy != null) UnityEngine.Object.DestroyImmediate(copy);
                EditorSceneManager.CloseScene(scene, true);
            }
        }

        static string GeometryHash(GameObject root)
        {
            var canonical = new StringBuilder();
            foreach (var filter in root.GetComponentsInChildren<MeshFilter>(true))
            {
                canonical.Append("mesh|").Append(PathOf(root.transform, filter.transform)).Append('|');
                var mesh = filter.sharedMesh;
                if(mesh==null) continue;
                foreach (var vertex in mesh.vertices)
                    canonical.Append(vertex.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                        .Append(vertex.y.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                        .Append(vertex.z.ToString("R", CultureInfo.InvariantCulture)).Append(';');
                foreach (var index in mesh.triangles) canonical.Append(index).Append(',');
            }
            foreach (var collider in root.GetComponentsInChildren<Collider>(true))
            {
                var bounds = collider.bounds;
                canonical.Append("collider|").Append(PathOf(root.transform, collider.transform)).Append('|')
                    .Append(collider.GetType().FullName).Append('|')
                    .Append(bounds.center.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(bounds.center.y.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(bounds.center.z.ToString("R", CultureInfo.InvariantCulture)).Append('|')
                    .Append(bounds.size.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(bounds.size.y.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(bounds.size.z.ToString("R", CultureInfo.InvariantCulture)).Append(';');
            }
            foreach (var item in root.GetComponentsInChildren<Transform>(true))
                canonical.Append("object|").Append(PathOf(root.transform, item)).Append('|')
                    .Append(item.localPosition.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(item.localPosition.y.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(item.localPosition.z.ToString("R", CultureInfo.InvariantCulture)).Append(';');
            using (var sha = SHA256.Create())
                return BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(canonical.ToString())))
                    .Replace("-", "").ToLowerInvariant();
        }

        static string PathOf(Transform root, Transform value)
        {
            var parts = new Stack<string>();
            for (var item = value; item != null; item = item.parent)
            {
                parts.Push(item.name);
                if (item == root) break;
            }
            return string.Join("/", parts);
        }
    }
}
