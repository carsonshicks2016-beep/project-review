using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using UnityEngine.Rendering;
using Core.Environment;
using Core.ML;
using Core.Physics;
using UI;

namespace Core.Presentation
{
    [DefaultExecutionOrder(10000)]
    public sealed class RallyVisualRenovation : MonoBehaviour
    {
        const int TextureSize = 512;
        static RallyVisualRenovation instance;

        TrackGenerator activeTrack;
        int activeSeed = int.MinValue;
        VehicleController activeVehicle;
        Material roadMaterial, terrainMaterial, forestMaterial, barkMaterial, rockMaterial;
        Material[] visualCanopyMaterials;
        Material[] horizonMaterials;
        Material tireMarkMaterial, postMaterial, postCapMaterial, dustMaterial;
        Material skyMaterial;
        Material gateFrameMaterial, gateBoardMaterial, gateStripeMaterial;
        Material marshalVestMaterial, marshalClothMaterial, marshalSkinMaterial, marshalFlagMaterial;
        Material plateFrameMaterial, plateMaterial, exhaustMaterial, exhaustInteriorMaterial;
        Font gateFont;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            if (!Application.isPlaying || Application.isBatchMode || instance != null) return;
            if (LabRuntime.Enabled && !LabRuntime.Config.viewer) return;
            var root = new GameObject("Rally Presentation");
            instance = root.AddComponent<RallyVisualRenovation>();
            DontDestroyOnLoad(root);
        }

        void OnEnable() => SceneManager.sceneLoaded += OnSceneLoaded;
        void OnDisable() => SceneManager.sceneLoaded -= OnSceneLoaded;
        void OnSceneLoaded(Scene scene, LoadSceneMode mode) => StartCoroutine(ApplyWhenReady());

        IEnumerator Start()
        {
            if (LabRuntime.Enabled && LabRuntime.Config.viewer)
            {
                // Desktop VSync overrides targetFrameRate; enforce the viewer-only cap.
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = 60;
            }
            yield return ApplyWhenReady();
            while (true)
            {
                yield return new WaitForSecondsRealtime(0.5f);
                ApplyIfChanged();
            }
        }

        IEnumerator ApplyWhenReady()
        {
            for (int i = 0; i < 40; i++)
            {
                if (ApplyIfChanged()) yield break;
                yield return null;
            }
        }

        bool ApplyIfChanged()
        {
            TrackGenerator track = FindObjectOfType<TrackGenerator>();
            VehicleController vehicle = FindObjectOfType<VehicleController>();
            if (track == null || vehicle == null || track.waypoints == null || track.waypoints.Count < 4)
                return false;

            if (track != activeTrack || track.CurrentSeed != activeSeed)
            {
                ApplyWorld(track);
                activeTrack = track;
                activeSeed = track.CurrentSeed;
            }

            if (vehicle != activeVehicle)
            {
                ApplyCar(vehicle);
                activeVehicle = vehicle;
            }

            ApplyCamera(vehicle, track);
            return true;
        }

        void ApplyWorld(TrackGenerator track)
        {
            CreateMaterials();
            ApplyLighting();
            if (track.importedCircuit)
            {
                StageAtmosphere.Apply(track);
                ReplaceTelemetry(track);
                return;
            }
            track.RefreshRoadRockVisuals(rockMaterial);

            MeshRenderer road = track.GetComponent<MeshRenderer>();
            if (road != null) road.sharedMaterial = roadMaterial;

            if (track.terrain != null)
            {
                var renderer = track.terrain.GetComponent<MeshRenderer>();
                if (renderer != null)
                {
                    var blendedGround = Resources.Load<Material>("StageDressing/Ground");
                    if (blendedGround != null)
                    {
                        blendedGround.SetFloat("_UVMetres", track.terrainTileMetres);
                        blendedGround.SetFloat("_RoadHalfWidth", track.roadWidth * .5f);
                        blendedGround.SetFloat("_Treeline", track.propClearance);
                    }
                    renderer.sharedMaterial = blendedGround != null ? blendedGround : terrainMaterial;
                }
            }

            BuildHorizon(track);

            if (track.props != null)
            {
                foreach (MeshRenderer renderer in track.props.GetComponentsInChildren<MeshRenderer>())
                {
                    if (renderer.gameObject.name == "Trees") renderer.sharedMaterial = forestMaterial;
                    else if (renderer.gameObject.name == "Boulders") renderer.sharedMaterial = rockMaterial;
                    renderer.shadowCastingMode = ShadowCastingMode.Off;
                    renderer.receiveShadows = false;
                }
            }

            DecorateCourse(track);
            ReplaceTelemetry(track);
        }

        void CreateMaterials()
        {
            if (roadMaterial != null) return;
            roadMaterial = MakeMaterial("Rally Gravel", BuildGravelTexture(), Color.white, 0.08f);
            terrainMaterial = MakeMaterial("Rally Forest Floor", BuildGroundTexture(), Color.white, 0.04f);
            SetNormalMap(roadMaterial, BuildSurfaceNormal("Rally Gravel Relief", 1129, 9, 0.34f), 0.38f);
            SetNormalMap(terrainMaterial, BuildSurfaceNormal("Rally Ground Relief", 1231, 5, 0.24f), 0.24f);
            forestMaterial = MakeMaterial("Rally Needles", BuildFoliageTexture(), Color.white, 0.04f);
            visualCanopyMaterials = new[]
            {
                MakeMaterial("Rally Pine Canopy Dark", null, new Color(0.12f, 0.21f, 0.12f), 0.02f),
                MakeMaterial("Rally Pine Canopy", null, new Color(0.17f, 0.28f, 0.15f), 0.02f),
                MakeMaterial("Rally Pine Canopy Sunlit", null, new Color(0.22f, 0.32f, 0.17f), 0.02f)
            };
            barkMaterial = MakeMaterial("Rally Bark", null, new Color(0.34f, 0.23f, 0.15f), 0.04f);
            rockMaterial = MakeMaterial("Rally Stone", BuildStoneTexture(), Color.white, 0.03f);
            Texture2D ridgeTexture = BuildStoneTexture();
            horizonMaterials = new[]
            {
                MakeMaterial("Rally Distant Ridge", ridgeTexture, new Color(0.63f, 0.68f, 0.58f), 0.02f),
                MakeMaterial("Rally Middle Ridge", ridgeTexture, new Color(0.70f, 0.76f, 0.79f), 0.02f),
                MakeMaterial("Rally Far Ridge", ridgeTexture, new Color(0.79f, 0.85f, 0.88f), 0.02f)
            };
            tireMarkMaterial = MakeTrailMaterial();
            postMaterial = MakeMaterial("Rally Marker", null, new Color(0.86f, 0.84f, 0.75f), 0.05f);
            postCapMaterial = MakeMaterial("Rally Marker Red", null, new Color(0.72f, 0.10f, 0.055f), 0.05f);
            // Alpha blending is essential here: Standard's opaque mode makes overlapping dust glow like solid billboards.
            dustMaterial = MakeTrailMaterial();
            dustMaterial.name = "Rally Dust";
            dustMaterial.mainTexture = BuildDustTexture();
            gateFrameMaterial = MakeMaterial("Rally Start Gate Blue", null, new Color(0.08f, 0.20f, 0.32f), 0.12f);
            gateBoardMaterial = MakeMaterial("Rally Start Gate Board", null, new Color(0.045f, 0.10f, 0.16f), 0.04f);
            gateStripeMaterial = MakeMaterial("Rally Start Gate Gold", null, new Color(0.97f, 0.70f, 0.12f), 0.10f);
            marshalVestMaterial = MakeMaterial("Marshal High Visibility", null, new Color(0.95f, 0.66f, 0.10f), 0.04f);
            marshalClothMaterial = MakeMaterial("Marshal Trousers", null, new Color(0.12f, 0.16f, 0.19f), 0.02f);
            marshalSkinMaterial = MakeMaterial("Marshal Skin", null, new Color(0.68f, 0.47f, 0.32f), 0.02f);
            marshalFlagMaterial = MakeMaterial("Marshal Green Flag", null, new Color(0.15f, 0.72f, 0.31f), 0.02f);
            plateFrameMaterial = MakeMaterial("Rally Plate Frame", null, new Color(0.04f, 0.045f, 0.04f), 0.08f);
            plateMaterial = MakeMaterial("Rally 555 Plate", null, new Color(0.98f, 0.78f, 0.12f), 0.08f);
            exhaustMaterial = MakeMaterial("Rally Exhaust Metal", null, new Color(0.48f, 0.50f, 0.48f), 0.48f);
            exhaustInteriorMaterial = MakeMaterial("Rally Exhaust Interior", null, new Color(0.035f, 0.04f, 0.04f), 0.02f);
        }

        static Material MakeMaterial(string name, Texture2D texture, Color tint, float smoothness)
        {
            Shader shader = Shader.Find("Standard");
            var material = new Material(shader != null ? shader : Shader.Find("Diffuse")) { name = name };
            if (material.HasProperty("_MainTex")) material.SetTexture("_MainTex", texture);
            if (material.HasProperty("_Color")) material.SetColor("_Color", tint);
            if (material.HasProperty("_Glossiness")) material.SetFloat("_Glossiness", smoothness);
            if (material.HasProperty("_Metallic")) material.SetFloat("_Metallic", 0f);
            return material;
        }

        static Material MakeTrailMaterial()
        {
            Shader shader = Shader.Find("Particles/Standard Unlit");
            if (shader == null) shader = Shader.Find("Sprites/Default");
            var material = new Material(shader) { name = "Driven Gravel Tracks" };
            if (material.HasProperty("_Color")) material.SetColor("_Color", Color.white);
            if (material.HasProperty("_Mode")) material.SetFloat("_Mode", 2f);
            if (material.HasProperty("_SrcBlend")) material.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
            if (material.HasProperty("_DstBlend")) material.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
            if (material.HasProperty("_ZWrite")) material.SetInt("_ZWrite", 0);
            material.SetOverrideTag("RenderType", "Transparent");
            material.EnableKeyword("_ALPHABLEND_ON");
            material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            material.renderQueue = 3000;
            return material;
        }

        static Texture2D BuildGravelTexture()
        {
            var texture = NewTile("Rally Gravel Albedo", TextureSize);
            var pixels = new Color32[TextureSize * TextureSize];
            for (int y = 0; y < TextureSize; y++)
            for (int x = 0; x < TextureSize; x++)
            {
                float u = x / (float)TextureSize, v = y / (float)TextureSize;
                float broad = PeriodicFbm(u, v, 5, 3, 101);
                float fine = PeriodicFbm(u, v, 7, 4, 211);
                float grit = Hash01(x, y, 307);
                Color c = Color.Lerp(new Color(0.28f, 0.25f, 0.21f),
                                     new Color(0.58f, 0.52f, 0.42f), Mathf.SmoothStep(0.20f, 0.84f, broad));
                c *= 0.78f + fine * 0.33f;

                // UV.x is road-relative width and repeats every six metres. These
                // shallow, broken lanes sit at the rally car's wheel offsets (about
                // 0.12 tile from centre) and give the road a used-in racing line.
                float across = u - Mathf.Floor(u);
                float laneDistance = Mathf.Min(Mathf.Abs(across - 0.12f), Mathf.Abs(across - 0.88f));
                float laneNoise = PeriodicNoise(u, v, 32, 1093);
                float compacted = (1f - Mathf.SmoothStep(0.014f, 0.065f, laneDistance)) *
                                  Mathf.SmoothStep(0.24f, 0.65f, laneNoise);
                c = Color.Lerp(c, c * new Color(0.79f, 0.78f, 0.75f), compacted * 0.34f);
                if (grit > 0.978f)
                    c = Color.Lerp(c, grit > 0.993f ? new Color(0.72f, 0.68f, 0.59f) : new Color(0.16f, 0.15f, 0.13f), 0.75f);
                pixels[y * TextureSize + x] = c;
            }
            texture.SetPixels32(pixels);
            texture.Apply(true, true);
            return texture;
        }

        static Texture2D BuildGroundTexture()
        {
            var texture = NewTile("Rally Forest Floor Albedo", TextureSize);
            var pixels = new Color32[TextureSize * TextureSize];
            for (int y = 0; y < TextureSize; y++)
            for (int x = 0; x < TextureSize; x++)
            {
                float u = x / (float)TextureSize, v = y / (float)TextureSize;
                float patch = PeriodicFbm(u, v, 4, 4, 401);
                float blades = PeriodicFbm(u, v, 8, 3, 503);
                float leaf = Hash01(x, y, 607);
                Color grass = Color.Lerp(new Color(0.105f, 0.17f, 0.085f),
                                         new Color(0.30f, 0.37f, 0.19f), Mathf.SmoothStep(0.24f, 0.82f, patch));
                grass *= 0.78f + blades * 0.34f;
                if (leaf > 0.988f) grass = Color.Lerp(grass, new Color(0.39f, 0.29f, 0.17f), 0.68f);
                pixels[y * TextureSize + x] = grass;
            }
            texture.SetPixels32(pixels);
            texture.Apply(true, true);
            return texture;
        }

        static Texture2D BuildFoliageTexture()
        {
            const int size = 256;
            var texture = NewTile("Rally Foliage Atlas", size);
            var pixels = new Color32[size * size];
            for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                float u = x / (float)size, v = y / (float)size;
                Color c;
                if (u < 0.5f)
                {
                    float fiber = Mathf.PerlinNoise(u * 180f + 4f, v * 20f + 9f);
                    float noise = Hash01(x, y, 811);
                    c = Color.Lerp(new Color(0.18f, 0.12f, 0.085f), new Color(0.40f, 0.29f, 0.19f), fiber * 0.68f + noise * 0.32f);
                }
                else
                {
                    float clump = Mathf.PerlinNoise(u * 31f + 17f, v * 34f + 29f);
                    float fleck = Hash01(x, y, 907);
                    c = Color.Lerp(new Color(0.055f, 0.13f, 0.065f), new Color(0.23f, 0.36f, 0.16f), clump * 0.75f + fleck * 0.25f);
                }
                pixels[y * size + x] = c;
            }
            texture.SetPixels32(pixels);
            texture.Apply(true, true);
            return texture;
        }

        static Texture2D BuildStoneTexture()
        {
            const int size = 256;
            var texture = NewTile("Rally Stone Albedo", size);
            var pixels = new Color32[size * size];
            for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                float n = PeriodicFbm(x / (float)size, y / (float)size, 4, 4, 1009);
                Color c = Color.Lerp(new Color(0.22f, 0.23f, 0.22f), new Color(0.53f, 0.52f, 0.48f), n);
                pixels[y * size + x] = c;
            }
            texture.SetPixels32(pixels);
            texture.Apply(true, true);
            return texture;
        }

        static Texture2D BuildDustTexture()
        {
            const int size = 64;
            var texture = new Texture2D(size, size, TextureFormat.RGBA32, true, true) { name = "Rally Dust Soft Particle" };
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.filterMode = FilterMode.Bilinear;
            var pixels = new Color32[size * size];
            for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                float dx = (x + 0.5f) / size * 2f - 1f;
                float dy = (y + 0.5f) / size * 2f - 1f;
                float alpha = Mathf.Pow(Mathf.Clamp01(1f - dx * dx - dy * dy), 2.2f);
                pixels[y * size + x] = new Color(1f, 1f, 1f, alpha);
            }
            texture.SetPixels32(pixels);
            texture.Apply(true, true);
            return texture;
        }

        static Texture2D BuildSurfaceNormal(string name, int seed, int baseGrid, float strength)
        {
            var texture = new Texture2D(TextureSize, TextureSize, TextureFormat.RGBA32, true, true)
            {
                name = name,
                wrapMode = TextureWrapMode.Repeat,
                filterMode = FilterMode.Trilinear,
                anisoLevel = 8
            };
            var pixels = new Color32[TextureSize * TextureSize];
            float step = 1f / TextureSize;
            for (int y = 0; y < TextureSize; y++)
            for (int x = 0; x < TextureSize; x++)
            {
                float u = (x + 0.5f) * step, v = (y + 0.5f) * step;
                float left = PeriodicFbm(u - step, v, seed, 4, baseGrid);
                float right = PeriodicFbm(u + step, v, seed, 4, baseGrid);
                float down = PeriodicFbm(u, v - step, seed, 4, baseGrid);
                float up = PeriodicFbm(u, v + step, seed, 4, baseGrid);
                var normal = new Vector3((left - right) * strength, (down - up) * strength, 1f).normalized;
                pixels[y * TextureSize + x] = new Color(normal.x * 0.5f + 0.5f,
                    normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f, 1f);
            }
            texture.SetPixels32(pixels);
            texture.Apply(true, true);
            return texture;
        }

        static void SetNormalMap(Material material, Texture2D normal, float scale)
        {
            material.SetTexture("_BumpMap", normal);
            material.SetFloat("_BumpScale", scale);
            material.EnableKeyword("_NORMALMAP");
        }

        static Texture2D NewTile(string name, int size)
        {
            return new Texture2D(size, size, TextureFormat.RGBA32, true, true)
            {
                name = name,
                wrapMode = TextureWrapMode.Repeat,
                filterMode = FilterMode.Trilinear,
                anisoLevel = 8
            };
        }

        static float PeriodicFbm(float u, float v, int seed, int octaves, int baseGrid)
        {
            float sum = 0f, weight = 0f, amp = 1f;
            for (int i = 0; i < octaves; i++)
            {
                int grid = baseGrid << i;
                sum += PeriodicNoise(u, v, grid, seed + i * 17) * amp;
                weight += amp;
                amp *= 0.52f;
            }
            return sum / weight;
        }

        static float PeriodicNoise(float u, float v, int grid, int seed)
        {
            float x = u * grid, y = v * grid;
            int ix = Mathf.FloorToInt(x), iy = Mathf.FloorToInt(y);
            float fx = Smooth01(x - ix), fy = Smooth01(y - iy);
            float a = Hash01(Mod(ix, grid), Mod(iy, grid), seed);
            float b = Hash01(Mod(ix + 1, grid), Mod(iy, grid), seed);
            float c = Hash01(Mod(ix, grid), Mod(iy + 1, grid), seed);
            float d = Hash01(Mod(ix + 1, grid), Mod(iy + 1, grid), seed);
            return Mathf.Lerp(Mathf.Lerp(a, b, fx), Mathf.Lerp(c, d, fx), fy);
        }

        static int Mod(int value, int divisor) => (value % divisor + divisor) % divisor;
        static float Smooth01(float value) => value * value * (3f - 2f * value);

        static float Hash01(int x, int y, int seed)
        {
            unchecked
            {
                uint h = (uint)(x * 374761393 + y * 668265263 + seed * 1442695041);
                h = (h ^ (h >> 13)) * 1274126177u;
                h ^= h >> 16;
                return (h & 0x00ffffffu) / 16777215f;
            }
        }

        void ApplyLighting()
        {
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.52f, 0.61f, 0.72f);
            RenderSettings.ambientEquatorColor = new Color(0.48f, 0.48f, 0.42f);
            RenderSettings.ambientGroundColor = new Color(0.24f, 0.22f, 0.18f);
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.Linear;
            RenderSettings.fogStartDistance = 170f;
            RenderSettings.fogEndDistance = 920f;
            RenderSettings.fogColor = new Color(0.59f, 0.68f, 0.72f);

            Shader skyShader = Shader.Find("Skybox/Procedural");
            if (skyMaterial == null && skyShader != null)
            {
                skyMaterial = new Material(skyShader) { name = "Rally Clear Mountain Sky" };
                skyMaterial.SetFloat("_AtmosphereThickness", 1.0f);
                skyMaterial.SetFloat("_Exposure", 1.08f);
                skyMaterial.SetColor("_SkyTint", new Color(0.56f, 0.68f, 0.80f));
                skyMaterial.SetColor("_GroundColor", new Color(0.40f, 0.42f, 0.38f));
                skyMaterial.SetFloat("_SunSize", 0.025f);
            }
            if (skyMaterial != null) RenderSettings.skybox = skyMaterial;

            foreach (Light light in FindObjectsOfType<Light>())
            {
                if (light.type != LightType.Directional) continue;
                light.color = new Color(1f, 0.96f, 0.88f);
                light.intensity = 1.18f;
                light.transform.rotation = Quaternion.Euler(42f, -32f, 0f);
                light.shadows = LightShadows.Soft;
                light.shadowStrength = 0.58f;
                light.shadowBias = 0.045f;
                light.shadowNormalBias = 0.32f;
            }

            QualitySettings.antiAliasing = 4;
        }

        void DecorateCourse(TrackGenerator track)
        {
            Transform previous = track.transform.Find("RallyVisualDetails");
            if (previous != null) DestroyVisualBranch(previous.gameObject);
            var root = new GameObject("RallyVisualDetails").transform;
            root.SetParent(track.transform, false);

            BuildStartGate(track, root);
            BuildForest(track, root);
            BuildRoadsideMarkers(track, root);
        }

        void BuildHorizon(TrackGenerator track)
        {
            if (track.backdrop == null) return;
            MeshRenderer legacyWall = track.backdrop.GetComponent<MeshRenderer>();
            if (legacyWall != null) legacyWall.enabled = false;
            Transform previous = track.backdrop.transform.Find("RallyMountainLayers");
            if (previous != null) DestroyVisualBranch(previous.gameObject);

            var root = new GameObject("RallyMountainLayers").transform;
            root.SetParent(track.backdrop.transform, false);
            BuildRidge(root, "Near Mountain Ridge", 520f, 20f, 96f, 2.8f, 1.1f, horizonMaterials[0]);
            BuildRidge(root, "Middle Mountain Ridge", 760f, 42f, 142f, 2.5f, 1.25f, horizonMaterials[1]);
            BuildRidge(root, "Far Mountain Ridge", 1040f, 72f, 198f, 2.2f, 1.35f, horizonMaterials[2]);
        }

        static void DestroyVisualBranch(GameObject branch)
        {
            foreach (MeshFilter filter in branch.GetComponentsInChildren<MeshFilter>(true))
            {
                Mesh mesh = filter.sharedMesh;
                if (mesh != null) Destroy(mesh);
                filter.sharedMesh = null;
            }
            Destroy(branch);
        }

        void BuildStartGate(TrackGenerator track, Transform parent)
        {
            if (track.waypoints == null || track.waypoints.Count <= track.spawnWaypointIndex + 1) return;
            if (!TryCoursePoint(track, 26f, out Vector3 position, out Vector3 forward)) return;

            var gate = new GameObject("Rally Stage Start Gate").transform;
            gate.SetParent(parent, false);
            gate.localPosition = position;
            forward.y = 0f;
            gate.localRotation = Quaternion.LookRotation(forward.normalized, Vector3.up);

            float halfSpan = track.roadWidth * 0.5f + 1.35f;
            GateBox(gate, "Left Gate Leg", new Vector3(-halfSpan, 2.20f, 0f), new Vector3(0.48f, 4.40f, 0.62f), gateFrameMaterial);
            GateBox(gate, "Right Gate Leg", new Vector3(halfSpan, 2.20f, 0f), new Vector3(0.48f, 4.40f, 0.62f), gateFrameMaterial);
            GateBox(gate, "Gate Header", new Vector3(0f, 4.40f, 0f), new Vector3(halfSpan * 2f + 0.48f, 0.44f, 0.68f), gateFrameMaterial);
            GateBox(gate, "Start Board", new Vector3(0f, 3.72f, 0.02f), new Vector3(7.4f, 0.92f, 0.22f), gateBoardMaterial);
            GateBox(gate, "Board Gold Edge", new Vector3(0f, 4.21f, -0.105f), new Vector3(7.4f, 0.045f, 0.035f), gateStripeMaterial);
            GateBox(gate, "Board Lower Edge", new Vector3(0f, 3.23f, -0.105f), new Vector3(7.4f, 0.045f, 0.035f), gateStripeMaterial);
            GateBox(gate, "Left Foot", new Vector3(-halfSpan, 0.12f, 0f), new Vector3(1.05f, 0.24f, 1.00f), gateBoardMaterial);
            GateBox(gate, "Right Foot", new Vector3(halfSpan, 0.12f, 0f), new Vector3(1.05f, 0.24f, 1.00f), gateBoardMaterial);
            GateBox(gate, "Left Red Leg Stripe", new Vector3(-halfSpan, 0.72f, -0.33f), new Vector3(0.50f, 0.20f, 0.035f), postCapMaterial);
            GateBox(gate, "Right Red Leg Stripe", new Vector3(halfSpan, 0.72f, -0.33f), new Vector3(0.50f, 0.20f, 0.035f), postCapMaterial);
            AddGateText(gate);
            AddMarshal(gate, -1f, halfSpan);
            AddMarshal(gate, 1f, halfSpan);
        }

        bool TryCoursePoint(TrackGenerator track, float distance, out Vector3 position, out Vector3 forward)
        {
            int start = Mathf.Clamp(track.spawnWaypointIndex, 0, track.waypoints.Count - 2);
            int finish = Mathf.Min(track.FinishWaypointIndex, track.waypoints.Count - 1);
            for (int i = start; i < finish; i++)
            {
                Vector3 a = track.waypoints[i];
                Vector3 b = track.waypoints[i + 1];
                Vector3 delta = b - a;
                float segment = new Vector2(delta.x, delta.z).magnitude;
                if (segment < 0.001f) continue;
                if (distance <= segment)
                {
                    float t = distance / segment;
                    position = Vector3.Lerp(a, b, t);
                    forward = delta.normalized;
                    return true;
                }
                distance -= segment;
            }
            position = Vector3.zero;
            forward = Vector3.forward;
            return false;
        }

        void GateBox(Transform parent, string name, Vector3 position, Vector3 scale, Material material)
        {
            GameObject box = GameObject.CreatePrimitive(PrimitiveType.Cube);
            box.name = name;
            box.transform.SetParent(parent, false);
            box.transform.localPosition = position;
            box.transform.localScale = scale;
            Collider collider = box.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
            MeshRenderer renderer = box.GetComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
        }

        void AddGateText(Transform gate)
        {
            if (gateFont == null)
                gateFont = Font.CreateDynamicFontFromOSFont(new[] { "Arial", "Helvetica Neue" }, 96);
            if (gateFont == null) return;
            var lettering = new GameObject("Start Gate Lettering", typeof(RectTransform), typeof(Canvas));
            lettering.transform.SetParent(gate, false);
            lettering.transform.localPosition = new Vector3(0f, 3.72f, -0.14f);
            lettering.transform.localRotation = Quaternion.identity;
            lettering.transform.localScale = Vector3.one * 0.01f;
            Canvas canvas = lettering.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            RectTransform canvasRect = lettering.GetComponent<RectTransform>();
            canvasRect.sizeDelta = new Vector2(720f, 88f);

            var labelObject = new GameObject("START", typeof(RectTransform), typeof(Text));
            labelObject.transform.SetParent(lettering.transform, false);
            RectTransform labelRect = labelObject.GetComponent<RectTransform>();
            labelRect.anchorMin = Vector2.zero;
            labelRect.anchorMax = Vector2.one;
            labelRect.offsetMin = Vector2.zero;
            labelRect.offsetMax = Vector2.zero;
            Text label = labelObject.GetComponent<Text>();
            label.text = "START";
            label.font = gateFont;
            label.fontSize = 76;
            label.fontStyle = FontStyle.Bold;
            label.color = new Color(1f, 0.84f, 0.34f);
            label.alignment = TextAnchor.MiddleCenter;
            label.raycastTarget = false;
        }

        void AddMarshal(Transform gate, float side, float halfSpan)
        {
            var marshal = new GameObject(side < 0f ? "Stage Marshal Left" : "Stage Marshal Right").transform;
            marshal.SetParent(gate, false);
            marshal.localPosition = new Vector3(side * (halfSpan + 1.85f), 0f, -0.35f);
            marshal.localRotation = Quaternion.Euler(0f, 180f, 0f);

            MarshalShape(marshal, PrimitiveType.Cube, "Jacket", new Vector3(0f, 1.20f, 0f), new Vector3(0.48f, 0.72f, 0.28f), marshalVestMaterial);
            MarshalShape(marshal, PrimitiveType.Cube, "Belt", new Vector3(0f, 0.79f, 0f), new Vector3(0.46f, 0.14f, 0.29f), marshalClothMaterial);
            MarshalShape(marshal, PrimitiveType.Cube, "Left Leg", new Vector3(-0.13f, 0.39f, 0f), new Vector3(0.17f, 0.66f, 0.20f), marshalClothMaterial);
            MarshalShape(marshal, PrimitiveType.Cube, "Right Leg", new Vector3(0.13f, 0.39f, 0f), new Vector3(0.17f, 0.66f, 0.20f), marshalClothMaterial);
            MarshalShape(marshal, PrimitiveType.Sphere, "Head", new Vector3(0f, 1.80f, 0f), new Vector3(0.29f, 0.32f, 0.27f), marshalSkinMaterial);
            MarshalShape(marshal, PrimitiveType.Cube, "Helmet", new Vector3(0f, 1.98f, 0f), new Vector3(0.34f, 0.12f, 0.31f), gateBoardMaterial);

            var arm = MarshalShape(marshal, PrimitiveType.Cube, "Flag Arm", new Vector3(-side * 0.34f, 1.58f, 0f), new Vector3(0.16f, 0.52f, 0.17f), marshalVestMaterial);
            arm.transform.localRotation = Quaternion.Euler(0f, 0f, side * -24f);
            float flagSide = -side;
            MarshalShape(marshal, PrimitiveType.Cylinder, "Flag Pole", new Vector3(flagSide * 0.50f, 2.02f, 0f), new Vector3(0.035f, 0.37f, 0.035f), gateBoardMaterial);
            MarshalShape(marshal, PrimitiveType.Cube, "Green Flag", new Vector3(flagSide * 0.66f, 2.29f, 0f), new Vector3(0.43f, 0.28f, 0.035f), marshalFlagMaterial);
        }

        GameObject MarshalShape(Transform parent, PrimitiveType type, string name, Vector3 position,
                                Vector3 scale, Material material)
        {
            GameObject shape = GameObject.CreatePrimitive(type);
            shape.name = name;
            shape.transform.SetParent(parent, false);
            shape.transform.localPosition = position;
            shape.transform.localScale = scale;
            Collider collider = shape.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
            MeshRenderer renderer = shape.GetComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            return shape;
        }

        static void BuildRidge(Transform parent, string name, float radius, float baseHeight,
                               float variation, float broadScale, float fineScale, Material material)
        {
            const int segments = 512;
            const float depth = 420f;
            var vertices = new List<Vector3>((segments + 1) * 2);
            var triangles = new List<int>(segments * 6);
            var uv = new List<Vector2>((segments + 1) * 2);
            for (int i = 0; i <= segments; i++)
            {
                float angle = i * (Mathf.PI * 2f / segments);
                float x = Mathf.Cos(angle), z = Mathf.Sin(angle);
                float broad = Mathf.PerlinNoise(14.3f + x * broadScale, 29.7f + z * broadScale);
                float detail = Mathf.PerlinNoise(41.9f + x * fineScale * 3f, 62.1f + z * fineScale * 3f);
                float profile = Mathf.Clamp01(broad * 0.68f + detail * 0.32f);
                float peaks = Mathf.Pow(Mathf.SmoothStep(0.18f, 0.92f, profile), 1.35f);
                float height = baseHeight + variation * peaks;
                Vector3 direction = new Vector3(x, 0f, z) * radius;
                vertices.Add(direction + Vector3.down * depth);
                vertices.Add(direction + Vector3.up * height);
                uv.Add(new Vector2(i / (float)segments, 0f));
                uv.Add(new Vector2(i / (float)segments, 1f));
                if (i == segments) continue;
                int bottom = i * 2, top = bottom + 1, nextBottom = bottom + 2, nextTop = bottom + 3;
                triangles.Add(bottom); triangles.Add(nextBottom); triangles.Add(top);
                triangles.Add(top); triangles.Add(nextBottom); triangles.Add(nextTop);
            }
            var mesh = new Mesh { name = name + " Mesh", indexFormat = IndexFormat.UInt32 };
            mesh.SetVertices(vertices);
            mesh.SetUVs(0, uv);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
        }

        void BuildForest(TrackGenerator track, Transform root)
        {
            if (StageDressing.UsesReviewKit(track)) return;
            if (track.terrain == null || track.terrain.GetComponent<MeshCollider>() == null) return;
            var terrainCollider = track.terrain.GetComponent<MeshCollider>();
            var barkVertices = new List<Vector3>(50000);
            var barkTriangles = new List<int>(90000);
            var leafVertices = new[] { new List<Vector3>(50000), new List<Vector3>(50000), new List<Vector3>(50000) };
            var leafTriangles = new[] { new List<int>(90000), new List<int>(90000), new List<int>(90000) };
            var rng = new System.Random(track.CurrentSeed ^ 0x5f3759df);
            int segmentCount = track.waypoints.Count - 3;
            float stepT = Mathf.Clamp(12f / Mathf.Max(1f, track.trackLength / Mathf.Max(1, track.controlPointsCount)), 0.10f, 0.45f);

            for (int segment = Mathf.Max(0, track.spawnWaypointIndex - 1); segment < segmentCount; segment++)
            {
                Vector3 p0 = track.waypoints[segment];
                Vector3 p1 = track.waypoints[segment + 1];
                Vector3 p2 = track.waypoints[segment + 2];
                Vector3 p3 = track.waypoints[segment + 3];
                for (float t = 0.03f; t < 0.98f; t += stepT)
                {
                    Vector3 center = SplineMath.GetCatmullRomPosition(t, p0, p1, p2, p3);
                    Vector3 tangent = SplineMath.GetTangent(t, p0, p1, p2, p3);
                    tangent.y = 0f;
                    if (tangent.sqrMagnitude < 0.001f) continue;
                    tangent.Normalize();
                    Vector3 right = Vector3.Cross(Vector3.up, tangent).normalized;

                    for (int sideIndex = -1; sideIndex <= 1; sideIndex += 2)
                    {
                        int count = rng.Next(0, 5);
                        for (int treeIndex = 0; treeIndex < count; treeIndex++)
                        {
                            float lateral = Mathf.Lerp(16f, track.terrainHalfWidth + 6f, (float)rng.NextDouble());
                            Vector3 candidate = center + right * (lateral * sideIndex)
                                + tangent * Mathf.Lerp(-6f, 6f, (float)rng.NextDouble());
                            if (!TryGround(terrainCollider, track.transform, candidate, out Vector3 ground)) continue;
                            float height = Mathf.Lerp(4.8f, 10.5f, (float)rng.NextDouble());
                            float radius = height * Mathf.Lerp(0.035f, 0.055f, (float)rng.NextDouble());
                            float yaw = (float)rng.NextDouble() * 360f;
                            int palette = rng.Next(visualCanopyMaterials.Length);
                            bool broadleaf = rng.NextDouble() < 0.35;
                            AppendForestTree(ground, height, radius, yaw, palette,
                                             barkVertices, barkTriangles, leafVertices[palette], leafTriangles[palette], broadleaf);
                        }
                    }
                }
            }

            AttachMesh(root, "Distant Forest Trunks", barkVertices, barkTriangles, barkMaterial);
            for (int i = 0; i < visualCanopyMaterials.Length; i++)
                AttachMesh(root, "Distant Forest Canopy " + (i + 1), leafVertices[i], leafTriangles[i], visualCanopyMaterials[i]);
        }

        static bool TryGround(MeshCollider collider, Transform track, Vector3 localPoint, out Vector3 localGround)
        {
            Vector3 world = track.TransformPoint(localPoint + Vector3.up * 250f);
            if (collider.Raycast(new Ray(world, Vector3.down), out RaycastHit hit, 600f))
            {
                localGround = track.InverseTransformPoint(hit.point);
                return true;
            }
            localGround = localPoint;
            return false;
        }

        static void AppendForestTree(Vector3 baseLocal, float height, float radius, float yaw, int palette,
                                     List<Vector3> barkV, List<int> barkT,
                                     List<Vector3> leafV, List<int> leafT, bool broadleaf)
        {
            Quaternion rotation = Quaternion.Euler(0f, yaw, 0f);
            Vector3 basePoint = baseLocal;
            AppendCylinder(barkV, barkT, basePoint, height * 0.53f, radius, radius * 0.72f, rotation, 7);

            if (!broadleaf)
            {
                int tiers = 3 + palette;
                for (int tier = 0; tier < tiers; tier++)
                {
                    float fraction = tier / (float)tiers;
                    float bottom = height * (0.20f + fraction * 0.59f);
                    float top = Mathf.Min(height * 1.03f, bottom + height * (0.29f - palette * 0.018f));
                    float canopy = height * (0.33f - fraction * 0.16f) * (0.84f + 0.12f * ((tier + palette) % 3));
                    Quaternion tierRotation = Quaternion.Euler(0f, yaw + tier * (11f + palette * 7f), 0f);
                    float lean = Mathf.Sin(yaw * Mathf.Deg2Rad + tier * 1.7f) * height * 0.018f;
                    Vector3 tierOffset = rotation * new Vector3(lean, 0f, Mathf.Cos(tier * 2.1f + yaw) * height * 0.014f);
                    AppendCone(leafV, leafT, basePoint + Vector3.up * bottom + tierOffset, top - bottom,
                               canopy, canopy * 0.07f, tierRotation, 9);
                }
            }
            else
            {
                AppendCylinder(barkV, barkT, basePoint, height * 0.68f, radius * 1.15f, radius * 0.60f, rotation, 7);
                Vector3 crown = basePoint + Vector3.up * height * 0.66f;
                for (int i = 0; i < 6; i++)
                {
                    float a = i * Mathf.PI / 3f + yaw * Mathf.Deg2Rad;
                    Vector3 offset = new Vector3(Mathf.Cos(a) * height * 0.14f, (i % 2) * height * 0.055f, Mathf.Sin(a) * height * 0.14f);
                    AppendOctahedron(leafV, leafT, crown + offset, height * 0.20f, height * 0.17f, height * 0.20f);
                }
            }
        }

        static void AppendCylinder(List<Vector3> vertices, List<int> triangles, Vector3 origin,
                                   float height, float bottomRadius, float topRadius, Quaternion rotation, int sides)
        {
            int start = vertices.Count;
            for (int i = 0; i < sides; i++)
            {
                float a0 = i * Mathf.PI * 2f / sides, a1 = (i + 1) * Mathf.PI * 2f / sides;
                vertices.Add(origin + rotation * new Vector3(Mathf.Cos(a0) * bottomRadius, 0f, Mathf.Sin(a0) * bottomRadius));
                vertices.Add(origin + rotation * new Vector3(Mathf.Cos(a1) * bottomRadius, 0f, Mathf.Sin(a1) * bottomRadius));
                vertices.Add(origin + rotation * new Vector3(Mathf.Cos(a1) * topRadius, height, Mathf.Sin(a1) * topRadius));
                vertices.Add(origin + rotation * new Vector3(Mathf.Cos(a0) * topRadius, height, Mathf.Sin(a0) * topRadius));
                int b = start + i * 4;
                triangles.Add(b); triangles.Add(b + 2); triangles.Add(b + 1);
                triangles.Add(b); triangles.Add(b + 3); triangles.Add(b + 2);
            }
        }

        static void AppendCone(List<Vector3> vertices, List<int> triangles, Vector3 origin,
                               float height, float radius, float topRadius, Quaternion rotation, int sides)
        {
            int start = vertices.Count;
            for (int i = 0; i < sides; i++)
            {
                float a0 = i * Mathf.PI * 2f / sides, a1 = (i + 1) * Mathf.PI * 2f / sides;
                vertices.Add(origin + rotation * new Vector3(Mathf.Cos(a0) * radius, 0f, Mathf.Sin(a0) * radius));
                vertices.Add(origin + rotation * new Vector3(Mathf.Cos(a1) * radius, 0f, Mathf.Sin(a1) * radius));
                vertices.Add(origin + rotation * new Vector3(Mathf.Cos(a1) * topRadius, height, Mathf.Sin(a1) * topRadius));
                vertices.Add(origin + rotation * new Vector3(Mathf.Cos(a0) * topRadius, height, Mathf.Sin(a0) * topRadius));
                int b = start + i * 4;
                triangles.Add(b); triangles.Add(b + 2); triangles.Add(b + 1);
                triangles.Add(b); triangles.Add(b + 3); triangles.Add(b + 2);
            }
        }

        static void AppendOctahedron(List<Vector3> vertices, List<int> triangles, Vector3 center,
                                     float rx, float ry, float rz)
        {
            Vector3[] p = {
                center + Vector3.up * ry, center + Vector3.down * ry,
                center + Vector3.right * rx, center + Vector3.left * rx,
                center + Vector3.forward * rz, center + Vector3.back * rz
            };
            int[] faces = { 0,2,4, 0,4,3, 0,3,5, 0,5,2, 1,4,2, 1,3,4, 1,5,3, 1,2,5 };
            for (int i = 0; i < faces.Length; i += 3)
            {
                Vector3 a = p[faces[i]], b = p[faces[i + 1]], c = p[faces[i + 2]];
                Vector3 normal = Vector3.Cross(b - a, c - a);
                if (Vector3.Dot(normal, (a + b + c) / 3f - center) < 0f)
                    (b, c) = (c, b);
                int start = vertices.Count;
                vertices.Add(a); vertices.Add(b); vertices.Add(c);
                triangles.Add(start); triangles.Add(start + 1); triangles.Add(start + 2);
            }
        }

        static void AttachMesh(Transform parent, string name, List<Vector3> vertices,
                               List<int> triangles, Material material)
        {
            if (vertices.Count == 0 || triangles.Count == 0) return;
            var mesh = new Mesh { name = name + " Mesh" };
            if (vertices.Count > 65000) mesh.indexFormat = IndexFormat.UInt32;
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
        }

        void BuildRoadsideMarkers(TrackGenerator track, Transform root)
        {
            if (track.terrain == null || track.terrain.GetComponent<MeshCollider>() == null) return;
            var collider = track.terrain.GetComponent<MeshCollider>();
            var paleV = new List<Vector3>(); var paleT = new List<int>();
            var redV = new List<Vector3>(); var redT = new List<int>();
            int segments = track.waypoints.Count - 3;
            float stepT = Mathf.Clamp(24f / Mathf.Max(1f, track.trackLength / Mathf.Max(1, track.controlPointsCount)), 0.2f, 0.8f);
            for (int segment = Mathf.Max(0, track.spawnWaypointIndex - 1); segment < segments; segment++)
            {
                for (float t = 0.08f; t < 0.99f; t += stepT)
                {
                    Vector3 p = SplineMath.GetCatmullRomPosition(t, track.waypoints[segment], track.waypoints[segment + 1],
                                                                  track.waypoints[segment + 2], track.waypoints[segment + 3]);
                    Vector3 tangent = SplineMath.GetTangent(t, track.waypoints[segment], track.waypoints[segment + 1],
                                                            track.waypoints[segment + 2], track.waypoints[segment + 3]);
                    tangent.y = 0f;
                    if (tangent.sqrMagnitude < 0.001f) continue;
                    Vector3 right = Vector3.Cross(Vector3.up, tangent.normalized);
                    for (int side = -1; side <= 1; side += 2)
                    {
                        Vector3 candidate = p + right * (7.7f * side);
                        if (!TryGround(collider, track.transform, candidate, out Vector3 ground)) continue;
                        AppendPost(ground, paleV, paleT, 0.76f, 0.055f);
                        AppendPost(ground + Vector3.up * 0.58f, redV, redT, 0.20f, 0.061f);
                    }
                }
            }
            AttachMesh(root, "Roadside Stakes", paleV, paleT, postMaterial);
            AttachMesh(root, "Roadside Reflectors", redV, redT, postCapMaterial);
        }

        static void AppendPost(Vector3 localBase, List<Vector3> vertices,
                               List<int> triangles, float height, float halfWidth)
        {
            Vector3 p = localBase;
            Vector3 up = Vector3.up * height;
            Vector3 right = new Vector3(halfWidth, 0f, 0f);
            Vector3 forward = new Vector3(0f, 0f, halfWidth);
            Vector3[] corners = { p - right - forward, p + right - forward, p + right + forward, p - right + forward,
                                  p + up - right - forward, p + up + right - forward, p + up + right + forward, p + up - right + forward };
            int[] faces = { 0,2,1, 0,3,2, 4,5,6, 4,6,7, 0,1,5, 0,5,4,
                            1,2,6, 1,6,5, 2,3,7, 2,7,6, 3,0,4, 3,4,7 };
            int start = vertices.Count;
            for (int i = 0; i < faces.Length; i++) vertices.Add(corners[faces[i]]);
            for (int i = 0; i < faces.Length; i++) triangles.Add(start + i);
        }

        void ReplaceTelemetry(TrackGenerator track)
        {
            TelemetryUI old = FindObjectOfType<TelemetryUI>();
            if (old != null)
            {
                old.enabled = false;
                Canvas oldCanvas = old.GetComponent<Canvas>();
                if (oldCanvas != null) oldCanvas.enabled = false;
                Destroy(old.gameObject);
            }
            Camera camera = Camera.main;
            VehicleController vehicle = FindObjectOfType<VehicleController>();
            if (camera == null || vehicle == null) return;
            RallyStageHUD previous = FindObjectOfType<RallyStageHUD>();
            if (previous != null) Destroy(previous.gameObject);
            var hud = new GameObject("Rally Stage HUD");
            hud.AddComponent<RallyStageHUD>().Configure(vehicle, track);
        }

        void ApplyCar(VehicleController vehicle)
        {
            foreach (Renderer renderer in vehicle.GetComponentsInChildren<Renderer>(true))
            {
                if (renderer is ParticleSystemRenderer) continue;
                Material[] originals = renderer.sharedMaterials;
                var upgraded = new Material[originals.Length];
                for (int i = 0; i < originals.Length; i++)
                {
                    Material source = originals[i];
                    if (source == null) continue;
                    var material = new Material(Shader.Find("Standard")) { name = source.name + " (Rally Finish)" };
                    if (source.mainTexture != null)
                    {
                        Texture texture = source.mainTexture;
                        texture.filterMode = FilterMode.Bilinear;
                        texture.anisoLevel = 8;
                        material.mainTexture = texture;
                    }
                    Color tint = source.HasProperty("_Color") ? source.color : Color.white;
                    bool tire = source.name.IndexOf("Tyre", StringComparison.OrdinalIgnoreCase) >= 0;
                    bool glass = source.name.IndexOf("Glass", StringComparison.OrdinalIgnoreCase) >= 0;
                    bool rim = source.name.IndexOf("Rim", StringComparison.OrdinalIgnoreCase) >= 0;
                    bool lamp = source.name.IndexOf("Lamp", StringComparison.OrdinalIgnoreCase) >= 0;
                    material.color = glass ? new Color(0.075f, 0.13f, 0.17f, 1f) : tint;
                    material.SetFloat("_Metallic", glass ? 0.32f : rim ? 0.62f : tire ? 0.02f : lamp ? 0.12f : 0.08f);
                    material.SetFloat("_Glossiness", glass ? 0.92f : rim ? 0.62f : tire ? 0.09f : lamp ? 0.76f : 0.42f);
                    if (lamp)
                    {
                        material.EnableKeyword("_EMISSION");
                        material.SetColor("_EmissionColor", tint * (source.name.IndexOf("Tail", StringComparison.OrdinalIgnoreCase) >= 0 ? 0.12f : 0.06f));
                    }
                    upgraded[i] = material;
                }
                renderer.sharedMaterials = upgraded;
                renderer.shadowCastingMode = ShadowCastingMode.On;
                renderer.receiveShadows = true;
            }
            BuildCarDetails(vehicle);
            CreateDust(vehicle);
            CreateBrakeLights(vehicle);
        }

        void BuildCarDetails(VehicleController vehicle)
        {
            if (vehicle.transform.Find("RallyCarDetails") != null) return;
            var details = new GameObject("RallyCarDetails").transform;
            details.SetParent(vehicle.transform, false);

            CarDetailShape(details, PrimitiveType.Cube, "Rear Plate Frame", new Vector3(0f, 0.535f, -2.247f),
                           new Vector3(0.64f, 0.19f, 0.035f), plateFrameMaterial, Quaternion.identity);
            CarDetailShape(details, PrimitiveType.Cube, "Yellow 555 Plate", new Vector3(0f, 0.535f, -2.27f),
                           new Vector3(0.58f, 0.14f, 0.018f), plateMaterial, Quaternion.identity);

            if (gateFont == null)
                gateFont = Font.CreateDynamicFontFromOSFont(new[] { "Arial", "Helvetica Neue" }, 96);
            if (gateFont != null)
            {
                var lettering = new GameObject("555 Plate Lettering", typeof(RectTransform), typeof(Canvas));
                lettering.transform.SetParent(details, false);
                lettering.transform.localPosition = new Vector3(0f, 0.535f, -2.284f);
                lettering.transform.localScale = Vector3.one * 0.001f;
                Canvas canvas = lettering.GetComponent<Canvas>();
                canvas.renderMode = RenderMode.WorldSpace;
                RectTransform canvasRect = lettering.GetComponent<RectTransform>();
                canvasRect.sizeDelta = new Vector2(580f, 140f);
                var labelObject = new GameObject("555", typeof(RectTransform), typeof(Text));
                labelObject.transform.SetParent(lettering.transform, false);
                RectTransform labelRect = labelObject.GetComponent<RectTransform>();
                labelRect.anchorMin = Vector2.zero;
                labelRect.anchorMax = Vector2.one;
                labelRect.offsetMin = Vector2.zero;
                labelRect.offsetMax = Vector2.zero;
                Text label = labelObject.GetComponent<Text>();
                label.text = "555";
                label.font = gateFont;
                label.fontSize = 112;
                label.fontStyle = FontStyle.Bold;
                label.color = new Color(0.05f, 0.055f, 0.05f);
                label.alignment = TextAnchor.MiddleCenter;
                label.raycastTarget = false;
            }

            GameObject exhaust = CarDetailShape(details, PrimitiveType.Cylinder, "Rear Exhaust",
                new Vector3(0.53f, 0.31f, -2.22f), new Vector3(0.075f, 0.075f, 0.075f), exhaustMaterial,
                Quaternion.Euler(90f, 0f, 0f));
            Collider exhaustCollider = exhaust.GetComponent<Collider>();
            if (exhaustCollider != null) Destroy(exhaustCollider);
            GameObject exhaustInterior = CarDetailShape(details, PrimitiveType.Cylinder, "Exhaust Outlet",
                new Vector3(0.53f, 0.31f, -2.296f), new Vector3(0.046f, 0.046f, 0.004f), exhaustInteriorMaterial,
                Quaternion.Euler(90f, 0f, 0f));
            Collider outletCollider = exhaustInterior.GetComponent<Collider>();
            if (outletCollider != null) Destroy(outletCollider);
        }

        GameObject CarDetailShape(Transform parent, PrimitiveType type, string name, Vector3 position,
                                  Vector3 scale, Material material, Quaternion rotation)
        {
            GameObject shape = GameObject.CreatePrimitive(type);
            shape.name = name;
            shape.transform.SetParent(parent, false);
            shape.transform.localPosition = position;
            shape.transform.localScale = scale;
            shape.transform.localRotation = rotation;
            Collider collider = shape.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
            MeshRenderer renderer = shape.GetComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            return shape;
        }

        void ApplyCamera(VehicleController vehicle, TrackGenerator track)
        {
            Camera camera = Camera.main;
            if (camera == null) return;
            // This method polls every half-second. SpectatorDirector owns animated
            // framing; resetting its FOV here produces periodic zoom pulses in every mode.
            camera.farClipPlane = 1200f;
            camera.allowHDR = true;
            camera.allowMSAA = true;
            var director = camera.GetComponent<SpectatorDirector>();
            if (director == null) return;
            director.targetCar = vehicle.transform;
            director.carRb = vehicle.GetComponent<Rigidbody>();
            director.orbitDistance = 6.35f;
            director.orbitHeight = 1.82f;
            director.chaseSideOffset = 0.48f;
            director.speedZoom = 0.012f;
            director.followSmoothness = 5.5f;
            director.ConfigureCourse(track);
        }

        void CreateDust(VehicleController vehicle)
        {
            if (vehicle.transform.Find("RallyDust_FrontLeft") != null) return;
            Shader shader = Shader.Find("Particles/Standard Unlit");
            if (shader == null) shader = Shader.Find("Sprites/Default");
            if (shader != null)
            {
                dustMaterial.shader = shader;
                if (dustMaterial.HasProperty("_MainTex")) dustMaterial.SetTexture("_MainTex", dustMaterial.mainTexture);
                dustMaterial.renderQueue = 3000;
            }
            PacejkaTireModel[] tires = vehicle.GetComponentsInChildren<PacejkaTireModel>(true);
            if (tires.Length == 0) return;
            int trackCount = tires.Length * RallyDustController.TrackSegmentsPerWheel;
            var dustSystems = new ParticleSystem[tires.Length];
            var gravelSystems = new ParticleSystem[tires.Length];
            var tracks = new TrailRenderer[trackCount];
            for (int i = 0; i < tires.Length; i++)
            {
                DynamicSuspension suspension = tires[i].GetComponent<DynamicSuspension>();
                if (suspension == null) continue;
                Vector3 anchor = suspension.transform.localPosition;
                string corner = suspension.corner.ToString();
                dustSystems[i] = CreateDustEmitter(vehicle.transform, "RallyDust_" + corner, anchor);
                gravelSystems[i] = CreateGravelEmitter(vehicle.transform, "RallyGravel_" + corner, anchor);
                for (int slot = 0; slot < RallyDustController.TrackSegmentsPerWheel; slot++)
                    tracks[i * RallyDustController.TrackSegmentsPerWheel + slot] =
                        CreateWheelTrack(vehicle.transform, "DrivenTrack_" + corner + "_" + slot);
            }
            var controller = vehicle.gameObject.AddComponent<RallyDustController>();
            controller.Configure(vehicle, tires, dustSystems, gravelSystems, tracks, dustMaterial);
        }

        void CreateBrakeLights(VehicleController vehicle)
        {
            RallyBrakeLightController controller = vehicle.GetComponent<RallyBrakeLightController>();
            if (controller == null) controller = vehicle.gameObject.AddComponent<RallyBrakeLightController>();
            controller.Configure(vehicle);
        }

        static ParticleSystem CreateDustEmitter(Transform car, string name, Vector3 localPosition)
        {
            var go = new GameObject(name);
            go.transform.SetParent(car, false);
            go.transform.localPosition = localPosition;
            var system = go.AddComponent<ParticleSystem>();
            var main = system.main;
            main.duration = 4f;
            main.loop = true;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.75f, 1.55f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(1.0f, 3.2f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.14f, 0.38f);
            main.startColor = new Color(0.58f, 0.55f, 0.49f, 0.38f);
            main.gravityModifier = 0.035f;
            main.maxParticles = 112;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            var emission = system.emission;
            emission.rateOverTime = 0f;
            var shape = system.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 18f;
            shape.radius = 0.14f;
            var color = system.colorOverLifetime;
            color.enabled = true;
            var gradient = new Gradient();
            gradient.SetKeys(
                new[] { new GradientColorKey(new Color(0.72f, 0.70f, 0.66f), 0f), new GradientColorKey(new Color(0.65f, 0.64f, 0.61f), 1f) },
                new[] { new GradientAlphaKey(0.0f, 0f), new GradientAlphaKey(0.32f, 0.12f), new GradientAlphaKey(0.0f, 1f) });
            color.color = gradient;
            var size = system.sizeOverLifetime;
            size.enabled = true;
            var curve = new AnimationCurve(new Keyframe(0f, 0.55f), new Keyframe(1f, 1.7f));
            size.size = new ParticleSystem.MinMaxCurve(1f, curve);
            var noise = system.noise;
            noise.enabled = true;
            noise.strength = 0.22f;
            noise.frequency = 0.34f;
            noise.scrollSpeed = 0.12f;
            noise.quality = ParticleSystemNoiseQuality.Low;
            system.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            return system;
        }

        static ParticleSystem CreateGravelEmitter(Transform car, string name, Vector3 localPosition)
        {
            var go = new GameObject(name);
            go.transform.SetParent(car, false);
            go.transform.localPosition = localPosition;
            var system = go.AddComponent<ParticleSystem>();
            var main = system.main;
            main.duration = 1f;
            main.loop = true;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.25f, 0.72f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(2.0f, 5.5f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.035f, 0.095f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                new Color(0.50f, 0.43f, 0.33f, 0.90f), new Color(0.78f, 0.68f, 0.51f, 1f));
            main.gravityModifier = 1.35f;
            main.maxParticles = 24;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            var emission = system.emission;
            emission.rateOverTime = 0f;
            var shape = system.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 28f;
            shape.radius = 0.065f;
            var color = system.colorOverLifetime;
            color.enabled = true;
            var gradient = new Gradient();
            gradient.SetKeys(
                new[]
                {
                    new GradientColorKey(new Color(0.62f, 0.52f, 0.39f), 0f),
                    new GradientColorKey(new Color(0.48f, 0.43f, 0.35f), 1f)
                },
                new[]
                {
                    new GradientAlphaKey(0f, 0f),
                    new GradientAlphaKey(0.92f, 0.08f),
                    new GradientAlphaKey(0.72f, 0.55f),
                    new GradientAlphaKey(0f, 1f)
                });
            color.color = gradient;
            var size = system.sizeOverLifetime;
            size.enabled = true;
            size.size = new ParticleSystem.MinMaxCurve(1f,
                new AnimationCurve(new Keyframe(0f, 0.72f), new Keyframe(1f, 0.18f)));
            var rotation = system.rotationOverLifetime;
            rotation.enabled = true;
            rotation.z = new ParticleSystem.MinMaxCurve(-7f, 7f);
            system.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            return system;
        }

        TrailRenderer CreateWheelTrack(Transform car, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(car, false);
            TrailRenderer trail = go.AddComponent<TrailRenderer>();
            trail.sharedMaterial = tireMarkMaterial;
            trail.time = RallyDustController.TrackLifetime;
            trail.widthMultiplier = 0.19f;
            trail.minVertexDistance = 0.14f;
            trail.numCapVertices = 2;
            trail.numCornerVertices = 2;
            trail.alignment = LineAlignment.TransformZ;
            trail.textureMode = LineTextureMode.Stretch;
            trail.shadowCastingMode = ShadowCastingMode.Off;
            trail.receiveShadows = false;
            trail.autodestruct = false;
            trail.emitting = false;

            var gradient = new Gradient();
            gradient.SetKeys(
                new[]
                {
                    new GradientColorKey(new Color(0.25f, 0.22f, 0.18f), 0f),
                    new GradientColorKey(new Color(0.34f, 0.29f, 0.22f), 1f)
                },
                new[]
                {
                    new GradientAlphaKey(0.30f, 0f),
                    new GradientAlphaKey(0.18f, 0.82f),
                    new GradientAlphaKey(0f, 1f)
                });
            trail.colorGradient = gradient;
            return trail;
        }
    }

    public sealed class RallyDustController : MonoBehaviour
    {
        public const int TrackSegmentsPerWheel = 4;
        public const float TrackLifetime = 38f;
        const float AirborneBreakSeconds = 0.24f;

        VehicleController vehicle;
        PacejkaTireModel[] tires;
        DynamicSuspension[] suspensions;
        ParticleSystem[] dustSystems;
        ParticleSystem[] gravelSystems;
        TrailRenderer[] tracks;
        int[] activeTrackSlots;
        float[] airborneSeconds;
        bool[] hasTrackSegment;
        bool[] wasGrounded;
        Rigidbody body;
        Vector3 lastVehiclePosition;
        bool hasLastVehiclePosition;
        RallyAgent episodeSource;

        public void Configure(VehicleController target, PacejkaTireModel[] wheelTires,
                              ParticleSystem[] dustEmitters, ParticleSystem[] gravelEmitters,
                              TrailRenderer[] wheelTracks, Material dustMaterial)
        {
            vehicle = target;
            tires = wheelTires;
            dustSystems = dustEmitters;
            gravelSystems = gravelEmitters;
            tracks = wheelTracks;
            body = target.GetComponent<Rigidbody>();
            suspensions = new DynamicSuspension[tires.Length];
            activeTrackSlots = new int[tires.Length];
            airborneSeconds = new float[tires.Length];
            hasTrackSegment = new bool[tires.Length];
            wasGrounded = new bool[tires.Length];
            for (int i = 0; i < tires.Length; i++)
            {
                if (tires[i] == null) continue;
                suspensions[i] = tires[i].GetComponent<DynamicSuspension>();
                ConfigureParticleRenderer(dustSystems[i], dustMaterial, 0.015f, 0.22f);
                ConfigureParticleRenderer(gravelSystems[i], dustMaterial, 0.01f, 0.09f);
            }
            if (body != null) lastVehiclePosition = body.position;
            hasLastVehiclePosition = body != null;
            // The contact-driven wheel emitters own runtime dust; retain the legacy
            // component only for offline review tooling.
            var legacy = target.GetComponent<DustPlume>();
            if (legacy != null) { legacy.ClearTrail(); legacy.enabled = false; }
            if (episodeSource != null) episodeSource.EpisodeEnded -= OnEpisodeEnded;
            episodeSource = target.GetComponent<RallyAgent>();
            if (episodeSource != null) episodeSource.EpisodeEnded += OnEpisodeEnded;
        }

        void OnEpisodeEnded(EpisodeOutcome outcome, int reached) => ClearEffects();
        void OnDisable() { if (tires != null) ClearEffects(); }
        void OnDestroy()
        {
            if (episodeSource != null) episodeSource.EpisodeEnded -= OnEpisodeEnded;
        }

        static void ConfigureParticleRenderer(ParticleSystem system, Material material,
                                              float minSize, float maxSize)
        {
            if (system == null) return;
            ParticleSystemRenderer renderer = system.GetComponent<ParticleSystemRenderer>();
            if (renderer == null) return;
            renderer.sharedMaterial = material;
            renderer.renderMode = ParticleSystemRenderMode.Billboard;
            renderer.minParticleSize = minSize;
            renderer.maxParticleSize = maxSize;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
        }

        void LateUpdate()
        {
            if (vehicle == null || tires == null || dustSystems == null || tracks == null) return;
            float dt = Mathf.Max(0.001f, Time.unscaledDeltaTime);
            Vector3 velocity = body != null ? body.linearVelocity : Vector3.zero;
            float speed = velocity.magnitude;
            if (body != null && hasLastVehiclePosition)
            {
                float teleportThreshold = Mathf.Max(12f, speed * dt * 3.5f + 8f);
                if ((body.position - lastVehiclePosition).magnitude > teleportThreshold) ClearEffects();
                lastVehiclePosition = body.position;
            }
            else if (body != null)
            {
                lastVehiclePosition = body.position;
                hasLastVehiclePosition = true;
            }

            for (int i = 0; i < tires.Length; i++)
            {
                PacejkaTireModel tire = tires[i];
                DynamicSuspension suspension = suspensions[i];
                if (suspension == null || tire == null) continue;

                float looseness = Mathf.Clamp01(suspension.surfaceLooseness);
                float loadFactor = Mathf.InverseLerp(250f, 2500f, tire.normalLoad);
                float slip = Mathf.Clamp01(tire.slipUtilisation);
                float sliding = Mathf.InverseLerp(0.45f, 5.5f, tire.slipSpeed);
                float lateralShare = Mathf.Clamp01(Mathf.Abs(tire.lateralForce) /
                    Mathf.Max(1f, Mathf.Abs(tire.lateralForce) + Mathf.Abs(tire.longitudinalForce)));
                float forceUse = Mathf.Clamp01(Mathf.Sqrt(
                    tire.longitudinalForce * tire.longitudinalForce +
                    tire.lateralForce * tire.lateralForce) / Mathf.Max(700f, tire.normalLoad * 1.45f));
                bool contact = tire.isGrounded && tire.normalLoad > 160f;
                Vector3 normal = contact ? suspension.tireHit.normal : Vector3.up;
                Vector3 forward = suspension.ContactForward.sqrMagnitude > 0.01f
                    ? suspension.ContactForward : vehicle.transform.forward;
                if (Vector3.Dot(forward, velocity) < 0f) forward = -forward;
                Vector3 sideForce = suspension.ContactRight * Mathf.Sign(tire.lateralForce);
                Vector3 plume = Vector3.ProjectOnPlane(
                    -forward - sideForce * (0.55f * lateralShare * sliding) + Vector3.up * 0.45f,
                    normal).normalized;
                if (plume.sqrMagnitude < 0.01f) plume = Vector3.ProjectOnPlane(-forward, normal).normalized;

                ParticleSystem dust = dustSystems[i];
                float rollSpray = Mathf.InverseLerp(2f, 18f, speed) * 11f;
                float slipSpray = sliding * (8f + slip * 22f + forceUse * 16f);
                float dustRate = contact && looseness > 0.02f
                    ? Mathf.Clamp((rollSpray + slipSpray) * looseness * loadFactor, 0f, 62f) : 0f;
                SetEmission(dust, dustRate);
                if (dust != null && contact && dustRate > 0.25f)
                {
                    PositionEmitter(dust, suspension.tireHit.point, normal, plume);
                    var main = dust.main;
                    main.startSpeed = new ParticleSystem.MinMaxCurve(0.75f + speed * 0.035f,
                        1.8f + speed * 0.075f + sliding * 1.35f);
                    main.startSize = new ParticleSystem.MinMaxCurve(0.13f + sliding * 0.04f,
                        0.34f + sliding * 0.16f);
                    var shape = dust.shape;
                    shape.angle = 17f + sliding * 18f;
                }

                ParticleSystem gravel = gravelSystems[i];
                float debrisRate = contact && looseness > 0.12f
                    ? Mathf.Clamp(looseness * loadFactor * Mathf.InverseLerp(0.48f, 0.92f,
                        Mathf.Max(slip, forceUse)) * Mathf.InverseLerp(0.45f, 4f, tire.slipSpeed) * 8f, 0f, 12f)
                    : 0f;
                SetEmission(gravel, debrisRate);
                if (gravel != null && contact && debrisRate > 0.05f)
                {
                    PositionEmitter(gravel, suspension.tireHit.point, normal, plume);
                    var main = gravel.main;
                    main.startSpeed = new ParticleSystem.MinMaxCurve(1.8f + speed * 0.05f,
                        3.6f + speed * 0.10f + forceUse * 2.5f);
                    var shape = gravel.shape;
                    shape.angle = 24f + lateralShare * sliding * 16f;
                }

                UpdateTrack(i, tire, suspension, contact, speed, looseness, loadFactor, slip);
            }
        }

        void UpdateTrack(int wheel, PacejkaTireModel tire, DynamicSuspension suspension,
                         bool contact, float speed, float looseness, float loadFactor, float slip)
        {
            bool grounded = tire.isGrounded;
            if (!grounded) airborneSeconds[wheel] += Time.unscaledDeltaTime;
            else if (grounded && !wasGrounded[wheel] && hasTrackSegment[wheel] &&
                     airborneSeconds[wheel] > AirborneBreakSeconds)
            {
                activeTrackSlots[wheel] = (activeTrackSlots[wheel] + 1) % TrackSegmentsPerWheel;
                TrailRenderer next = TrackAt(wheel, activeTrackSlots[wheel]);
                if (next != null) next.Clear();
            }

            bool shouldMark = contact && speed > 1.25f && looseness > 0.025f;
            TrailRenderer trail = TrackAt(wheel, activeTrackSlots[wheel]);
            if (trail == null) return;
            if (shouldMark)
            {
                Vector3 normal = suspension.tireHit.normal;
                Vector3 direction = Vector3.ProjectOnPlane(suspension.ContactForward, normal).normalized;
                if (direction.sqrMagnitude < 0.01f) direction = Vector3.ProjectOnPlane(vehicle.transform.forward, normal).normalized;
                trail.transform.SetPositionAndRotation(
                    suspension.tireHit.point + normal * 0.024f,
                    Quaternion.LookRotation(normal, direction));
                trail.widthMultiplier = 0.13f + loadFactor * 0.035f + slip * 0.025f;
                float disturbance = Mathf.Clamp01(loadFactor * (0.22f + slip * 0.65f));
                float grain = Mathf.PerlinNoise(suspension.tireHit.point.x * 2.3f,
                    suspension.tireHit.point.z * 2.3f);
                trail.startColor = new Color(0.34f, 0.31f, 0.27f,
                    Mathf.Lerp(0.045f, 0.20f, disturbance) * Mathf.Lerp(0.55f, 1f, grain));
                trail.endColor = new Color(0.39f, 0.36f, 0.32f, 0f);
                trail.emitting = true;
                hasTrackSegment[wheel] = true;
                airborneSeconds[wheel] = 0f;
            }
            else
            {
                trail.emitting = false;
                if (grounded) airborneSeconds[wheel] = 0f;
            }
            wasGrounded[wheel] = grounded;
        }

        TrailRenderer TrackAt(int wheel, int slot)
            => tracks[wheel * TrackSegmentsPerWheel + slot];

        static void PositionEmitter(ParticleSystem system, Vector3 point, Vector3 normal, Vector3 direction)
        {
            if (direction.sqrMagnitude < 0.001f) return;
            system.transform.SetPositionAndRotation(point + normal * 0.035f,
                Quaternion.LookRotation(direction, normal));
        }

        static void SetEmission(ParticleSystem system, float rate)
        {
            if (system == null) return;
            var emission = system.emission;
            emission.rateOverTime = rate;
            if (rate > 0.05f && !system.isPlaying) system.Play();
            else if (rate <= 0.05f && system.isPlaying)
                system.Stop(true, ParticleSystemStopBehavior.StopEmitting);
        }

        void ClearEffects()
        {
            for (int i = 0; i < tires.Length; i++)
            {
                activeTrackSlots[i] = 0;
                airborneSeconds[i] = 0f;
                hasTrackSegment[i] = false;
                wasGrounded[i] = false;
                if (dustSystems[i] != null) dustSystems[i].Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
                if (gravelSystems[i] != null) gravelSystems[i].Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
                for (int slot = 0; slot < TrackSegmentsPerWheel; slot++)
                {
                    TrailRenderer trail = TrackAt(i, slot);
                    if (trail == null) continue;
                    trail.emitting = false;
                    trail.Clear();
                }
            }
        }
    }

    public sealed class RallyBrakeLightController : MonoBehaviour
    {
        static readonly int EmissionColor = Shader.PropertyToID("_EmissionColor");
        readonly List<Material> lenses = new List<Material>(2);
        VehicleController vehicle;
        float brakeLevel;

        public void Configure(VehicleController target)
        {
            vehicle = target;
            lenses.Clear();
            foreach (MeshRenderer renderer in vehicle.GetComponentsInChildren<MeshRenderer>(true))
            {
                foreach (Material material in renderer.sharedMaterials)
                {
                    if (material == null || material.name.IndexOf("Tail", StringComparison.OrdinalIgnoreCase) < 0 ||
                        !material.HasProperty(EmissionColor)) continue;
                    material.EnableKeyword("_EMISSION");
                    lenses.Add(material);
                }
            }
        }

        void LateUpdate()
        {
            if (vehicle == null || lenses.Count == 0) return;
            brakeLevel = Mathf.MoveTowards(brakeLevel, vehicle.brakeInput, Time.unscaledDeltaTime * 9f);
            Color emission = Color.white * Mathf.Lerp(0.12f, 1.35f, brakeLevel);
            for (int i = 0; i < lenses.Count; i++)
                if (lenses[i] != null) lenses[i].SetColor(EmissionColor, emission);
        }
    }
}
