using UnityEngine;
using UnityEngine.Rendering;

namespace Core.Environment
{
    /// <summary>
    /// The light a forest stage is seen in: a low, warm late-afternoon sun, a sky to match,
    /// haze in the distance, and glare when the camera looks toward the sun.
    ///
    /// Look only, like the rest of the dressing. Nothing the AI observes depends on light —
    /// it sees raycasts and vectors, and headless training never renders — so this changes
    /// the picture and nothing else. It is applied at load rather than saved in the scene so
    /// that it travels with the dressing to frozen courses and the offline review.
    ///
    /// THE SUN'S DIRECTION is set from the stage, not fixed in the world. Stages run broadly
    /// down +Z but each one bends its own way, so a fixed sun is sometimes ahead and
    /// sometimes behind the camera for a whole run. This puts it low and a little off to one
    /// side AHEAD of the stage's overall heading. The chase camera then spends much of a
    /// run driving toward it: long shadows reaching back across the road, the trees backlit
    /// and glowing at their edges, and the sun strobing through the canopy. That is the
    /// forest-stage shot.
    /// </summary>
    public static class StageAtmosphere
    {
        const string SkyPath = "StageDressing/Sky";
        const string GlarePath = "StageDressing/SunGlare";

        const float SunElevation = 23f;   // degrees above the horizon
        const float SunOffset = 32f;      // degrees left of the stage's heading

        static readonly Color SunColor = new Color(1.0f, 0.86f, 0.68f);
        static readonly Color Haze = new Color(0.68f, 0.71f, 0.74f);

        public static void Apply(TrackGenerator track)
        {
            Light sun = RenderSettings.sun;
            if (sun == null)
                foreach (var l in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
                    if (l.type == LightType.Directional) { sun = l; break; }

            if (sun != null)
            {
                Vector3 heading = Heading(track);
                Vector3 toSun = Quaternion.AngleAxis(-SunOffset, Vector3.up) * heading;
                toSun = (toSun * Mathf.Cos(SunElevation * Mathf.Deg2Rad) + Vector3.up * Mathf.Sin(SunElevation * Mathf.Deg2Rad)).normalized;
                sun.transform.rotation = Quaternion.LookRotation(-toSun);
                sun.color = SunColor;
                sun.intensity = 1.2f;
                sun.shadows = LightShadows.Soft;
                sun.shadowStrength = 0.82f;
                RenderSettings.sun = sun;
            }

            // Cool from the sky, warm bounce from the ground. Generous, because with the sun
            // this low the forest shades most of the road most of the time, and shade has to
            // stay warm and readable — a stage that goes grey-blue whenever the trees close in
            // looks like dusk, not a sunny afternoon.
            Color skyFill = new Color(0.70f, 0.70f, 0.70f);
            Color horizonFill = new Color(0.60f, 0.56f, 0.49f);
            Color groundFill = new Color(0.30f, 0.26f, 0.20f);
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Custom;
            RenderSettings.ambientProbe = GradientProbe(skyFill, horizonFill, groundFill);

            // Thicker than before: with a forest either side, haze is what tells you the far
            // end of a straight is far away.
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogDensity = 0.0042f;
            RenderSettings.fogColor = Haze;

            var sky = Resources.Load<Material>(SkyPath);
            if (sky != null)
            {
                sky.SetColor("_Horizon", Haze);
                RenderSettings.skybox = sky;
            }


            Camera camera = Camera.main;
            var glare = Resources.Load<Material>(GlarePath);
            if (camera != null && glare != null)
            {
                var effect = camera.GetComponent<SunGlare>();
                if (effect == null) effect = camera.gameObject.AddComponent<SunGlare>();
                effect.material = glare;
                // Set here as well as in the effect's OnEnable, which does not run in the
                // offline review. The glare and the dust's soft edges both read it.
                camera.depthTextureMode |= DepthTextureMode.Depth;
                effect.sunColor = SunColor;
            }
        }

        /// <summary>
        /// A sky / horizon / ground ambient gradient written straight into the ambient probe.
        ///
        /// The world shaders read ambient through ShadeSH9, which samples this probe. Setting
        /// Unity's trilight colours from script does not rebuild it — that is deferred to
        /// DynamicGI, which in the offline renders never got the chance — so the colours
        /// changed and the picture did not. Building the probe here applies it the moment it
        /// is set, in the game and in the review alike. A constant term for the horizon, and
        /// an up/down lobe pulling toward the sky above and the ground below.
        /// </summary>
        static SphericalHarmonicsL2 GradientProbe(Color sky, Color horizon, Color ground)
        {
            var sh = new SphericalHarmonicsL2();
            sh.AddAmbientLight(horizon);
            Color lift = (sky - ground) * 0.5f;
            sh.AddDirectionalLight(Vector3.up, lift, 1.15f);
            return sh;
        }

        /// <summary>The stage's overall direction, start to finish, flattened.</summary>
        static Vector3 Heading(TrackGenerator track)
        {
            if (track == null || track.waypoints.Count < 2) return Vector3.forward;
            Vector3 a = track.transform.TransformPoint(track.waypoints[0]);
            Vector3 b = track.transform.TransformPoint(track.waypoints[track.waypoints.Count - 1]);
            Vector3 d = b - a;
            d.y = 0f;
            return d.sqrMagnitude > 1f ? d.normalized : Vector3.forward;
        }
    }

    /// <summary>
    /// Runs the glare-and-grade pass over the finished frame. See SunGlare.shader for how
    /// the glare measures the sun through the trees.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class SunGlare : MonoBehaviour
    {
        public Material material;
        public Color sunColor = Color.white;
        [Range(0f, 2f)] public float strength = 1f;
        [Range(0f, 1f)] public float vignette = 0.28f;

        Camera cam;

        void OnEnable()
        {
            cam = GetComponent<Camera>();
            // The glare reads the depth buffer to see whether the sun is behind a tree.
            cam.depthTextureMode |= DepthTextureMode.Depth;
        }

        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            if (material == null) { Graphics.Blit(source, destination); return; }

            Light sun = RenderSettings.sun;
            Vector4 sunScreen = Vector4.zero;
            if (sun != null)
            {
                Vector3 toSun = -sun.transform.forward;
                Vector3 vp = cam.WorldToViewportPoint(cam.transform.position + toSun * 1000f);
                // Full strength looking at the sun, gone by about 70 degrees off it, and
                // fading as it leaves the frame so it never pops at the screen edge.
                float facing = Mathf.Clamp01((Vector3.Dot(cam.transform.forward, toSun) - 0.35f) / 0.65f);
                float edge = Mathf.Clamp01(1f - Mathf.Max(Mathf.Abs(vp.x - 0.5f), Mathf.Abs(vp.y - 0.5f)) * 2f + 0.6f);
                sunScreen = new Vector4(vp.x, vp.y, vp.z > 0f ? facing * edge : 0f, 0f);
            }

            material.SetVector("_SunScreen", sunScreen);
            material.SetColor("_SunColor", sunColor);
            material.SetFloat("_Aspect", cam.aspect);
            material.SetFloat("_GlareStrength", strength);
            material.SetFloat("_Vignette", vignette);
            Graphics.Blit(source, destination, material);
        }
    }
}
