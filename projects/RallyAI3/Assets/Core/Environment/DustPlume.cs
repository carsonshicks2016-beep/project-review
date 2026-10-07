using UnityEngine;
using UnityEngine.Rendering;
using Core.Physics;

namespace Core.Environment
{
    /// <summary>
    /// The dust a car throws up on gravel — on a dry forest stage, half of the picture.
    ///
    /// Emitted per wheel, at the contact patch, every frame. How much depends on the three
    /// things that actually make dust: how loose the surface under that tyre is (nothing on
    /// tarmac), how fast the car is going, and how much the tyre is sliding — a car sideways
    /// through a corner throws far more than one rolling straight. Rear wheels throw the most:
    /// they are driving, and the fronts' dust is half swallowed by the car.
    ///
    /// The puffs start about wheel-sized, billow to several metres, slow almost to a stop and
    /// hang in the air for seconds. That hang is what makes a dust trail read as a trail: the
    /// road behind the car stays hazy long after it has gone.
    ///
    /// Look only. Installed by StageDressing, which only runs with a graphics device, so a
    /// training worker never builds one. No collider, nothing the AI can sense.
    /// </summary>
    [RequireComponent(typeof(VehicleController))]
    public class DustPlume : MonoBehaviour
    {
        const string MaterialPath = "StageDressing/Dust";
        const int MaxParticles = 2500;

        // Particles per second, per wheel, before looseness and wheel weighting.
        const float PerMetrePerSecond = 1.7f;   // from speed, above a walking-pace threshold
        const float PerSlipMetre = 11f;         // from sliding
        const float SpeedThreshold = 3f;        // m/s below which rolling makes no dust

        static readonly Color DustColor = new Color(0.58f, 0.55f, 0.49f);

        VehicleController vehicle;
        ParticleSystem system;
        float[] carry = new float[4];
        System.Random rng = new System.Random(4271);

        void Awake() => EnsureBuilt();

        // Built on first use as well as in Awake: the offline review drives a plume in edit
        // mode, where Awake never runs.
        void EnsureBuilt()
        {
            if (vehicle == null) vehicle = GetComponent<VehicleController>();
            if (system == null) system = Build(transform);
        }

        void Update()
        {
            if (system == null || vehicle == null || vehicle.Suspensions == null) return;
            float dt = Time.deltaTime;
            if (dt <= 0f) return;

            Vector3 velocity = vehicle.Body != null ? vehicle.Body.linearVelocity : Vector3.zero;
            var corners = vehicle.Suspensions;
            if (carry.Length < corners.Length) carry = new float[corners.Length];

            for (int i = 0; i < corners.Length; i++)
            {
                var sus = corners[i];
                if (sus == null || !sus.isGrounded) continue;
                var tyre = sus.GetComponent<PacejkaTireModel>();
                float slip = tyre != null ? tyre.slipSpeed : 0f;
                EmitWheel(i, dt, sus.tireHit.point, velocity, slip, sus.surfaceLooseness, sus.IsFront ? 0.5f : 1f);
            }
        }

        /// <summary>
        /// Emit one wheel's dust for one step. Public so the offline review can drive a car
        /// along the road without physics and still get the trail it would have left.
        /// </summary>
        public void EmitWheel(int wheel, float dt, Vector3 contact, Vector3 velocity, float slip, float looseness, float weight)
        {
            EnsureBuilt();
            if (system == null) return;
            if (wheel >= carry.Length) System.Array.Resize(ref carry, wheel + 1);

            float speed = velocity.magnitude;
            float rate = looseness * weight
                       * (Mathf.Max(0f, speed - SpeedThreshold) * PerMetrePerSecond + slip * PerSlipMetre);
            carry[wheel] += rate * dt;

            // How thick each puff is follows how hard the ground is being worked, so a gentle
            // roll lays a thin haze and a slide lays a wall.
            float density = Mathf.Clamp01(speed / 28f + slip / 10f);

            while (carry[wheel] >= 1f)
            {
                carry[wheel] -= 1f;
                var p = new ParticleSystem.EmitParams
                {
                    position = contact + Vector3.up * 0.25f + Jitter(0.35f),
                    // Dragged along behind the car a little, kicked up, sprayed sideways.
                    velocity = velocity * 0.22f + Vector3.up * Range(0.9f, 2.2f) + Jitter(1.0f) + Jitter(0.25f * slip),
                    startSize = Range(1.4f, 2.6f) * (0.7f + 0.6f * looseness),
                    startLifetime = Range(3.5f, 5.8f),
                    rotation = Range(0f, 360f),
                    startColor = new Color(DustColor.r, DustColor.g, DustColor.b, Range(0.16f, 0.30f) * density),
                    applyShapeToPosition = false
                };
                system.Emit(p, 1);
            }
        }

        /// <summary>Advance the particles by hand. Only the offline review needs this.</summary>
        public void Advance(float dt) { if (system != null) system.Simulate(dt, true, false, false); }

        public void ClearTrail() { if (system != null) system.Clear(); for (int i = 0; i < carry.Length; i++) carry[i] = 0f; }

        float Range(float a, float b) => a + (b - a) * (float)rng.NextDouble();
        Vector3 Jitter(float r) => new Vector3(Range(-r, r), Range(-r, r) * 0.4f, Range(-r, r));

        /// <summary>The particle system, configured in code so it needs no prefab or scene asset.</summary>
        static ParticleSystem Build(Transform parent)
        {
            var material = Resources.Load<Material>(MaterialPath);
            if (material == null)
            {
                Debug.LogWarning($"[DustPlume] Resources/{MaterialPath} missing — no dust.");
                return null;
            }

            var go = new GameObject("DustPlume") { layer = 2 };
            go.transform.SetParent(parent, false);
            var ps = go.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);

            var main = ps.main;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.maxParticles = MaxParticles;
            main.playOnAwake = false;
            main.loop = true;
            main.gravityModifier = -0.015f;   // warm dust lifts, very slowly
            main.startSpeed = 0f;

            var emission = ps.emission;
            emission.enabled = false;         // every particle is emitted by hand, per wheel

            var shape = ps.shape;
            shape.enabled = false;

            // Billows fast at first and keeps spreading.
            var size = ps.sizeOverLifetime;
            size.enabled = true;
            size.size = new ParticleSystem.MinMaxCurve(1f, new AnimationCurve(
                new Keyframe(0f, 0.6f), new Keyframe(0.25f, 2.4f), new Keyframe(1f, 5.0f)));

            // Fades in quickly, thins out slowly.
            var color = ps.colorOverLifetime;
            color.enabled = true;
            var gradient = new Gradient();
            gradient.SetKeys(
                new[] { new GradientColorKey(Color.white, 0f), new GradientColorKey(Color.white, 1f) },
                new[] { new GradientAlphaKey(0f, 0f), new GradientAlphaKey(1f, 0.06f),
                        new GradientAlphaKey(0.55f, 0.45f), new GradientAlphaKey(0f, 1f) });
            color.color = gradient;

            // Air drag: the kick dies away and the cloud hangs.
            var limit = ps.limitVelocityOverLifetime;
            limit.enabled = true;
            limit.limit = 100f;
            limit.drag = 1.4f;

            // Turbulence, so the cloud curls instead of drifting as a block.
            var noise = ps.noise;
            noise.enabled = true;
            noise.strength = 0.55f;
            noise.frequency = 0.22f;
            noise.scrollSpeed = 0.15f;
            noise.damping = true;

            var rotation = ps.rotationOverLifetime;
            rotation.enabled = true;
            rotation.z = new ParticleSystem.MinMaxCurve(-0.4f, 0.4f);

            var renderer = go.GetComponent<ParticleSystemRenderer>();
            renderer.sharedMaterial = material;
            renderer.renderMode = ParticleSystemRenderMode.Billboard;
            renderer.sortMode = ParticleSystemSortMode.Distance;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.maxParticleSize = 3f;   // fraction of screen; a puff at the lens can fill it

            ps.Play();
            return ps;
        }
    }
}
