using UnityEngine;
using Core.Physics;

namespace Audio
{
    public sealed class ImpactContactRelay : MonoBehaviour
    {
        public ImpactSynth voice;
        void OnCollisionEnter(Collision contact)
        {
            if (voice != null) voice.Contact(contact);
        }
    }

    public sealed class ImpactSynth : ProceduralAudio
    {
        public VehicleController vehicle;
        readonly AudioEventQueue events = new AudioEventQueue();
        float thump, crash, airTime, fallingSpeed, lastContact = -10;
        float thumpDecay, crashDecay;
        bool initialized, grounded;
        Resonator chassis, panel;
        OnePole muffled;
        public int DroppedEvents => events.Dropped;

        protected override void Configure()
        {
            volume = AudioMixProfile.ImpactVolume;
            chassis.Set(105, 100, sampleRate); panel.Set(1800, 700, sampleRate);
            thumpDecay = Mathf.Exp(-1f / (.045f * sampleRate));
            crashDecay = Mathf.Exp(-1f / (.065f * sampleRate));
        }
        protected override void ReadState() { }
        protected override void ReadFixedState()
        {
            if (vehicle == null || vehicle.Suspensions == null) return;
            int contacts = 0; float load = 0;
            foreach (var wheel in vehicle.Suspensions)
                if (wheel != null && wheel.isGrounded) { contacts++; load += wheel.normalLoad; }
            bool now = contacts > 0;
            var body = vehicle.GetComponent<Rigidbody>();
            if (!now)
            {
                airTime += Time.fixedDeltaTime;
                if (body != null) fallingSpeed = Mathf.Max(fallingSpeed, -body.linearVelocity.y);
            }
            if (initialized && now && !grounded && airTime > .06f && fallingSpeed > .8f)
            {
                float strength = Mathf.Clamp01(fallingSpeed / 12f +
                    Mathf.Max(0, load / Mathf.Max(1, vehicle.mass * 9.81f) - 1) * .1f);
                events.Enqueue(VehicleSoundEvent.Landing, strength, Generation);
            }
            if (now) { airTime = 0; fallingSpeed = 0; }
            grounded = now; initialized = true;
        }
        public void Contact(Collision contact)
        {
            if (!AudioGate.ShouldRun || vehicle == null || Time.unscaledTime - lastContact < .12f) return;
            float intensity = Mathf.Max(contact.relativeVelocity.magnitude / 18f,
                contact.impulse.magnitude / Mathf.Max(1, vehicle.mass * 8f));
            if (intensity < .08f) return;
            lastContact = Time.unscaledTime;
            events.Enqueue(VehicleSoundEvent.Collision, Mathf.Clamp01(intensity), Generation);
        }
        protected override void ResetMainState()
        {
            initialized = grounded = false; airTime = fallingSpeed = 0; lastContact = -10;
        }
        protected override void ResetAudioState()
        {
            thump = crash = 0; chassis = panel = default(Resonator); muffled = default(OnePole);
            Configure();
        }
        protected override void Render(float[] data)
        {
            while (!ResetInProgress && events.TryDequeue(out var ev))
            {
                if (ev.generation != AudioGeneration) continue;
                if (ev.kind == VehicleSoundEvent.Landing) thump = Mathf.Max(thump, ev.strength);
                else crash = Mathf.Max(crash, ev.strength);
            }
            for (int i = 0; i < data.Length; i++)
            {
                float noise = Noise();
                data[i] = SoftClip(chassis.Process(muffled.Process(noise, .08f)) * thump * .85f +
                    panel.Process(noise) * crash * .25f + noise * crash * .06f);
                thump *= thumpDecay; crash *= crashDecay;
            }
        }
    }
}
