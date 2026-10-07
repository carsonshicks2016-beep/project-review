using UnityEngine;
using System.Threading;

namespace Audio
{
    /// <summary>
    /// Shared plumbing for a source that generates its own samples.
    ///
    /// Sound comes out of an AudioClip created in STREAMING mode, not out of
    /// OnAudioFilterRead. Two reasons. A streaming clip plays through an ordinary
    /// AudioSource, so distance attenuation and panning come for free; and it needs no
    /// dummy clip, whereas a filter chain only runs on a source that is already playing,
    /// which is why the usual OnAudioFilterRead recipe involves feeding it silence.
    ///
    /// THREADING, and this is the part that bites. <see cref="Render"/> runs on the audio
    /// thread. It must not touch the Unity API — no transforms, no Time, no GetComponent,
    /// no logging. Everything it needs is copied into plain float fields by
    /// <see cref="ReadState"/> on the main thread, once a frame. Those fields are marked
    /// volatile so the reads are not hoisted; a 32-bit aligned float cannot tear, so the
    /// worst case is one buffer built from values captured a millisecond apart from each
    /// other. That is inaudible, and it is much better than taking a lock on the audio
    /// thread, which is how you get dropouts.
    /// </summary>
    [DefaultExecutionOrder(250)]
    [RequireComponent(typeof(AudioSource))]
    public abstract class ProceduralAudio : MonoBehaviour
    {
        [Tooltip("Master level for this source.")]
        [Range(0f, 1f)] public float volume = 0.8f;

        [Tooltip("0 is heard everywhere at full level, 1 is fully positional.")]
        [Range(0f, 1f)] public float spatialBlend = 0.85f;

        [Tooltip("Metres over which the source falls silent.")]
        public float maxDistance = 140f;

        /// <summary>Output rate of the audio thread. Read once, on the main thread.</summary>
        protected int sampleRate { get; private set; } = 48000;

        AudioSource source;
        AudioClip clip;
        bool live;
        int resetRequested, resetApplied;
        float resetGain = 1f;
        bool resetting;
        public AudioSource Source => source;
        public bool IsLive => live;
        protected int Generation => Volatile.Read(ref resetRequested);
        protected int AudioGeneration => resetApplied;
        protected bool ResetInProgress => resetting || resetApplied != Volatile.Read(ref resetRequested);

        // Main thread writes, audio thread reads. See the threading note above.
        volatile float gateTarget;
        float gateLevel;

        uint rng = 0x9E3779B9u;

        /// <summary>
        /// Give each source its own noise stream.
        ///
        /// Without this every instance starts from the same constant and runs the same
        /// xorshift, so engine, tyres and wind all draw the SAME numbers. Three correlated
        /// noise sources do not sum to a fuller texture — they sum to one source at triple
        /// amplitude, which is a very different and much thinner sound.
        /// </summary>
        void SeedNoise()
        {
            uint seed = (uint)GetInstanceID() * 2654435761u + 0x9E3779B9u;
            rng = seed == 0u ? 0x9E3779B9u : seed;   // xorshift is stuck at zero forever
        }

        // ══════════════════════════════════════════════════════════════
        //  SUBCLASS CONTRACT
        // ══════════════════════════════════════════════════════════════

        /// <summary>Main thread, once a frame. Copy whatever Render will need into fields.</summary>
        protected abstract void ReadState();

        /// <summary>
        /// Audio thread. Fill the buffer with mono samples in roughly [-1, 1]. The master
        /// level and the gate ramp are applied afterwards, so ignore both.
        /// </summary>
        protected abstract void Render(float[] data);

        /// <summary>Called on the main thread once the sample rate is known.</summary>
        protected virtual void Configure() { }
        protected virtual void ReadFixedState() { }
        protected virtual void ResetAudioState() => Configure();
        protected virtual void ResetMainState() { }

        public void ResetPlayback()
        {
            ResetMainState();
            Interlocked.Increment(ref resetRequested);
        }

        // ══════════════════════════════════════════════════════════════
        //  LIFECYCLE
        // ══════════════════════════════════════════════════════════════

        protected virtual void OnEnable()
        {
            // Guarded because a script reload can run OnEnable outside play mode, and
            // batch mode has no device to allocate a clip against.
            if (!Application.isPlaying || Application.isBatchMode ||
                (Core.ML.LabRuntime.Enabled && !Core.ML.LabRuntime.Config.viewer)) return;

            int rate = AudioSettings.outputSampleRate;
            sampleRate = rate > 0 ? rate : 48000;
            SeedNoise();
            Configure();
            resetApplied = resetRequested;
            resetGain = 1f;
            resetting = false;

            source = GetComponent<AudioSource>();
            source.playOnAwake = false;
            source.loop = true;
            source.spatialBlend = spatialBlend;
            source.rolloffMode = AudioRolloffMode.Linear;
            source.minDistance = 4f;
            source.maxDistance = maxDistance;

            // Doppler off, deliberately. The chase camera lags the car under acceleration
            // (SpectatorDirector smooths its follow), so camera and car have a relative
            // velocity that has nothing to do with the world — and Unity would faithfully
            // render that as pitch warble every time the throttle moved.
            source.dopplerLevel = 0f;

            // stream:true is load-bearing. With it false the callback is invoked once to
            // fill a fixed buffer and never called again.
            clip = AudioClip.Create(GetType().Name, sampleRate, 1, sampleRate, true, OnAudioRead);
            source.clip = clip;
            source.Play();
            live = true;
        }

        protected virtual void OnDisable()
        {
            live = false;
            if (source != null) { source.Stop(); source.clip = null; }
            if (clip != null) { Destroy(clip); clip = null; }
        }

        void Update()
        {
            if (!live) return;
            gateTarget = AudioGate.ShouldRun ? 1f : 0f;
            ReadState();
        }

        void FixedUpdate()
        {
            if (live && AudioGate.ShouldRun) ReadFixedState();
        }

        void OnAudioRead(float[] data)
        {
            // ─── audio thread from here down ───
            float target = gateTarget;
            if (resetApplied != Volatile.Read(ref resetRequested)) resetting = true;
            if (resetting && resetGain <= 0f)
            {
                ResetAudioState();
                resetApplied = Volatile.Read(ref resetRequested);
                resetting = false;
            }

            // Silent AND free: while a trainer is attached the synthesis never runs at all,
            // rather than running and being multiplied by zero.
            if (gateLevel < 1e-4f && target < 1e-4f)
            {
                if (resetting)
                {
                    ResetAudioState(); resetApplied = Volatile.Read(ref resetRequested);
                    resetting = false; resetGain = 0;
                }
                System.Array.Clear(data, 0, data.Length);
                return;
            }

            Render(data);

            float coef = SmoothingCoef(0.03f);   // ~30 ms, so gating never clicks
            for (int i = 0; i < data.Length; i++)
            {
                gateLevel += (target - gateLevel) * coef;
                resetGain = Mathf.Clamp01(resetGain + (resetting ? -1f : 1f) / (sampleRate * .02f));
                data[i] *= gateLevel * volume * resetGain;
            }
        }

        // ══════════════════════════════════════════════════════════════
        //  DSP HELPERS
        // ══════════════════════════════════════════════════════════════

        /// <summary>Per-sample coefficient for a one-pole that settles in about this long.</summary>
        protected float SmoothingCoef(float seconds) =>
            1f - Mathf.Exp(-1f / Mathf.Max(1e-4f, seconds * sampleRate));

        /// <summary>Per-sample coefficient for a one-pole lowpass at this corner frequency.</summary>
        protected float LowpassCoef(float hz) =>
            Mathf.Clamp01(1f - Mathf.Exp(-2f * Mathf.PI * Mathf.Max(1f, hz) / sampleRate));

        /// <summary>White noise in [-1, 1]. Audio thread only — the state is not shared.</summary>
        protected float Noise()
        {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            return (rng & 0xFFFFFFu) / 8388607.5f - 1f;
        }

        /// <summary>
        /// Cubic soft clip. The resonators below can ring past full scale on a hard
        /// throttle transient, and hard clipping there is a click; this folds the peaks
        /// over smoothly instead. Unity gain is restored by the 1.5x.
        /// </summary>
        protected static float SoftClip(float x)
        {
            if (x <= -1f) return -1f;
            if (x >=  1f) return  1f;
            return 1.5f * (x - x * x * x / 3f);
        }

        /// <summary>A one-pole lowpass. Holds its own state; copy semantics are the point.</summary>
        protected struct OnePole
        {
            float z;
            public float Process(float x, float coef)
            {
                z += (x - z) * coef;
                return z;
            }
        }

        /// <summary>
        /// A two-pole resonator, used here as a fixed formant.
        ///
        /// This is what makes the engine an engine rather than a pitch-shifted loop. An
        /// exhaust system's resonances do not move when the revs rise — only the rate at
        /// which the cylinders excite them does. Holding these frequencies still while the
        /// pulse train speeds up is a source-filter model, the same shape as a vowel, and
        /// it is why a real engine changes character across its range instead of merely
        /// getting higher.
        /// </summary>
        protected struct Resonator
        {
            float y1, y2, a1, a2, norm;

            public void Set(float freq, float bandwidth, int sampleRate)
            {
                float r = Mathf.Exp(-Mathf.PI * bandwidth / sampleRate);
                a1 = 2f * r * Mathf.Cos(2f * Mathf.PI * freq / sampleRate);
                a2 = -r * r;
                // Peak gain of a two-pole runs to 1/(1-r), which for a narrow band is a
                // factor of hundreds. Normalise on the way in or the first impulse
                // saturates everything downstream.
                norm = 1f - r;
            }

            public float Process(float x)
            {
                float y = x * norm + a1 * y1 + a2 * y2;
                y2 = y1; y1 = y;
                return y;
            }
        }
    }
}
