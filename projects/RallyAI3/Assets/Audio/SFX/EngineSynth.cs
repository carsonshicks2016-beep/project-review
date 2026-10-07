using UnityEngine;
using Core.Physics;

namespace Audio
{
    /// <summary>
    /// The engine, synthesised rather than sampled.
    ///
    /// WHY NOT A SAMPLE. This used to pitch-shift one AudioSource from RPM. That approach
    /// cannot work over a real rev range: idle is 1400 and the limiter is 7600, so the
    /// clip has to be stretched better than five to one, and it arrives as a mosquito at
    /// one end and a tractor at the other. Worse, shifting a loop scales its formants too,
    /// which is exactly the thing a real engine does NOT do.
    ///
    /// WHAT THIS DOES INSTEAD. A source-filter model, the same shape as synthesised
    /// speech. The SOURCE is a train of exhaust pulses whose rate follows the crank; the
    /// FILTER is a set of fixed resonances standing in for the exhaust system. Revs move
    /// the pulse rate, which is heard as pitch. The resonances never move, which is heard
    /// as the engine keeping its voice. At low revs you make out individual thumps and at
    /// high revs they fuse into a tone, all of it out of one mechanism — and that
    /// transition is most of what makes an engine sound alive.
    ///
    /// THE BOXER RUMBLE. A four-stroke four fires four times per two crank revolutions,
    /// so an even-firing engine puts a pulse every quarter of a cycle. The classic
    /// Impreza's unequal-length headers meant pulses from one bank had further to travel
    /// and arrived late, so what reaches the tailpipe is unevenly spaced even though the
    /// firing is not. That unevenness repeats once per CYCLE rather than once per pulse,
    /// which plants energy an octave below the firing frequency — and that subharmonic is
    /// the burble. <see cref="unevenFiring"/> is that offset. Set it to zero and this
    /// becomes an ordinary inline four.
    /// </summary>
    public class EngineSynth : ProceduralAudio
    {
        /// <summary>
        /// Which voice renders. Rally is the default; Legacy is the original voice, kept for
        /// comparison. F8 toggles between them in play mode, with a 20 ms crossfade through
        /// silence so the switch does not click.
        /// </summary>
        public enum VoiceProfile { Rally, Legacy }
        public VoiceProfile voice = VoiceProfile.Rally;
        public VehicleController vehicle;

        readonly RallyEngineVoice rallyVoice = new RallyEngineVoice();
        bool activeRally = true;       // audio thread: the voice actually rendering
        float profileGain = 1f;        // audio thread: crossfade level

        /// <summary>
        /// Where this voice sits in the mix. Lives here rather than in the builder so that
        /// the three levels are comparable in one place, and so the offline balance check
        /// reads the same numbers the car is actually assembled with.
        /// </summary>
        public const float DefaultVolume = AudioMixProfile.EngineVolume;
        public volatile float perspectiveInduction = 1f;
        readonly AudioEventQueue events = new AudioEventQueue();
        int observedGear;
        volatile bool observed;
        bool observedShift;
        float observedThrottle, observedBoost;
        public int DroppedEvents => events.Dropped;
        public int ShiftEvents => rallyVoice.ShiftEvents;
        public int ReleaseEvents => rallyVoice.ReleaseEvents;

        [Header("Character")]
        [Tooltip("How far the second and fourth pulses arrive ahead of even spacing, as a " +
                 "fraction of the firing cycle. 0 is an even-fire four; 0.06 is the boxer " +
                 "burble. Past about 0.1 it stops being a rumble and starts being a misfire.")]
        [Range(0f, 0.12f)] public float unevenFiring = 0.06f;

        [Tooltip("Level at closed throttle, relative to full load.")]
        [Range(0f, 1f)] public float idleLevel = 0.30f;

        [Tooltip("Induction and turbo rush. Rises with throttle and revs.")]
        [Range(0f, 1f)] public float inductionLevel = 0.35f;

        [Tooltip("Anti-lag crackle on a closed throttle. The car runs anti-lag, so the " +
                 "bangs on overrun are not decoration — they are what it would do.")]
        [Range(0f, 1f)] public float antiLagLevel = 0.5f;

        // ── Exhaust resonances, in Hz, with their bandwidths. Fixed on purpose: see the
        //    class note. Low is the boom you feel, mid is the body of the sound, high is
        //    the rasp that only shows up under load.
        const float F1 = 78f,  BW1 = 55f;
        const float F2 = 210f, BW2 = 130f;
        const float F3 = 620f, BW3 = 400f;

        /// <summary>Pulses per second at which the level compensation is unity — 3000 rpm.</summary>
        const float RefPulseRate = 100f;

        // ── Snapshot: main thread writes, audio thread reads ─────────────
        volatile float sRpm = 900f;
        volatile float sNormRpm;
        volatile float sThrottle;
        volatile float sBoost;
        volatile float sShift;
        volatile int   sGear;
        volatile bool  sAntiLag;
        volatile bool  useRally = true;

        // ── Audio thread state ───────────────────────────────────────────
        float cyclePhase;          // 0..1 across one full four-stroke cycle (two crank revs)
        float cycleRate;           // smoothed, Hz
        float level, brightness;   // smoothed
        Resonator r1, r2, r3;
        OnePole inductionLp, crackleLp;
        float crackleEnv;
        int crackleCountdown;
        float dcPole;

        // No two cylinders are quite equal, and a perfectly uniform pulse train reads as
        // synthetic immediately. These are fixed rather than random so the engine has a
        // consistent character instead of shimmering.
        static readonly float[] CylinderGain = { 1.00f, 0.91f, 0.97f, 0.88f };

        protected override void Configure()
        {
            rallyVoice.Reset(sampleRate);
            useRally = activeRally = voice == VoiceProfile.Rally;
            r1.Set(F1, BW1, sampleRate);
            r2.Set(F2, BW2, sampleRate);
            r3.Set(F3, BW3, sampleRate);
            cycleRate = 900f / 120f;

            // Corner the DC blocker at 20 Hz, derived from the rate rather than written as
            // a literal pole. The obvious 0.995 is a 38 Hz corner at 48 kHz, which is not a
            // DC blocker at all — it sits right on top of the half-order burble (50 Hz at
            // 3000 rpm) and takes 2 dB off the one thing this engine exists to produce.
            dcPole = 1f - 2f * Mathf.PI * 20f / sampleRate;
        }

        protected override void ReadState()
        {
            if (vehicle == null) return;
            if (Input.GetKeyDown(KeyCode.F8))
                voice = voice == VoiceProfile.Rally ? VoiceProfile.Legacy : VoiceProfile.Rally;
            useRally  = voice == VoiceProfile.Rally;
            sGear     = vehicle.currentGear;
            sAntiLag  = vehicle.antiLag;
            sRpm      = vehicle.currentRpm;
            sNormRpm  = vehicle.NormalisedRpm;
            sThrottle = vehicle.throttleInput;
            sBoost    = vehicle.currentBoost;
            sShift    = vehicle.isShifting ? 1f : 0f;
        }

        protected override void ReadFixedState()
        {
            if (vehicle == null) return;
            if (observed)
            {
                if (vehicle.currentGear != observedGear || (vehicle.isShifting && !observedShift))
                    events.Enqueue(VehicleSoundEvent.Shift, 1, Generation);
                if (observedThrottle > .45f && vehicle.throttleInput < .15f && observedBoost > .2f)
                    events.Enqueue(VehicleSoundEvent.TurboRelease, observedBoost, Generation);
            }
            observed = true;
            observedGear = vehicle.currentGear;
            observedShift = vehicle.isShifting;
            observedThrottle = vehicle.throttleInput;
            observedBoost = vehicle.currentBoost;
            sGear = vehicle.currentGear; sAntiLag = vehicle.antiLag;
            sRpm = vehicle.currentRpm; sNormRpm = vehicle.NormalisedRpm;
            sThrottle = vehicle.throttleInput; sBoost = vehicle.currentBoost;
            sShift = vehicle.isShifting ? 1f : 0f;
        }

        protected override void ResetMainState() { observed = false; }
        protected override void ResetAudioState()
        {
            Configure();
            cyclePhase = level = brightness = crackleEnv = 0;
            crackleCountdown = 0;
            inductionLp = crackleLp = default(OnePole);
        }

        protected override void Render(float[] data)
        {
            // Switch voices only once the fade-out has reached silence.
            bool switching = useRally != activeRally;
            if (switching && profileGain <= 0f) { activeRally = useRally; switching = false; }

            if (activeRally)
            {
                // Runtime events are captured every physics tick; offline callers retain snapshot detection.
                rallyVoice.SetState(sRpm, sThrottle, sBoost, sShift > 0.5f, sGear, sAntiLag,
                    unevenFiring, data.Length, !observed);
                rallyVoice.Induction = perspectiveInduction;
                if (!ResetInProgress)
                    while (events.TryDequeue(out var ev))
                        if (ev.generation == AudioGeneration) rallyVoice.Trigger(ev.kind, ev.strength);
                for (int i = 0; i < data.Length; i++) data[i] = rallyVoice.Next();
            }
            else
            {
                while (events.TryDequeue(out var ignored)) { }
                RenderLegacy(data);
            }

            float step = 1f / (sampleRate * 0.02f);
            for (int i = 0; i < data.Length; i++)
            {
                profileGain = Mathf.Clamp01(profileGain + (switching ? -step : step));
                data[i] *= profileGain;
            }
        }

        void RenderLegacy(float[] data)
        {
            // ─── audio thread ───
            float rpm      = Mathf.Max(300f, sRpm);
            float normRpm  = Mathf.Clamp01(sNormRpm);
            float throttle = Mathf.Clamp01(sThrottle);
            float shifting = sShift;

            // A shift is a torque cut, so treat it as a closed throttle for both the level
            // and the crackle. Cutting the pulse train for those few tens of milliseconds
            // is a small thing that does more for realism than another octave of harmonics.
            float effThrottle = throttle * (1f - shifting);

            // One full four-stroke cycle is two crank revolutions: rpm/60/2 per second.
            float rateTarget = rpm / 120f;

            // Loud under load, quiet on a trailing throttle — and compensated for revs
            // rather than boosted by them.
            //
            // The low resonator rings for about 6 ms. At idle the exhaust pulses are 21 ms
            // apart, so each one decays before the next arrives; at the limiter they are
            // 3.9 ms apart, so they overlap and add COHERENTLY, and the resonator builds up
            // roughly in proportion to the pulse rate. Left alone that is a 28 dB climb
            // from idle to the limiter, which is not an engine coming on song, it is a
            // synth pinned against the clipper for the whole top half of the rev range.
            //
            // Dividing by rate^0.85 cancels most of it and leaves roughly 2 dB of growth
            // from idle to the limiter at a fixed throttle — the rest of the range comes
            // from load, which is where it should come from. The build-up is tamed, not
            // removed: it is exactly why a real engine gains body as it revs, and it is
            // still the thing doing the work here.
            float pulseRate = Mathf.Max(20f, rpm / 30f);
            float overlapComp = Mathf.Pow(RefPulseRate / pulseRate, 0.85f);
            float levelTarget = Mathf.Lerp(idleLevel, 1f, effThrottle) * overlapComp;

            // Rasp is a load effect, not a rev effect — an engine at 6000 rpm off the
            // throttle is not making the noise it makes at 6000 rpm pulling.
            float brightTarget = 0.25f + 0.75f * effThrottle;

            bool overrun = effThrottle < 0.15f && normRpm > 0.30f;

            float rateCoef  = SmoothingCoef(0.020f);   // revs move fast; do not lag them
            float levelCoef = SmoothingCoef(0.040f);
            float indCoef   = LowpassCoef(Mathf.Lerp(700f, 3200f, normRpm));
            float crackCoef = LowpassCoef(1800f);

            float uneven = unevenFiring;
            float p0 = 0f, p1 = 0.25f - uneven, p2 = 0.5f, p3 = 0.75f - uneven;

            for (int i = 0; i < data.Length; i++)
            {
                cycleRate  += (rateTarget  - cycleRate)  * rateCoef;
                level      += (levelTarget - level)      * levelCoef;
                brightness += (brightTarget - brightness) * levelCoef;

                float prev = cyclePhase;
                cyclePhase += cycleRate / sampleRate;
                if (cyclePhase >= 1f) cyclePhase -= 1f;

                // ── Source: the exhaust pulse train ──
                float excite = 0f;
                if (Crossed(prev, cyclePhase, p0)) excite += CylinderGain[0];
                if (Crossed(prev, cyclePhase, p1)) excite += CylinderGain[1];
                if (Crossed(prev, cyclePhase, p2)) excite += CylinderGain[2];
                if (Crossed(prev, cyclePhase, p3)) excite += CylinderGain[3];

                // A real pulse is not a clean impulse — the charge is not identical every
                // time round. A little roughness on each one keeps the tone from going
                // glassy at high revs.
                if (excite > 0f) excite *= level * (0.85f + 0.15f * Noise());

                // ── Anti-lag: unburnt fuel lighting off in the exhaust on a closed
                //    throttle. Scheduled rather than rolled per sample, so the pops come
                //    at a controllable rate instead of as a wash of noise.
                float crackleDrive = 0f;
                if (antiLagLevel > 0f)
                {
                    if (--crackleCountdown <= 0)
                    {
                        if (overrun || shifting > 0.5f)
                        {
                            float intensity = Mathf.Clamp01(0.35f + 0.65f * sBoost) * normRpm;
                            crackleEnv = intensity * antiLagLevel;
                            float gap = Mathf.Lerp(0.28f, 0.05f, intensity);
                            crackleCountdown = (int)(sampleRate * gap * (0.55f + 0.9f * Random01()));
                        }
                        else
                        {
                            crackleCountdown = sampleRate / 20;   // idle the scheduler
                        }
                    }

                    if (crackleEnv > 1e-4f)
                    {
                        // Deliberately far smaller than a cylinder pulse, because it does
                        // not arrive as one. A pulse is an impulse and the narrow 78 Hz
                        // resonator answers it with a peak of about 0.35; this is a BURST,
                        // and sustained noise into that same resonator has a gain near 2.8
                        // — eight times as much. Matching the two by eye is how the
                        // overrun ends up clipping while the pulls sound fine.
                        crackleDrive = crackleLp.Process(Noise(), crackCoef) * crackleEnv * 0.45f;
                        crackleEnv *= 0.9965f;                    // ~8 ms crack, not a puff
                    }
                }

                // ── Filter: the fixed exhaust resonances ──
                float drive = excite + crackleDrive;
                float body = r1.Process(drive) * 1.00f
                           + r2.Process(drive) * 0.75f
                           + r3.Process(drive) * (0.20f + 0.55f * brightness);

                // ── Induction. Broadband rush that opens up with throttle and revs; it
                //    is what fills the gaps between pulses once they are far apart.
                float induction = inductionLp.Process(Noise(), indCoef)
                                * inductionLevel * effThrottle * (0.25f + 0.75f * normRpm);

                float outSample = body * 0.75f + induction * 0.42f;

                // DC block. A pulse train has a standing offset, and at idle the cycle
                // fundamental is under 12 Hz — inaudible, but it eats headroom that the
                // audible part needs.
                dcY = outSample - dcX + dcPole * dcY;
                dcX = outSample;

                data[i] = SoftClip(dcY);
            }
        }

        float dcX, dcY;

        // Cheap uniform [0,1) off the shared noise generator. Audio thread only.
        float Random01() => Noise() * 0.5f + 0.5f;

        /// <summary>
        /// Did the phase pass <paramref name="target"/> during this sample? The step is
        /// always far below one cycle — 63 Hz against a 48 kHz rate at the limiter — so
        /// the only case needing care is the wrap, where prev sits just under 1 and cur
        /// just above 0.
        /// </summary>
        static bool Crossed(float prev, float cur, float target)
        {
            if (cur >= prev) return prev < target && cur >= target;
            return target > prev || target <= cur;
        }
    }
}
