using System;

namespace Audio
{
    /// <summary>
    /// The rally voice for <see cref="EngineSynth"/>. Same source-filter idea as the legacy
    /// voice — a crank-locked pulse train into fixed exhaust resonances — tuned for a
    /// competition car rather than a road car.
    ///
    /// WHAT IS DIFFERENT, and why each one is there:
    ///
    ///  * Resonances moved up and widened (92 / 245 / 820 Hz against 78 / 210 / 620). The
    ///    boom moves up slightly so it survives laptop speakers; the rasp opens out so
    ///    full load sounds hard rather than just louder. The depth comes from the
    ///    explicit half-order rumble below, not from the filter.
    ///  * A short gas tail after every pulse. A bare impulse train goes glassy at high
    ///    revs; a couple of milliseconds of hot gas behind each pulse turns it into a bark.
    ///  * Load does most of the work. Lifting at high revs drops the drive, the rasp and
    ///    the rumble together, so on and off throttle are two sounds, not one sound at two
    ///    volumes. The offline check holds that gap above 8 dB.
    ///  * Gear changes are events. A change of gear starts a 75 ms torque cut that drains
    ///    the load in about 6 ms — far faster than a throttle lift — plus a short
    ///    mechanical clack, so the shift is heard as a break in the drive.
    ///  * Turbo release on a sharp lift while boosted, and anti-lag pops on overrun.
    ///
    /// THREADING. Plain DSP state: no Unity API, no allocation. <see cref="SetState"/> is
    /// called once per buffer and <see cref="Next"/> once per sample, both on the audio
    /// thread, so neither needs to be thread-safe against the other.
    /// </summary>
    public sealed class RallyEngineVoice
    {
        int rate, gear, cutSamples, popCountdown;
        bool initialized, wasShifting;
        double phase;                  // 0..1 across one four-stroke cycle (two crank revs)
        uint random = 0x73AD91EF;

        // Targets arrive once a buffer. Each one is ramped linearly from the previous value
        // across the buffer it arrives with, then lightly smoothed per sample.
        //
        // The ramp is what matters. A streaming clip asks for audio about 4096 samples
        // (85 ms) at a time, so a new target turns up only once per chunk. Smoothing alone
        // reaches it in a fraction of that and then sits on it for the rest of the chunk,
        // and a steady rev rise comes out as a staircase. Ramping across the chunk keeps
        // revs, load and boost moving the whole time, whatever size the chunks are.
        float rpmTarget, loadTarget, boostTarget, rpm = 1400, load, boost;
        float rpmFrom, loadFrom, boostFrom;
        int rampLength = 1, rampPosition;
        float rpmBlend, loadBlend, cutBlend;

        // Per-sample decay factors, derived from times in Reset so they hold at any rate.
        float decay, dcPole, shiftDecay, releaseDecay, popDecay;

        float shiftEnvelope, releaseEnvelope, popEnvelope, gas, uneven = .06f;
        float intake, popNoise, dcX, dcY;
        bool overrun, antiLag;
        Resonance low, mid, rasp;

        /// <summary>Smoothed revs, exposed for the offline stair-step check.</summary>
        public float Rpm => rpm;
        public float Induction { get; set; } = 1f;
        public int ShiftEvents { get; private set; }
        public int ReleaseEvents { get; private set; }

        public void Trigger(VehicleSoundEvent kind, float strength)
        {
            if (kind == VehicleSoundEvent.Shift)
            {
                shiftEnvelope = .65f;
                cutSamples = (int)(rate * .075f);
                ShiftEvents++;
            }
            else if (kind == VehicleSoundEvent.TurboRelease)
            {
                releaseEnvelope = Clamp(strength) * .12f;
                ReleaseEvents++;
            }
        }

        public void Reset(int sampleRate)
        {
            rate = sampleRate;
            initialized = wasShifting = false;
            phase = 0; rpm = 1400; load = boost = gas = 0;
            shiftEnvelope = releaseEnvelope = popEnvelope = intake = popNoise = dcX = dcY = 0;
            cutSamples = popCountdown = 0;
            rampLength = 1; rampPosition = 0;
            random = 0x73AD91EF;
            ShiftEvents = ReleaseEvents = 0;
            Induction = 1f;

            rpmBlend  = Blend(.018f);
            loadBlend = Blend(.025f);
            cutBlend  = Blend(.006f);   // a torque cut is abrupt; a throttle lift is not

            decay        = (float)Math.Exp(-1.0 / (.002 * rate));   // gas tail behind a pulse
            dcPole       = (float)Math.Exp(-2 * Math.PI * 20 / rate);
            shiftDecay   = (float)Math.Exp(-1.0 / (.028 * rate));
            releaseDecay = (float)Math.Exp(-1.0 / (.055 * rate));
            popDecay     = (float)Math.Exp(-1.0 / (.007 * rate));

            low.Set(92, 65, rate);
            mid.Set(245, 170, rate);
            rasp.Set(820, 650, rate);
        }

        /// <summary>
        /// Once per buffer, with the length of the buffer about to be rendered. Detects shifts
        /// and lifts by comparing against the last call.
        /// </summary>
        public void SetState(float revs, float throttle, float pressure, bool shifting,
                             int newGear, bool enabledAntiLag, float unevenFiring, int samples,
                             bool detectEvents = true)
        {
            uneven = Math.Max(0, Math.Min(.12f, unevenFiring));
            revs = Math.Max(300, Math.Min(9000, revs));
            throttle = Clamp(throttle);
            pressure = Clamp(pressure);

            if (initialized && detectEvents)
            {
                // The vehicle sets the new gear and isShifting in the same frame, so this
                // fires once per change rather than twice.
                if (newGear != gear || (shifting && !wasShifting))
                {
                    Trigger(VehicleSoundEvent.Shift, 1);
                }

                // Snapping shut from load with boost up: the charge has nowhere to go.
                if (loadTarget > .45f && throttle < .15f && boostTarget > .2f)
                    Trigger(VehicleSoundEvent.TurboRelease, boostTarget);
            }

            // Start this ramp from wherever the last one had got to, so a buffer cut short
            // never turns into a jump.
            float progress = RampProgress();
            rpmFrom   = initialized ? rpmFrom   + (rpmTarget   - rpmFrom)   * progress : revs;
            loadFrom  = initialized ? loadFrom  + (loadTarget  - loadFrom)  * progress : 0;
            boostFrom = initialized ? boostFrom + (boostTarget - boostFrom) * progress : pressure;
            rampLength = Math.Max(1, samples);
            rampPosition = 0;

            initialized = true;
            gear = newGear;
            wasShifting = shifting;
            rpmTarget = revs;
            loadTarget = shifting ? 0 : throttle;
            boostTarget = pressure;
            overrun = throttle < .12f && revs > 3300 && !shifting;
            antiLag = enabledAntiLag;
        }

        public float Next()
        {
            if (rampPosition < rampLength) rampPosition++;
            float ramp = RampProgress();
            float rpmGoal   = rpmFrom   + (rpmTarget   - rpmFrom)   * ramp;
            float loadGoal  = loadFrom  + (loadTarget  - loadFrom)  * ramp;
            float boostGoal = boostFrom + (boostTarget - boostFrom) * ramp;

            rpm += (rpmGoal - rpm) * rpmBlend;
            bool cut = cutSamples > 0;
            if (cut) cutSamples--;
            load += ((cut ? 0 : loadGoal) - load) * (cut ? cutBlend : loadBlend);
            boost += (boostGoal - boost) * loadBlend;

            // ── Source: four pulses a cycle, the second and fourth early by `uneven` ──
            double previous = phase;
            phase += rpm / (120.0 * rate);
            if (phase >= 1) phase -= 1;
            float pulse = 0;
            if (Crossed(previous, phase, 0)) pulse += 1;
            if (Crossed(previous, phase, .25 - uneven)) pulse += .91f;
            if (Crossed(previous, phase, .50)) pulse += .97f;
            if (Crossed(previous, phase, .75 - uneven)) pulse += .88f;

            // Resonator build-up grows with pulse rate; take most of it back out so revs
            // add body and load adds level. Same reasoning as the legacy voice.
            float compensation = (float)Math.Pow(100 / Math.Max(20, rpm / 30), .78);
            float level = (.26f + .74f * load) * compensation;
            if (pulse > 0) gas += pulse * level * .007f;
            gas *= decay;
            float noise = Noise();
            float source = pulse * level * (.94f + .06f * noise) + gas * (.7f + .3f * noise);

            // ── Anti-lag: scheduled pops on overrun while there is boost to burn ──
            if (--popCountdown <= 0)
            {
                if (overrun && antiLag && boost > .25f)
                    popEnvelope = .10f * boost;
                popCountdown = (int)(rate * (.18f + .25f * (Noise() * .5f + .5f)));
            }
            popNoise += (noise - popNoise) * .21f;

            // ── Filter: fixed exhaust resonances; the rasp only opens under load ──
            float exhaust = low.Next(source) * .60f + mid.Next(source) * .48f
                          + rasp.Next(source) * (.12f + .34f * load);

            // The unequal-header burble, made explicit. Scaled by `uneven` so an even-fire
            // setting really is even-fire, which the offline boxer check depends on.
            float rumble = (float)(Math.Sin(phase * Math.PI * 2) * .075
                                 + Math.Sin(phase * Math.PI * 4) * .035) * uneven / .06f;
            intake += (noise - intake) * (.12f + .17f * load);

            float sample = exhaust * .78f + rumble * (.3f + .7f * load)
                         + intake * (.028f + .050f * boost) * load * Induction
                         + popNoise * popEnvelope + noise * releaseEnvelope
                         + (popNoise * .09f + (float)Math.Sin(phase * Math.PI * 8) * .025f) * shiftEnvelope;
            shiftEnvelope *= shiftDecay;
            releaseEnvelope *= releaseDecay;
            popEnvelope *= popDecay;

            dcY = sample - dcX + dcPole * dcY;
            dcX = sample;
            float bounded = Math.Max(-1, Math.Min(1, dcY));
            return 1.5f * (bounded - bounded * bounded * bounded / 3);   // cubic soft clip
        }

        float RampProgress() => (float)rampPosition / rampLength;
        float Blend(float seconds) => 1 - (float)Math.Exp(-1.0 / (seconds * rate));
        static float Clamp(float value) => Math.Max(0, Math.Min(1, value));

        /// <summary>Did the phase pass <paramref name="target"/> this sample, allowing for the wrap?</summary>
        static bool Crossed(double a, double b, double target) =>
            b >= a ? a < target && b >= target : target > a || target <= b;

        float Noise()
        {
            random ^= random << 13; random ^= random >> 17; random ^= random << 5;
            return (random & 0xFFFFFF) / 8388607.5f - 1;
        }

        /// <summary>Two-pole resonator at a fixed centre and bandwidth.</summary>
        struct Resonance
        {
            float a, b, gain, y1, y2;
            public void Set(float hz, float width, int rate)
            {
                float r = (float)Math.Exp(-Math.PI * width / rate);
                a = 2 * r * (float)Math.Cos(2 * Math.PI * hz / rate);
                b = -r * r;
                gain = 1 - r;
                y1 = y2 = 0;
            }
            public float Next(float x)
            {
                float y = x * gain + a * y1 + b * y2;
                y2 = y1; y1 = y;
                return y;
            }
        }
    }
}
