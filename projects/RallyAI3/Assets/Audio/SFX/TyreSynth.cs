using UnityEngine;
using Core.Physics;

namespace Audio
{
    /// <summary>
    /// What the tyres are doing to the ground, synthesised from the contact patches.
    ///
    /// THREE VOICES, because tyre noise is three different mechanisms and a single
    /// filtered hiss can only ever be one of them:
    ///
    ///   ROLL   the tread pattern pumping air, a broadband bed whose pitch and level
    ///          follow how fast the patch is moving over the surface.
    ///   GRAIN  loose material — individual stones struck by the tread and flung at the
    ///          arches. This is the voice that makes gravel sound like gravel.
    ///   SCRUB  the patch sliding rather than rolling. On tarmac that is a narrow
    ///          resonance you hear as a squeal; on gravel the same slip just scatters
    ///          stones, so the resonance widens into a rush.
    ///
    /// GRAINS, AND WHY NOT NOISE. The grain voice is a Poisson train of impulses whose
    /// RATE is what tracks speed — not its level. At a crawl that is a few dozen ticks a
    /// second and you hear separate stones; at 120 km/h it is a couple of thousand and
    /// they fuse into a roar. One mechanism covers both, and the transition between them
    /// is free. Filtered noise cannot do the low end of that at all: it is already fused.
    ///
    /// TWO MORE, for gravel specifically, switchable with <see cref="gravelDetail"/> (F9
    /// in play mode) so they can be judged against the plain three:
    ///
    ///   STRIKE  the stones that do not just tick off the tread but hit the CAR — a dull
    ///           thud from the arch liner and a short metallic ring from the panels.
    ///           Sparse, loud and irregular, and far more of them in a slide, because a
    ///           sliding tyre is a shovel. Grain is the hiss; this is the clatter.
    ///   RUMBLE  the ground coming up through the suspension. Driven partly by speed and
    ///           mostly by how fast the suspension is actually moving, so a rough stretch
    ///           is heard as rough and a smooth one goes quiet.
    ///
    /// WEIGHTED BY LOAD, NOT BY WHEEL COUNT. Every corner contributes in proportion to
    /// the vertical load it is carrying, which is the same thing that sets how hard the
    /// tread is being worked. Two consequences fall out for nothing: a wheel light over a
    /// crest goes quiet on its own, and an airborne car goes silent — so the landing
    /// arrives as a slam. That silence is one of the most recognisable sounds in rally
    /// and here it costs no code at all.
    /// </summary>
    public class TyreSynth : ProceduralAudio
    {
        public VehicleController vehicle;

        /// <summary>See <see cref="EngineSynth.DefaultVolume"/>.</summary>
        public const float DefaultVolume = AudioMixProfile.TyreVolume;

        [Header("Voice levels")]
        [Tooltip("Tread bed. Rises with contact speed.")]
        [Range(0f, 1f)] public float rollLevel = 0.45f;

        [Tooltip("Loose material. Scales with the surface's looseness, so tarmac silences it.")]
        [Range(0f, 1f)] public float grainLevel = 0.85f;

        [Tooltip("Sliding. Tarmac squeal at one end of the surface range, gravel scatter at the other.")]
        [Range(0f, 1f)] public float scrubLevel = 0.7f;

        [Tooltip("Stones hitting the arches and underbody. Scales with looseness, and with slip most of all.")]
        [Range(0f, 1f)] public float strikeLevel = 0.6f;

        [Tooltip("Low rumble from the suspension working over the surface.")]
        [Range(0f, 1f)] public float rumbleLevel = 0.5f;

        [Tooltip("Stone strikes, surface rumble and the uneven stone sizes. Off is the plain " +
                 "three-voice sound, kept for comparison. F9 toggles it in play mode.")]
        public bool gravelDetail = true;

        // ── Reference speeds for the normalisations, in m/s. 35 is about 125 km/h, at
        //    which the roll bed is at full level; 12 m/s of slip is a big, committed slide.
        const float RollRef = 35f;
        const float SlipRef = 12f;

        /// <summary>Suspension speed, m/s, at which the rumble is at full level. A hard rut.</summary>
        const float BumpRef = 0.6f;

        // ── Snapshot: main thread writes, audio thread reads ─────────────
        volatile float sContact;    // m/s, load-weighted mean patch speed
        volatile float sSlip;       // m/s, load-weighted mean slide speed
        volatile float sLoose;      // 0..1, load-weighted mean surface looseness
        volatile float sGround;     // 0..1, fraction of the car's weight on the ground
        volatile float sBump;       // m/s, load-weighted mean suspension speed
        volatile bool  sDetail = true;

        // ── Audio thread state ───────────────────────────────────────────
        Resonator rollRes, grainLo, grainHi, scrubRes;
        Resonator archThud, panelLo, panelHi, rumbleRes;
        OnePole rollLp, rumbleLp;
        float mContact, mSlip, mLoose, mGround, mBump;   // values at the end of the last buffer
        float rollGain, grainGain, scrubGain, rumbleGain; // sample-rate smoothing of the levels
        float strikeAmp;
        int strikeBurst;

        /// <summary>Stone strikes started so far. Audio thread writes; read by the offline check.</summary>
        public int StrikeCount { get; private set; }

        // Cached pairing of each suspension with the tyre on the same node. Done once
        // rather than trusting VehicleController's two arrays to be in the same order.
        DynamicSuspension[] corners;
        PacejkaTireModel[] wheels;
        float[] lastTravel;
        float bumpSmoothed;
        float weightN = 12000f;
        bool fixedCaptured;

        protected override void Configure()
        {
            // Fixed centres. As with the engine, the surface has a voice that does not
            // move — only how hard it is excited does.
            grainLo.Set(430f, 190f, sampleRate);
            grainHi.Set(1750f, 320f, sampleRate);

            // A stone on the car: the plastic arch liner is a dull, heavily damped thud;
            // the steel around it rings briefly at two panel modes. Narrow bands are what
            // make it read as metal rather than as a louder grain.
            archThud.Set(320f, 140f, sampleRate);
            panelLo.Set(2300f, 90f, sampleRate);
            panelHi.Set(3700f, 140f, sampleRate);

            rumbleRes.Set(62f, 70f, sampleRate);
            SetVariableResonators();
        }

        // ══════════════════════════════════════════════════════════════
        //  MAIN THREAD
        // ══════════════════════════════════════════════════════════════

        protected override void ReadState()
        {
            if (Input.GetKeyDown(KeyCode.F9)) gravelDetail = !gravelDetail;
            sDetail = gravelDetail;
            if (!fixedCaptured) CaptureState(Time.deltaTime);
        }

        protected override void ReadFixedState()
        {
            fixedCaptured = true;
            CaptureState(Time.fixedDeltaTime);
        }

        protected override void ResetMainState()
        {
            if (corners != null)
                for (int i = 0; i < corners.Length; i++) lastTravel[i] = corners[i] != null ? corners[i].travel : 0;
            bumpSmoothed = 0;
            sContact = sSlip = sLoose = sGround = sBump = 0;
        }

        protected override void ResetAudioState()
        {
            rollRes = grainLo = grainHi = scrubRes = archThud = panelLo = panelHi = rumbleRes = default(Resonator);
            rollLp = rumbleLp = default(OnePole);
            mContact = mSlip = mLoose = mGround = mBump = 0;
            rollGain = grainGain = scrubGain = rumbleGain = strikeAmp = 0;
            strikeBurst = StrikeCount = 0;
            Configure();
        }

        void CaptureState(float dt)
        {
            if (vehicle == null) return;
            if (corners == null && !Pair()) return;
            sDetail = gravelDetail;

            float loadSum = 0f, looseSum = 0f, contactSum = 0f, slipSum = 0f, bumpSum = 0f;
            for (int i = 0; i < corners.Length; i++)
            {
                DynamicSuspension sus = corners[i];
                if (sus == null) continue;

                // Track travel on every corner, grounded or not, so the first frame back
                // on the ground is not read as a huge suspension speed.
                float travel = sus.travel;
                float bump = dt > 0f ? Mathf.Abs(travel - lastTravel[i]) / dt : 0f;
                lastTravel[i] = travel;
                if (!sus.isGrounded) continue;

                float load = sus.normalLoad;
                if (load <= 0f) continue;

                loadSum += load;
                looseSum += sus.surfaceLooseness * load;
                bumpSum += bump * load;

                PacejkaTireModel tyre = wheels[i];
                if (tyre == null) continue;

                // Patch speed from the WHEEL, not from the body. A locked wheel under
                // braking has a body moving at 30 m/s and a patch moving at zero: the roll
                // bed has to stop and hand over to scrub, which is exactly what you hear.
                contactSum += Mathf.Abs(tyre.angularVelocity) * sus.wheelRadius * load;
                slipSum += tyre.slipSpeed * load;
            }

            if (loadSum <= 1f)
            {
                sGround = 0f;                    // airborne — everything falls silent
                return;
            }

            float inv = 1f / loadSum;
            sContact = contactSum * inv;
            sSlip = slipSum * inv;
            sLoose = looseSum * inv;
            // Physics moves the suspension on its own fixed step, which does not line up
            // with frames: one frame sees no movement, the next sees two steps' worth. Read
            // raw, that is a rumble that flutters at the beat between the two rates.
            if (dt > 0f) bumpSmoothed += (bumpSum * inv - bumpSmoothed) * (1f - Mathf.Exp(-dt / 0.06f));
            sBump = bumpSmoothed;
            sGround = Mathf.Clamp01(loadSum / weightN);
        }

        bool Pair()
        {
            DynamicSuspension[] found = vehicle.Suspensions;
            if (found == null || found.Length == 0) return false;

            corners = found;
            wheels = new PacejkaTireModel[found.Length];
            lastTravel = new float[found.Length];
            for (int i = 0; i < found.Length; i++)
            {
                wheels[i] = found[i] != null ? found[i].GetComponent<PacejkaTireModel>() : null;
                lastTravel[i] = found[i] != null ? found[i].travel : 0f;
            }

            weightN = Mathf.Max(1f, vehicle.mass * 9.81f);
            return true;
        }

        // ══════════════════════════════════════════════════════════════
        //  AUDIO THREAD
        // ══════════════════════════════════════════════════════════════

        /// <summary>The per-buffer levels and rates, worked out from one set of control values.</summary>
        struct Controls
        {
            public float roll, grainProb, scrub, strikeProb, rumble;
        }

        Controls ControlsFor(float contact, float slip, float loose, float ground, float bump, bool detail)
        {
            float rollNorm = Mathf.Clamp01(contact / RollRef);
            float slipNorm = Mathf.Clamp01(slip / SlipRef);
            var c = new Controls();

            // Tread noise grows fast at first and then flattens; a square root is a decent
            // fit and, more usefully, keeps a car rolling at walking pace audible.
            c.roll = rollLevel * ground * Mathf.Sqrt(rollNorm);

            // Grains: the RATE carries the speed, so the gain stays near constant and is
            // deliberately NOT density-normalised. Power rises with the rate on its own,
            // which is the +16 dB or so between crawling and flat out — the real spread.
            //
            // Strictly proportional to movement, with no constant term. A stone is thrown
            // because the tread moved over it; a car sitting still on gravel is silent, and
            // a rate floor here would have it quietly crunching on the start line.
            float rate = loose * ground * (66f * contact + 260f * slip);
            c.grainProb = Mathf.Min(rate, 5000f) / sampleRate;

            c.scrub = scrubLevel * ground * Mathf.Pow(slipNorm, 0.8f);

            if (detail)
            {
                // Tens a second, not thousands: about 15 at 120 km/h rolling, near 40 in a
                // committed slide. Slip is weighted eight times speed because a sliding tyre
                // throws stones sideways into the arch instead of back along the road.
                float strikes = loose * ground * (0.6f * contact + 5f * slip);
                c.strikeProb = Mathf.Min(strikes, 80f) / sampleRate;

                float bumpNorm = Mathf.Clamp01(bump / BumpRef);
                c.rumble = rumbleLevel * ground * (0.25f + 0.75f * loose)
                         * Mathf.Clamp01(0.5f * Mathf.Sqrt(rollNorm) + bumpNorm);
            }
            return c;
        }

        protected override void Render(float[] data)
        {
            // Control values arrive once per buffer, and a streaming clip asks for about
            // 4096 samples (85 ms) at a time. Every level and rate is therefore ramped
            // linearly from where the last buffer ended to where this one should end, so
            // nothing jumps at a buffer boundary and then sits still until the next one —
            // the same staircase the engine had. The filter shapes are still set once per
            // buffer: recomputing a resonator per sample costs an exp and a cos each time
            // for a parameter that cannot move audibly within one buffer.
            bool detail = sDetail;
            Controls from = ControlsFor(mContact, mSlip, mLoose, mGround, mBump, detail);
            mContact = sContact;
            mSlip = sSlip;
            mLoose = sLoose;
            mGround = sGround;
            mBump = sBump;
            Controls to = ControlsFor(mContact, mSlip, mLoose, mGround, mBump, detail);

            SetVariableResonators();

            float rollNorm = Mathf.Clamp01(mContact / RollRef);
            float grainTarget = grainLevel * 0.8f;

            // Gravel is duller than tarmac at the same speed: less high-frequency tread
            // pumping, more low-frequency displacement of loose material.
            float rollCoef = LowpassCoef(Mathf.Lerp(700f, 4200f, rollNorm) * Mathf.Lerp(1f, 0.5f, mLoose));
            float rumbleCoef = LowpassCoef(140f);

            float levelCoef = SmoothingCoef(0.030f);
            float step = 1f / data.Length;
            int burstLength = 1 + sampleRate / 1250;   // 0.8 ms of grit per strike

            for (int i = 0; i < data.Length; i++)
            {
                float t = (i + 1) * step;
                float rollTarget   = from.roll   + (to.roll   - from.roll)   * t;
                float scrubTarget  = from.scrub  + (to.scrub  - from.scrub)  * t;
                float rumbleTarget = from.rumble + (to.rumble - from.rumble) * t;
                float grainProb    = from.grainProb  + (to.grainProb  - from.grainProb)  * t;
                float strikeProb   = from.strikeProb + (to.strikeProb - from.strikeProb) * t;

                rollGain += (rollTarget - rollGain) * levelCoef;
                grainGain += (grainTarget - grainGain) * levelCoef;
                scrubGain += (scrubTarget - scrubGain) * levelCoef;
                rumbleGain += (rumbleTarget - rumbleGain) * levelCoef;

                // ── Roll ──
                float roll = rollRes.Process(rollLp.Process(Noise(), rollCoef)) * rollGain * 0.42f;

                // ── Grain. Signed impulses: a stone strike is a pressure transient with no
                //    preferred direction, and letting the sign vary also means the train
                //    carries no DC into the resonators, so nothing downstream needs to
                //    remove it.
                //
                //    With detail on, sizes follow a cube law — mostly fine grit, now and
                //    then a real stone — rescaled so the mean power matches the uniform
                //    sizes it replaces. Uniform sizes are what make a grain bed sound like
                //    rain rather than gravel.
                float grain = 0f;
                if (Random01() < grainProb)
                {
                    float u = Random01();
                    float size = detail ? (0.2f + 0.8f * u * u * u) * 1.525f : 0.35f + 0.65f * u;
                    grain = size * (Noise() < 0f ? -1f : 1f) * grainGain;
                }
                float grainVoice = grainLo.Process(grain) * 1.0f + grainHi.Process(grain) * 0.8f;

                // ── Scrub ──
                float scrub = scrubRes.Process(Noise()) * scrubGain * 0.70f;

                // ── Strike. A short burst of grit rather than one impulse, because a stone
                //    hitting a panel is a scrape of contact, not an ideal click.
                float strikeVoice = 0f;
                if (detail)
                {
                    if (strikeBurst == 0 && Random01() < strikeProb)
                    {
                        float u = Random01();
                        strikeAmp = strikeLevel * (0.3f + 0.7f * u * u);
                        strikeBurst = burstLength;
                        StrikeCount++;
                    }
                    float hit = 0f;
                    if (strikeBurst > 0) { strikeBurst--; hit = Noise() * strikeAmp; }
                    strikeVoice = archThud.Process(hit) * 1.0f
                                + panelLo.Process(hit) * 0.9f
                                + panelHi.Process(hit) * 0.6f;
                }

                // ── Rumble ──
                float rumble = rumbleRes.Process(rumbleLp.Process(Noise(), rumbleCoef)) * rumbleGain;

                // Mix weights are set from measured RMS, not by ear-guessing: at the
                // loudest case the project has — locked wheels on gravel at 100 km/h —
                // this lands near -14 dB with peaks around 0.6, so SoftClip is only ever
                // catching genuine transients. A voice permanently against the clipper is
                // not "loud", it is a square wave.
                float mixed = roll + grainVoice * 0.43f + scrub * 0.60f
                            + strikeVoice * StrikeMix + rumble * RumbleMix;
                // Preserve small-signal gain without flattening rare coincident stone transients.
                data[i] = (float)System.Math.Tanh(1.5f * mixed);
            }
        }

        // Output weights for the two detail voices, set from the offline check.
        //
        // The rumble weight is small because its input is not: noise held below 140 Hz
        // into a 62 Hz resonator piles up energy the way sustained noise always does in a
        // narrow band (see the crackle note in EngineSynth). At 1.0 it sat 13 dB over the
        // whole tyre bed. At this weight it adds about 1 dB on a rough gravel road.
        //
        // Strikes are judged by their peaks, not their RMS. They are sparse, so even at full
        // weight they barely move the average, while the big ones land about 14 dB above
        // the grain bed, which is what makes them read as separate stones.
        const float StrikeMix = 1f;
        const float RumbleMix = 0.12f;

        /// <summary>
        /// The two resonators whose shape depends on the surface. Called once per buffer.
        ///
        /// The scrub bandwidth is the interesting one. Narrow on tarmac is a squeal — a
        /// tone, because a tarmac patch slips and grips in a regular cycle. Wide on gravel
        /// is a rush, because there is nothing for it to grip against. Same voice, and the
        /// surface decides which one you get.
        /// </summary>
        void SetVariableResonators()
        {
            float rollNorm = Mathf.Clamp01(mContact / RollRef);
            float slipNorm = Mathf.Clamp01(mSlip / SlipRef);

            rollRes.Set(Mathf.Lerp(1250f, 640f, mLoose) * (0.75f + 0.45f * rollNorm),
                        Mathf.Lerp(900f, 1400f, mLoose),
                        sampleRate);

            scrubRes.Set(Mathf.Lerp(1150f, 470f, mLoose) * (0.9f + 0.25f * slipNorm),
                         Mathf.Lerp(95f, 850f, mLoose),
                         sampleRate);
        }

        float Random01() => Noise() * 0.5f + 0.5f;
    }
}
