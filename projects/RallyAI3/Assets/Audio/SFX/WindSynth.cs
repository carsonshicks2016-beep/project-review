using UnityEngine;
using Core.Physics;

namespace Audio
{
    /// <summary>
    /// Air over the bodywork.
    ///
    /// WHY IT EARNS ITS PLACE. The engine cannot tell you how fast you are going. In a
    /// tall gear at a steady 4000 rpm it sounds identical at 90 km/h and at 160, and on a
    /// long straight that is most of the time. Wind is the voice that only speed moves,
    /// so it is what makes the straights feel quick — and, on a fast corner entry, what
    /// makes lifting feel like slowing down.
    ///
    /// THE LEVEL CURVE MATTERS MORE THAN THE TIMBRE. Aerodynamic noise power climbs
    /// steeply with speed — far faster than linearly — which is why wind is inaudible
    /// around town and dominant at rally speeds. Mapping it linearly gives you a hiss
    /// that is always half-present and never says anything; the exponent below is doing
    /// the real work here.
    ///
    /// BUFFETING. Steady filtered noise reads as tape hiss, not as air. Real airflow over
    /// an open rally car is turbulent and its level wanders. Two slow modulators at
    /// deliberately incommensurate rates give a wander that never audibly repeats, which
    /// costs two sines and is the whole difference between hiss and wind.
    /// </summary>
    public class WindSynth : ProceduralAudio
    {
        public VehicleController vehicle;

        /// <summary>See <see cref="EngineSynth.DefaultVolume"/>.</summary>
        public const float DefaultVolume = AudioMixProfile.WindVolume;

        [Header("Character")]
        [Tooltip("Speed in m/s at which the wind is at full level. 55 is about 200 km/h.")]
        public float referenceSpeed = 55f;

        [Tooltip("How steeply level climbs with speed. 1 is linear and sounds wrong — " +
                 "wind should be nearly absent at town speeds and loud at rally speeds.")]
        [Range(1f, 3f)] public float speedExponent = 1.8f;

        [Tooltip("Turbulence off the arches and wing when the car is sideways. This is what " +
                 "makes a big slide audible even with the throttle steady.")]
        [Range(0f, 1f)] public float yawRoar = 0.6f;

        [Tooltip("Depth of the slow level wander.")]
        [Range(0f, 0.6f)] public float buffet = 0.3f;

        // ── Snapshot: main thread writes, audio thread reads ─────────────
        volatile float sSpeed;      // m/s
        volatile float sYaw;        // 0..1, how sideways

        // ── Audio thread state ───────────────────────────────────────────
        OnePole rush, body;
        Resonator whistle;
        float mSpeed, mYaw;
        float gain, whistleGain;
        float lfoA, lfoB;

        protected override void ResetMainState() { sSpeed = sYaw = 0; }
        protected override void ResetAudioState()
        {
            rush = body = default(OnePole); whistle = default(Resonator);
            mSpeed = mYaw = gain = whistleGain = lfoA = lfoB = 0;
        }

        protected override void ReadState()
        {
            if (vehicle == null) return;
            sSpeed = Mathf.Abs(vehicle.currentSpeedKmh) / 3.6f;

            // Past about 35 degrees of drift the car is presenting its flank to the
            // airflow and there is nothing more to give, so the curve saturates there
            // rather than rewarding ever-wilder angles.
            sYaw = Mathf.Clamp01(Mathf.Abs(vehicle.driftAngleDeg) / 35f);
        }

        protected override void Render(float[] data)
        {
            float k = 1f - Mathf.Exp(-data.Length / (0.10f * sampleRate));
            mSpeed += (sSpeed - mSpeed) * k;
            mYaw += (sYaw - mYaw) * k;

            float speedNorm = Mathf.Clamp01(mSpeed / Mathf.Max(1f, referenceSpeed));
            float loudness = Mathf.Pow(speedNorm, speedExponent);

            float gainTarget = loudness * (1f + yawRoar * mYaw);

            // The rush gets brighter as well as louder: at low speed the flow is smooth
            // and what reaches the cabin is a low roar, and the top end only arrives once
            // the flow has energy enough to break up.
            float rushCoef = LowpassCoef(Mathf.Lerp(280f, 2800f, speedNorm));

            // A second, much lower pole under everything. Without it the whole voice sits
            // in the upper mids and sounds like a radio between stations rather than like
            // pressure against the shell of a car.
            float bodyCoef = LowpassCoef(Mathf.Lerp(90f, 320f, speedNorm));

            // The whistle is edge tone — air tearing off the A-pillar and the wing. It
            // barely exists head-on and is most of the sound when the car is sideways.
            whistle.Set(Mathf.Lerp(760f, 1580f, speedNorm), Mathf.Lerp(420f, 190f, mYaw), sampleRate);
            float whistleTarget = loudness * (0.10f + 0.75f * mYaw);

            // Incommensurate on purpose: 0.37 and 0.61 Hz share no simple ratio, so the
            // combined envelope takes minutes to come back round and never sounds looped.
            float rateA = 0.37f * (1f + speedNorm) / sampleRate;
            float rateB = 0.61f * (1f + speedNorm) / sampleRate;

            float levelCoef = SmoothingCoef(0.050f);

            for (int i = 0; i < data.Length; i++)
            {
                gain += (gainTarget - gain) * levelCoef;
                whistleGain += (whistleTarget - whistleGain) * levelCoef;

                lfoA += rateA; if (lfoA >= 1f) lfoA -= 1f;
                lfoB += rateB; if (lfoB >= 1f) lfoB -= 1f;
                float mod = 1f + buffet * (0.62f * Mathf.Sin(lfoA * 2f * Mathf.PI)
                                         + 0.38f * Mathf.Sin(lfoB * 2f * Mathf.PI));

                float n = Noise();
                float air = rush.Process(n, rushCoef) * 0.60f + body.Process(n, bodyCoef) * 0.86f;
                float edge = whistle.Process(n) * whistleGain * 0.72f;

                data[i] = SoftClip((air * gain + edge) * mod);
            }
        }
    }
}
