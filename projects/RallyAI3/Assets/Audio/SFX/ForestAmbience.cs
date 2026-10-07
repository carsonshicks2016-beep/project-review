using UnityEngine;

namespace Audio
{
    public sealed class ForestAmbience : ProceduralAudio
    {
        OnePole air;
        float phase, chirpPhase, chirpEnvelope, decay;
        int untilChirp;
        protected override void Configure()
        {
            volume = AudioMixProfile.AmbienceVolume;
            decay = Mathf.Exp(-1f / (.12f * sampleRate));
            untilChirp = sampleRate * 5;
        }
        protected override void ReadState() { }
        protected override void Render(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                phase += .17f / sampleRate;
                if (phase >= 1) phase -= 1;
                if (--untilChirp <= 0)
                {
                    chirpEnvelope = .08f;
                    untilChirp = (int)(sampleRate * (7 + (Noise() + 1) * 6));
                }
                chirpPhase += (1900 + chirpEnvelope * 8000) / sampleRate;
                if (chirpPhase >= 1) chirpPhase -= 1;
                data[i] = air.Process(Noise(), .035f) * (.25f + .08f * Mathf.Sin(phase * 2 * Mathf.PI)) +
                    Mathf.Sin(chirpPhase * 2 * Mathf.PI) * chirpEnvelope;
                chirpEnvelope *= decay;
            }
        }
    }
}
