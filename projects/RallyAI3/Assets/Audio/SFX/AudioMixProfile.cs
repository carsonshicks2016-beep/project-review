using System;

namespace Audio
{
    public enum ListeningPerspective { Chase, Hood, Driver, Trackside, Helicopter }

    public struct PerspectiveMix
    {
        public float engine, tyres, wind, impacts, ambience, cutoff, spatial, induction;
        public bool doppler;
    }

    public static class AudioMixProfile
    {
        public const float EngineVolume = .95f, TyreVolume = .19f, WindVolume = .17f;
        public const float ImpactVolume = .24f, AmbienceVolume = .035f;
        public static PerspectiveMix For(ListeningPerspective view)
        {
            var mix = new PerspectiveMix { engine = 1, tyres = 1, wind = .65f,
                impacts = 1, ambience = .6f, cutoff = 18000, spatial = .85f, induction = 1 };
            switch (view)
            {
                case ListeningPerspective.Hood:
                    mix.engine = .9f; mix.wind = .9f; mix.induction = 1.6f; break;
                case ListeningPerspective.Driver:
                    mix.engine = .8f; mix.tyres = 1.3f; mix.wind = .5f;
                    mix.impacts = 1.2f; mix.ambience = .25f; mix.cutoff = 2200;
                    mix.spatial = .15f; break;
                case ListeningPerspective.Trackside:
                    mix.wind = 0; mix.ambience = 1; mix.spatial = 1; mix.doppler = true; break;
                case ListeningPerspective.Helicopter:
                    mix.engine = .9f; mix.tyres = .6f; mix.impacts = .6f;
                    mix.wind = 0; mix.ambience = .8f; mix.spatial = 1;
                    mix.cutoff = 10000; mix.doppler = true; break;
            }
            return mix;
        }

        // Positive velocity is toward listener->source. Approach raises pitch, recession lowers it.
        public static float Doppler(float sourceRadial, float listenerRadial)
        {
            const float soundSpeed = 343f;
            float ratio = (soundSpeed + Clamp(listenerRadial, -100, 100)) /
                          (soundSpeed + Clamp(sourceRadial, -100, 100));
            return Clamp(ratio, .75f, 1.35f);
        }
        static float Clamp(float value, float min, float max) => Math.Max(min, Math.Min(max, value));
    }
}
