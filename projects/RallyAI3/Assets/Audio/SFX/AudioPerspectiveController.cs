using System;
using System.IO;
using UnityEngine;
using UnityEngine.Audio;
using Core.ML;
using Core.Physics;
using UI;

namespace Audio
{
    public sealed class AudioPerspectiveController : MonoBehaviour
    {
        SpectatorDirector director;
        VehicleController vehicle;
        Rigidbody body;
        EngineSynth engine;
        TyreSynth tyres;
        WindSynth wind;
        ImpactSynth impacts;
        ForestAmbience ambience;
        AudioLowPassFilter engineFilter, tyreFilter, impactFilter;
        AudioMixer mixer;
        Vector3 previousListener, previousVehicle;
        bool hasMotion;
        float pitch = 1, nextTrace;
        StreamWriter trace;
        public ListeningPerspective Perspective { get; private set; }

        [Serializable] class AudioFrame
        {
            public float time, pitch, sourceRadial, listenerRadial, engineGain, tyreGain, windGain;
            public int perspective, sources, listeners, droppedEngineEvents, droppedImpactEvents;
            public int shiftEvents, releaseEvents;
            public float engineCutoff;
            public bool cut, reset;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            if (Application.isBatchMode || (LabRuntime.Enabled && !LabRuntime.Config.viewer)) return;
            var camera = FindAnyObjectByType<SpectatorDirector>();
            if (camera != null && camera.GetComponent<AudioPerspectiveController>() == null)
                camera.gameObject.AddComponent<AudioPerspectiveController>();
        }

        void Start()
        {
            director = GetComponent<SpectatorDirector>();
            var ownListener = GetComponent<AudioListener>();
            if (ownListener == null) ownListener = gameObject.AddComponent<AudioListener>();
            ownListener.enabled = true;
            foreach (var listener in FindObjectsByType<AudioListener>(FindObjectsSortMode.None))
                if (listener != ownListener) listener.enabled = false;
            mixer = Resources.Load<AudioMixer>("Audio/RallyMix");
            if (mixer == null) Debug.LogError("Rally audio mixer missing; rebuild the audio assets before viewing.");
            string path = Environment.GetEnvironmentVariable("RALLY_AUDIO_TRACE");
            if (!string.IsNullOrEmpty(path)) trace = new StreamWriter(path) { AutoFlush = true };
            director.AudioPerspectiveChanged += OnPerspective;
            Bind();
        }

        void Bind()
        {
            if (vehicle != null) return;
            vehicle = director.targetCar != null ? director.targetCar.GetComponent<VehicleController>() :
                FindAnyObjectByType<VehicleController>();
            if (vehicle == null) return;
            if (ambience != null) Destroy(ambience.gameObject);
            body = vehicle.GetComponent<Rigidbody>();
            engine = vehicle.GetComponentInChildren<EngineSynth>();
            tyres = vehicle.GetComponentInChildren<TyreSynth>();
            wind = vehicle.GetComponentInChildren<WindSynth>();
            impacts = vehicle.GetComponentInChildren<ImpactSynth>();
            if (impacts == null)
            {
                var node = new GameObject("Audio_Impacts"); node.transform.SetParent(vehicle.transform, false);
                node.transform.localPosition = new Vector3(0, .3f, 0);
                impacts = node.AddComponent<ImpactSynth>(); impacts.vehicle = vehicle;
                var relay = vehicle.GetComponent<ImpactContactRelay>();
                if (relay == null) relay = vehicle.gameObject.AddComponent<ImpactContactRelay>();
                relay.voice = impacts;
            }
            var ambientNode = new GameObject("Audio_Forest"); ambientNode.transform.SetParent(transform, false);
            ambience = ambientNode.AddComponent<ForestAmbience>();
            Route(engine, "Engine"); Route(tyres, "Tyres"); Route(wind, "Wind");
            Route(impacts, "Impacts"); Route(ambience, "Ambience");
            engineFilter = Filter(engine); tyreFilter = Filter(tyres); impactFilter = Filter(impacts);
            hasMotion = false;
        }

        void Route(ProceduralAudio voice, string group)
        {
            if (voice == null || voice.Source == null || mixer == null) return;
            var matches = mixer.FindMatchingGroups(group);
            if (matches.Length != 1) { Debug.LogError("Missing or ambiguous audio group: " + group); return; }
            voice.Source.outputAudioMixerGroup = matches[0];
        }
        static AudioLowPassFilter Filter(ProceduralAudio voice)
        {
            if (voice == null) return null;
            var filter = voice.GetComponent<AudioLowPassFilter>();
            if (filter == null) filter = voice.gameObject.AddComponent<AudioLowPassFilter>();
            filter.cutoffFrequency = 18000; filter.lowpassResonanceQ = 1;
            return filter;
        }

        void OnPerspective(SpectatorDirector.CameraMode mode, bool cut, bool reset)
        {
            Bind();
            if (vehicle == null) return;
            Perspective = mode == SpectatorDirector.CameraMode.Driver ? ListeningPerspective.Driver :
                mode == SpectatorDirector.CameraMode.Hood ? ListeningPerspective.Hood :
                mode == SpectatorDirector.CameraMode.Trackside ? ListeningPerspective.Trackside :
                mode == SpectatorDirector.CameraMode.Helicopter ? ListeningPerspective.Helicopter : ListeningPerspective.Chase;
            var mix = AudioMixProfile.For(Perspective);
            float dt = Mathf.Clamp(Time.unscaledDeltaTime, .001f, .1f);
            float blend = 1f - Mathf.Exp(-dt / .15f);
            Vector3 position = vehicle.transform.position;
            bool jump = hasMotion && (position - previousVehicle).sqrMagnitude > 100f;
            if (reset || jump)
            {
                if (engine != null) engine.ResetPlayback();
                if (tyres != null) tyres.ResetPlayback();
                if (wind != null) wind.ResetPlayback();
                if (impacts != null) impacts.ResetPlayback();
            }
            Vector3 listenerVelocity = hasMotion && !cut && !reset && !jump
                ? Vector3.ClampMagnitude((transform.position - previousListener) / dt, 100f) : Vector3.zero;
            Vector3 direction = (position - transform.position).normalized;
            float sourceRadial = body != null ? Vector3.Dot(body.linearVelocity, direction) : 0;
            float listenerRadial = Vector3.Dot(listenerVelocity, direction);
            if (cut || reset || jump || !hasMotion) pitch = 1;
            else pitch = Mathf.Lerp(pitch, mix.doppler ? AudioMixProfile.Doppler(sourceRadial, listenerRadial) : 1, blend);
            if (!mix.doppler) pitch = 1;
            Apply(engine, mix.engine, mix.spatial, pitch, blend, engineFilter, mix.cutoff);
            Apply(tyres, mix.tyres, mix.spatial, pitch, blend, tyreFilter, mix.cutoff);
            Apply(impacts, mix.impacts, mix.spatial, pitch, blend, impactFilter, mix.cutoff);
            Apply(wind, mix.wind, 0, 1, blend, null, 18000);
            Apply(ambience, mix.ambience, 0, 1, blend, null, 18000);
            if (engine != null) engine.perspectiveInduction = Mathf.Lerp(engine.perspectiveInduction, mix.induction, blend);
            previousListener = transform.position; previousVehicle = position; hasMotion = true;
            if (trace != null && (cut || reset || jump || Time.realtimeSinceStartup >= nextTrace))
            {
                nextTrace = Time.realtimeSinceStartup + .1f;
                trace.WriteLine(JsonUtility.ToJson(new AudioFrame {
                    time = Time.realtimeSinceStartup, perspective = (int)Perspective, pitch = pitch,
                    sourceRadial = sourceRadial, listenerRadial = listenerRadial, cut = cut, reset = reset || jump,
                    engineGain = engine != null && engine.Source != null ? engine.Source.volume : -1,
                    tyreGain = tyres != null && tyres.Source != null ? tyres.Source.volume : -1,
                    windGain = wind != null && wind.Source != null ? wind.Source.volume : -1,
                    sources = FindObjectsByType<AudioSource>(FindObjectsSortMode.None).Length,
                    listeners = ActiveListeners(), droppedEngineEvents = engine != null ? engine.DroppedEvents : 0,
                    shiftEvents = engine != null ? engine.ShiftEvents : 0,
                    releaseEvents = engine != null ? engine.ReleaseEvents : 0,
                    engineCutoff = engineFilter != null ? engineFilter.cutoffFrequency : -1,
                    droppedImpactEvents = impacts != null ? impacts.DroppedEvents : 0 }));
            }
        }

        static int ActiveListeners()
        {
            int count = 0;
            foreach (var listener in FindObjectsByType<AudioListener>(FindObjectsSortMode.None))
                if (listener.isActiveAndEnabled) count++;
            return count;
        }
        static void Apply(ProceduralAudio voice, float gain, float spatial, float pitch,
            float blend, AudioLowPassFilter filter, float cutoff)
        {
            if (voice == null || voice.Source == null) return;
            var source = voice.Source;
            source.volume = Mathf.Lerp(source.volume, gain, blend);
            source.spatialBlend = Mathf.Lerp(source.spatialBlend, spatial, blend);
            source.pitch = pitch; source.dopplerLevel = 0;
            source.rolloffMode = AudioRolloffMode.Logarithmic;
            if (filter != null) filter.cutoffFrequency = Mathf.Lerp(filter.cutoffFrequency, cutoff, blend);
        }

        void OnDestroy()
        {
            if (director != null) director.AudioPerspectiveChanged -= OnPerspective;
            if (ambience != null) Destroy(ambience.gameObject);
            trace?.Dispose();
        }
    }
}
