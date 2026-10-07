// Minimal stand-ins for the Unity API the synths touch, so the REAL synth sources can be
// compiled and run outside the editor. Nothing here reimplements any DSP — the files under
// test are compiled verbatim, which is the point: a harness that copied the maths would
// drift away from the thing it is supposed to be checking.
using System;
using System.Collections.Generic;

namespace UnityEngine
{
    public static class Mathf
    {
        public const float PI = 3.14159265358979f;
        public static float Exp(float f) => (float)Math.Exp(f);
        public static float Cos(float f) => (float)Math.Cos(f);
        public static float Sin(float f) => (float)Math.Sin(f);
        public static float Sqrt(float f) => (float)Math.Sqrt(f);
        public static float Pow(float f, float p) => (float)Math.Pow(f, p);
        public static float Abs(float f) => Math.Abs(f);
        public static float Max(float a, float b) => a > b ? a : b;
        public static float Min(float a, float b) => a < b ? a : b;
        public static float Clamp01(float f) => f < 0f ? 0f : (f > 1f ? 1f : f);
        public static float Clamp(float f, float a, float b) => f < a ? a : (f > b ? b : f);
        public static float Lerp(float a, float b, float t) => a + (b - a) * Clamp01(t);
        public static bool Approximately(float a, float b) => Math.Abs(b - a) < 1e-6f;
    }

    public class Object
    {
        static int nextId = 1000;
        readonly int id = nextId += 7717;
        public int GetInstanceID() => id;
        public static void Destroy(Object o) { }
    }

    public class Component : Object
    {
        public readonly List<Component> siblings = new List<Component>();
        public T GetComponent<T>() where T : Component
        {
            foreach (var c in siblings) if (c is T t) return t;
            return null;
        }
        public T AddSibling<T>() where T : Component, new()
        {
            var t = new T();
            foreach (var c in siblings) { t.siblings.Add(c); c.siblings.Add(t); }
            siblings.Add(t); t.siblings.Add(this);
            return t;
        }
    }

    public class MonoBehaviour : Component { }
    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public float magnitude => (float)Math.Sqrt(x * x + y * y + z * z);
    }
    public class Rigidbody : Component { public Vector3 linearVelocity; }
    public class Collision { public Vector3 relativeVelocity, impulse; }

    public delegate void PCMReaderCallback(float[] data);

    public class AudioClip : Object
    {
        public static AudioClip Create(string n, int len, int ch, int freq, bool stream, PCMReaderCallback cb)
            => new AudioClip();
    }

    public enum AudioRolloffMode { Logarithmic, Linear, Custom }

    public class AudioSource : Component
    {
        public bool playOnAwake, loop;
        public float spatialBlend, minDistance, maxDistance, dopplerLevel;
        public AudioRolloffMode rolloffMode;
        public AudioClip clip;
        public void Play() { }
        public void Stop() { }
    }

    public static class AudioSettings { public static int outputSampleRate = 48000; }
    public static class Application { public static bool isPlaying = true, isBatchMode = false; }
    public static class Time { public static float timeScale = 1f, deltaTime = 1024f / 48000f, fixedDeltaTime = .01f, unscaledTime; }
    public static class Debug { public static void Log(object o) => Console.WriteLine(o); }
    public enum KeyCode { F8, F9 }
    public static class Input { public static bool GetKeyDown(KeyCode k) => false; }

    [AttributeUsage(AttributeTargets.All, AllowMultiple = true)]
    public class RequireComponent : Attribute { public RequireComponent(Type t) { } }
    public class TooltipAttribute : Attribute { public TooltipAttribute(string s) { } }
    public class RangeAttribute : Attribute { public RangeAttribute(float a, float b) { } }
    public class HeaderAttribute : Attribute { public HeaderAttribute(string s) { } }
    public class HideInInspector : Attribute { }
    public class DefaultExecutionOrder : Attribute { public DefaultExecutionOrder(int order) { } }
}

namespace Core.ML
{
    public static class LabRuntime
    {
        public static bool Enabled = false;
        public static Launch Config = new Launch();
        public class Launch { public bool viewer = true; }
    }
}

namespace Unity.MLAgents
{
    public class Academy
    {
        public static bool IsInitialized = false;
        public static Academy Instance = new Academy();
        public bool IsCommunicatorOn = false;
    }
}

// Data-only stand-ins. Field NAMES and types match the real components, so if those are
// renamed this harness stops compiling — which is exactly the alarm we want.
namespace Core.Physics
{
    using UnityEngine;

    public class PacejkaTireModel : MonoBehaviour
    {
        public float angularVelocity;
        public float slipSpeed;
    }

    public class DynamicSuspension : MonoBehaviour
    {
        public bool isGrounded = true;
        public float normalLoad;
        public float surfaceLooseness = 0.6f;
        public float wheelRadius = 0.330f;
        public float travel;
    }

    public class VehicleController : MonoBehaviour
    {
        public float mass = 1230f;
        public float idleRpm = 1400f, revLimitRpm = 7600f;
        public float currentRpm = 1400f;
        public float currentSpeedKmh;
        public float currentBoost;
        public float driftAngleDeg;
        public float throttleInput;
        public bool isShifting;
        public int currentGear = 1;
        public bool antiLag = true;
        public float NormalisedRpm => Mathf.Clamp01((currentRpm - idleRpm) / (revLimitRpm - idleRpm));

        public DynamicSuspension[] suspensions;
        public DynamicSuspension[] Suspensions => suspensions;
    }
}
