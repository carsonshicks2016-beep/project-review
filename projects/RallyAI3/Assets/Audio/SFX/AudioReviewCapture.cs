using System;
using System.IO;
using System.Threading;
using UnityEngine;
using Core.ML;

namespace Audio
{
    // Opt-in final listener output. The callback only copies into bounded preallocated storage.
    public sealed class AudioReviewCapture : MonoBehaviour
    {
        const int Seconds = 45;
        float[] samples;
        int written, channels, rate;
        volatile bool closed;
        string folder;
        bool saved;

        [Serializable] class CaptureManifest
        {
            public string course, checkpoint, unity, camera;
            public int sampleRate, channels, samples, bufferLength, bufferCount, listeners;
            public double seconds, rms, peak;
            public int nonFinite, fullScaleSamples;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            string path = Environment.GetEnvironmentVariable("RALLY_AUDIO_CAPTURE");
            if (string.IsNullOrEmpty(path) || Application.isBatchMode ||
                !LabRuntime.Enabled || !LabRuntime.Config.viewer) return;
            var listener = FindAnyObjectByType<AudioListener>();
            if (listener == null) throw new InvalidOperationException("Audio review needs a listener");
            var capture = listener.gameObject.AddComponent<AudioReviewCapture>();
            capture.folder = path;
        }

        void Awake()
        {
            rate = AudioSettings.outputSampleRate;
            samples = new float[Math.Max(1, rate) * Seconds * 8];
        }

        void OnAudioFilterRead(float[] data, int channelCount)
        {
            if (closed) return;
            channels = channelCount;
            int offset = written;
            int count = Math.Min(data.Length, Math.Min(samples.Length,
                rate * Seconds * channelCount) - offset);
            if (count <= 0) return;
            Array.Copy(data, 0, samples, offset, count);
            Volatile.Write(ref written, offset + count);
        }

        void Save()
        {
            if (saved || string.IsNullOrEmpty(folder)) return;
            saved = true;
            closed = true;
            int count = Volatile.Read(ref written);
            Directory.CreateDirectory(folder);
            double square = 0, peak = 0;
            int invalid = 0, clipped = 0;
            using (var writer = new BinaryWriter(File.Create(Path.Combine(folder, "listener.wav"))))
            {
                int bytes = count * 2;
                writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write(36 + bytes);
                writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); writer.Write(16);
                writer.Write((short)1); writer.Write((short)Math.Max(1, channels));
                writer.Write(rate); writer.Write(rate * Math.Max(1, channels) * 2);
                writer.Write((short)(Math.Max(1, channels) * 2)); writer.Write((short)16);
                writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write(bytes);
                for (int i = 0; i < count; i++)
                {
                    float value = samples[i];
                    if (float.IsNaN(value) || float.IsInfinity(value)) { invalid++; value = 0; }
                    square += value * value;
                    peak = Math.Max(peak, Math.Abs(value));
                    if (Math.Abs(value) >= .999f) clipped++;
                    writer.Write((short)(Math.Max(-1, Math.Min(1, value)) * 32767));
                }
            }
            AudioSettings.GetDSPBufferSize(out int length, out int buffers);
            File.WriteAllText(Path.Combine(folder, "audio.json"), JsonUtility.ToJson(new CaptureManifest {
                course = LabRuntime.Config.courseId, checkpoint = LabRuntime.Config.checkpointId,
                unity = Application.unityVersion,
                camera = Environment.GetEnvironmentVariable("RALLY_REVIEW_CAMERA"),
                sampleRate = rate, channels = channels, samples = count,
                bufferLength = length, bufferCount = buffers,
                listeners = FindObjectsByType<AudioListener>(FindObjectsSortMode.None).Length,
                seconds = count / (double)Math.Max(1, rate * channels),
                rms = count > 0 ? Math.Sqrt(square / count) : 0, peak = peak,
                nonFinite = invalid, fullScaleSamples = clipped }, true));
        }

        void OnApplicationQuit() => Save();
        void OnDestroy() => Save();
    }
}
