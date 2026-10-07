using System.Threading;

namespace Audio
{
    public enum VehicleSoundEvent { Shift, TurboRelease, Landing, Collision }

    public struct SoundEvent
    {
        public VehicleSoundEvent kind;
        public float strength;
        public int generation;
    }

    // Single main-thread producer and audio-thread consumer. Overflow drops new events, never blocks.
    public sealed class AudioEventQueue
    {
        readonly SoundEvent[] events = new SoundEvent[32];
        int read, write, dropped;
        public int Dropped => Volatile.Read(ref dropped);

        public bool Enqueue(VehicleSoundEvent kind, float strength, int generation)
        {
            int current = write;
            int next = (current + 1) % events.Length;
            if (next == Volatile.Read(ref read)) { Interlocked.Increment(ref dropped); return false; }
            events[current] = new SoundEvent { kind = kind, strength = strength, generation = generation };
            Volatile.Write(ref write, next);
            return true;
        }

        public bool TryDequeue(out SoundEvent value)
        {
            int current = read;
            if (current == Volatile.Read(ref write)) { value = default(SoundEvent); return false; }
            value = events[current];
            Volatile.Write(ref read, (current + 1) % events.Length);
            return true;
        }
    }
}
