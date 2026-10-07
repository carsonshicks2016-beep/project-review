using UnityEngine;

namespace Aegis
{
    public enum MotionState { Idle, Walk, Slash, Block, Hit }

    /// <summary>
    /// Holds the named reference clips and switches which one the ReferencePoseProvider plays,
    /// based on the agent's current combat state. The imitation reward then tracks whatever
    /// motion matches what the agent is trying to do (idle/walk/slash/block).
    /// SOLID (M2): wraps the existing provider; just data + clip switching.
    /// </summary>
    public class MotionLibrary : MonoBehaviour
    {
        public ReferencePoseProvider provider;

        [Header("Clips (Mixamo sword & shield)")]
        public AnimationClip idle;
        public AnimationClip walk;
        public AnimationClip slash;
        public AnimationClip block;
        public AnimationClip hitReact;

        public MotionState State { get; private set; } = MotionState.Idle;

        public void SetState(MotionState s)
        {
            if (s == State) return;
            State = s;
            provider.clip = ClipFor(s);
            provider.Sample(0f);   // restart the new clip from its first frame
        }

        public void Sample(float phase01) => provider.Sample(phase01);
        public Quaternion LocalRot(HumanBodyBones b) => provider.LocalRot(b);
        public float Duration => provider.Duration;

        AnimationClip ClipFor(MotionState s)
        {
            switch (s)
            {
                case MotionState.Walk: return walk != null ? walk : idle;
                case MotionState.Slash: return slash != null ? slash : idle;
                case MotionState.Block: return block != null ? block : idle;
                case MotionState.Hit: return hitReact != null ? hitReact : idle;
                default: return idle;
            }
        }
    }
}
