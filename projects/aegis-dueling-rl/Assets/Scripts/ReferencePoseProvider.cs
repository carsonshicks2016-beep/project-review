using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// A kinematic Humanoid clone that plays the reference motion clip. The agent samples
    /// it each physics step to get target bone rotations for the DeepMimic-style imitation
    /// reward. This object is logic-only — hide its renderers.
    ///
    /// NOTE: AnimationClip.SampleAnimation on Humanoid clips can be finicky. If poses look
    /// wrong, switch to an AnimatorController with one state and use
    /// animator.Play(state, 0, phase01); animator.Update(0f);  (see M1_GUIDE.md).
    /// </summary>
    public class ReferencePoseProvider : MonoBehaviour
    {
        [Tooltip("Animator of the kinematic clone (Humanoid).")]
        public Animator referenceAnimator;

        [Tooltip("Reference motion (start with an idle clip).")]
        public AnimationClip clip;

        public float Phase01 { get; private set; }
        public float Duration => clip != null ? clip.length : 1f;

        public void Sample(float time01)
        {
            Phase01 = Mathf.Repeat(time01, 1f);
            if (clip != null && referenceAnimator != null)
                clip.SampleAnimation(referenceAnimator.gameObject, Phase01 * clip.length);
        }

        public Quaternion LocalRot(HumanBodyBones b)
        {
            var t = referenceAnimator.GetBoneTransform(b);
            return t ? t.localRotation : Quaternion.identity;
        }

        public Vector3 WorldPos(HumanBodyBones b)
        {
            var t = referenceAnimator.GetBoneTransform(b);
            return t ? t.position : Vector3.zero;
        }

        public Transform Root => referenceAnimator != null ? referenceAnimator.GetBoneTransform(HumanBodyBones.Hips) : null;
    }
}
