using System;
using System.Collections.Generic;
using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// Deterministic motion feature vector shared by the AMP discriminator dataset (reference
    /// clone) and the policy (ragdoll). Same bones, same order → comparable transitions.
    /// SOLID: pure feature extraction. The feature CHOICE is a design knob you can revise.
    /// </summary>
    public static class MotionFeaturizer
    {
        public static readonly HumanBodyBones[] Bones =
        {
            HumanBodyBones.Spine, HumanBodyBones.Chest, HumanBodyBones.Head,
            HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand,
            HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand,
            HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot,
            HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot,
        };

        // root(up.y=1 + height=1) + per-bone localRotation(4) + 4 end-effectors rel root (3)
        public static int Size => 2 + Bones.Length * 4 + 4 * 3;

        public static void Write(Transform root, Func<HumanBodyBones, Transform> resolve, List<float> outBuf)
        {
            outBuf.Clear();
            outBuf.Add(Vector3.Dot(root.up, Vector3.up));
            outBuf.Add(root.position.y);
            foreach (var b in Bones)
            {
                var t = resolve(b);
                Quaternion q = t ? t.localRotation : Quaternion.identity;
                outBuf.Add(q.x); outBuf.Add(q.y); outBuf.Add(q.z); outBuf.Add(q.w);
            }
            AddRel(root, resolve(HumanBodyBones.LeftHand), outBuf);
            AddRel(root, resolve(HumanBodyBones.RightHand), outBuf);
            AddRel(root, resolve(HumanBodyBones.LeftFoot), outBuf);
            AddRel(root, resolve(HumanBodyBones.RightFoot), outBuf);
        }

        static void AddRel(Transform root, Transform t, List<float> buf)
        {
            Vector3 p = t ? root.InverseTransformPoint(t.position) : Vector3.zero;
            buf.Add(p.x); buf.Add(p.y); buf.Add(p.z);
        }
    }
}
