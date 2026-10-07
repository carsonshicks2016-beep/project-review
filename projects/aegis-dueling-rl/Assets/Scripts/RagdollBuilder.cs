using System.Collections.Generic;
using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// Builds an ArticulationBody active-ragdoll rig from a Humanoid Animator.
    /// Add this to the imported Mixamo character (Rig = Humanoid), then use the
    /// component context menu ("Build Ragdoll") in the editor. Re-runnable: "Clear Ragdoll"
    /// strips a previous build.
    ///
    /// IMPORTANT (tuning): anchor *rotations*, joint *axes* (esp. knees/elbows hinge axis),
    /// limits, and drive gains depend on your rig and WILL need editor tuning. Build first,
    /// then refine in the Inspector. See docs/M1_GUIDE.md.
    /// </summary>
    [RequireComponent(typeof(Animator))]
    public class RagdollBuilder : MonoBehaviour
    {
        enum Kind { Root, Spherical, Revolute, Fixed }

        struct Spec
        {
            public HumanBodyBones bone;
            public HumanBodyBones child;   // for capsule direction/length; same as bone if leaf
            public Kind kind;
            public float mass;
            public float radius;
            public float twist;            // X half-range (deg)
            public float swingY;           // Y half-range (deg)
            public float swingZ;           // Z half-range (deg)
            public float lowRev, highRev;  // revolute range (deg)
            public float stiffness, damping, forceLimit;
            public bool lateralHinge;      // revolute: align hinge to world lateral (X) axis at bind
            public bool footSole;          // use a flat box sole + high friction instead of a capsule
        }

        // Root-first ordering matters (root AB must exist before children).
        static readonly Spec[] Skeleton =
        {
            S(HumanBodyBones.Hips,          HumanBodyBones.Spine,         Kind.Root,      11f, 0.13f),
            Sph(HumanBodyBones.Spine,       HumanBodyBones.Chest,         8f, 0.12f, 20,20,20, 2000,200),
            Sph(HumanBodyBones.Chest,       HumanBodyBones.Head,          16f,0.14f, 20,20,20, 3000,300),
            Sph(HumanBodyBones.Head,        HumanBodyBones.Head,          5f, 0.10f, 30,30,30,  500, 50),

            Sph(HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, 2.5f,0.05f, 45,90,90, 800,80),
            Rev(HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand,     1.6f,0.04f, 0,150,     600,60),
            Fix(HumanBodyBones.LeftHand,     HumanBodyBones.LeftHand,     0.6f,0.04f),

            Sph(HumanBodyBones.RightUpperArm,HumanBodyBones.RightLowerArm,2.5f,0.05f, 45,90,90, 800,80),
            Rev(HumanBodyBones.RightLowerArm,HumanBodyBones.RightHand,    1.6f,0.04f, 0,150,     600,60),
            Fix(HumanBodyBones.RightHand,    HumanBodyBones.RightHand,    0.6f,0.04f),

            Sph(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, 7f, 0.08f, 30,80,40, 4000,400),
            Rev(HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot,     3.2f,0.06f, 0,150,    3000,300, true),
            Sph(HumanBodyBones.LeftFoot,     HumanBodyBones.LeftToes,     1.0f,0.05f, 20,30,20,  2500,250, true),

            Sph(HumanBodyBones.RightUpperLeg,HumanBodyBones.RightLowerLeg,7f, 0.08f, 30,80,40, 4000,400),
            Rev(HumanBodyBones.RightLowerLeg,HumanBodyBones.RightFoot,    3.2f,0.06f, 0,150,    3000,300, true),
            Sph(HumanBodyBones.RightFoot,    HumanBodyBones.RightToes,    1.0f,0.05f, 20,30,20,  2500,250, true),
        };

        [ContextMenu("Build Ragdoll")]
        public void Build()
        {
            var anim = GetComponent<Animator>();
            if (anim == null || !anim.isHuman)
            {
                Debug.LogError("[Aegis] RagdollBuilder needs a Humanoid Animator on this object.");
                return;
            }
            Clear();

            foreach (var s in Skeleton)
            {
                var t = anim.GetBoneTransform(s.bone);
                if (t == null) { Debug.LogWarning($"[Aegis] Missing bone {s.bone}, skipping."); continue; }

                var ab = t.gameObject.AddComponent<ArticulationBody>();
                ab.mass = s.mass;

                if (s.kind == Kind.Root)
                {
                    ab.immovable = false;                 // floating base, free to fall
                    ab.useGravity = true;
                }
                else
                {
                    ab.anchorPosition = Vector3.zero;     // bone pivot == joint (true for Mixamo rigs)
                    // Knees use a lateral hinge so they bend in the sagittal plane (no sideways/back
                    // fold); other joints keep identity. TUNE: flip the knee range sign if it bends
                    // the wrong way.
                    ab.anchorRotation = s.lateralHinge ? Quaternion.Inverse(t.rotation) : Quaternion.identity;
                    ab.matchAnchors = true;

                    switch (s.kind)
                    {
                        case Kind.Spherical:
                            ab.jointType = ArticulationJointType.SphericalJoint;
                            ab.twistLock = ArticulationDofLock.LimitedMotion;
                            ab.swingYLock = ArticulationDofLock.LimitedMotion;
                            ab.swingZLock = ArticulationDofLock.LimitedMotion;
                            ab.xDrive = Drive(-s.twist, s.twist, s.stiffness, s.damping, s.forceLimit);
                            ab.yDrive = Drive(-s.swingY, s.swingY, s.stiffness, s.damping, s.forceLimit);
                            ab.zDrive = Drive(-s.swingZ, s.swingZ, s.stiffness, s.damping, s.forceLimit);
                            break;
                        case Kind.Revolute:
                            ab.jointType = ArticulationJointType.RevoluteJoint;
                            ab.twistLock = ArticulationDofLock.LimitedMotion;
                            ab.xDrive = Drive(s.lowRev, s.highRev, s.stiffness, s.damping, s.forceLimit);
                            break;
                        case Kind.Fixed:
                            ab.jointType = ArticulationJointType.FixedJoint;
                            break;
                    }
                }

                if (s.footSole) AddFootSole(t, anim.GetBoneTransform(s.child), FootMaterial);
                else AddCapsule(t, anim.GetBoneTransform(s.child), s.radius);
            }

            Debug.Log("[Aegis] Ragdoll built. Press Play to confirm it flops believably, then tune.");
        }

        [ContextMenu("Clear Ragdoll")]
        public void Clear()
        {
            foreach (var ab in GetComponentsInChildren<ArticulationBody>()) DestroyImmediate(ab);
            foreach (var c in GetComponentsInChildren<Collider>()) DestroyImmediate(c);
        }

        static PhysicsMaterial _footMat;
        static PhysicsMaterial FootMaterial =>
            _footMat != null ? _footMat
            : (_footMat = new PhysicsMaterial("AegisFoot") { dynamicFriction = 1.2f, staticFriction = 1.4f });

        // Flat, world-aligned box "sole" under the foot — a real base of support (capsules roll).
        static void AddFootSole(Transform foot, Transform toes, PhysicsMaterial mat)
        {
            Vector3 ankleW = foot.position;
            Vector3 toeW = toes != null ? toes.position : foot.position + foot.forward * 0.18f;
            Vector3 fwd = toeW - ankleW; fwd.y = 0f;
            float len = Mathf.Max(fwd.magnitude, 0.05f) + 0.10f;
            fwd = fwd.sqrMagnitude > 1e-6f ? fwd.normalized : Vector3.forward;
            Vector3 center = (ankleW + toeW) * 0.5f; center.y = 0.03f;
            var sole = new GameObject("Sole");
            sole.transform.SetParent(foot, true);
            sole.transform.position = center;
            sole.transform.rotation = Quaternion.LookRotation(fwd, Vector3.up);
            var box = sole.AddComponent<BoxCollider>();
            box.size = new Vector3(0.10f, 0.06f, len);
            if (mat != null) box.sharedMaterial = mat;
        }

        static void AddCapsule(Transform t, Transform child, float radius)
        {
            var col = t.gameObject.AddComponent<CapsuleCollider>();
            col.radius = radius;
            if (child != null && child != t)
            {
                Vector3 local = t.InverseTransformPoint(child.position);
                col.height = Mathf.Max(local.magnitude, radius * 2f);
                col.center = local * 0.5f;
                col.direction = DominantAxis(local);
            }
            else
            {
                col.height = radius * 2.5f;
                col.center = Vector3.zero;
                col.direction = 1; // Y
            }
        }

        static int DominantAxis(Vector3 v)
        {
            v = new Vector3(Mathf.Abs(v.x), Mathf.Abs(v.y), Mathf.Abs(v.z));
            if (v.x >= v.y && v.x >= v.z) return 0;
            return v.y >= v.z ? 1 : 2;
        }

        static ArticulationDrive Drive(float low, float high, float stiff, float damp, float forceLimit)
        {
            return new ArticulationDrive
            {
                lowerLimit = low, upperLimit = high,
                stiffness = stiff, damping = damp,
                forceLimit = forceLimit, target = 0f, targetVelocity = 0f
            };
        }

        // --- spec constructors ---
        static Spec S(HumanBodyBones b, HumanBodyBones c, Kind k, float m, float r) =>
            new Spec { bone = b, child = c, kind = k, mass = m, radius = r, forceLimit = 1000f };
        static Spec Sph(HumanBodyBones b, HumanBodyBones c, float m, float r, float tw, float sy, float sz, float st, float dp, bool footSole = false) =>
            new Spec { bone = b, child = c, kind = Kind.Spherical, mass = m, radius = r, twist = tw, swingY = sy, swingZ = sz, stiffness = st, damping = dp, forceLimit = 1500f, footSole = footSole };
        static Spec Rev(HumanBodyBones b, HumanBodyBones c, float m, float r, float lo, float hi, float st, float dp, bool lateralHinge = false) =>
            new Spec { bone = b, child = c, kind = Kind.Revolute, mass = m, radius = r, lowRev = lo, highRev = hi, stiffness = st, damping = dp, forceLimit = 1500f, lateralHinge = lateralHinge };
        static Spec Fix(HumanBodyBones b, HumanBodyBones c, float m, float r) =>
            new Spec { bone = b, child = c, kind = Kind.Fixed, mass = m, radius = r, forceLimit = 1000f };
    }
}
