using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// M1 agent: learn to BALANCE and track a reference motion under PD torque control.
    /// No opponent yet (that's M3). Reward = pose imitation + upright + small energy penalty.
    /// Reset is to the spawn pose (robust); reference-state-init (random phase) is an
    /// optional upgrade described in M1_GUIDE.md.
    /// </summary>
    public class DuelistAgent : Agent
    {
        [Header("Wiring")]
        public PDJointController controller;
        public ReferencePoseProvider reference;
        [Tooltip("Animator on the ragdoll (Humanoid) — used only to resolve bone transforms. Disable its updates.")]
        public Animator ragdollAnimator;
        public ArticulationBody root;            // pelvis; auto = controller.root

        [Header("Reward weights (balance-focused)")]
        public float wUpright = 0.30f;   // head above pelvis, world axis (robust to bone orientation)
        public float wHeight = 0.30f;    // pelvis near its standing height (don't sink)
        public float wCOM = 0.20f;       // pelvis horizontally over the feet (base of support)
        public float wPose = 0.20f;      // idle imitation, kept secondary
        public float aliveBonus = 0.02f; // per-step survival incentive
        public float wEnergy = 0.001f;

        [Header("Termination")]
        public float maxEpisodeSeconds = 20f;

        static readonly HumanBodyBones[] CompareBones =
        {
            HumanBodyBones.Spine, HumanBodyBones.Chest, HumanBodyBones.Head,
            HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm,
            HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm,
            HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot,
            HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot,
        };
        static readonly HumanBodyBones[] EndEffectors =
        {
            HumanBodyBones.LeftHand, HumanBodyBones.RightHand,
            HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot,
        };

        // caches
        readonly Dictionary<HumanBodyBones, Transform> _ragBones = new Dictionary<HumanBodyBones, Transform>();
        readonly List<float> _initJointPos = new List<float>();
        readonly List<float> _zeroJointVel = new List<float>();
        Vector3 _initRootPos; Quaternion _initRootRot;
        float _phase, _elapsed;

        void Awake()
        {
            // Auto-size Behavior Parameters from the actual rig, so observation/action counts
            // never need hand-entering and are guaranteed to match what CollectObservations emits.
            // Runs before the Agent's policy is created (OnEnable).
            if (controller == null) return;
            controller.EnsureDiscovered();
            var bp = GetComponent<BehaviorParameters>();
            if (bp != null)
            {
                bp.BrainParameters.VectorObservationSize = 10 + controller.JointStateCount + EndEffectors.Length * 3;
                bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(controller.ActionSize);
            }
        }

        public override void Initialize()
        {
            if (root == null) root = controller.root;

            foreach (var b in CompareBones) Cache(b);
            foreach (var b in EndEffectors) Cache(b);

            _initRootPos = root.transform.position;
            _initRootRot = root.transform.rotation;
            root.GetJointPositions(_initJointPos);
            for (int i = 0; i < _initJointPos.Count; i++) _zeroJointVel.Add(0f);

            int obs = 10 + controller.JointStateCount + EndEffectors.Length * 3;
            Debug.Log($"[Aegis] Behavior Parameters → Continuous Actions = {controller.ActionSize}, " +
                      $"Vector Observation Space Size = {obs}");
        }

        void Cache(HumanBodyBones b)
        {
            if (_ragBones.ContainsKey(b)) return;
            var t = ragdollAnimator != null ? ragdollAnimator.GetBoneTransform(b) : null;
            if (t != null) _ragBones[b] = t;
        }

        public override void OnEpisodeBegin()
        {
            _elapsed = 0f;
            _phase = 0f;

            // Reset physics state to the spawn pose.
            root.TeleportRoot(_initRootPos, _initRootRot);
            var pos = new List<float>(_initJointPos);
            root.SetJointPositions(pos);
            root.SetJointVelocities(_zeroJointVel);
            root.linearVelocity = Vector3.zero;
            root.angularVelocity = Vector3.zero;
            controller.ResetTargets();

            reference.Sample(0f);
        }

        void FixedUpdate()
        {
            // Advance the reference clip independent of the decision cadence.
            _elapsed += Time.fixedDeltaTime;
            _phase += Time.fixedDeltaTime / Mathf.Max(reference.Duration, 0.01f);
            reference.Sample(_phase);
            if (_elapsed >= maxEpisodeSeconds) EndEpisode();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            var tr = root.transform;
            sensor.AddObservation(Vector3.Dot(tr.up, Vector3.up));          // balance
            sensor.AddObservation(tr.position.y);                           // height
            sensor.AddObservation(tr.InverseTransformDirection(root.linearVelocity) * 0.1f);
            sensor.AddObservation(tr.InverseTransformDirection(root.angularVelocity) * 0.1f);
            sensor.AddObservation(Mathf.Sin(_phase * 2f * Mathf.PI));       // phase
            sensor.AddObservation(Mathf.Cos(_phase * 2f * Mathf.PI));
            controller.AddJointObservations(sensor.AddObservation);
            foreach (var b in EndEffectors)
                if (_ragBones.TryGetValue(b, out var t))
                    sensor.AddObservation(tr.InverseTransformPoint(t.position));
                else
                    sensor.AddObservation(Vector3.zero);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            var act = actions.ContinuousActions;
            controller.ApplyActions(act);

            Vector3 pelvis = root.transform.position;
            Vector3 head = BonePos(HumanBodyBones.Head, pelvis + Vector3.up * 0.5f);
            Vector3 lf = BonePos(HumanBodyBones.LeftFoot, pelvis);
            Vector3 rf = BonePos(HumanBodyBones.RightFoot, pelvis);

            // Uprightness from world positions (head above pelvis) — independent of bone axes.
            Vector3 axis = head - pelvis;
            float aLen = axis.magnitude;
            float rUpright = aLen > 1e-4f ? Mathf.Clamp01(Vector3.Dot(axis / aLen, Vector3.up)) : 0f;

            // Stay near standing height (don't crumple).
            float dH = pelvis.y - _initRootPos.y;
            float rHeight = Mathf.Exp(-5f * dH * dH);

            // Keep the pelvis over the base of support (between the feet), horizontally.
            Vector3 feetMid = (lf + rf) * 0.5f;
            float dx = pelvis.x - feetMid.x, dz = pelvis.z - feetMid.z;
            float rCOM = Mathf.Exp(-4f * (dx * dx + dz * dz));

            float rPose = PoseReward();

            float energy = 0f;
            for (int i = 0; i < act.Length; i++) energy += act[i] * act[i];
            energy = act.Length > 0 ? energy / act.Length : 0f;

            AddReward(wUpright * rUpright + wHeight * rHeight + wCOM * rCOM
                      + wPose * rPose + aliveBonus - wEnergy * energy);

            // Robust fall termination: head no longer above pelvis, or pelvis dropped a lot.
            if (rUpright < 0.4f || pelvis.y < _initRootPos.y * 0.55f)
            {
                AddReward(-1f);
                EndEpisode();
            }
        }

        Vector3 BonePos(HumanBodyBones b, Vector3 fallback)
        {
            return _ragBones.TryGetValue(b, out var t) ? t.position : fallback;
        }

        float PoseReward()
        {
            float sumSq = 0f; int n = 0;
            foreach (var b in CompareBones)
            {
                if (!_ragBones.TryGetValue(b, out var t)) continue;
                float deg = Quaternion.Angle(t.localRotation, reference.LocalRot(b));
                float rad = deg * Mathf.Deg2Rad;
                sumSq += rad * rad;
                n++;
            }
            if (n == 0) return 0f;
            return Mathf.Exp(-2f * sumSq / n);   // 1 = perfect match, →0 as it diverges
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var ca = actionsOut.ContinuousActions;
            for (int i = 0; i < ca.Length; i++) ca[i] = 0f; // hold bind pose
        }
    }
}
