using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// Full combat agent (supersedes DuelistAgent once you reach M2). Hybrid action space:
    ///   • continuous = PD joint targets (motor control)
    ///   • discrete[0] = combat intent {0 none, 1 attack, 2 block}
    /// It imitates the clip that matches its current intent (via MotionLibrary), stays upright,
    /// and pursues the task: hit the opponent, don't get hit.
    ///
    /// M2: opponent = a TrainingDummy. M3: opponent = another DuelAgent, with a MatchManager
    /// assigning terminal win/lose rewards. Leave `opponent`/`opponentHealth` null for solo drills.
    ///
    /// SOLID structurally; reward weights, swing timing, and thresholds need tuning.
    /// </summary>
    public class DuelAgent : Agent
    {
        [Header("Body / motion")]
        public PDJointController controller;
        public MotionLibrary motion;
        public Animator ragdollAnimator;     // bone resolver (disable its updates)
        public ArticulationBody root;

        [Header("Combat")]
        public Health health;
        public Weapon sword;
        [Tooltip("Active-window timing within an attack (seconds): windup, active, recovery.")]
        public Vector3 attackTiming = new Vector3(0.20f, 0.15f, 0.35f);
        public float attackStaminaCost = 15f;

        [Header("Opponent (null for solo drills)")]
        public Transform opponent;
        public Health opponentHealth;

        [Header("Reward weights")]
        public float wUpright = 0.05f;
        public float wPose = 0.05f;
        public float wFace = 0.02f;       // face the opponent
        public float wDealt = 1.0f;       // per point of damage dealt
        public float wTaken = 1.0f;       // per point of damage taken
        public float wEnergy = 0.0005f;

        [Header("Termination")]
        public float minRootHeight = 0.6f;
        public float maxEpisodeSeconds = 30f;

        static readonly HumanBodyBones[] EndEffectors =
        { HumanBodyBones.LeftHand, HumanBodyBones.RightHand, HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot };

        // spawn-pose reset cache
        readonly Dictionary<HumanBodyBones, Transform> _bones = new Dictionary<HumanBodyBones, Transform>();
        readonly List<float> _initJointPos = new List<float>();
        readonly List<float> _zeroVel = new List<float>();
        Vector3 _initRootPos; Quaternion _initRootRot;

        // combat state
        enum Phase { Neutral, Windup, Active, Recovery }
        Phase _phase = Phase.Neutral;
        float _phaseTimer, _clipPhase, _elapsed, _prevDealt, _prevTaken;

        public MatchManager match;   // set by MatchManager in M3 (optional)

        public override void Initialize()
        {
            if (root == null) root = controller.root;
            foreach (var b in EndEffectors) CacheBone(b);
            _initRootPos = root.transform.position;
            _initRootRot = root.transform.rotation;
            root.GetJointPositions(_initJointPos);
            for (int i = 0; i < _initJointPos.Count; i++) _zeroVel.Add(0f);

            int obs = 8 + controller.JointStateCount + EndEffectors.Length * 3 + 8; // +8 opponent block
            Debug.Log($"[Aegis] DuelAgent → Continuous Actions = {controller.ActionSize}, " +
                      $"Discrete Branch = 1 of size 3, Vector Observation Space Size = {obs}");
        }

        void CacheBone(HumanBodyBones b)
        {
            var t = ragdollAnimator ? ragdollAnimator.GetBoneTransform(b) : null;
            if (t) _bones[b] = t;
        }

        public override void OnEpisodeBegin()
        {
            _elapsed = 0f; _clipPhase = 0f; _phase = Phase.Neutral; _phaseTimer = 0f;
            root.TeleportRoot(_initRootPos, _initRootRot);
            root.SetJointPositions(new List<float>(_initJointPos));
            root.SetJointVelocities(_zeroVel);
            root.linearVelocity = Vector3.zero; root.angularVelocity = Vector3.zero;
            controller.ResetTargets();
            if (health) health.ResetState();
            if (sword) sword.EndSwing();
            motion.SetState(MotionState.Idle);
            _prevDealt = _prevTaken = 0f;
        }

        void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;
            _elapsed += dt;
            _clipPhase += dt / Mathf.Max(motion.Duration, 0.01f);
            motion.Sample(_clipPhase);
            TickAttack(dt);
            if (_elapsed >= maxEpisodeSeconds && match == null) EndEpisode();
        }

        void TickAttack(float dt)
        {
            if (_phase == Phase.Neutral) return;
            _phaseTimer -= dt;
            if (_phaseTimer > 0f) return;
            switch (_phase)
            {
                case Phase.Windup:   _phase = Phase.Active;   _phaseTimer = attackTiming.y; sword.BeginSwing(); break;
                case Phase.Active:   _phase = Phase.Recovery; _phaseTimer = attackTiming.z; sword.EndSwing();   break;
                case Phase.Recovery: _phase = Phase.Neutral;  motion.SetState(MotionState.Idle);                break;
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            var tr = root.transform;
            sensor.AddObservation(Vector3.Dot(tr.up, Vector3.up));
            sensor.AddObservation(tr.position.y);
            sensor.AddObservation(tr.InverseTransformDirection(root.linearVelocity) * 0.1f);   // 3
            sensor.AddObservation((int)_phase / 3f);
            sensor.AddObservation(health ? health.Health01 : 1f);
            sensor.AddObservation(health ? health.Stamina01 : 1f);

            controller.AddJointObservations(sensor.AddObservation);
            foreach (var b in EndEffectors)
                sensor.AddObservation(_bones.TryGetValue(b, out var t) ? tr.InverseTransformPoint(t.position) : Vector3.zero);

            // opponent block (8): rel pos(3), rel dir facing dot(1), opp vel(3 -> approx 0), opp hp(1)
            if (opponent != null)
            {
                Vector3 rel = tr.InverseTransformPoint(opponent.position);
                sensor.AddObservation(rel);
                sensor.AddObservation(Vector3.Dot(tr.forward, (opponent.position - tr.position).normalized));
                sensor.AddObservation(opponent.position - tr.position); // world offset (3)
                sensor.AddObservation(opponentHealth ? opponentHealth.Health01 : 1f);
            }
            else
            {
                for (int i = 0; i < 8; i++) sensor.AddObservation(0f);
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            controller.ApplyActions(actions.ContinuousActions);
            HandleIntent(actions.DiscreteActions[0]);

            // --- reward ---
            float r = 0f;
            r += wUpright * Mathf.Clamp01(Vector3.Dot(root.transform.up, Vector3.up));
            if (opponent != null)
                r += wFace * Mathf.Clamp01(Vector3.Dot(root.transform.forward, (opponent.position - root.transform.position).normalized));

            if (health)
            {
                float dDealt = health.DamageDealt - _prevDealt; _prevDealt = health.DamageDealt;
                float dTaken = health.DamageTaken - _prevTaken; _prevTaken = health.DamageTaken;
                r += wDealt * dDealt - wTaken * dTaken;
            }

            float energy = 0f; var ca = actions.ContinuousActions;
            for (int i = 0; i < ca.Length; i++) energy += ca[i] * ca[i];
            r -= wEnergy * (ca.Length > 0 ? energy / ca.Length : 0f);

            AddReward(r);

            if (root.transform.position.y < minRootHeight) { AddReward(-1f); if (match) match.OnFighterFell(this); else EndEpisode(); }
        }

        void HandleIntent(int intent)
        {
            // 0 = none, 1 = attack, 2 = block
            if (health != null && health.IsStaggered) { health.IsBlockingInput = false; return; }

            if (intent == 2) { health.IsBlockingInput = true; if (_phase == Phase.Neutral) motion.SetState(MotionState.Block); return; }
            health.IsBlockingInput = false;

            if (intent == 1 && _phase == Phase.Neutral && health.TrySpendStamina(attackStaminaCost))
            {
                _phase = Phase.Windup; _phaseTimer = attackTiming.x;
                motion.SetState(MotionState.Slash);
            }
            else if (_phase == Phase.Neutral && motion.State != MotionState.Idle)
            {
                motion.SetState(MotionState.Idle);
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var ca = actionsOut.ContinuousActions;
            for (int i = 0; i < ca.Length; i++) ca[i] = 0f;
            var da = actionsOut.DiscreteActions;
            da[0] = 0;
        }
    }
}
