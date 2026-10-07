using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// Maps a flat action vector to ArticulationDrive targets (PD position control),
    /// and exposes reduced-coordinate joint state for observations.
    /// Auto-discovers actuated DoFs from the rig built by RagdollBuilder, so the
    /// action size is whatever the rig has — read it from ActionSize and set the
    /// Behavior Parameters "Continuous Actions" to match.
    /// </summary>
    public class PDJointController : MonoBehaviour
    {
        [Tooltip("The pelvis ArticulationBody (articulation root). Auto-found if left empty.")]
        public ArticulationBody root;

        struct Axis { public ArticulationBody body; public int axis; public float low, high; } // axis 0=x,1=y,2=z

        readonly List<Axis> _axes = new List<Axis>();
        readonly List<float> _pos = new List<float>();
        readonly List<float> _vel = new List<float>();

        public int ActionSize => _axes.Count;
        public int JointStateCount { get { root.GetJointPositions(_pos); return _pos.Count * 2; } }

        bool _discovered;

        void Awake() => EnsureDiscovered();

        /// Idempotent; safe to call from another component's Awake to force discovery early.
        public void EnsureDiscovered()
        {
            if (_discovered) return;
            if (root == null) root = GetComponentInChildren<ArticulationBody>();
            Discover();
            _discovered = true;
        }

        void Discover()
        {
            _axes.Clear();
            foreach (var ab in root.GetComponentsInChildren<ArticulationBody>())
            {
                if (ab == root) continue;
                switch (ab.jointType)
                {
                    case ArticulationJointType.RevoluteJoint:
                        if (ab.twistLock != ArticulationDofLock.LockedMotion)
                            _axes.Add(new Axis { body = ab, axis = 0, low = ab.xDrive.lowerLimit, high = ab.xDrive.upperLimit });
                        break;
                    case ArticulationJointType.SphericalJoint:
                        if (ab.twistLock != ArticulationDofLock.LockedMotion)
                            _axes.Add(new Axis { body = ab, axis = 0, low = ab.xDrive.lowerLimit, high = ab.xDrive.upperLimit });
                        if (ab.swingYLock != ArticulationDofLock.LockedMotion)
                            _axes.Add(new Axis { body = ab, axis = 1, low = ab.yDrive.lowerLimit, high = ab.yDrive.upperLimit });
                        if (ab.swingZLock != ArticulationDofLock.LockedMotion)
                            _axes.Add(new Axis { body = ab, axis = 2, low = ab.zDrive.lowerLimit, high = ab.zDrive.upperLimit });
                        break;
                }
            }
        }

        /// Action in [-1,1] per DoF -> drive target lerped within that DoF's limits.
        public void ApplyActions(ActionSegment<float> act)
        {
            int n = Mathf.Min(act.Length, _axes.Count);
            for (int i = 0; i < n; i++)
            {
                var a = _axes[i];
                float t = Mathf.Lerp(a.low, a.high, 0.5f * (Mathf.Clamp(act[i], -1f, 1f) + 1f));
                SetTarget(a.body, a.axis, t);
            }
        }

        /// Reset all drive targets to neutral (0 deg = bind pose for symmetric joints).
        public void ResetTargets()
        {
            foreach (var a in _axes) SetTarget(a.body, a.axis, 0f);
        }

        static void SetTarget(ArticulationBody b, int axis, float target)
        {
            switch (axis)
            {
                case 0: { var d = b.xDrive; d.target = target; b.xDrive = d; break; }
                case 1: { var d = b.yDrive; d.target = target; b.yDrive = d; break; }
                case 2: { var d = b.zDrive; d.target = target; b.zDrive = d; break; }
            }
        }

        /// Push reduced-coordinate joint positions + (scaled) velocities into a sensor callback.
        public void AddJointObservations(System.Action<float> add)
        {
            root.GetJointPositions(_pos);
            root.GetJointVelocities(_vel);
            for (int i = 0; i < _pos.Count; i++) add(_pos[i]);
            for (int i = 0; i < _vel.Count; i++) add(Mathf.Clamp(_vel[i] * 0.1f, -5f, 5f));
        }
    }
}
