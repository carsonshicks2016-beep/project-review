using System.Collections.Generic;
using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// Sword blade hit detection. Put a trigger collider along the blade and assign this.
    /// Damage only lands while the swing is in its ACTIVE window AND the tip is moving fast
    /// enough — so a slow/parried swing does nothing (physics-aware, the point of going ragdoll).
    /// SOLID (M2) apart from tip-speed / damage numbers, which you'll tune.
    /// </summary>
    [RequireComponent(typeof(Collider))]
    public class Weapon : MonoBehaviour
    {
        [Header("Damage")]
        public float damage = 18f;
        public float poiseDamage = 20f;
        public float minTipSpeed = 2.0f;   // m/s required to deal damage — TUNE

        [Tooltip("Health of the fighter wielding this weapon (so we never self-hit).")]
        public Health owner;

        [Tooltip("Tip transform for speed measurement (empty = use this collider's transform).")]
        public Transform tip;

        public bool Active { get; private set; }      // set true during the attack active frames

        readonly HashSet<Health> _hitThisSwing = new HashSet<Health>();
        Vector3 _prevTipPos;
        float _tipSpeed;

        void Awake()
        {
            if (tip == null) tip = transform;
            GetComponent<Collider>().isTrigger = true;
            _prevTipPos = tip.position;
        }

        void FixedUpdate()
        {
            _tipSpeed = (tip.position - _prevTipPos).magnitude / Mathf.Max(Time.fixedDeltaTime, 1e-5f);
            _prevTipPos = tip.position;
        }

        public void BeginSwing() { Active = true; _hitThisSwing.Clear(); }
        public void EndSwing() { Active = false; }

        void OnTriggerEnter(Collider other)
        {
            if (!Active || _tipSpeed < minTipSpeed) return;

            var target = other.GetComponentInParent<Health>();
            if (target == null || target == owner) return;          // ignore self
            if (_hitThisSwing.Contains(target)) return;             // one hit per swing
            _hitThisSwing.Add(target);

            Vector3 dir = (target.transform.position - tip.position).normalized;
            var result = target.ApplyHit(damage, poiseDamage, dir);
            if (owner != null && result != HitResult.Blocked) owner.DamageDealt += (result == HitResult.GuardBroken ? damage * 0.5f : damage);
        }
    }
}
