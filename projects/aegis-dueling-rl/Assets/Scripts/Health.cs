using System;
using UnityEngine;

namespace Aegis
{
    public enum HitResult { Clean, Blocked, GuardBroken }

    /// <summary>
    /// Combat resources for one fighter: HP, stamina, poise, and block resolution.
    /// SOLID (M2): plain game-logic, no physics-tuning dependence. Counters here feed the
    /// duel reward (damageDealt/damageTaken) in M3.
    /// </summary>
    public class Health : MonoBehaviour
    {
        [Header("Resources")]
        public float maxHealth = 100f;
        public float maxStamina = 100f;
        public float staminaRegenPerSec = 15f;
        public float maxPoise = 50f;
        public float poiseRegenPerSec = 20f;

        [Header("Blocking")]
        [Tooltip("Forward of the shield (assign the shield transform).")]
        public Transform shieldForward;
        [Tooltip("Half-angle of the block arc in degrees.")]
        public float blockArcDeg = 70f;
        public float blockStaminaCost = 20f;

        public float Health01 => health / maxHealth;
        public float Stamina01 => stamina / maxStamina;
        public bool IsAlive => health > 0f;
        public bool IsStaggered { get; private set; }
        public bool IsBlockingInput { get; set; }   // set by the agent each step

        public float DamageDealt { get; set; }       // bookkeeping for reward (attacker writes)
        public float DamageTaken { get; private set; }

        public event Action<HitResult, float> OnHit; // (result, damage)
        public event Action OnDeath;

        float health, stamina, poise, staggerTimer;

        void OnEnable() => ResetState();

        public void ResetState()
        {
            health = maxHealth; stamina = maxStamina; poise = maxPoise;
            IsStaggered = false; staggerTimer = 0f;
            DamageDealt = 0f; DamageTaken = 0f; IsBlockingInput = false;
        }

        void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;
            stamina = Mathf.Min(maxStamina, stamina + staminaRegenPerSec * dt);
            poise = Mathf.Min(maxPoise, poise + poiseRegenPerSec * dt);
            if (IsStaggered) { staggerTimer -= dt; if (staggerTimer <= 0f) IsStaggered = false; }
        }

        public bool TrySpendStamina(float amount)
        {
            if (stamina < amount) return false;
            stamina -= amount;
            return true;
        }

        /// Apply an incoming hit. attackDir = world direction the blow travels.
        public HitResult ApplyHit(float damage, float poiseDamage, Vector3 attackDir)
        {
            if (!IsAlive) return HitResult.Clean;

            if (IsBlockingInput && WithinBlockArc(attackDir))
            {
                if (TrySpendStamina(blockStaminaCost))
                {
                    OnHit?.Invoke(HitResult.Blocked, 0f);
                    return HitResult.Blocked;
                }
                // out of stamina -> guard break: take partial damage + stagger
                Stagger(0.8f);
                Damage(damage * 0.5f);
                OnHit?.Invoke(HitResult.GuardBroken, damage * 0.5f);
                return HitResult.GuardBroken;
            }

            Damage(damage);
            poise -= poiseDamage;
            if (poise <= 0f) { poise = maxPoise; Stagger(0.6f); }
            OnHit?.Invoke(HitResult.Clean, damage);
            return HitResult.Clean;
        }

        void Damage(float d)
        {
            health -= d;
            DamageTaken += d;
            if (health <= 0f) { health = 0f; OnDeath?.Invoke(); }
        }

        void Stagger(float seconds) { IsStaggered = true; staggerTimer = Mathf.Max(staggerTimer, seconds); }

        bool WithinBlockArc(Vector3 attackDir)
        {
            if (shieldForward == null) return false;
            // Block if the shield faces toward the incoming blow.
            return Vector3.Angle(shieldForward.forward, -attackDir.normalized) <= blockArcDeg;
        }
    }
}
