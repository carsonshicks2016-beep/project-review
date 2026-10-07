using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// M3 duel orchestrator: owns two DuelAgents (TeamId 0 / 1), resets both at episode start,
    /// detects win conditions (death, ring-out, timeout), assigns terminal rewards, and ends
    /// both episodes together. Pair with config/duel_ppo_selfplay.yaml (ML-Agents tracks ELO).
    ///
    /// SOLID structurally. Tune: spawn distance, ring bounds, terminal reward magnitudes.
    /// </summary>
    public class MatchManager : MonoBehaviour
    {
        public DuelAgent fighterA;
        public DuelAgent fighterB;

        [Header("Spawn")]
        public Transform spawnA;
        public Transform spawnB;

        [Header("Ring-out")]
        public float ringRadius = 6f;       // from this transform's position
        public float ringFloorY = 0f;
        public float fallBelowY = -1.5f;

        [Header("Match")]
        public float maxMatchSeconds = 30f;
        public float winReward = 1f;
        public float loseReward = -1f;
        public float drawReward = -0.1f;

        float _timer;
        bool _resolved;

        void Start() => WireCrossReferences();

        void WireCrossReferences()
        {
            fighterA.opponent = fighterB.root != null ? fighterB.root.transform : fighterB.transform;
            fighterB.opponent = fighterA.root != null ? fighterA.root.transform : fighterA.transform;
            fighterA.opponentHealth = fighterB.health;
            fighterB.opponentHealth = fighterA.health;
            fighterA.match = this; fighterB.match = this;
        }

        public void BeginMatch()
        {
            _timer = 0f; _resolved = false;
            if (spawnA) { fighterA.root.TeleportRoot(spawnA.position, spawnA.rotation); }
            if (spawnB) { fighterB.root.TeleportRoot(spawnB.position, spawnB.rotation); }
        }

        void FixedUpdate()
        {
            if (_resolved) return;
            _timer += Time.fixedDeltaTime;

            bool aOut = IsRingedOut(fighterA);
            bool bOut = IsRingedOut(fighterB);
            bool aDead = fighterA.health && !fighterA.health.IsAlive;
            bool bDead = fighterB.health && !fighterB.health.IsAlive;

            if (aDead || aOut) { Resolve(winner: fighterB, loser: fighterA); return; }
            if (bDead || bOut) { Resolve(winner: fighterA, loser: fighterB); return; }
            if (_timer >= maxMatchSeconds) { ResolveDraw(); }
        }

        bool IsRingedOut(DuelAgent f)
        {
            var p = f.root.transform.position;
            if (p.y < fallBelowY) return true;
            Vector3 c = transform.position; c.y = p.y;
            return Vector3.Distance(p, c) > ringRadius;
        }

        public void OnFighterFell(DuelAgent fallen)
        {
            // Called by a DuelAgent when its root drops below its own min height.
            if (_resolved) return;
            Resolve(winner: fallen == fighterA ? fighterB : fighterA, loser: fallen);
        }

        void Resolve(DuelAgent winner, DuelAgent loser)
        {
            _resolved = true;
            winner.AddReward(winReward);
            loser.AddReward(loseReward);
            winner.EndEpisode();
            loser.EndEpisode();
        }

        void ResolveDraw()
        {
            _resolved = true;
            // Edge to whoever has more health on a timeout.
            float ha = fighterA.health ? fighterA.health.Health01 : 0f;
            float hb = fighterB.health ? fighterB.health.Health01 : 0f;
            if (Mathf.Abs(ha - hb) < 0.05f) { fighterA.AddReward(drawReward); fighterB.AddReward(drawReward); }
            else if (ha > hb) { fighterA.AddReward(winReward * 0.5f); fighterB.AddReward(loseReward * 0.5f); }
            else { fighterB.AddReward(winReward * 0.5f); fighterA.AddReward(loseReward * 0.5f); }
            fighterA.EndEpisode(); fighterB.EndEpisode();
        }
    }
}
