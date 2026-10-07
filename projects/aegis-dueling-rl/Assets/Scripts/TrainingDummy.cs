using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// A scripted opponent for M2 so the agent has something to learn against before self-play.
    /// Modes: Passive (just stands), Blocker (raises shield periodically), Aggressor (basic swings).
    /// Needs a Health (+ optional Weapon for Aggressor). SOLID (M2).
    /// </summary>
    public class TrainingDummy : MonoBehaviour
    {
        public enum Mode { Passive, Blocker, Aggressor }
        public Mode mode = Mode.Passive;

        public Health health;
        public Weapon sword;           // only used by Aggressor
        public float actionInterval = 1.5f;
        public Vector3 swingTiming = new Vector3(0.25f, 0.15f, 0.5f);

        float _timer, _swingT; int _swingPhase; // 0 idle,1 windup,2 active,3 recovery

        void OnEnable() { _timer = Random.Range(0f, actionInterval); _swingPhase = 0; }

        void FixedUpdate()
        {
            if (health == null || !health.IsAlive) return;
            float dt = Time.fixedDeltaTime;

            switch (mode)
            {
                case Mode.Passive:
                    break;
                case Mode.Blocker:
                    _timer -= dt;
                    if (_timer <= 0f) { health.IsBlockingInput = !health.IsBlockingInput; _timer = actionInterval; }
                    break;
                case Mode.Aggressor:
                    TickSwing(dt);
                    break;
            }
        }

        void TickSwing(float dt)
        {
            _swingT -= dt;
            if (_swingPhase == 0) { _timer -= dt; if (_timer <= 0f) { _swingPhase = 1; _swingT = swingTiming.x; } return; }
            if (_swingT > 0f) return;
            switch (_swingPhase)
            {
                case 1: _swingPhase = 2; _swingT = swingTiming.y; if (sword) sword.BeginSwing(); break;
                case 2: _swingPhase = 3; _swingT = swingTiming.z; if (sword) sword.EndSwing(); break;
                case 3: _swingPhase = 0; _timer = actionInterval; break;
            }
        }
    }
}
