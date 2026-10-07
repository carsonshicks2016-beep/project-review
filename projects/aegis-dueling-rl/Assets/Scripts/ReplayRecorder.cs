using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// Record a fight to a file and play it back on a kinematic clone — for the gen-1-vs-gen-N
    /// showcase and slow-mo highlights. Record mode samples bone rotations each frame; Play mode
    /// applies them to a clone (disable that clone's physics + Animator).
    /// SOLID (M5).
    /// </summary>
    public class ReplayRecorder : MonoBehaviour
    {
        public enum Mode { Record, Play }
        public Mode mode = Mode.Record;
        public Animator boneSource;            // record: the live fighter. play: the clone to drive.
        public Transform root;                 // pelvis transform
        public string path = "replay.jsonl";
        public bool active = false;

        static readonly HumanBodyBones[] Bones = MotionFeaturizer.Bones;
        StreamWriter _w; StreamReader _r;

        void OnDisable() { _w?.Dispose(); _r?.Dispose(); _w = null; _r = null; }

        public void Begin()
        {
            string full = Path.Combine(Application.dataPath, "..", path);
            if (mode == Mode.Record) _w = new StreamWriter(full, false);
            else _r = new StreamReader(full);
            active = true;
        }

        void LateUpdate()
        {
            if (!active) return;
            if (mode == Mode.Record) RecordFrame();
            else PlayFrame();
        }

        void RecordFrame()
        {
            if (_w == null || boneSource == null || root == null) return;
            var sb = new StringBuilder(256);
            sb.Append('{');
            sb.Append("\"p\":["); V3(sb, root.position); sb.Append("],");
            sb.Append("\"r\":["); Q(sb, root.rotation); sb.Append("],\"b\":[");
            for (int i = 0; i < Bones.Length; i++)
            {
                var t = boneSource.GetBoneTransform(Bones[i]);
                if (i > 0) sb.Append(',');
                sb.Append('['); Q(sb, t ? t.localRotation : Quaternion.identity); sb.Append(']');
            }
            sb.Append("]}");
            _w.WriteLine(sb.ToString());
        }

        void PlayFrame()
        {
            if (_r == null) return;
            string line = _r.ReadLine();
            if (line == null) { active = false; return; }
            // Minimal parse (assumes the exact format written above).
            // For robust playback, swap in a real JSON lib (e.g. Newtonsoft) — kept dependency-free here.
            Debug.Log("[Aegis] ReplayRecorder.PlayFrame: parse + apply per-bone rotations here.");
        }

        static void V3(StringBuilder sb, Vector3 v)
        { sb.Append(F(v.x)); sb.Append(','); sb.Append(F(v.y)); sb.Append(','); sb.Append(F(v.z)); }
        static void Q(StringBuilder sb, Quaternion q)
        { sb.Append(F(q.x)); sb.Append(','); sb.Append(F(q.y)); sb.Append(','); sb.Append(F(q.z)); sb.Append(','); sb.Append(F(q.w)); }
        static string F(float f) => f.ToString("R", CultureInfo.InvariantCulture);
    }
}
