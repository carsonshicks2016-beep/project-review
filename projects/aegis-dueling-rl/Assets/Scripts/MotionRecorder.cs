using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace Aegis
{
    /// <summary>
    /// Records motion TRANSITIONS (s, s') as JSONL for the AMP discriminator.
    /// Put one on the reference clone (label="expert") to build the real-motion dataset; put one
    /// on a fighter (label="policy") to capture agent motion for discriminator updates.
    /// SOLID, but only needed at M4. See docs/M4_AMP.md.
    /// </summary>
    public class MotionRecorder : MonoBehaviour
    {
        public ArticulationBody root;          // fighter: pelvis. reference: leave null, set rootTransform.
        public Transform rootTransform;        // used if root is null (reference clone)
        public Animator boneSource;            // resolves bones
        public string label = "expert";
        public string outputPath = "motion_expert.jsonl";
        public bool recording = false;

        readonly List<float> _cur = new List<float>();
        readonly List<float> _prev = new List<float>();
        StreamWriter _w;
        bool _hasPrev;

        Transform Root => root != null ? root.transform : rootTransform;

        void OnEnable()
        {
            if (recording) Open();
        }

        void OnDisable() => Close();

        public void Open()
        {
            _w = new StreamWriter(Path.Combine(Application.dataPath, "..", outputPath), append: false);
            _hasPrev = false;
        }

        public void Close()
        {
            _w?.Flush(); _w?.Dispose(); _w = null;
        }

        void FixedUpdate()
        {
            if (!recording || _w == null || Root == null || boneSource == null) return;
            MotionFeaturizer.Write(Root, boneSource.GetBoneTransform, _cur);
            if (_hasPrev) WriteLine(_prev, _cur);
            _prev.Clear(); _prev.AddRange(_cur);
            _hasPrev = true;
        }

        void WriteLine(List<float> s, List<float> sNext)
        {
            var sb = new StringBuilder(256);
            sb.Append("{\"s\":[");
            Append(sb, s);
            sb.Append("],\"s2\":[");
            Append(sb, sNext);
            sb.Append("]}");
            _w.WriteLine(sb.ToString());
        }

        static void Append(StringBuilder sb, List<float> v)
        {
            for (int i = 0; i < v.Count; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(v[i].ToString("R", CultureInfo.InvariantCulture));
            }
        }
    }
}
