using System.Collections.Generic;
using UnityEngine;

namespace Core.Environment
{
    [ExecuteAlways]
    public sealed class DressingMeshOwner : MonoBehaviour
    {
        readonly List<Mesh> generated = new List<Mesh>();
        public void Register(Mesh mesh) { if (mesh != null && !generated.Contains(mesh)) generated.Add(mesh); }
        void OnDestroy()
        {
            foreach (Mesh mesh in generated)
            {
                if (mesh == null) continue;
                if (Application.isPlaying) Destroy(mesh);
                else DestroyImmediate(mesh);
            }
            generated.Clear();
        }
    }
}
