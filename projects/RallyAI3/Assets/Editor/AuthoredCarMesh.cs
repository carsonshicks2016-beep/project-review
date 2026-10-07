using System;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace EditorScripts
{
    public static class AuthoredCarMesh
    {
        [Serializable] class Submesh { public int[] indices; }
        [Serializable] class Definition
        {
            public int schema;
            public string name;
            public Vector3[] positions, normals;
            public Vector2[] uvs;
            public Submesh[] submeshes;
        }

        public static Mesh Body() => Load("GC8_Body", 5);
        public static Mesh Wheel() => Load("GC8_Wheel", 2);
        public static Mesh Cockpit() => Load("GC8_Cockpit", 5);

        static Mesh Load(string name, int slots)
        {
            string path = $"Assets/Art/Models/Vehicles/{name}.json";
            var data = JsonUtility.FromJson<Definition>(File.ReadAllText(path));
            if (data.schema != 1 || data.submeshes.Length != slots ||
                data.positions.Length != data.normals.Length || data.positions.Length != data.uvs.Length)
                throw new InvalidDataException($"Invalid authored car mesh: {path}");
            var mesh = new Mesh { name = name, indexFormat = IndexFormat.UInt32 };
            mesh.vertices = data.positions;
            mesh.normals = data.normals;
            mesh.uv = data.uvs;
            mesh.subMeshCount = slots;
            for (int i = 0; i < slots; i++)
                mesh.SetTriangles(data.submeshes[i].indices, i);
            mesh.RecalculateBounds();
            if (mesh.bounds.size.x > 2.1f || mesh.bounds.size.y > 1.6f || mesh.bounds.size.z > 4.6f)
                throw new InvalidDataException($"Authored mesh exceeds visual envelope: {path}");
            return mesh;
        }
    }
}
