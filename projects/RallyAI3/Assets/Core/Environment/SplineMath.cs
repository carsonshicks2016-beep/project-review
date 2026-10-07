using UnityEngine;
using System.Collections.Generic;

namespace Core.Environment
{
    public static class SplineMath
    {
        // Catmull-Rom interpolation
        public static Vector3 GetCatmullRomPosition(float t, Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
        {
            Vector3 a = 2f * p1;
            Vector3 b = p2 - p0;
            Vector3 c = 2f * p0 - 5f * p1 + 4f * p2 - p3;
            Vector3 d = -p0 + 3f * p1 - 3f * p2 + p3;
            
            Vector3 pos = 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
            return pos;
        }

        // Tangent approximation
        public static Vector3 GetTangent(float t, Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
        {
            float dt = 0.01f;
            Vector3 pos1 = GetCatmullRomPosition(t, p0, p1, p2, p3);
            Vector3 pos2 = GetCatmullRomPosition(t + dt, p0, p1, p2, p3);
            return (pos2 - pos1).normalized;
        }
    }
}
