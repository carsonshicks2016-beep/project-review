using UnityEngine;

namespace Core.Environment
{
    /// <summary>
    /// Keeps the distant ridge line centred on the camera in the horizontal plane,
    /// so it stays the same distance away wherever the car is on the stage.
    ///
    /// A static ring cannot work here. The stage is a kilometre long, so a ring big
    /// enough to enclose it would be a hundred metres away at one end and a
    /// kilometre away at the other — a wall on one lap and invisible on the next.
    /// Translating with the viewer is what distant scenery has always done: hills
    /// that never get closer, which is exactly how real ones behave at this range.
    ///
    /// Position only, and only in X and Z. Rotation is left alone or the ridge line
    /// would swing round with the camera, and Y is held fixed or cresting a rise
    /// would drag the whole horizon up with the car.
    /// </summary>
    [ExecuteAlways]
    public class BackdropFollower : MonoBehaviour
    {
        [Tooltip("Camera to track. Falls back to Camera.main when empty.")]
        public Transform target;

        [Tooltip("World height the base of the ridge sits at. Held constant on purpose.")]
        public float fixedHeight;

        void LateUpdate()
        {
            Transform t = target;

            // ?. would skip Unity's == overload and call through a destroyed object.
            if (t == null)
            {
                Camera cam = Camera.main;
                if (cam == null) return;
                t = cam.transform;
            }

            Vector3 p = t.position;
            transform.position = new Vector3(p.x, fixedHeight, p.z);
        }
    }
}
