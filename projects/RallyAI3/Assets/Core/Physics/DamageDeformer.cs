using UnityEngine;

namespace Core.Physics
{
    [RequireComponent(typeof(MeshFilter))]
    public class DamageDeformer : MonoBehaviour
    {
        public ComputeShader deformationCompute;
        public float impactThreshold = 10f; // Minimum impulse to trigger damage
        public float deformationMultiplier = 0.05f;
        public float maxDeformation = 0.5f;

        private Mesh filterMesh;
        private Mesh originalMesh;
        private Vector3[] originalVertices;
        private Vector3[] currentVertices;
        
        private ComputeBuffer originalVerticesBuffer;
        private ComputeBuffer currentVerticesBuffer;
        private int kernelIndex;

        void Start()
        {
            // ── Refuse to run rather than half-run.
            //
            //    This component needs a compute shader and a GPU to dispatch it on. It had
            //    neither wired up for most of the project's life, and the way it failed was
            //    the worst kind: OnCollisionEnter returned on a null check, so damage was a
            //    feature that existed, compiled, appeared in the inspector, and did nothing.
            //
            //    A headless training worker is the other case. Six players under
            //    -nographics have no compute support at all, and dispatching into that
            //    logs an error per impact — thousands of them across a run — for a visual
            //    effect nobody is there to see.
            if (deformationCompute == null)
            {
                Debug.LogWarning("[Damage] No compute shader assigned; deformation is off. " +
                                 "Assets/Art/Shaders/Deformation.compute is the one it wants.", this);
                enabled = false;
                return;
            }

            if (!SystemInfo.supportsComputeShaders)
            {
                Debug.Log("[Damage] No compute support on this device (headless?); deformation is off.", this);
                enabled = false;
                return;
            }

            MeshFilter mf = GetComponent<MeshFilter>();
            originalMesh = mf.sharedMesh;

            // Create an instance of the mesh to deform
            filterMesh = Instantiate(originalMesh);
            mf.mesh = filterMesh;

            originalVertices = originalMesh.vertices;
            currentVertices = new Vector3[originalVertices.Length];
            originalVertices.CopyTo(currentVertices, 0);

            // Both of these are guaranteed by the checks at the top of Start.
            kernelIndex = deformationCompute.FindKernel("CSMain");

            originalVerticesBuffer = new ComputeBuffer(originalVertices.Length, sizeof(float) * 3);
            originalVerticesBuffer.SetData(originalVertices);

            currentVerticesBuffer = new ComputeBuffer(currentVertices.Length, sizeof(float) * 3);
            currentVerticesBuffer.SetData(currentVertices);

            deformationCompute.SetBuffer(kernelIndex, "originalVertices", originalVerticesBuffer);
            deformationCompute.SetBuffer(kernelIndex, "vertices", currentVerticesBuffer);
        }

        void OnCollisionEnter(Collision collision)
        {
            // Start disables the component outright when it cannot work, so reaching here
            // means the buffers exist.
            if (collision.impulse.magnitude > impactThreshold)
            {
                // Simple average of contact points
                Vector3 avgContactPoint = Vector3.zero;
                foreach (var contact in collision.contacts)
                {
                    avgContactPoint += contact.point;
                }
                avgContactPoint /= collision.contacts.Length;
                
                // Convert to local space
                Vector3 localImpact = transform.InverseTransformPoint(avgContactPoint);
                float force = (collision.impulse.magnitude - impactThreshold) * deformationMultiplier;
                float radius = 1.0f + force * 0.1f;

                DeformMesh(localImpact, radius, force);
                CheckComponentDetachment(collision.impulse.magnitude);
            }
        }

        private void DeformMesh(Vector3 localImpact, float radius, float force)
        {
            deformationCompute.SetVector("impactPoint", localImpact);
            deformationCompute.SetFloat("impactRadius", radius);
            deformationCompute.SetFloat("impactForce", force);
            deformationCompute.SetFloat("maxDeformation", maxDeformation);

            int threadGroups = Mathf.CeilToInt(currentVertices.Length / 64.0f);
            deformationCompute.Dispatch(kernelIndex, threadGroups, 1, 1);

            // Read back data
            currentVerticesBuffer.GetData(currentVertices);
            
            // Apply to mesh
            filterMesh.vertices = currentVertices;
            filterMesh.RecalculateNormals(); // Recompute faceted normals for PS1 style
            filterMesh.RecalculateBounds();
        }

        private void CheckComponentDetachment(float impulseMag)
        {
            // Example hook for snapping hinges
            // if (impulseMag > 500f) BreakHinges();
        }

        void OnDestroy()
        {
            if (originalVerticesBuffer != null) originalVerticesBuffer.Release();
            if (currentVerticesBuffer != null) currentVerticesBuffer.Release();
        }
    }
}
