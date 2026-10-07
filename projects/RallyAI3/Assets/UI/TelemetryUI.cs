using UnityEngine;
using UnityEngine.UI;
using Core.Physics;

namespace UI
{
    public class TelemetryUI : MonoBehaviour
    {
        public VehicleController vehicle;
        public PacejkaTireModel frontLeftTire;
        
        [Header("UI Elements")]
        public Text speedText;
        public Text gearText;
        public Text rpmText;
        public RectTransform gForceDot; // For friction circle

        [Header("Friction Circle Config")]
        public float gForceUIMultiplier = 50f;
        public float maxUIOffset = 100f;

        private Rigidbody rb;
        private Vector3 lastVelocity;

        void Start()
        {
            if (vehicle != null)
            {
                rb = vehicle.GetComponent<Rigidbody>();
            }
        }

        void FixedUpdate()
        {
            if (rb == null) return;
            
            // Calculate global G-force
            Vector3 acceleration = (rb.linearVelocity - lastVelocity) / Time.fixedDeltaTime;
            lastVelocity = rb.linearVelocity;

            // Convert to local space of the car
            Vector3 localAccel = rb.transform.InverseTransformDirection(acceleration);

            UpdateFrictionCircle(localAccel.x, localAccel.z);
        }

        void Update()
        {
            if (vehicle == null) return;

            if (speedText) speedText.text = $"{Mathf.RoundToInt(vehicle.currentSpeedKmh / 1.609344f)} MPH";
            if (gearText) gearText.text = $"GEAR {vehicle.currentGear}";
            if (rpmText) rpmText.text = $"{Mathf.RoundToInt(vehicle.currentRpm)} RPM";
        }

        private void UpdateFrictionCircle(float latG, float lonG)
        {
            if (gForceDot == null) return;

            // X is lateral, Y (UI) is longitudinal (acceleration/braking)
            Vector2 targetPos = new Vector2(latG, lonG) * gForceUIMultiplier;
            targetPos = Vector2.ClampMagnitude(targetPos, maxUIOffset);

            gForceDot.anchoredPosition = Vector2.Lerp(gForceDot.anchoredPosition, targetPos, Time.deltaTime * 15f);
        }
    }
}
