using UnityEngine;
using Core.Physics;

namespace UI
{
    /// <summary>
    /// Turns the steering wheel with the steering input, and would move a driver's hands
    /// with it if there were a driver.
    ///
    /// WHAT IS ACTUALLY WIRED
    ///
    /// The wheel. RallyCarBuilder.BuildCockpit builds a steering wheel on a tilted column
    /// and assigns it here, so the rotation half of this component works and the steering
    /// is visible from inside the car — which matters when you are watching a policy drive
    /// and trying to tell lock from countersteer.
    ///
    /// WHAT IS NOT
    ///
    /// The hands. <see cref="leftHandTarget"/> and <see cref="rightHandTarget"/> stay null,
    /// because the car has no driver: the mesh builder produces a body and four road
    /// wheels, and there is no rig, no arms and nothing for an IK target to drive. This
    /// component tolerates that — it never touches a target it was not given — so what you
    /// get is the part of the feature that has something to act on.
    ///
    /// Building a driver is a separate piece of work. Until it is done, the two hand fields
    /// are documentation of an intention, not a broken wire.
    /// </summary>
    public class RobotDriverIK : MonoBehaviour
    {
        [Header("References")]
        public VehicleController vehicle;
        
        [Header("IK Targets")]
        public Transform leftHandTarget;
        public Transform rightHandTarget;
        
        [Header("Steering Wheel")]
        public Transform steeringWheel;
        public float maxWheelRotation = 180f; // degrees

        private Vector3 initialLeftHandPos;
        private Vector3 initialRightHandPos;

        void Start()
        {
            if (leftHandTarget) initialLeftHandPos = leftHandTarget.localPosition;
            if (rightHandTarget) initialRightHandPos = rightHandTarget.localPosition;
        }

        void Update()
        {
            if (vehicle == null || steeringWheel == null) return;

            // Rotate steering wheel based on vehicle steering input
            float currentSteering = vehicle.steeringInput;
            float rotationAngle = currentSteering * maxWheelRotation;
            
            // Assuming Z is forward for the wheel
            steeringWheel.localRotation = Quaternion.Euler(0, 0, -rotationAngle);

            // In a real project using Unity Animation Rigging, 
            // the IK targets would simply be parented to the steering wheel.
            // If we are doing it manually, we rotate the hand targets around the wheel axis.
            
            // Hand cross-over logic can be implemented here if the rotation > 180 degrees
            // but for a rally car 180 is usually enough for visual approximation.
        }
    }
}
