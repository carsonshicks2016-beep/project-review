using System;
using System.Collections;
using System.IO;
using System.Linq;
using Core.ML;
using Core.Physics;
using UnityEngine;

namespace Core.Presentation
{
    public sealed class CockpitReviewAudit : MonoBehaviour
    {
        [Serializable] class Counts
        {
            public int cockpits, meshes, materials, colliders;
            public bool steeringBound;
        }
        [Serializable] class Frame
        {
            public float time, rpm, needleDegrees;
            public float cockpitLocalOffset, cockpitLocalAngle;
            public int gear, speed;
            public string renderedGear, renderedSpeed;
        }
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            if (Application.isBatchMode || (LabRuntime.Enabled && !LabRuntime.Config.viewer) ||
                string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("RALLY_COCKPIT_AUDIT"))) return;
            new GameObject("Cockpit review audit").AddComponent<CockpitReviewAudit>();
        }
        IEnumerator Start()
        {
            string folder=System.Environment.GetEnvironmentVariable("RALLY_COCKPIT_AUDIT");
            Directory.CreateDirectory(folder);
            yield return new WaitForSecondsRealtime(4);
            var vehicle=FindAnyObjectByType<VehicleController>();
            if(vehicle==null) yield break;
            using(var log=new StreamWriter(Path.Combine(folder,"cockpit-lifecycle.jsonl")))
            {
                for(int i=0;i<8;i++)
                {
                    var old=vehicle.GetComponentInChildren<RallyCockpit>();
                    if(old!=null)
                    {
                        old.enabled=false;
                        Destroy(old.gameObject);
                    }
                    yield return null; yield return null;
                    RallyCockpit.Create(vehicle);
                    yield return null; yield return null;
                    // Warm every numeric glyph before comparing resource counts.
                    yield return new WaitForSecondsRealtime(.2f);
                    var counts=new Counts {
                        cockpits=FindObjectsByType<RallyCockpit>(FindObjectsSortMode.None).Length,
                        meshes=Resources.FindObjectsOfTypeAll<Mesh>().Count(m=>m.name.StartsWith("Cockpit")),
                        materials=Resources.FindObjectsOfTypeAll<Material>().Count(m=>m.name.StartsWith("Cockpit")),
                        colliders=FindObjectsByType<Collider>(FindObjectsSortMode.None).Length,
                        steeringBound=vehicle.GetComponent<UI.RobotDriverIK>().steeringWheel!=null
                    };
                    log.WriteLine(JsonUtility.ToJson(counts)); log.Flush();
                }
            }
            using(var log=new StreamWriter(Path.Combine(folder,"cockpit-telemetry.jsonl")))
                for(int i=0;i<40;i++)
                {
                    var cockpit=vehicle.GetComponentInChildren<RallyCockpit>();
                    var needle=cockpit.transform.Find("RPM needle pivot");
                    log.WriteLine(JsonUtility.ToJson(new Frame {time=Time.unscaledTime,rpm=vehicle.currentRpm,
                        needleDegrees=needle.localEulerAngles.z,gear=vehicle.currentGear,
                        cockpitLocalOffset=cockpit.transform.localPosition.magnitude,
                        cockpitLocalAngle=Quaternion.Angle(cockpit.transform.localRotation,Quaternion.identity),
                        renderedGear=cockpit.GearReadout,renderedSpeed=cockpit.SpeedReadout,
                        speed=Mathf.RoundToInt(vehicle.GetComponent<Rigidbody>().linearVelocity.magnitude*3.6f / 1.609344f)}));
                    yield return new WaitForSecondsRealtime(.25f);
                }
            Destroy(gameObject);
        }
    }
}
