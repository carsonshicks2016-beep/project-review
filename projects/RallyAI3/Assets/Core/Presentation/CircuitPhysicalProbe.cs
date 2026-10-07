using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using Core.Physics;
using Core.Environment;
namespace Core.Presentation
{
    public sealed class CircuitPhysicalProbe:MonoBehaviour
    {
        [Serializable] class Contact {public string wheel,collider,region;public bool grounded;public float grip,looseness,load;}
        [Serializable] class Sample {public float simulationTime,speed;public Vector3 position;public Contact[] contacts;}
        string path;VehicleController car;StreamWriter output;float next;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install(){if(!string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_PHYSICAL_PROBE")))new GameObject("Circuit Physical Probe").AddComponent<CircuitPhysicalProbe>();}
        void Update()
        {
            if(car==null)car=FindAnyObjectByType<VehicleController>();if(car==null||Time.time<next)return;next=Time.time+.1f;
            if(output==null){path=System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_PHYSICAL_PROBE");output=new StreamWriter(path);}
            var contacts=new List<Contact>();foreach(var wheel in car.GetComponentsInChildren<DynamicSuspension>())
            {
                var collider=wheel.tireHit.collider;var region=collider!=null?collider.GetComponent<CircuitSurfaceRegion>():null;
                contacts.Add(new Contact{wheel=wheel.corner.ToString(),collider=collider!=null?collider.name:"",region=region!=null?region.region:"",grounded=wheel.isGrounded,grip=wheel.surfaceGrip,looseness=wheel.surfaceLooseness,load=wheel.normalLoad});
            }
            output.WriteLine(JsonUtility.ToJson(new Sample{simulationTime=Time.time,speed=car.ForwardSpeed,position=car.transform.position,contacts=contacts.ToArray()}));output.Flush();
        }
        void OnDestroy(){output?.Dispose();}
    }
}
