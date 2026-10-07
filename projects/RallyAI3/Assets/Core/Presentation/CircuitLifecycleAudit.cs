using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Core.Environment;
using Core.ML;
using Core.Physics;

namespace Core.Presentation
{
    public sealed class CircuitLifecycleAudit:MonoBehaviour
    {
        [Serializable] class Counts {public string phase;public int cycle,objects,meshes,materials,textures,colliders,effects,audio;public long memory;}
        [Serializable] class Report {public string course;public Counts[] samples;}
        List<Counts> counts=new List<Counts>();
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install(){if(!string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_LIFECYCLE")))new GameObject("Circuit Lifecycle Audit").AddComponent<CircuitLifecycleAudit>();}
        IEnumerator Start()
        {
            yield return new WaitForSecondsRealtime(4);
            var car=FindAnyObjectByType<VehicleController>();car.enabled=false;car.GetComponent<Rigidbody>().isKinematic=true;
            foreach(var wheel in car.GetComponentsInChildren<DynamicSuspension>())wheel.enabled=false;
            var agent=FindAnyObjectByType<RallyAgent>();agent.enabled=false;
            var track=FindAnyObjectByType<TrackGenerator>();
            StageDressing.DressCircuitNearby(track,true);yield return Settle();Sample("frozen baseline",0);
            // Reconstruct the complete geometry, not merely teleport/reset the car.
            for(int i=1;i<=8;i++)
            {
                track.frozenCourse=false;track.GenerateTrack();track.frozenCourse=true;
                StageDressing.DressCircuitNearby(track,true);yield return Settle();Sample("reconstruction",i);
            }
            for(int i=1;i<=3;i++)
            {
                track=LabRuntime.ReloadCourseForInspection();StageDressing.DressCircuitNearby(track,true);
                yield return Settle();Sample("unload/reload",i);
            }
            File.WriteAllText(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_LIFECYCLE"),JsonUtility.ToJson(new Report{course=LabRuntime.Config.courseId,samples=counts.ToArray()},true));
            Application.Quit();
        }
        IEnumerator Settle(){yield return null;yield return Resources.UnloadUnusedAssets();GC.Collect();yield return new WaitForSecondsRealtime(1);}
        void Sample(string phase,int cycle)
        {
            counts.Add(new Counts{phase=phase,cycle=cycle,objects=FindObjectsByType<Transform>(FindObjectsSortMode.None).Length,
                meshes=Resources.FindObjectsOfTypeAll<Mesh>().Length,materials=Resources.FindObjectsOfTypeAll<Material>().Length,
                textures=Resources.FindObjectsOfTypeAll<Texture>().Length,colliders=FindObjectsByType<Collider>(FindObjectsSortMode.None).Length,
                effects=FindObjectsByType<ParticleSystem>(FindObjectsSortMode.None).Length,audio=FindObjectsByType<AudioSource>(FindObjectsSortMode.None).Length,
                memory=System.Diagnostics.Process.GetCurrentProcess().WorkingSet64});
        }
    }
}
