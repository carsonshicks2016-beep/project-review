using System;
using System.IO;
using System.Reflection;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Actuators;
using Core.ML;
using Core.Environment;

namespace Core.Presentation
{
    // Explicit, unranked native fixtures. Inert in normal viewing/training.
    public sealed class CircuitRuntimeAcceptance : MonoBehaviour
    {
        [Serializable] class Case { public string name; public bool passed; public string detail; }
        [Serializable] class Report { public string course,revision; public int maxStep; public float fixedDelta; public Case[] cases; }
        bool done; RallyAgent agent; TrackGenerator track; Rigidbody body;
        readonly List<Case> cases=new List<Case>();
        const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install(){if(!string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_RUNTIME_ACCEPTANCE")))new GameObject("Circuit runtime fixtures").AddComponent<CircuitRuntimeAcceptance>();}
        void Set(string name,object value){typeof(RallyAgent).GetField(name,Private).SetValue(agent,value);}
        T Get<T>(string name){return (T)typeof(RallyAgent).GetField(name,Private).GetValue(agent);}
        bool Gate(int index,Vector3 before,Vector3 after,float speed=0)
        {
            Set("currentWaypointIndex",index);Set("labPrevious",before);Set("labInvalid",false);
            body.position=after;agent.transform.position=after;body.linearVelocity=Vector3.forward*speed;
            UnityEngine.Physics.SyncTransforms();
            return (bool)typeof(RallyAgent).GetMethod("LabCrossedGate",Private).Invoke(agent,new object[]{track.transform.TransformPoint(track.waypoints[index])});
        }
        void Check(string name,bool passed,string detail=""){cases.Add(new Case{name=name,passed=passed,detail=detail});}
        void Update()
        {
            if(done||Time.realtimeSinceStartup<1)return;
            agent=FindAnyObjectByType<RallyAgent>();track=FindAnyObjectByType<TrackGenerator>();
            if(agent==null||track==null||!track.importedCircuit)return;
            done=true;body=agent.GetComponent<Rigidbody>();
            try
            {
                int index=Mathf.Min(20,track.FinishWaypointIndex-2);
                Vector3 gate=track.transform.TransformPoint(track.waypoints[index]);
                Vector3 tangent=track.transform.TransformDirection(track.CircuitTangent(track.CircuitGateStation(index))).normalized;
                Vector3 side=Vector3.Cross(Vector3.up,tangent).normalized;
                Check("forward gate",Gate(index,gate-tangent,gate+tangent));
                Check("reverse gate rejected",!Gate(index,gate+tangent,gate-tangent));
                Check("missed gate rejected",!Gate(index,gate+tangent,gate+tangent*2));
                Check("outside envelope rejected",!Gate(index,gate-tangent+side*20,gate+tangent+side*20));
                Check("altitude-separated crossing rejected",!Gate(index,gate-tangent+Vector3.up*10,gate+tangent+Vector3.up*10));
                Check("teleport rejected permanently",!Gate(index,gate-tangent*100,gate+tangent*100)&&Get<bool>("labInvalid"));
                Set("labStartGate",gate);Set("labStartCrossed",false);Set("labTicks",123);
                Set("labPrevious",gate+tangent);body.position=gate-tangent;agent.transform.position=body.position;
                typeof(RallyAgent).GetMethod("LabCrossedStart",Private).Invoke(agent,null);
                Check("reverse start rejected",!Get<bool>("labStartCrossed"));
                Set("labPrevious",gate-tangent);body.position=gate+tangent;agent.transform.position=body.position;
                typeof(RallyAgent).GetMethod("LabCrossedStart",Private).Invoke(agent,null);
                Check("directional start fixed-tick timer",Get<bool>("labStartCrossed")&&Get<int>("labStartTick")==123);
                float seam=Vector3.Distance(track.CircuitSurfacePoint(0,0),track.CircuitSurfacePoint(track.circuitLength,0));
                Check("periodic seam and tangent",seam<.01f&&Vector3.Dot(track.CircuitTangent(-.1f),track.CircuitTangent(track.circuitLength-.1f))>.999f);
                foreach(float station in new[]{750f,4250f,11250f,18300f})
                {
                    int hint=1;while(hint<track.FinishWaypointIndex&&track.CircuitGateStation(hint)<station)hint++;
                    var frame=track.SampleRoadFrame(track.transform.TransformPoint(track.CircuitPosition(station)),hint);
                    Check("bounded projection "+station,frame.valid&&Mathf.Abs(frame.distanceAlong-station)<2f);
                }
                int straight=1;while(straight<track.FinishWaypointIndex-4&&track.CircuitGateStation(straight)<18300)straight++;
                Vector3 first=track.transform.TransformPoint(track.waypoints[straight]);
                Vector3 last=track.transform.TransformPoint(track.waypoints[straight+2]);
                var direction=(last-first).normalized;Vector3 previous=first-direction,now=last+direction;
                Set("currentWaypointIndex",straight);Set("labPrevious",previous);Set("labStartCrossed",true);Set("labInvalid",false);Set("episodeStartTime",Time.time-2f);
                body.position=now;agent.transform.position=now;body.linearVelocity=direction*(Vector3.Distance(previous,now)/Time.fixedDeltaTime);
                agent.OnActionReceived(new ActionBuffers(new ActionSegment<float>(new float[]{0,0}),new ActionSegment<int>(new int[0])));
                int reached=Get<int>("currentWaypointIndex");
                Check("multiple ordered gates in one decision",reached>=straight+3);
                int splits=Get<List<float>>("labSplits").Count;
                agent.OnActionReceived(new ActionBuffers(new ActionSegment<float>(new float[]{0,0}),new ActionSegment<int>(new int[0])));
                Check("repeated stationary crossing no gate reward",Get<int>("currentWaypointIndex")==reached&&Get<List<float>>("labSplits").Count==splits);
                Check("episode budget",agent.MaxStep==Mathf.CeilToInt(LabRuntime.Config.episodeSeconds/Time.fixedDeltaTime),"MaxStep="+agent.MaxStep);
            }
            catch(Exception error){Check("fixture execution",false,error.ToString());}
            File.WriteAllText(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_RUNTIME_ACCEPTANCE"),JsonUtility.ToJson(new Report{course=LabRuntime.Config.courseId,revision=track.circuitRevision,maxStep=agent.MaxStep,fixedDelta=Time.fixedDeltaTime,cases=cases.ToArray()},true));
            Application.Quit(cases.Exists(c=>!c.passed)?2:0);
        }
    }
}
