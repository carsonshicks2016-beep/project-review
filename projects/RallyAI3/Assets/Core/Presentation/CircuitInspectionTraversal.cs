using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using Core.Environment;
using Core.Physics;
using Core.ML;
using UI;

namespace Core.Presentation
{
    // Opt-in kinematic inspection, deliberately excluded from attempt records.
    [DefaultExecutionOrder(-100)]
    public sealed class CircuitInspectionTraversal:MonoBehaviour
    {
        [Serializable] public class Shot {public float start,end,duration=15,lateral;public string camera="ThirdPerson";public bool overview;public string label;}
        [Serializable] public class Plan {public Shot[] shots;public float captureFps=12;public bool capture=true;}
        [Serializable] class Coverage {public string kind="kinematic diagnostic inspection; no physical driving evidence",course;public int shot,frames;public float station,start,end,wallTime;public string label,camera;}
        Plan plan;TrackGenerator track;VehicleController car;Rigidbody body;Camera view;SpectatorDirector director;
        string folder;float start,nextFrame;int shotIndex,frame;StreamWriter trace;bool ready;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install(){if(LabRuntime.CircuitInspection&&string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_LIFECYCLE")))new GameObject("Circuit Inspection Traversal").AddComponent<CircuitInspectionTraversal>();}
        void Update()
        {
            if(!ready)
            {
                track=FindAnyObjectByType<TrackGenerator>();car=FindAnyObjectByType<VehicleController>();view=Camera.main;director=FindAnyObjectByType<SpectatorDirector>();
                if(track==null||!track.importedCircuit||car==null||view==null||Time.realtimeSinceStartup<3)return;
                string audit=System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_AUDIT");if(!string.IsNullOrEmpty(audit)&&!File.Exists(audit))return;
                plan=JsonUtility.FromJson<Plan>(File.ReadAllText(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_INSPECTION")));
                folder=System.Environment.GetEnvironmentVariable("RALLY_PRESENTATION_REVIEW");Directory.CreateDirectory(Path.Combine(folder,"inspection"));
                trace=new StreamWriter(Path.Combine(folder,"coverage.jsonl"));
                car.enabled=false;body=car.GetComponent<Rigidbody>();body.isKinematic=true;
                foreach(var wheel in car.GetComponentsInChildren<DynamicSuspension>())wheel.enabled=false;
                foreach(var agent in car.GetComponentsInChildren<RallyAgent>())agent.enabled=false;
                start=Time.realtimeSinceStartup;nextFrame=start+1;ready=true;SetShot();
            }
            if(shotIndex>=plan.shots.Length)return;
            var shot=plan.shots[shotIndex];float elapsed=Time.realtimeSinceStartup-start;
            // One second lets the existing cameras settle after each diagnostic reposition.
            float u=Mathf.Clamp01((elapsed-1)/shot.duration);float station=Mathf.Lerp(shot.start,shot.end,u);
            var surface=track.CircuitSurfacePoint(station,shot.lateral);
            var forward=track.CircuitSurfacePoint(station+.2f,shot.lateral)-track.CircuitSurfacePoint(station-.2f,shot.lateral);
            var across=track.CircuitSurfacePoint(station,shot.lateral+.1f)-track.CircuitSurfacePoint(station,shot.lateral-.1f);
            var normal=Vector3.Cross(forward,across).normalized;
            car.transform.SetPositionAndRotation(track.transform.TransformPoint(surface)+normal*.6f,Quaternion.LookRotation(forward,normal));
            if(elapsed>shot.duration+1)
            {
                trace.Flush();shotIndex++;start=Time.realtimeSinceStartup;nextFrame=start+1;
                if(shotIndex>=plan.shots.Length){trace.Dispose();trace=null;Application.Quit();return;}SetShot();
            }
        }
        void SetShot()
        {
            var shot=plan.shots[shotIndex];Enum.TryParse(shot.camera,out SpectatorDirector.CameraMode mode);director.SetMode(mode);director.enabled=!shot.overview;
        }
        void LateUpdate()
        {
            if(!ready||shotIndex>=plan.shots.Length)return;
            var shot=plan.shots[shotIndex];float elapsed=Time.realtimeSinceStartup-start;
            float station=Mathf.Lerp(shot.start,shot.end,Mathf.Clamp01((elapsed-1)/shot.duration));
            if(shot.overview)
            {
                var at=track.transform.TransformPoint(track.CircuitSurfacePoint(station,0));var tangent=track.CircuitTangent(station);
                view.transform.position=at+Vector3.up*55-tangent*20;view.transform.rotation=Quaternion.LookRotation(at-view.transform.position,Vector3.up);view.fieldOfView=65;
            }
            if(elapsed<1||Time.realtimeSinceStartup<nextFrame)return;
            if(plan.capture)ScreenCapture.CaptureScreenshot(Path.Combine(folder,"inspection",$"shot-{shotIndex:00}-frame-{frame:000000}.png"));
            trace.WriteLine(JsonUtility.ToJson(new Coverage{course=LabRuntime.Config.courseId,shot=shotIndex,frames=frame++,station=station,start=shot.start,end=shot.end,wallTime=Time.realtimeSinceStartup,label=shot.label,camera=shot.overview?"TerrainOverview":shot.camera}));
            nextFrame=Time.realtimeSinceStartup+1/Mathf.Max(1,plan.captureFps);
        }
        void OnDestroy(){trace?.Dispose();}
    }
}
