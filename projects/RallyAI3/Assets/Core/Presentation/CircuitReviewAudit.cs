using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using Core.Environment;
using Core.Physics;
using Core.ML;

namespace Core.Presentation
{
    /// <summary>Opt-in inspection of the actual frozen standalone collider geometry.</summary>
    public class CircuitReviewAudit : MonoBehaviour
    {
        [Serializable] class Report
        {
            public string revision,course;public int centerSamples,centerFailures,kerbFailures,barrierFailures,projectionFailures;
            public int asphaltContacts,wrongSurfaceContacts,colliders,episodeSteps;
            public int vegetationSamples,vegetationIntrusions,vegetationColliders;
            public int liveTarmacWheels,groundSamples,groundFailures,neighbourRoadSupportSamples,stackedGroundContacts,meshColliderMismatches,degenerateFaces,nonfiniteVertices;
            public float length,fixedDelta,seamError;public string[] failures;
        }
        bool done;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            if(!string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_AUDIT")))
                new GameObject("Circuit Review Audit").AddComponent<CircuitReviewAudit>();
        }
        static bool TrackHit(Vector3 origin,Vector3 direction,out RaycastHit selected,float distance)
        {
            selected=default;float closest=float.MaxValue;bool found=false;
            foreach(var hit in UnityEngine.Physics.RaycastAll(origin,direction,distance))
            {
                string name=hit.collider.name;if(!name.StartsWith("Road_")&&!name.StartsWith("Kerb_")&&!name.StartsWith("Ground"))continue;
                if(hit.distance>=closest)continue;closest=hit.distance;selected=hit;found=true;
            }
            return found;
        }
        void Update()
        {
            if(done||Time.realtimeSinceStartup<3) return;
            var track=FindAnyObjectByType<TrackGenerator>();if(track==null||!track.importedCircuit)return;
            done=true;UnityEngine.Physics.SyncTransforms();
            var car=FindAnyObjectByType<VehicleController>();
            if(car.GetComponent<CircuitCollisionProbe>()==null)car.gameObject.AddComponent<CircuitCollisionProbe>();
            var failures=new List<string>();var r=new Report {revision=track.circuitRevision,course=LabRuntime.Config.courseId,
                length=track.circuitLength,fixedDelta=Time.fixedDeltaTime,episodeSteps=FindAnyObjectByType<RallyAgent>().MaxStep,
                colliders=track.GetComponentsInChildren<Collider>().Length,
                seamError=Vector3.Distance(track.CircuitSurfacePoint(0,0),track.CircuitSurfacePoint(track.circuitLength,0))};
            foreach(var wheel in car.GetComponentsInChildren<DynamicSuspension>())
                if(wheel.isGrounded&&wheel.surfaceLooseness==0)r.liveTarmacWheels++;
            foreach(var filter in track.GetComponentsInChildren<MeshFilter>())
            {
                var mesh=filter.sharedMesh;if(mesh==null)continue;var vertices=mesh.vertices;var indices=mesh.triangles;
                foreach(var vertex in vertices)if(float.IsNaN(vertex.x)||float.IsInfinity(vertex.x)||float.IsNaN(vertex.y)||float.IsInfinity(vertex.y)||float.IsNaN(vertex.z)||float.IsInfinity(vertex.z))r.nonfiniteVertices++;
                for(int i=0;i<indices.Length;i+=3)if(Vector3.Cross(vertices[indices[i+1]]-vertices[indices[i]],vertices[indices[i+2]]-vertices[indices[i]]).sqrMagnitude<1e-16f)r.degenerateFaces++;
            }
            foreach(var collider in track.GetComponentsInChildren<MeshCollider>())
            {
                var filter=collider.GetComponent<MeshFilter>();if(filter==null||filter.sharedMesh!=collider.sharedMesh)r.meshColliderMismatches++;
            }
            StageDressing.DressCircuitNearby(track);
            var forest=track.transform.Find("Forest");
            if(forest!=null)
            {
                r.vegetationColliders=forest.GetComponentsInChildren<Collider>().Length;
                foreach(var filter in forest.GetComponentsInChildren<MeshFilter>())
                {
                    var vertices=filter.sharedMesh.vertices;
                    for(int i=0;i<vertices.Length;i+=32)
                    {
                        r.vegetationSamples++;var point=track.transform.InverseTransformPoint(filter.transform.TransformPoint(vertices[i]));
                        track.CircuitNearest(point,out float s);var at=track.CircuitPosition(s);
                        float lateral=Vector3.Dot(point-at,Vector3.Cross(Vector3.up,track.CircuitTangent(s)));
                        if(Mathf.Abs(lateral)<track.LegalHalfWidthAt(s,lateral))r.vegetationIntrusions++;
                    }
                }
            }
            float spacing=System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_AUDIT_COARSE")=="1"?50:5;
            var auditStations=new SortedSet<float>();
            for(float s=0;s<track.circuitLength;s+=spacing)auditStations.Add(s);
            for(float s=0;s<track.circuitLength;s+=200)auditStations.Add(s);
            foreach(float s in track.CircuitAuditBoundaries())auditStations.Add(s);
            foreach(float station in auditStations)
            {
                r.centerSamples++;
                var p=track.transform.TransformPoint(track.CircuitSurfacePoint(station,0));
                if(!TrackHit(p+Vector3.up*5,Vector3.down,out var hit,10)||!hit.collider.name.StartsWith("Road_"))
                {r.centerFailures++;if(failures.Count<40)failures.Add("missing center road "+station);}
                else
                {
                    SurfaceProperties.Sample(hit.collider,out float grip,out _,out float loose);
                    if(Mathf.Abs(grip-1.25f)>.001f||loose!=0)r.wrongSurfaceContacts++;else r.asphaltContacts++;
                }
                for(int side=-1;side<=1;side+=2)
                {
                    p=track.transform.TransformPoint(track.CircuitSurfacePoint(station,side*(track.RoadHalfWidthAt(station,side)+track.KerbWidthAt(station,side)*.5f)));
                    if(track.KerbWidthAt(station,side)>.02f&&(!TrackHit(p+Vector3.up*5,Vector3.down,out hit,10)||!hit.collider.name.StartsWith("Kerb_")))
                    {r.kerbFailures++;if(failures.Count<40)failures.Add("missing kerb "+station+" side "+side);}
                    var lateralOffsets=new List<float>{.01f,.05f,.15f,.35f,.65f};for(float offset=1;offset<=35;offset+=2)lateralOffsets.Add(offset);
                    foreach(float offset in lateralOffsets)
                    {
                        r.groundSamples++;
                        var ground=track.transform.TransformPoint(track.CircuitSurfacePoint(station,side*(track.LegalHalfWidthAt(station,side)+offset)));
                        int supports=0;float firstHeight=float.NaN;bool stacked=false;bool adjacentRoad=false;
                        foreach(var contact in UnityEngine.Physics.RaycastAll(ground+Vector3.up*60,Vector3.down,160))
                            if(contact.collider.name.StartsWith("Ground")||contact.collider.name.StartsWith("Road_")||contact.collider.name.StartsWith("Kerb_")){if(!contact.collider.name.StartsWith("Ground"))adjacentRoad=true;supports++;if(float.IsNaN(firstHeight))firstHeight=contact.point.y;else if(Mathf.Abs(firstHeight-contact.point.y)>.02f)stacked=true;}
                        if(adjacentRoad)r.neighbourRoadSupportSamples++;
                        if(stacked)r.stackedGroundContacts++;
                        if(supports==0){r.groundFailures++;if(failures.Count<40)failures.Add("terrain gap "+station+" lateral "+side*offset);}
                    }
                    Vector3 right=track.transform.TransformDirection(Vector3.Cross(Vector3.up,track.CircuitTangent(station)))*side;
                    p=track.transform.TransformPoint(track.CircuitSurfacePoint(station,side*(track.BarrierOffsetAt(station,side)-1.5f)))+Vector3.up*.5f;
                    if(!UnityEngine.Physics.Raycast(p,right,out hit,3)||!hit.collider.name.StartsWith("Barrier_")||!hit.collider.CompareTag("Obstacle"))
                    {r.barrierFailures++;if(failures.Count<40)failures.Add("missing barrier "+station+" side "+side);}
                }
            }
            // Check gate-index projection on this variant and seam-wrapped lookahead.
            for(int i=2;i<=track.FinishWaypointIndex;i++)
            {
                float s=track.CircuitGateStation(i)-20;
                var frame=track.SampleRoadFrame(track.transform.TransformPoint(track.CircuitPosition(s)),i);
                if(!frame.valid||Mathf.Abs(frame.distanceAlong-s)>1||Mathf.Abs(frame.signedOffset)>.1f)r.projectionFailures++;
            }
            r.failures=failures.ToArray();
            var output=System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_AUDIT");
            File.WriteAllText(output,JsonUtility.ToJson(r,true));
            Debug.Log("[Circuit Audit] "+JsonUtility.ToJson(r));
        }
    }

    public class CircuitCollisionProbe : MonoBehaviour
    {
        [Serializable] class ContactReport {public string collider,tag;public float simulationTime,relativeSpeed;public Vector3 point;}
        void OnCollisionEnter(Collision collision)
        {
            if(!collision.collider.name.StartsWith("Barrier_"))return;
            string path=System.Environment.GetEnvironmentVariable("RALLY_CIRCUIT_AUDIT");
            if(string.IsNullOrEmpty(path))return;
            File.AppendAllText(Path.Combine(Path.GetDirectoryName(path),"barrier-contacts.jsonl"),
                JsonUtility.ToJson(new ContactReport{collider=collision.collider.name,tag=collision.collider.tag,
                    simulationTime=Time.time,relativeSpeed=collision.relativeVelocity.magnitude,
                    point=collision.contactCount>0?collision.GetContact(0).point:transform.position})+"\n");
        }
    }
}
