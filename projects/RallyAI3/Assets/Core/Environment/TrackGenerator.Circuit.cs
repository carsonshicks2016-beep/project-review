using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Core.Physics;

namespace Core.Environment
{
    public partial class TrackGenerator
    {
        [Serializable] public class CircuitLandmark { public string name; public float station; }
        [Serializable] public class CircuitInterval
        {
            public string landmark, kind, confidence, evidence;
            public float start, end, side, bankDegrees, channelWidth;
        }
        [Serializable] public class CircuitRoadProfile
        {
            public float left=5.5f,right=5.5f,crossfall,leftKerb,rightKerb,kerbHeight=.07f,kerbBevel=.2f;
            public bool independentKerbDimensions;
            public float leftKerbHeight=.07f,rightKerbHeight=.07f,leftKerbBevel=.2f,rightKerbBevel=.2f;
            public float leftShoulder=3,rightShoulder=3,leftBarrier=3.5f,rightBarrier=3.5f,barrierHeight=.85f;
            public float leftMarking=1,rightMarking=1;public string ground="grass";
        }
        [Serializable] public class CircuitTerrainChunk {public string name;public Vector3[] vertices,normals;public int[] triangles;public Vector3[] skirtVertices;public int[] skirtTriangles;}
        [Serializable] public class CircuitSkirt {public Vector3[] points;}
        [Serializable] public class CircuitTerrainData {public string revision,generatorVersion;public CircuitTerrainChunk[] chunks;}
        [Serializable] public class CircuitSection {public float start,end;}
        [Serializable] public class CircuitPatch {public string name;public float start,end,left,right;}
        [Serializable] public class CircuitStructure {public string name;public float station,deckClearance,deckWidth,span;}
        [Serializable] public class CircuitData
        {
            public int schema; public string name, revision, sourceHash, profileHash, confidence, startReference;
            public string generatorVersion;public CircuitRoadProfile[] roadProfiles;public CircuitStructure[] structures;public CircuitPatch[] patches;
            public float length, width; public Vector3[] points; public float[] stations;
            public CircuitSection[] sections;public CircuitLandmark[] landmarks; public CircuitInterval[] intervals;
        }
        public IEnumerable<float> CircuitAuditBoundaries()
        {
            if(circuitSections!=null)foreach(var section in circuitSections){yield return section.start;yield return section.end;}
            if(circuitRoadProfiles!=null)for(int i=1;i<circuitRoadProfiles.Length;i++)
                if(Mathf.Abs(circuitRoadProfiles[i].leftKerb-circuitRoadProfiles[i-1].leftKerb)>.01f||Mathf.Abs(circuitRoadProfiles[i].rightKerb-circuitRoadProfiles[i-1].rightKerb)>.01f)yield return circuitStations[i];
        }
        public bool importedCircuit;
        public string circuitRevision, circuitSourceHash, circuitProfileHash;
        public string circuitGeneratorVersion;
        [SerializeField] CircuitSection[] circuitSections;
        [SerializeField] CircuitRoadProfile[] circuitRoadProfiles;
        [SerializeField] CircuitStructure[] circuitStructures;
        [SerializeField] CircuitPatch[] circuitPatches;
        public float circuitLength, circuitStart, circuitEnd;
        public int circuitEpisodeSeconds = 1800;
        [SerializeField] Vector3[] circuitPoints;
        [SerializeField] float[] circuitStations;
        [SerializeField] CircuitInterval[] circuitIntervals;
        public CircuitLandmark[] circuitLandmarks;
        [SerializeField] float[] circuitGateStations;
        readonly Dictionary<Vector2Int, List<int>> circuitGrid = new Dictionary<Vector2Int, List<int>>();
        CircuitResourceOwner circuitOwner;
        const float GridMetres = 24;
        readonly Dictionary<Vector2Int,float> circuitGroundHeights = new Dictionary<Vector2Int,float>();
        const float GroundStep=8;
        float GroundHeight(Vector3 p)
        {
            int x=Mathf.FloorToInt(p.x/GroundStep),z=Mathf.FloorToInt(p.z/GroundStep);
            float H(int dx,int dz)=>circuitGroundHeights.TryGetValue(new Vector2Int(x+dx,z+dz),out float y)?y:p.y-1;
            float u=p.x/GroundStep-x,w=p.z/GroundStep-z;
            // Match the actual two triangles, rather than a bilinear curved surface.
            return u+w<=1?H(0,0)+(H(1,0)-H(0,0))*u+(H(0,1)-H(0,0))*w:
                H(1,1)+(H(0,1)-H(1,1))*(1-u)+(H(1,0)-H(1,1))*(1-w);
        }

        public void InstallCircuit(CircuitData data, int firstSector, int sectorCount)
        {
            if (data.schema != 2 || data.points.Length != data.stations.Length || firstSector < 0 ||
                sectorCount < 1 || firstSector + sectorCount > 16) throw new ArgumentException("Invalid imported circuit");
            importedCircuit = true; circuitRevision = data.revision;
            circuitSourceHash = data.sourceHash; circuitProfileHash = data.profileHash;
            circuitLength = data.length; circuitPoints = data.points; circuitStations = data.stations;
            circuitSections=data.sections;circuitPatches=data.patches;circuitStructures=data.structures;circuitRoadProfiles=data.roadProfiles;circuitGeneratorVersion=data.generatorVersion;
            circuitIntervals = data.intervals; circuitLandmarks = data.landmarks;
            circuitStart = firstSector * circuitLength / 16; circuitEnd = (firstSector+sectorCount)*circuitLength/16;
            circuitEpisodeSeconds = sectorCount == 16 ? 1800 : Mathf.Max(240, Mathf.CeilToInt((circuitEnd-circuitStart)/12+60));
            trackLength = circuitEnd-circuitStart; roadWidth = data.width;
            spawnWaypointIndex = 1; randomiseSeed = false; scatterProps = false; spawnObstacles = false;
            obstaclesPer100m = 0; frozenCourse = false;
        }

        int CircuitIndex(float station)
        {
            station = Mathf.Repeat(station, circuitLength);
            int lo=0, hi=circuitStations.Length-1;
            while (lo+1<hi) {int mid=(lo+hi)/2; if(circuitStations[mid]<=station) lo=mid; else hi=mid;}
            return lo;
        }
        public Vector3 CircuitPosition(float station)
        {
            station = Mathf.Repeat(station,circuitLength);
            int i=CircuitIndex(station);
            return Vector3.Lerp(circuitPoints[i],circuitPoints[i+1],
                (station-circuitStations[i])/Mathf.Max(.001f,circuitStations[i+1]-circuitStations[i]));
        }
        public Vector3 CircuitTangent(float station)
        {
            Vector3 t=CircuitPosition(station+1.5f)-CircuitPosition(station-1.5f);
            return new Vector3(t.x,0,t.z).normalized;
        }
        public float CircuitGateStation(int index) => circuitGateStations[Mathf.Clamp(index,0,circuitGateStations.Length-1)];
        public CircuitRoadProfile ProfileAt(float station)
        {
            if(circuitRoadProfiles==null||circuitRoadProfiles.Length==0)return new CircuitRoadProfile{left=roadWidth*.5f,right=roadWidth*.5f,leftKerb=.65f,rightKerb=.65f};
            int i=CircuitIndex(station);var a=circuitRoadProfiles[i];var b=circuitRoadProfiles[i+1];
            float u=(Mathf.Repeat(station,circuitLength)-circuitStations[i])/Mathf.Max(.0001f,circuitStations[i+1]-circuitStations[i]);
            return new CircuitRoadProfile {
                left=Mathf.Lerp(a.left,b.left,u),right=Mathf.Lerp(a.right,b.right,u),crossfall=Mathf.Lerp(a.crossfall,b.crossfall,u),
                leftKerb=Mathf.Lerp(a.leftKerb,b.leftKerb,u),rightKerb=Mathf.Lerp(a.rightKerb,b.rightKerb,u),
                kerbHeight=Mathf.Lerp(a.kerbHeight,b.kerbHeight,u),kerbBevel=Mathf.Lerp(a.kerbBevel,b.kerbBevel,u),
                independentKerbDimensions=true,
                leftKerbHeight=Mathf.Lerp(a.independentKerbDimensions?a.leftKerbHeight:a.kerbHeight,b.independentKerbDimensions?b.leftKerbHeight:b.kerbHeight,u),
                rightKerbHeight=Mathf.Lerp(a.independentKerbDimensions?a.rightKerbHeight:a.kerbHeight,b.independentKerbDimensions?b.rightKerbHeight:b.kerbHeight,u),
                leftKerbBevel=Mathf.Lerp(a.independentKerbDimensions?a.leftKerbBevel:a.kerbBevel,b.independentKerbDimensions?b.leftKerbBevel:b.kerbBevel,u),
                rightKerbBevel=Mathf.Lerp(a.independentKerbDimensions?a.rightKerbBevel:a.kerbBevel,b.independentKerbDimensions?b.rightKerbBevel:b.kerbBevel,u),
                leftShoulder=Mathf.Lerp(a.leftShoulder,b.leftShoulder,u),rightShoulder=Mathf.Lerp(a.rightShoulder,b.rightShoulder,u),
                leftBarrier=Mathf.Lerp(a.leftBarrier,b.leftBarrier,u),rightBarrier=Mathf.Lerp(a.rightBarrier,b.rightBarrier,u),
                barrierHeight=Mathf.Lerp(a.barrierHeight,b.barrierHeight,u),leftMarking=a.leftMarking,rightMarking=a.rightMarking,ground=a.ground};
        }
        public float RoadHalfWidthAt(float station) => Mathf.Max(ProfileAt(station).left,ProfileAt(station).right);
        public float RoadHalfWidthAt(float station,float signedOffset) => signedOffset<0?ProfileAt(station).left:ProfileAt(station).right;
        public float KerbWidthAt(float station,int side) => side<0?ProfileAt(station).leftKerb:ProfileAt(station).rightKerb;
        public float LegalHalfWidthAt(float station) => Mathf.Max(LegalHalfWidthAt(station,-1),LegalHalfWidthAt(station,1));
        public float LegalHalfWidthAt(float station,float signedOffset)
        {var p=ProfileAt(station);return signedOffset<0?p.left+p.leftKerb:p.right+p.rightKerb;}
        public float BarrierOffsetAt(float station,int side)
        {var p=ProfileAt(station);return side<0?p.left+p.leftBarrier:p.right+p.rightBarrier;}
        public float CircuitClearance(Vector3 point)
        {CircuitNearest(point,out float s);var at=CircuitPosition(s);float lateral=Vector3.Dot(point-at,Vector3.Cross(Vector3.up,CircuitTangent(s)));return Mathf.Abs(lateral)-BarrierOffsetAt(s,lateral<0?-1:1);}

        void EnsureCircuitGrid()
        {
            if(circuitGrid.Count>0) return;
            for(int i=0;i<circuitPoints.Length-1;i++)
            {
                var p=circuitPoints[i]; var key=new Vector2Int(Mathf.FloorToInt(p.x/GridMetres),Mathf.FloorToInt(p.z/GridMetres));
                if(!circuitGrid.TryGetValue(key,out var values)) circuitGrid[key]=values=new List<int>();
                values.Add(i);
            }
        }
        public float CircuitNearest(Vector3 point, out float station, bool useHeight=false)
        {
            EnsureCircuitGrid(); float best=float.MaxValue; station=0;
            var key=new Vector2Int(Mathf.FloorToInt(point.x/GridMetres),Mathf.FloorToInt(point.z/GridMetres));
            for(int x=-3;x<=3;x++) for(int z=-3;z<=3;z++)
            {
                if(!circuitGrid.TryGetValue(key+new Vector2Int(x,z),out var indices)) continue;
                foreach(int i in indices)
                {
                    Vector3 a=circuitPoints[i], b=circuitPoints[i+1], ab=b-a, ap=point-a;
                    Vector3 flat=new Vector3(ab.x,0,ab.z);
                    float t=Mathf.Clamp01((ap.x*ab.x+ap.z*ab.z)/Mathf.Max(.0001f,flat.sqrMagnitude));
                    Vector3 delta=point-(a+ab*t);
                    float d=delta.x*delta.x+delta.z*delta.z+(useHeight?delta.y*delta.y:0);
                    if(d>=best) continue; best=d; station=Mathf.Lerp(circuitStations[i],circuitStations[i+1],t);
                }
            }
            if(best==float.MaxValue)
            {
                // Outer terrain tile corners can lie beyond the fine lookup radius.
                // This fallback is not used by bounded runtime road projection.
                for(int i=0;i<circuitPoints.Length-1;i+=12)
                {
                    var delta=point-circuitPoints[i];float d=delta.x*delta.x+delta.z*delta.z;
                    if(d<best){best=d;station=circuitStations[i];}
                }
            }
            return Mathf.Sqrt(best);
        }
        public RoadFrame CircuitFrame(Vector3 world, int hint)
        {
            Vector3 point=transform.InverseTransformPoint(world);
            float expected=CircuitGateStation(hint), best=float.MaxValue, station=expected, offset=0;
            // Station-bounded segment projection protects parallel roads and the lap seam.
            for(float s=expected-100;s<=expected+25;s+=1.5f)
            {
                int i=CircuitIndex(s); Vector3 a=circuitPoints[i], b=circuitPoints[i+1], ab=b-a, ap=point-a;
                float t=Mathf.Clamp01((ap.x*ab.x+ap.z*ab.z)/Mathf.Max(.0001f,ab.x*ab.x+ab.z*ab.z));
                Vector3 at=a+ab*t, delta=point-at;
                float d=delta.x*delta.x+delta.z*delta.z+delta.y*delta.y*.25f;
                if(d>=best) continue;
                best=d; float wrapped=Mathf.Lerp(circuitStations[i],circuitStations[i+1],t);
                station=expected+Mathf.DeltaAngle(expected/circuitLength*360,wrapped/circuitLength*360)/360*circuitLength;
                offset=Vector3.Dot(delta,Vector3.Cross(Vector3.up,CircuitTangent(wrapped)));
            }
            return new RoadFrame {valid=best<float.MaxValue,tangent=CircuitTangent(station),signedOffset=offset,distanceAlong=station};
        }
        public float CircuitSurfaceHeight(float station,float lateral)
        {
            float height=lateral*Mathf.Tan(ProfileAt(station).crossfall*Mathf.Deg2Rad);
            foreach(var section in circuitIntervals)
            {
                float s=Mathf.Repeat(station,circuitLength);
                if(s<section.start||s>section.end||section.kind!="bowl") continue;
                float blend=Mathf.SmoothStep(0,1,Mathf.Min((s-section.start)/40,(section.end-s)/40));
                float into=Mathf.Clamp((lateral*section.side-(RoadHalfWidthAt(station,section.side)-section.channelWidth)),0,section.channelWidth);
                // Concrete descends toward the inside, with a smooth join to the higher asphalt bypass.
                if((circuitGeneratorVersion=="circuit-geometry-v3"||circuitGeneratorVersion=="circuit-geometry-v4"||circuitGeneratorVersion=="circuit-geometry-v5"||circuitGeneratorVersion=="circuit-geometry-v6"||circuitGeneratorVersion=="circuit-geometry-v7"))
                    height-=section.channelWidth*Mathf.Tan(section.bankDegrees*Mathf.Deg2Rad)/1.3f*Mathf.Pow(into/section.channelWidth,1.3f)*blend;
                else height+=into*Mathf.Tan(section.bankDegrees*Mathf.Deg2Rad)*blend; // Preserve legacy frozen-course profile semantics.
            }
            return height;
        }
        Vector3 CrossPoint(float station,float lateral,float extra=0)
        {
            Vector3 p=CircuitPosition(station)+Vector3.Cross(Vector3.up,CircuitTangent(station))*lateral;
            p.y+=CircuitSurfaceHeight(station,lateral)+extra; return p;
        }
        public Vector3 CircuitSurfacePoint(float station,float lateral) => CrossPoint(station,lateral);

        Material CircuitMaterial(string name,Color color)
        {
            var shader=Shader.Find("Standard");
            var m=new Material(shader){name=name,color=color}; m.SetFloat("_Glossiness",.12f);
            if(name=="Circuit Asphalt"||name=="Circuit Kerb"||name=="Circuit Concrete")
            {
                var texture=new Texture2D(128,128,TextureFormat.RGB24,true){name=name+" Texture",wrapMode=TextureWrapMode.Repeat};
                var pixels=new Color[128*128];var rng=new System.Random(7041);
                for(int y=0;y<128;y++)for(int x=0;x<128;x++)
                {
                    if(name=="Circuit Kerb")pixels[y*128+x]=y<64?new Color(.63f,.12f,.09f):new Color(.8f,.78f,.7f);
                    else
                    {
                        float aggregate=.8f+(float)rng.NextDouble()*.35f;
                        float wear=1-.06f*Mathf.PerlinNoise(x*.08f,y*.08f);
                        if(name=="Circuit Concrete"&&(y<2||x<1))wear*=.68f;
                        pixels[y*128+x]=new Color(aggregate*wear,aggregate*wear,aggregate*wear);
                    }
                }
                texture.SetPixels(pixels);texture.Apply();m.mainTexture=texture;
                if(name=="Circuit Kerb")m.color=Color.white;
            }
            if(circuitOwner!=null){circuitOwner.Register(m);if(m.mainTexture!=null)circuitOwner.Register(m.mainTexture);}
            return m;
        }
        GameObject CircuitMesh(string name,List<Vector3> vertices,List<int> triangles,List<Vector2> uv,Material material,
            Transform parent, bool collider=true, bool obstacle=false)
        {
            var go=new GameObject(name); go.transform.SetParent(parent,false);
            // Tiny clipped slivers can collapse when converted to Unity float coordinates.
            // Remove only zero-area faces after conversion; their shared vertices remain.
            var finiteTriangles=new List<int>(triangles.Count);
            for(int i=0;i<triangles.Count;i+=3)
                if(Vector3.Cross(vertices[triangles[i+1]]-vertices[triangles[i]],vertices[triangles[i+2]]-vertices[triangles[i]]).sqrMagnitude>=1e-16f)
                {finiteTriangles.Add(triangles[i]);finiteTriangles.Add(triangles[i+1]);finiteTriangles.Add(triangles[i+2]);}
            var mesh=new Mesh{name=name,indexFormat=IndexFormat.UInt32}; mesh.SetVertices(vertices);
            mesh.SetTriangles(finiteTriangles,0); mesh.SetUVs(0,uv); mesh.RecalculateNormals(); mesh.RecalculateBounds();
            if(circuitOwner!=null)circuitOwner.Register(mesh);
            go.AddComponent<MeshFilter>().sharedMesh=mesh; go.AddComponent<MeshRenderer>().sharedMaterial=material;
            if(collider)
            {
                go.AddComponent<MeshCollider>().sharedMesh=mesh;
                // Match the procedural stages the policy learned on: only the land beside the road is
                // TrackBoundary. Tagging pavement/kerbs made every rise read as a wall to the ray sensors.
                if(obstacle)go.tag="Obstacle";else if(name.StartsWith("Ground"))go.tag="TrackBoundary";
                var sp=go.AddComponent<SurfaceProperties>();
                sp.surface=name.StartsWith("GroundSoil")?SurfaceProperties.Surface.Dirt:name.StartsWith("Ground")?SurfaceProperties.Surface.Grass:SurfaceProperties.Surface.Tarmac;
                sp.ApplyPreset();
                var region=go.AddComponent<CircuitSurfaceRegion>();region.region=name.StartsWith("GroundSoil")?"soil":name.StartsWith("Ground")?"grass":name.StartsWith("Kerb")?"kerb":material.name=="Circuit Concrete"?"concrete":obstacle?"barrier":"asphalt";
            }
            return go;
        }
        void CircuitStrip(string name,float start,float end,float[] lateral,float[] rise,Material material,Transform parent,bool obstacle=false,bool collider=true)
        {
            var v=new List<Vector3>(); var t=new List<int>(); var uv=new List<Vector2>();
            var samples=new List<float>{start};
            for(int k=0;k<circuitStations.Length;k++)if(circuitStations[k]>start+.00001f&&circuitStations[k]<end-.00001f)samples.Add(circuitStations[k]);
            samples.Add(end);int lanes=lateral.Length;var normals=new List<Vector3>();
            for(int i=0;i<samples.Count;i++)
            {
                float station=samples[i];var profile=ProfileAt(station);
                for(int j=0;j<lanes;j++)
                {
                    float l=lateral[j],h=roadWidth*.5f;int side=l<0?-1:1;
                    float edge=side<0?profile.left:profile.right;
                    float r=rise[j];
                    if(name.StartsWith("Kerb"))
                    {float fraction=(Mathf.Abs(l)-h)/.65f;float width=KerbWidthAt(station,side);float distance=j==1?Mathf.Max(0,width-Mathf.Min(profile.independentKerbDimensions?(side<0?profile.leftKerbBevel:profile.rightKerbBevel):profile.kerbBevel,width*.5f)):fraction*width;l=side*(edge+distance);r*=(profile.independentKerbDimensions?(side<0?profile.leftKerbHeight:profile.rightKerbHeight):profile.kerbHeight)/.08f*Mathf.Clamp01(width/.65f);}
                    else if(name.StartsWith("Barrier")||name.StartsWith("ArmcoRail"))
                    {l=side*BarrierOffsetAt(station,side)+(l-side*(h+3.5f));r*=profile.barrierHeight/.85f;}
                    else if(name.StartsWith("Marking")) l=side*(edge-Mathf.Abs(Mathf.Abs(l)-h));
                    else l=side*edge*Mathf.Abs(l)/h;
                    var point=CrossPoint(station,l,r);v.Add(point);uv.Add(new Vector2(l/3,station/3));
                    Vector3 along=CrossPoint(station+.2f,l,r)-CrossPoint(station-.2f,l,r);
                    Vector3 across=CrossPoint(station,l+.02f,r)-CrossPoint(station,l-.02f,r);
                    normals.Add(Vector3.Cross(along,across).normalized);
                }
                if(i==0)continue;
                for(int j=0;j<lanes-1;j++)
                {
                    int x=(i-1)*lanes+j,y=x+lanes;
                    if(name.StartsWith("Marking")&&((lateral[j]<0?profile.leftMarking:profile.rightMarking)<.5f))continue;
                    if(Vector3.Cross(v[y]-v[x],v[x+1]-v[x]).sqrMagnitude>1e-12f){t.Add(x);t.Add(y);t.Add(x+1);}
                    if(Vector3.Cross(v[y]-v[x+1],v[y+1]-v[x+1]).sqrMagnitude>1e-12f){t.Add(x+1);t.Add(y);t.Add(y+1);}
                }
            }
            if(t.Count==0)return;
            var go=CircuitMesh(name,v,t,uv,material,parent,collider,obstacle);
            if(name.StartsWith("Road")||name.StartsWith("Kerb")||name.StartsWith("Marking"))go.GetComponent<MeshFilter>().sharedMesh.SetNormals(normals);
        }
        void CircuitPosts(float start,float end,int side,Material material,Transform parent)
        {
            var v=new List<Vector3>();var t=new List<int>();var uv=new List<Vector2>();
            for(float s=Mathf.Ceil(start/4)*4;s<end;s+=4)
            {
                Vector3 center=CrossPoint(s,side*BarrierOffsetAt(s,side));
                Vector3 right=Vector3.Cross(Vector3.up,CircuitTangent(s))*.06f;
                Vector3 forward=CircuitTangent(s)*.06f;int b=v.Count;
                for(int y=0;y<2;y++)for(int z=0;z<2;z++)for(int x=0;x<2;x++)
                {v.Add(center+right*(x==0?-1:1)+forward*(z==0?-1:1)+Vector3.up*(y*ProfileAt(s).barrierHeight));uv.Add(Vector2.zero);}
                int[] faces={0,4,2,2,4,6,1,3,5,3,7,5,0,1,4,1,5,4,2,6,3,3,6,7,4,5,6,5,7,6};
                foreach(int index in faces)t.Add(b+index);
            }
            CircuitMesh($"ArmcoPosts_{start}_{side}",v,t,uv,material,parent,false);
        }
        void GenerateCircuit()
        {
            CurrentSeed=seed; ClearObstacles(); circuitGrid.Clear(); EnsureCircuitGrid();
            for(int i=transform.childCount-1;i>=0;i--) RetireImmediately(transform.GetChild(i).gameObject);
            GetComponent<MeshFilter>().sharedMesh=null;
            var originalCollider=GetComponent<MeshCollider>(); if(originalCollider!=null) originalCollider.sharedMesh=null;
            waypoints.Clear(); var gates=new List<float>{circuitStart-50,circuitStart};
            for(float s=circuitStart+50;s<circuitEnd-.1f;s+=50) gates.Add(s);
            gates.Add(circuitEnd); gates.Add(circuitEnd+50);
            circuitGateStations=gates.ToArray(); foreach(float s in gates) waypoints.Add(CircuitPosition(s));
            var roadRoot=new GameObject("CircuitRoad");roadRoot.transform.SetParent(transform,false);circuitOwner=roadRoot.AddComponent<CircuitResourceOwner>();
            var asphalt=CircuitMaterial("Circuit Asphalt",new Color(.19f,.20f,.21f));
            var concrete=CircuitMaterial("Circuit Concrete",new Color(.47f,.46f,.43f));
            var kerb=CircuitMaterial("Circuit Kerb",new Color(.66f,.64f,.61f));
            var barrier=CircuitMaterial("Circuit Armco",new Color(.43f,.46f,.47f));
            var marking=CircuitMaterial("Circuit Edge Marking",new Color(.74f,.73f,.66f));
            var grass=CircuitMaterial("Circuit Ground",new Color(.24f,.28f,.17f));
            terrain=new GameObject("Terrain");terrain.transform.SetParent(transform,false);
            BuildCircuitGround(grass);
            // Build periodic geometry independently of the selected training sector.
            var chunkCuts=new SortedSet<float>{0,circuitLength};
            for(float s=200;s<circuitLength;s+=200)chunkCuts.Add(s);
            if(circuitGeneratorVersion=="circuit-geometry-v6"||circuitGeneratorVersion=="circuit-geometry-v7")
            {
                foreach(var section in circuitSections){chunkCuts.Add(section.start);chunkCuts.Add(section.end);}
                foreach(var interval in circuitIntervals){chunkCuts.Add(interval.start);chunkCuts.Add(interval.end);}
            }
            var chunkBounds=new List<float>(chunkCuts);int chunks=chunkBounds.Count-1;float h=roadWidth*.5f;
            for(int c=0;c<chunks;c++)
            {
                float a=chunkBounds[c],b=chunkBounds[c+1];
                // Uniform transverse samples let the concrete channel have its own material.
                float[] laneEdges={-h,-4.5f,-3.5f,-2.3f,-1.3f,0,2.75f,h};
                for(int lane=0;lane<laneEdges.Length-1;lane++)
                {
                    float l0=laneEdges[lane],l1=laneEdges[lane+1];
                    // Split longitudinally at profile boundaries so materials follow the bowl.
                    var cuts=new List<float>{a,b};
                    foreach(var section in circuitIntervals) {if(section.start>a&&section.start<b)cuts.Add(section.start);if(section.end>a&&section.end<b)cuts.Add(section.end);}
                    cuts.Sort();
                    for(int k=0;k<cuts.Count-1;k++)
                    {
                        Material m=asphalt;float mid=(cuts[k]+cuts[k+1])*.5f;
                        foreach(var section in circuitIntervals) if(mid>=section.start&&mid<=section.end&&
                            (l0+l1)*.5f*section.side>h-section.channelWidth)m=concrete;
                        CircuitStrip($"Road_{c}_{lane}_{k}",cuts[k],cuts[k+1],new[]{l0,l1},new[]{0f,0f},m,roadRoot.transform);
                    }
                }
                for(int side=-1;side<=1;side+=2)
                {
                    float[] curb=side<0?new[]{-h-.65f,-h-.45f,-h}:new[]{h,h+.45f,h+.65f};
                    CircuitStrip($"Kerb_{c}_{side}",a,b,curb,side<0?new[]{0f,.08f,0f}:new[]{0f,.08f,0f},kerb,roadRoot.transform);
                    float x=side*(h+3.5f);
                    CircuitStrip($"Barrier_{c}_{side}",a,b,new[]{x-.08f,x-.08f,x+.08f,x+.08f},
                        new[]{.3f,.85f,.85f,.3f},barrier,roadRoot.transform,true);
                    // Smooth collision envelope around the physical rail, with a corrugated
                    // presentation skin and batched posts rather than a solid visual wall.
                    roadRoot.transform.GetChild(roadRoot.transform.childCount-1).GetComponent<MeshRenderer>().enabled=false;
                    for(int face=-1;face<=1;face+=2)
                    {
                        float[] profile={x+face*.08f,x+face*.04f,x+face*.08f,x+face*.04f,x+face*.08f};
                        float[] heights={.3f,.43f,.57f,.71f,.85f};
                        if(face>0){Array.Reverse(profile);Array.Reverse(heights);}
                        CircuitStrip($"ArmcoRail_{c}_{side}_{face}",a,b,profile,
                            heights,barrier,roadRoot.transform,false,false);
                    }
                    CircuitPosts(a,b,side,barrier,roadRoot.transform);
                    float[] line=side<0?new[]{-h+.12f,-h+.24f}:new[]{h-.24f,h-.12f};
                    CircuitStrip($"Marking_{c}_{side}",a,b,line,new[]{.008f,.008f},marking,roadRoot.transform,false,false);
                }
            }
            if(circuitPatches!=null&&circuitPatches.Length>0)
            {
                var repair=CircuitMaterial("Circuit Repair",new Color(.16f,.17f,.18f));
                foreach(var patch in circuitPatches)CircuitStrip("Repair_"+patch.name,patch.start,patch.end,new[]{patch.left,patch.right},new[]{.006f,.006f},repair,roadRoot.transform,false,false);
            }
            if(circuitStructures!=null)foreach(var structure in circuitStructures)BuildCircuitStructure(structure,barrier,roadRoot.transform);
            Debug.Log($"[Circuit] {circuitRevision} length={circuitLength} chunks={chunks} gates={gates.Count-3} budget={circuitEpisodeSeconds}s");
        }
        void BuildCircuitStructure(CircuitStructure structure,Material material,Transform parent)
        {
            void Box(string name,Vector3 center,Vector3 size)
            {
                var go=GameObject.CreatePrimitive(PrimitiveType.Cube);go.name=name;go.transform.SetParent(parent,false);
                go.transform.position=transform.TransformPoint(center);go.transform.rotation=Quaternion.LookRotation(CircuitTangent(structure.station));go.transform.localScale=size;
                go.GetComponent<Renderer>().sharedMaterial=material;go.tag="Obstacle";
            }
            var center=CircuitPosition(structure.station);
            Box("OverpassDeck_"+structure.name,center+Vector3.up*(structure.deckClearance+.3f),new Vector3(structure.span,.6f,structure.deckWidth));
            var right=Vector3.Cross(Vector3.up,CircuitTangent(structure.station));
            for(int side=-1;side<=1;side+=2)
                Box("OverpassSupport_"+side,center+right*(side*(structure.span*.5f-.5f))+Vector3.up*structure.deckClearance*.5f,new Vector3(.5f,structure.deckClearance,.5f));
        }
        void BuildCircuitGround(Material grass)
        {
            if(circuitRoadProfiles==null||circuitRoadProfiles.Length==0){BuildLegacyCircuitGround(grass);return;}
            var asset=Resources.Load<TextAsset>("Circuits/NordschleifeTerrain");
            if(asset==null)throw new InvalidOperationException("Missing offline constrained terrain");
            var data=JsonUtility.FromJson<CircuitTerrainData>(asset.text);
            if(data.revision!=circuitRevision)throw new InvalidOperationException("Terrain/circuit revision mismatch");
            foreach(var chunk in data.chunks)
            {
                var uv=new List<Vector2>();foreach(var p in chunk.vertices)uv.Add(new Vector2(p.x/8,p.z/8));
                var ground=CircuitMesh(chunk.name,new List<Vector3>(chunk.vertices),new List<int>(chunk.triangles),uv,grass,terrain.transform);
                if(chunk.normals!=null&&chunk.normals.Length==chunk.vertices.Length)ground.GetComponent<MeshFilter>().sharedMesh.normals=chunk.normals;
                if(chunk.skirtVertices!=null&&chunk.skirtVertices.Length>0)
                {
                    uv.Clear();foreach(var p in chunk.skirtVertices)uv.Add(new Vector2(p.x/8,p.z/8));
                    CircuitMesh("TerrainSkirt_"+chunk.name,new List<Vector3>(chunk.skirtVertices),new List<int>(chunk.skirtTriangles),uv,grass,terrain.transform,false);
                }
            }
        }
        void BuildLegacyCircuitGround(Material grass)
        {
            // One shared world-space grid, not overlapping swept terrain ribbons.
            const float step=8; const int cells=8; const float tile=step*cells;
            var tiles=new HashSet<Vector2Int>();
            for(int i=0;i<circuitPoints.Length-1;i+=12)
            {
                var p=circuitPoints[i];var key=new Vector2Int(Mathf.FloorToInt(p.x/tile),Mathf.FloorToInt(p.z/tile));
                for(int x=-1;x<=1;x++)for(int z=-1;z<=1;z++) tiles.Add(key+new Vector2Int(x,z));
            }
            circuitGroundHeights.Clear();
            foreach(var key in tiles)
                for(int z=0;z<=cells;z++)for(int x=0;x<=cells;x++)
                {
                    var index=new Vector2Int(key.x*cells+x,key.y*cells+z);
                    if(circuitGroundHeights.ContainsKey(index))continue;
                    var p=new Vector3(index.x*step,0,index.y*step);
                    float d=CircuitNearest(p,out float station);
                    var at=CircuitPosition(station);
                    var right=Vector3.Cross(Vector3.up,CircuitTangent(station));
                    float lateral=Vector3.Dot(p-at,right);
                    float height=at.y+CircuitSurfaceHeight(station,Mathf.Clamp(lateral,-roadWidth*.5f,roadWidth*.5f))-.3f;
                    height+=Mathf.Clamp01((d-roadWidth*.5f)/30)*((Mathf.PerlinNoise(p.x*.007f+31,p.z*.007f+17)-.5f)*4);
                    // Keep the shared foundation below nearby pavement; the precise shoulder
                    // connects the kerb to this field. Never cut holes in the foundation.
                    if(d<roadWidth*.5f+12)height-=2;
                    circuitGroundHeights[index]=height;
                }
            // Nearby parts of the circuit can have different heights. Bound terrain slopes
            // together instead of creating vertical cliffs where nearest-road identity changes.
            for(int pass=0;pass<16;pass++)
            {
                var next=new Dictionary<Vector2Int,float>(circuitGroundHeights);
                foreach(var pair in circuitGroundHeights)
                    foreach(var delta in new[]{Vector2Int.left,Vector2Int.right,Vector2Int.up,Vector2Int.down})
                        if(circuitGroundHeights.TryGetValue(pair.Key+delta,out float neighbor))
                            next[pair.Key]=Mathf.Min(next[pair.Key],neighbor+step*.6f);
                circuitGroundHeights.Clear();foreach(var pair in next)circuitGroundHeights.Add(pair.Key,pair.Value);
            }
            foreach(var key in tiles)
            {
                var v=new List<Vector3>();var t=new List<int>();var uv=new List<Vector2>();
                for(int z=0;z<=cells;z++)for(int x=0;x<=cells;x++)
                {
                    Vector3 p=new Vector3(key.x*tile+x*step,0,key.y*tile+z*step);
                    p.y=circuitGroundHeights[new Vector2Int(key.x*cells+x,key.y*cells+z)];
                    v.Add(p);uv.Add(new Vector2(p.x/8,p.z/8));
                }
                for(int z=0;z<cells;z++)for(int x=0;x<cells;x++)
                {
                    int a=z*(cells+1)+x,b=a+cells+1;
                    t.Add(a);t.Add(b);t.Add(a+1);
                    t.Add(a+1);t.Add(b);t.Add(b+1);
                }
                CircuitMesh($"Ground_{key.x}_{key.y}",v,t,uv,grass,terrain.transform);
            }
        }
    }
}
