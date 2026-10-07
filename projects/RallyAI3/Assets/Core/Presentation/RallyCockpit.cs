using Core.ML;
using Core.Physics;
using UnityEngine;
using System.Collections.Generic;

namespace Core.Presentation
{
    [DefaultExecutionOrder(200)]
    public sealed class RallyCockpit : MonoBehaviour
    {
        VehicleController vehicle;
        Transform needle;
        Renderer[] gear, speed;
        readonly Dictionary<char,Mesh> glyphs = new Dictionary<char,Mesh>();
        Material whiteMaterial, goldMaterial, greyMaterial;
        Mesh dialMesh;
        Mesh roundDialMesh;
        Material[] interiorMaterials;
        Material dialMaterial, needleMaterial;
        Renderer shiftLight;
        Transform steeringColumn;
        Vector3 originalColumn;
        Rigidbody body;
        int lastGear = int.MinValue, lastSpeed = int.MinValue;
        public static readonly Vector3 DriverEye = new Vector3(.36f, 1.10f, -.18f);
        public static readonly Vector3 HoodEye = new Vector3(0, 1.42f, 1.38f);
        public string GearReadout => gear[0].GetComponent<MeshFilter>().sharedMesh.name.Substring("Cockpit glyph ".Length);
        public string SpeedReadout => string.Concat(System.Array.ConvertAll(speed,
            r=>r.GetComponent<MeshFilter>().sharedMesh.name.Substring("Cockpit glyph ".Length)));

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            if (Application.isBatchMode || (LabRuntime.Enabled && !LabRuntime.Config.viewer)) return;
            var vehicle = FindAnyObjectByType<VehicleController>();
            if (vehicle != null && vehicle.GetComponentInChildren<RallyCockpit>() == null)
                Create(vehicle);
        }

        public static RallyCockpit Create(VehicleController vehicle)
        {
            var existing=vehicle.GetComponentInChildren<RallyCockpit>();
            if(existing!=null) return existing;
            var mesh = Resources.Load<Mesh>("Vehicles/GC8_Cockpit");
            if (mesh == null) { Debug.LogError("[Cockpit] Missing authored interior mesh"); return null; }
            var root = new GameObject("ViewerCockpit");
            root.transform.SetParent(vehicle.transform, false);
            var cockpit = root.AddComponent<RallyCockpit>();
            cockpit.vehicle = vehicle;
            cockpit.body = vehicle.GetComponent<Rigidbody>();
            var body = vehicle.transform.Find("VisualChassis");
            var source = body != null ? body.GetComponentInChildren<MeshRenderer>() : null;
            if (source == null || source.sharedMaterials.Length != 5)
            {
                Debug.LogError("[Cockpit] Missing five-slot body materials");
                Destroy(root);
                return null;
            }
            root.AddComponent<MeshFilter>().sharedMesh = mesh;
            cockpit.interiorMaterials = new Material[5];
            var colors = new[] {new Color(.72f,.53f,.12f),new Color(.09f,.095f,.105f),
                new Color(.055f,.062f,.068f),new Color(.34f,.37f,.39f),new Color(.025f,.028f,.03f)};
            var gloss = new[] {.12f,.02f,.16f,.38f,.05f};
            for(int i=0;i<5;i++)
            {
                var material = new Material(source.sharedMaterials[2]) {name="Cockpit surface "+i,color=colors[i]};
                material.mainTexture=null;
                material.SetFloat("_Glossiness",gloss[i]);
                material.SetColor("_SpecColor",Color.white*(i==3?.15f:.025f));
                cockpit.interiorMaterials[i]=material;
            }
            root.AddComponent<MeshRenderer>().sharedMaterials = cockpit.interiorMaterials;
            cockpit.BuildInstruments();
            cockpit.steeringColumn = vehicle.transform.Find("SteeringColumn");
            if (cockpit.steeringColumn != null)
            {
                cockpit.originalColumn = cockpit.steeringColumn.localPosition;
                cockpit.steeringColumn.SetParent(root.transform,false);
                cockpit.steeringColumn.localPosition = new Vector3(.36f,.805f,.10f);
            }
            return cockpit;
        }

        void BuildInstruments()
        {
            var instrument = Resources.Load<Material>("Vehicles/Instrument");
            dialMaterial = new Material(instrument) { color = new Color(.035f,.04f,.045f) };
            needleMaterial = new Material(instrument) { color = new Color(1f,.24f,.06f) };
            whiteMaterial = new Material(instrument) {color=Color.white};
            goldMaterial = new Material(instrument) {color=new Color(1,.72f,.18f)};
            greyMaterial = new Material(instrument) {color=Color.gray};
            dialMaterial.name="Cockpit dial"; needleMaterial.name="Cockpit needle";
            whiteMaterial.name="Cockpit white"; goldMaterial.name="Cockpit gold"; greyMaterial.name="Cockpit grey";
            foreach(char glyph in "0123456789NRPMXKH/ ") Glyph(glyph);
            dialMesh = new Mesh { name = "Cockpit instrument quad" };
            dialMesh.vertices = new[] {new Vector3(-.5f,-.5f,0),new Vector3(.5f,-.5f,0),
                new Vector3(.5f,.5f,0),new Vector3(-.5f,.5f,0)};
            dialMesh.triangles = new[] {0,2,1,0,3,2};
            dialMesh.RecalculateNormals();
            roundDialMesh = new Mesh {name="Cockpit round dial"};
            var circle = new Vector3[49]; var indices = new int[48*3];
            for(int i=0;i<48;i++)
            {
                float angle=i*Mathf.PI*2/48;
                circle[i+1]=new Vector3(Mathf.Cos(angle)*.5f,Mathf.Sin(angle)*.5f,0);
                indices[i*3]=0; indices[i*3+1]=(i+1)%48+1; indices[i*3+2]=i+1;
            }
            roundDialMesh.vertices=circle; roundDialMesh.triangles=indices; roundDialMesh.RecalculateNormals();
            var bezel = Panel("Tachometer bezel",new Vector3(.33f,.958f,.325f),new Vector3(.102f,.102f,1),greyMaterial);
            bezel.GetComponent<MeshFilter>().sharedMesh=roundDialMesh;
            var face = Panel("Tachometer", new Vector3(.33f,.958f,.321f), new Vector3(.092f,.092f,1), dialMaterial);
            face.GetComponent<MeshFilter>().sharedMesh=roundDialMesh;
            for (int i=0;i<=8;i++)
            {
                float angle = (225f-i*33.75f)*Mathf.Deg2Rad;
                Label(i.ToString(),face.localPosition+new Vector3(Mathf.Cos(angle)*.035f,
                    Mathf.Sin(angle)*.035f,-.003f),.010f,Color.white);
                var tick=Panel("Tachometer tick",face.localPosition+new Vector3(Mathf.Cos(angle)*.041f,
                    Mathf.Sin(angle)*.041f,-.004f),new Vector3(.006f,.0015f,1),i>=7?needleMaterial:whiteMaterial);
                tick.localRotation=Quaternion.Euler(0,0,angle*Mathf.Rad2Deg);
            }
            Label("RPM x1000",new Vector3(.33f,.944f,.314f),.006f,Color.gray);
            needle = new GameObject("RPM needle pivot").transform;
            needle.SetParent(transform,false);
            needle.localPosition = face.localPosition+new Vector3(0,0,-.008f);
            var pointer = Panel("RPM needle",Vector3.zero,new Vector3(.032f,.002f,1),needleMaterial);
            pointer.SetParent(needle,false);
            pointer.localPosition = new Vector3(.016f,0,0);
            Panel("Digital instrument recess",new Vector3(.452f,.956f,.320f),new Vector3(.09f,.087f,1),dialMaterial);
            gear = Label("N",new Vector3(.452f,.977f,.315f),.029f,new Color(1,.72f,.18f));
            speed = Label("000",new Vector3(.452f,.943f,.315f),.017f,Color.white);
            Label("MPH",new Vector3(.452f,.923f,.315f),.007f,Color.gray);
            shiftLight = Panel("Shift light",new Vector3(.33f,1.012f,.315f),new Vector3(.025f,.005f,1),needleMaterial).GetComponent<Renderer>();
            var eye = new GameObject("DriverEye").transform;
            eye.SetParent(transform,false); eye.localPosition = DriverEye;
            var hood = new GameObject("HoodEye").transform;
            hood.SetParent(transform,false); hood.localPosition = HoodEye;
        }

        Transform Panel(string name, Vector3 position, Vector3 scale, Material material)
        {
            var panel = new GameObject(name);
            panel.transform.SetParent(transform,false);
            panel.transform.localPosition = position; panel.transform.localScale = scale;
            panel.AddComponent<MeshFilter>().sharedMesh = dialMesh;
            var renderer = panel.AddComponent<MeshRenderer>(); renderer.sharedMaterial = material;
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            return panel.transform;
        }

        Renderer[] Label(string value, Vector3 position, float size, Color color)
        {
            var renderers = new Renderer[value.Length];
            for (int i=0;i<value.Length;i++)
            {
                var root = new GameObject("Instrument glyph");
                root.transform.SetParent(transform,false);
                root.transform.localPosition = position+new Vector3((i-(value.Length-1)*.5f)*size*.8f,0,0);
                root.transform.localScale = Vector3.one*size;
                root.AddComponent<MeshFilter>().sharedMesh = Glyph(value[i]);
                var renderer = root.AddComponent<MeshRenderer>();
                renderer.sharedMaterial = color == Color.white ? whiteMaterial : color == Color.gray ? greyMaterial : goldMaterial;
                renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                renderers[i] = renderer;
            }
            return renderers;
        }

        Mesh Glyph(char c)
        {
            c=char.ToUpperInvariant(c);
            if (glyphs.TryGetValue(c,out var mesh)) return mesh;
            const string alphabet = "0123456789NRPMXKH/ ";
            string[] patterns = {"111101101101111","010110010010111","111001111100111","111001111001111",
                "101101111001001","111100111001111","111100111101111","111001010010010","111101111101111",
                "111101111001111","101111111111101","110101110101101","110101110100100","101111111101101",
                "101101010101101","101110100110101","101101111101101","001001010100100","000000000000000"};
            int index=alphabet.IndexOf(c);
            string pattern=patterns[index<0?patterns.Length-1:index];
            var vertices=new List<Vector3>(); var triangles=new List<int>();
            for(int y=0;y<5;y++) for(int x=0;x<3;x++) if(pattern[y*3+x]=='1')
            {
                int start=vertices.Count; float px=(x-1)*.2f, py=(2-y)*.2f;
                vertices.Add(new Vector3(px-.09f,py-.09f,0)); vertices.Add(new Vector3(px+.09f,py-.09f,0));
                vertices.Add(new Vector3(px+.09f,py+.09f,0)); vertices.Add(new Vector3(px-.09f,py+.09f,0));
                triangles.AddRange(new[]{start,start+2,start+1,start,start+3,start+2});
            }
            mesh=new Mesh{name="Cockpit glyph "+c}; mesh.SetVertices(vertices); mesh.SetTriangles(triangles,0);
            mesh.RecalculateNormals(); glyphs.Add(c,mesh); return mesh;
        }

        void Update()
        {
            if (vehicle == null || needle == null) return;
            needle.localRotation = Quaternion.Euler(0,0,225f-Mathf.Clamp(vehicle.currentRpm/1000f,0,8)*33.75f);
            if (vehicle.currentGear != lastGear)
            {
                lastGear = vehicle.currentGear;
                gear[0].GetComponent<MeshFilter>().sharedMesh = Glyph(lastGear < 0 ? 'R' : lastGear == 0 ? 'N' : (char)('0'+lastGear));
            }
            int mph = body != null ? Mathf.RoundToInt(body.linearVelocity.magnitude*3.6f / 1.609344f) : 0;
            if (mph != lastSpeed)
            {
                lastSpeed = mph; int value = Mathf.Clamp(mph,0,999);
                for(int i=2;i>=0;i--) {speed[i].GetComponent<MeshFilter>().sharedMesh=Glyph((char)('0'+value%10)); value/=10;}
            }
            shiftLight.enabled = vehicle.currentRpm >= 6800f;
        }

        void OnDestroy()
        {
            if (dialMesh != null) Destroy(dialMesh);
            if (roundDialMesh != null) Destroy(roundDialMesh);
            if (interiorMaterials != null) foreach(var material in interiorMaterials) Destroy(material);
            RestoreColumn();
            if (dialMaterial != null) Destroy(dialMaterial);
            if (needleMaterial != null) Destroy(needleMaterial);
            if (whiteMaterial != null) Destroy(whiteMaterial);
            if (goldMaterial != null) Destroy(goldMaterial);
            if (greyMaterial != null) Destroy(greyMaterial);
            foreach(var mesh in glyphs.Values) Destroy(mesh);
        }

        void OnDisable() => RestoreColumn();
        void RestoreColumn()
        {
            if (steeringColumn == null || vehicle == null) return;
            steeringColumn.SetParent(vehicle.transform,false);
            steeringColumn.localPosition = originalColumn;
        }
    }
}
