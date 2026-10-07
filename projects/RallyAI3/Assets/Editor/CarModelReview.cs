using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace EditorScripts
{
    public static class CarModelReview
    {
        static string Output => System.Environment.GetEnvironmentVariable("RALLY_CAR_REVIEW_OUTPUT")
            ?? ".rally/visual-review/car";
        public static void Capture()
        {
            Directory.CreateDirectory(Output);
            EditorSceneManager.OpenScene("Assets/Scenes/RallyTraining.unity", OpenSceneMode.Single);
            if (System.Environment.GetEnvironmentVariable("RALLY_COCKPIT_REVIEW") == "1")
            {
                var vehicle = Object.FindAnyObjectByType<Core.Physics.VehicleController>();
                if (vehicle != null) Core.Presentation.RallyCockpit.Create(vehicle);
            }
            Render("capture");
        }

        public static void AuditRebuilds()
        {
            Directory.CreateDirectory(Output);
            EditorSceneManager.OpenScene("Assets/Scenes/RallyTraining.unity", OpenSceneMode.Single);
            var car=GameObject.Find("RallyCar");
            var source=car.GetComponent<Core.Physics.VehicleController>();
            int colliders=car.GetComponentsInChildren<Collider>(true).Length;
            var suspension=car.GetComponentsInChildren<Core.Physics.DynamicSuspension>();
            var anchors=System.Array.ConvertAll(suspension,s=>s.transform.localPosition);
            int objects=-1;
            using(var writer=new StreamWriter(Path.Combine(Output,"editor-rebuilds.jsonl")))
                for(int i=0;i<8;i++)
                {
                    RallyCarBuilder.RebuildVisuals();
                    int count=car.GetComponentsInChildren<Transform>(true).Length;
                    if(i>0 && count!=objects) throw new System.InvalidOperationException("Car object count grew");
                    objects=count;
                    if(car.GetComponentsInChildren<Collider>(true).Length!=colliders)
                        throw new System.InvalidOperationException("Visual rebuild changed collision shell");
                    for(int j=0;j<suspension.Length;j++)
                        if(suspension[j].transform.localPosition!=anchors[j])
                            throw new System.InvalidOperationException("Visual rebuild moved suspension");
                    if(car.GetComponent<UI.RobotDriverIK>().steeringWheel==null)
                        throw new System.InvalidOperationException("Steering binding lost");
                    writer.WriteLine($"{{\"cycle\":{i+1},\"objects\":{count},\"colliders\":{colliders},\"mass\":{source.mass.ToString(System.Globalization.CultureInfo.InvariantCulture)}}}");
                }
            AssetDatabase.SaveAssets();
        }

        public static void Execute()
        {
            Directory.CreateDirectory(".rally/visual-review/car");
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            if (!File.Exists(".rally/visual-review/car/before-side-720.png")) Render("before");
            EditorSceneManager.OpenScene("Assets/Scenes/RallyTraining.unity", OpenSceneMode.Single);
            RallyCarBuilder.RebuildVisuals();
            AssetDatabase.SaveAssets();
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            Render("after");
        }

        static void Render(string label)
        {
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(.60f, .64f, .70f);
            var source = GameObject.Find("RallyCar");
            var car = source != null ? Object.Instantiate(source) : new GameObject("Review car");
            car.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            foreach (var behaviour in car.GetComponentsInChildren<MonoBehaviour>()) behaviour.enabled = false;
            foreach (var collider in car.GetComponentsInChildren<Collider>()) collider.enabled = false;
            foreach (var rigidbody in car.GetComponentsInChildren<Rigidbody>()) rigidbody.isKinematic = true;
            if (source != null)
                foreach (var root in source.scene.GetRootGameObjects()) root.SetActive(false);
            car.SetActive(true);
            if (source == null)
            {
            var body = car.AddComponent<MeshFilter>();
            body.sharedMesh = AssetDatabase.LoadAssetAtPath<Mesh>("Assets/Art/Generated/Impreza_Body.asset");
            var renderer = car.AddComponent<MeshRenderer>();
            string[] names = {"Body", "Glass", "Trim", "Lamp", "TailLamp"};
            var mats = new Material[5];
            for (int i=0;i<5;i++) mats[i] = AssetDatabase.LoadAssetAtPath<Material>($"Assets/Art/Generated/Impreza_{names[i]}.mat");
            renderer.sharedMaterials = mats;
            foreach (float x in new[] {-.75f,.75f})
                foreach (float z in new[] {-1.26f,1.26f})
                {
                    var wheel = new GameObject("Wheel");
                    wheel.transform.SetParent(car.transform);
                    wheel.transform.localPosition = new Vector3(x,.33f,z);
                    wheel.AddComponent<MeshFilter>().sharedMesh = AssetDatabase.LoadAssetAtPath<Mesh>("Assets/Art/Generated/Impreza_Wheel.asset");
                    wheel.AddComponent<MeshRenderer>().sharedMaterials = new[] {
                        AssetDatabase.LoadAssetAtPath<Material>("Assets/Art/Generated/Impreza_Rim.mat"),
                        AssetDatabase.LoadAssetAtPath<Material>("Assets/Art/Generated/Impreza_Tyre.mat")};
                }
            }
            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.transform.localScale = Vector3.one*4;
            var floorMat = new Material(Shader.Find("Standard"));
            floorMat.color = new Color(.23f,.26f,.29f);
            floor.GetComponent<Renderer>().sharedMaterial = floorMat;
            var light = new GameObject("Soft daylight").AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.3f;
            light.transform.rotation = Quaternion.Euler(40,-35,0);
            light.shadows = LightShadows.Soft;
            var camera = new GameObject("Review camera").AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(.15f,.18f,.22f);
            camera.fieldOfView = 33;
            camera.nearClipPlane = .05f;
            Vector3[] positions = {new Vector3(5,2.25f,-6),new Vector3(7,1.7f,0),new Vector3(4,2.1f,6),
                new Vector3(-5,2.25f,-6),new Vector3(0,1.4f,8),new Vector3(0,1.4f,-8),
                new Vector3(0,1.42f,1.38f),Core.Presentation.RallyCockpit.DriverEye};
            string[] views = {"rear-quarter","side","front-quarter","left-rear-quarter","front","rear","hood","driver"};
            for (int size=0;size<2;size++)
                for (int i=0;i<positions.Length;i++)
                {
                    camera.transform.position = positions[i];
                    camera.fieldOfView = i >= 6 ? 78 : 33;
                    camera.transform.LookAt(i >= 6 ? positions[i] + Vector3.forward * 20 : new Vector3(0,.70f,0));
                    int width = size==0 ? 1280 : 1920, height = size==0 ? 720 : 1080;
                    var target = new RenderTexture(width,height,24);
                    camera.targetTexture = target;
                    camera.Render();
                    RenderTexture.active = target;
                    var image = new Texture2D(width,height,TextureFormat.RGB24,false);
                    image.ReadPixels(new Rect(0,0,width,height),0,0);
                    image.Apply();
                    File.WriteAllBytes(Path.Combine(Output, $"{label}-{views[i]}-{height}.png"),image.EncodeToPNG());
                    RenderTexture.active = null;
                    camera.targetTexture = null;
                    Object.DestroyImmediate(image);
                    Object.DestroyImmediate(target);
                }
            Object.DestroyImmediate(car);
            Object.DestroyImmediate(floor);
            Object.DestroyImmediate(floorMat);
            Object.DestroyImmediate(light.gameObject);
            Object.DestroyImmediate(camera.gameObject);
            Debug.Log($"Car review rendered: {label}");
        }
    }
}
