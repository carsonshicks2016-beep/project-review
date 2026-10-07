using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Aegis.EditorTools
{
    /// <summary>
    /// Assembles the M1 balance-training scene from the imported Paladin + idle clip:
    /// ground, camera, an ArticulationBody ragdoll (RagdollBuilder), a logic-only reference
    /// clone playing the idle, and a DuelistAgent wired up. Obs/action sizes auto-set at runtime.
    /// Run: menu **Aegis ▸ Build M1 Balance Scene**, or -executeMethod
    /// Aegis.EditorTools.BuildM1Scene.Build.
    /// </summary>
    public static class BuildM1Scene
    {
        const string PaladinPath = "Assets/Mixamo/Paladin.fbx";
        const string IdlePath = "Assets/Mixamo/idle.fbx";
        const string ScenePath = "Assets/Scenes/Arena.unity";

        [MenuItem("Aegis/Build M1 Balance Scene")]
        public static void Build()
        {
            var paladin = AssetDatabase.LoadAssetAtPath<GameObject>(PaladinPath);
            if (paladin == null) { Debug.LogError("[Aegis] Paladin.fbx not found at " + PaladinPath); return; }
            var idle = LoadClip(IdlePath);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(5, 1, 5);

            if (Camera.main != null)
            {
                Camera.main.transform.position = new Vector3(0, 1.4f, -3.5f);
                Camera.main.transform.rotation = Quaternion.Euler(8, 0, 0);
            }

            // --- Fighter (active ragdoll) ---
            var fighter = (GameObject)Object.Instantiate(paladin);
            fighter.name = "Fighter";
            fighter.transform.position = Vector3.zero;
            var anim = fighter.GetComponentInChildren<Animator>();

            var builder = fighter.AddComponent<RagdollBuilder>();
            builder.Build(); // adds ArticulationBodies + colliders to the bones

            var hipsT = anim.GetBoneTransform(HumanBodyBones.Hips);
            var hips = hipsT != null ? hipsT.GetComponent<ArticulationBody>() : null;
            if (hips == null) { Debug.LogError("[Aegis] No ArticulationBody on Hips — ragdoll build failed."); return; }

            var pd = fighter.AddComponent<PDJointController>();
            pd.root = hips;

            // --- Reference clone (logic-only; plays idle for the imitation reward) ---
            var refClone = (GameObject)Object.Instantiate(paladin);
            refClone.name = "Reference";
            refClone.transform.position = new Vector3(3, 0, 0);
            foreach (var r in refClone.GetComponentsInChildren<Renderer>()) r.enabled = false;
            var rpp = refClone.AddComponent<ReferencePoseProvider>();
            rpp.referenceAnimator = refClone.GetComponentInChildren<Animator>();
            rpp.clip = idle;

            // --- Agent ---
            var agent = fighter.AddComponent<DuelistAgent>();
            agent.controller = pd;
            agent.reference = rpp;
            agent.ragdollAnimator = anim;
            agent.root = hips;

            var dr = fighter.AddComponent<DecisionRequester>();
            dr.DecisionPeriod = 5;
            dr.TakeActionsBetweenDecisions = true;

            // BehaviorParameters is auto-added by DuelistAgent's [RequireComponent] — grab THAT
            // instance (adding a second is ignored by the Agent → name/size mismatch).
            var bp = fighter.GetComponent<BehaviorParameters>();
            if (bp == null) bp = fighter.AddComponent<BehaviorParameters>();
            // Set the serialized name directly so it persists into the scene and matches the
            // trainer config key (config/balance_ppo.yaml → "Duelist"). Sizes auto-set at runtime.
            var bpSO = new SerializedObject(bp);
            bpSO.FindProperty("m_BehaviorName").stringValue = "Duelist";
            bpSO.ApplyModifiedPropertiesWithoutUndo();

            Directory.CreateDirectory("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.Refresh();

            int bodies = fighter.GetComponentsInChildren<ArticulationBody>().Length;
            Debug.Log($"[Aegis] Built M1 scene '{ScenePath}'. Ragdoll ArticulationBodies: {bodies}. " +
                      $"Reference clip: {(idle != null ? idle.name : "MISSING")}.");
        }

        static AnimationClip LoadClip(string path)
        {
            foreach (var o in AssetDatabase.LoadAllAssetsAtPath(path))
                if (o is AnimationClip c && !c.name.StartsWith("__")) return c;
            Debug.LogError("[Aegis] No AnimationClip found in " + path);
            return null;
        }
    }
}
