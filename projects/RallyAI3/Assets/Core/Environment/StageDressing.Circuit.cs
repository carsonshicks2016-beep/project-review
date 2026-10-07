using UnityEngine;
using System;
using System.Collections.Generic;

namespace Core.Environment
{
    public partial class StageDressing
    {
        static TrackGenerator circuitDressed;
        static GameObject circuitTerrain;
        static readonly Dictionary<int,GameObject> circuitForestCells=new Dictionary<int,GameObject>();
        public static void DressCircuitNearby(TrackGenerator track,bool force=false)
        {
            var vehicle=UnityEngine.Object.FindAnyObjectByType<Core.Physics.VehicleController>();
            float station=track.circuitStart;
            if(vehicle!=null)track.CircuitNearest(track.transform.InverseTransformPoint(vehicle.transform.position),out station,true);
            int count=Mathf.CeilToInt(track.circuitLength/200),center=Mathf.FloorToInt(station/200);
            if(force||circuitDressed!=track||circuitTerrain!=track.terrain||track.transform.Find(ForestName)==null)
            {
                Transform old=track.transform.Find(ForestName);
                if(old!=null){old.gameObject.SetActive(false);if(Application.isPlaying)Destroy(old.gameObject);else DestroyImmediate(old.gameObject);}
                circuitForestCells.Clear();circuitDressed=track;circuitTerrain=track.terrain;
                var root=new GameObject(ForestName){layer=2};root.transform.SetParent(track.transform,false);
                StageAtmosphere.Apply(track);
            }
            if(ReviewKit==null)return;
            var needed=new HashSet<int>();for(int offset=-3;offset<=3;offset++)needed.Add((center+offset+count)%count);
            var retiring=new List<int>();foreach(var pair in circuitForestCells)if(!needed.Contains(pair.Key))retiring.Add(pair.Key);
            foreach(int key in retiring){var cell=circuitForestCells[key];if(cell!=null){cell.SetActive(false);if(Application.isPlaying)Destroy(cell);else DestroyImmediate(cell);}circuitForestCells.Remove(key);}
            var bark=Resources.Load<Material>(MaterialPath);var parent=track.transform.Find(ForestName);
            foreach(int key in needed)
            {
                if(circuitForestCells.ContainsKey(key))continue;
                var root=new GameObject("ForestCell_"+key){layer=2};root.transform.SetParent(parent,false);root.AddComponent<DressingMeshOwner>();
                var forest=new ReviewForest(root.transform,bark);
                for(int k=key*20;k<(key+1)*20&&k*10<track.circuitLength;k++)
                {
                    float s=k*10f;var rng=new System.Random(unchecked(track.seed*7919+k*104729));
                    for(int side=-1;side<=1;side+=2)for(int band=0;band<5;band++)
                    {
                        float lateral=side*(15+band*7+(float)rng.NextDouble()*5);
                        Vector3 foot=track.CircuitSurfacePoint(s+(float)rng.NextDouble()*8,lateral);
                        if(track.CircuitClearance(foot)<4)continue;
                        Vector3 world=track.transform.TransformPoint(foot);
                        if(!UnityEngine.Physics.Raycast(world+Vector3.up*60,Vector3.down,out var hit,160,~(1<<2)))continue;
                        if(!hit.collider.transform.IsChildOf(track.terrain.transform))continue;
                        foot=track.transform.InverseTransformPoint(hit.point);
                        forest.Add(foot,9+(float)rng.NextDouble()*7,(float)rng.NextDouble()*360,.2f,.6f,false,.3f,s);
                    }
                }
                forest.Attach();circuitForestCells.Add(key,root);
            }
        }
    }
}
