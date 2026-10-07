using System.Collections.Generic;
using UnityEngine;
namespace Core.Environment
{
    public sealed class CircuitResourceOwner:MonoBehaviour
    {
        readonly List<Object> owned=new List<Object>();
        public void Register(Object value){if(value!=null&&!owned.Contains(value))owned.Add(value);}
        void OnDestroy(){foreach(var value in owned)if(value!=null){if(Application.isPlaying)Destroy(value);else DestroyImmediate(value);}owned.Clear();}
    }
}
