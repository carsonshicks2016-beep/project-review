using UnityEditor;
using UnityEngine;

namespace EditorScripts
{
    /// <summary>
    /// Get-or-add for components, written so that it actually works on Unity objects.
    ///
    /// The obvious one-liner is a trap:
    ///
    ///     var box = go.GetComponent&lt;BoxCollider&gt;() ?? go.AddComponent&lt;BoxCollider&gt;();
    ///
    /// GetComponent hands back a "fake null" when the component is not there — a live C#
    /// wrapper around a native object that does not exist. Unity overloads == so that
    /// wrapper compares equal to null, but ?? and ?. are compiled against real null and
    /// skip the overload entirely. So the wrapper counts as a hit, AddComponent never
    /// runs, and the first field you touch throws MissingComponentException.
    ///
    /// Comparing with != null is what routes through the overload. Hence this helper.
    /// </summary>
    public static class ComponentEnsure
    {
        /// <summary>Existing component if there is one, otherwise a new one.</summary>
        public static T Ensure<T>(GameObject go) where T : Component
        {
            T existing = go.GetComponent<T>();
            return existing != null ? existing : go.AddComponent<T>();
        }

        /// <summary>Same, but an addition is recorded on the undo stack.</summary>
        public static T EnsureUndo<T>(GameObject go) where T : Component
        {
            T existing = go.GetComponent<T>();
            return existing != null ? existing : Undo.AddComponent<T>(go);
        }
    }
}
