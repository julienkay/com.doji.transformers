#if UNITY_EDITOR || UNITY_STANDALONE || UNITY_ANDROID || UNITY_IOS || UNITY_WSA || UNITY_WEBGL || UNITY_LINUX
#define UNITY
#endif

namespace Doji.AI.Transformers {

    internal static class Debug{

        public static void Assert(bool condition) {
#if UNITY
            UnityEngine.Debug.Assert(condition);
#else
            System.Diagnostics.Debug.Assert(condition);
#endif
        }
        public static void Assert(bool condition, string message) {
#if UNITY
            UnityEngine.Debug.Assert(condition, message);
#else
            System.Diagnostics.Debug.Assert(condition, message);
#endif
        }
    }
}