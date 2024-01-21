#if UNITY_EDITOR || UNITY_STANDALONE || UNITY_ANDROID || UNITY_IOS || UNITY_WSA || UNITY_WEBGL || UNITY_LINUX
#define UNITY
#endif

using System.Diagnostics;

namespace Doji.AI.Transformers {

    public static class Log{

        [Conditional("DOJI_LL_DEBUG")]
        [Conditional("DDOJI_LL_WARNING")]
        [Conditional("DOJI_LL_ERROR")]
        public static void Info(string message) {
#if DOJI_LL_ERROR
            UnityEngine.Debug.Log(message);
#else
;
#endif
#if UNITY
            UnityEngine.Debug.Log(message);
#else
            System.Console.WriteLine(message);
#endif
        }

        [Conditional("DDOJI_LL_WARNING")]
        [Conditional("DOJI_LL_ERROR")]
        public static void Warning(string message) {
#if UNITY
            UnityEngine.Debug.Log(message);
#else
            System.Console.WriteLine(message);
#endif
        }

        [Conditional("DOJI_LL_ERROR")]
        public static void Error(string message) {
#if UNITY
            UnityEngine.Debug.Log(message);
#else
            System.Console.WriteLine(message);
#endif
        }
    }
}