#if UNITY_EDITOR || UNITY_STANDALONE || UNITY_ANDROID || UNITY_IOS || UNITY_WSA || UNITY_WEBGL || UNITY_LINUX
#define UNITY
#endif

using System.Diagnostics;

namespace Doji.AI.Transformers {

    public static class Log{

        [Conditional("LOG_INFO")]
        public static void Info(string message) {
#if UNITY
            UnityEngine.Debug.Log(message);
#else
            System.Console.WriteLine(message);
#endif
        }

        [Conditional("LOG_INFO")]
        public static void Info(object message) {
#if UNITY
            UnityEngine.Debug.Log(message);
#else
            System.Console.WriteLine(message);
#endif
        }

        [Conditional("LOG_INFO")]
        [Conditional("LOG_WARNINGS")]
        public static void Warning(string message) {
#if UNITY
            UnityEngine.Debug.LogWarning(message);
#else
            System.Console.WriteLine(message);
#endif
        }

        [Conditional("LOG_INFO")]
        [Conditional("LOG_WARNINGS")]
        [Conditional("LOG_ERRORS")]
        public static void Error(string message) {
#if UNITY
            UnityEngine.Debug.LogError(message);
#else
            System.Console.WriteLine(message);
#endif
        }
    }
}