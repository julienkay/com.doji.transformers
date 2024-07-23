using System.IO;
using UnityEngine;

namespace Doji.AI.Transformers {

    public static class PathUtils {

        public static string StreamingAssetsPath(this string subPath) {
            return Path.Combine(Application.streamingAssetsPath, subPath);
        }

        public static string ResourcePath(this string subPath) {
            return Path.ChangeExtension(subPath, null);
        }

        public static string StreamingAssetsPathForModel(this string subPath, string modelFileName) {
            return Path.Combine(Application.streamingAssetsPath, subPath, $"{modelFileName}.sentis");
        }

        public static string ResourcePathForModel(this string subPath, string modelFileName) {
            return Path.Combine(subPath, $"{modelFileName}.onnx");
        }
    }
}