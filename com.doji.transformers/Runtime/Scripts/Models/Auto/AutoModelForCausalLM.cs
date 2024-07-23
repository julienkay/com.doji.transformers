using Newtonsoft.Json;
using System.IO;
using System;
using UnityEngine;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public static class AutoModelForCausalLM {

        public static string CONFIG_FILE = "config.json";

#if UNITY_EDITOR
        public static event Action<string> OnModelRequested = (x) => { };
#endif

        internal static PretrainedConfig LoadPretrainedConfig(string pretrainedModelNameOrPath) {
            return LoadFromJson<PretrainedConfig>(pretrainedModelNameOrPath);
        }

        /// <summary>
        /// Loads an object of type <typeparamref name="T"/> from a json file
        /// either in StreamingAssets or Resources.
        /// </summary>
        internal static T LoadFromJson<T>(string pretrainedModelNameOrPath) {
            string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, pretrainedModelNameOrPath, CONFIG_FILE);
            if (File.Exists(streamingAssetsPath)) {
                return LoadJsonFromFile<T>(streamingAssetsPath);
            }
            string resourcePath = Path.Combine(pretrainedModelNameOrPath, Path.ChangeExtension(CONFIG_FILE, null));
            return LoadJsonFromTextAsset<T>(resourcePath);
        }

        /// <summary>
        /// Loads an object of type <typeparamref name="T"/> from a json file
        /// by deserializing using <see cref="Newtonsoft.Json.JsonConvert"/>.
        /// </summary>
        private static T LoadJsonFromFile<T>(string path) {
#if !UNITY_STANDALONE
            throw new NotImplementedException();
#endif
            if (!File.Exists(path)) {
                throw new FileNotFoundException($"The .json file was not found at: '{path}'");
            }
            string json = File.ReadAllText(path);
            T deserializedObject = JsonConvert.DeserializeObject<T>(json);
            return deserializedObject;
        }

        /// <summary>
        /// Loads an object of type <typeparamref name="T"/> from a text asset in Resources
        /// by deserializing using <see cref="Newtonsoft.Json.JsonConvert"/>.
        /// </summary>
        /// <param name="path">The path to the text file in the Resources folder</param>
        private static T LoadJsonFromTextAsset<T>(string path) {
            TextAsset textAsset = Resources.Load<TextAsset>(path);
            if (textAsset == null) {
                throw new FileNotFoundException($"The TextAsset file was not found at: '{path}'");
            }
            T deserializedObject = JsonConvert.DeserializeObject<T>(textAsset.text);
            Resources.UnloadAsset(textAsset);
            return deserializedObject;
        }

        /// <summary>
        /// Loads the given pretrained model (with a causal language modeling head).
        /// </summary>
        public static PretrainedModel FromPretrained(string pretrainedModelNameOrPath, BackendType backend = BackendType.GPUCompute) {
#if UNITY_EDITOR
            OnModelRequested?.Invoke(pretrainedModelNameOrPath);
#endif
            // use the tokenizer_config file to get the specific tokenizer class.
            PretrainedConfig config = LoadPretrainedConfig(pretrainedModelNameOrPath);
            if (config.Architectures == null || config.Architectures.Count  == 0) {
                throw new Exception($"No architecture found in the config for '{pretrainedModelNameOrPath}'.");
            }
            string arch = config.Architectures[0];

            return arch switch {
                "Phi3ForCausalLM" => Phi3ForCausalLM.FromPretrained(pretrainedModelNameOrPath, backend),
                _ => throw new NotImplementedException($"'{arch}'architecture not yet implemented."),
            };
        }
    }
}