using Newtonsoft.Json;
using System.IO;
using UnityEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// All configuration parameters are stored under <see cref="Configurable{T}.Config"/>.
    /// Also provides a <see cref="Configurable.FromConfig{C}(Config, BackendType)"/>
    /// method for loading classes that inherit from <see cref="Configurable{T}"/>.
    /// </summary>
    public abstract class Configurable<T> where T : PretrainedConfig {

        public const string CONFIG_NAME = "config.json";

        public T Config { get; }

        public Configurable(T config) {
            Config = config;
        }

        /// <summary>
        /// Load a config file from a Resources folder.
        /// </summary>
        protected static U LoadConfigFromTextAsset<U>(string resourcePath) {
            TextAsset textAsset = Resources.Load<TextAsset>(resourcePath);
            if (textAsset == null) {
                //Debug.LogError($"The TextAsset file was not found at: '{path}'");
                return default;
            }

            U deserializedObject = JsonConvert.DeserializeObject<U>(textAsset.text);
            Resources.UnloadAsset(textAsset);
            return deserializedObject;
        }

        /// <summary>
        /// Load a config file from either StreamingAssets or Resources.
        /// If no config is found, null is returned.
        /// </summary>
        protected static PretrainedConfig LoadConfig(string file) {
            if (File.Exists(file.StreamingAssetsPath())) {
                return JsonConvert.DeserializeObject<PretrainedConfig>(File.ReadAllText(file.StreamingAssetsPath()));
            }
            return LoadConfigFromTextAsset<PretrainedConfig>(file.ResourcePath());
        }

        /// <summary>
        /// Load a given type from a file in either StreamingAssets or Resources.
        /// If no file is found, null is returned.
        /// </summary>
        protected static U Load<U>(string file) {
            if (File.Exists(file.StreamingAssetsPath())) {
                return JsonConvert.DeserializeObject<U>(File.ReadAllText(file.StreamingAssetsPath()));
            }
            return LoadConfigFromTextAsset<U>(file.ResourcePath());
        }
    }
}