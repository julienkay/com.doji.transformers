using Newtonsoft.Json;
using System;
using System.IO;
using UnityEngine;
using static Doji.AI.Transformers.PreTrainedTokenizerBase;

namespace Doji.AI.Transformers {

    public static class AutoTokenizer {

#if UNITY_EDITOR
        public static event Action<string> OnModelRequested = (x) => { };
#endif

        internal static TokenizerConfig LoadTokenizerConfig(string pretrainedModelNameOrPath) {
            return LoadFromJson<TokenizerConfig>(pretrainedModelNameOrPath);
        }

        /// <summary>
        /// Loads an object of type <typeparamref name="T"/> from a json file
        /// either in StreamingAssets or Resources.
        /// </summary>
        internal static T LoadFromJson<T>(string pretrainedModelNameOrPath) {
            string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, pretrainedModelNameOrPath, TOKENIZER_CONFIG_FILE);
            if (File.Exists(streamingAssetsPath)) {
                return LoadJsonFromFile<T>(streamingAssetsPath);
            }
            string resourcePath = Path.Combine(pretrainedModelNameOrPath, Path.ChangeExtension(TOKENIZER_CONFIG_FILE, null));
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
        /// Returns a path for the tokenizer.model file in either StreamingAssets or Resources.
        /// </summary>
        private static string GetTokenizerModelPath(string pretrainedModelNameOrPath) {
            string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, pretrainedModelNameOrPath, "tokenizer.model");
            if (File.Exists(streamingAssetsPath)) {
                return streamingAssetsPath;
            }
            string resourcePath = Path.Combine(pretrainedModelNameOrPath, "tokenizer");
            var tokenizerModel = Resources.Load<TokenizerModelAsset>(resourcePath);
            if (tokenizerModel == null) {
                throw new Exception($"Tokenizer model asset for '{pretrainedModelNameOrPath} could not be loaded'");
            }

            // since a path is required we extract the tokenizer model loaded from Resources and return that path.
            string tmpModelPath = CreateTokenizerModelAtPath(tokenizerModel, pretrainedModelNameOrPath);
            return tmpModelPath;
        }

        /// <summary>
        /// Writes the contents of a tokenizer model to a temporary file and returns the path.
        /// </summary>
        private static string CreateTokenizerModelAtPath(TokenizerModelAsset tokenizerModel, string pretrainedModelNameOrPath) {
            if (tokenizerModel == null) {
                UnityEngine.Debug.LogError("TextAsset is null. Cannot write to temp file.");
                return null;
            }
            string folderPath = Path.Combine(Application.temporaryCachePath, pretrainedModelNameOrPath);
            string tempPath = Path.Combine(folderPath, "tokenizer.model");
            if (!Directory.Exists(folderPath)) {
                Directory.CreateDirectory(folderPath);
            }
            File.WriteAllBytes(tempPath, tokenizerModel.ModelData);
            return tempPath;
        }

        private static string ReplaceFileName(string originalFilePath, string newFileName) {
            string directory = Path.GetDirectoryName(originalFilePath);
            string newFilePath = Path.Combine(directory, newFileName);
            return newFilePath;
        }

        /// <summary>
        /// Loads the given tokenizer from a pretrained model vocabulary.
        /// </summary>
        public static PreTrainedTokenizerBase FromPretrained(string pretrainedModelNameOrPath) {
#if UNITY_EDITOR
            OnModelRequested?.Invoke(pretrainedModelNameOrPath);
#endif
            // use the tokenizer_config file to get the specific tokenizer class.
            TokenizerConfig config = LoadTokenizerConfig(pretrainedModelNameOrPath);

            switch (config.TokenizerClass) {
                case "CLIPTokenizer":
                    return new ClipTokenizer(null, null);
                case "LlamaTokenizer":
                    string llamaVocabPath = GetTokenizerModelPath(pretrainedModelNameOrPath);
                    return new LlamaTokenizer(llamaVocabPath, config);
                default:
                    throw new NotImplementedException($"'{config.TokenizerClass}' not yet implemented.");
            }
        }
    }
}