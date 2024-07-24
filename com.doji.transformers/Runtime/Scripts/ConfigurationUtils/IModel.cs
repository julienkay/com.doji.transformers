using System.IO;
using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Transformers {

    public abstract class PretrainedModel : Configurable<PretrainedConfig>, IDisposable {

        public const string MODEL_NAME = "model";

        /// <summary>
        /// Which <see cref="BackendType"/> to run the model with.
        /// </summary>
        private BackendType Backend { get; set; } = BackendType.GPUCompute;

        /// <summary>
        /// The runtime model.
        /// </summary>
        private Model _model;
        private IWorker _worker;

        public PretrainedModel(Model model, PretrainedConfig config, BackendType backend) : base(config) {
            Backend = backend;
            InitializeNetwork(model);
        }

        protected virtual void InitializeNetwork(Model model) {
            if (model == null) {
                throw new ArgumentException("Unet Model was null", nameof(model));
            }

            _model = model;
            _worker = WorkerFactory.CreateWorker(Backend, _model);
        }

        public virtual void Dispose() {
            _worker?.Dispose();
        }

        /// <summary>
        /// Loads a Sentis <see cref="Model"/> from a <see cref="ModelAsset"/> in Resources.
        /// </summary>
        /// <param name="path">The path to the model file in the Resources folder</param>
        private static Model LoadFromModelAsset(string path) {
            ModelAsset modelAsset = Resources.Load<ModelAsset>(path);
            if (modelAsset == null) {
                return null;
            }
            Model model = ModelLoader.Load(modelAsset);
            Resources.UnloadAsset(modelAsset);
            return model;
        }

        /// <summary>
        /// Load a pretrained model either from StreamingAssets (in .sentis format)
        /// or from a Resources folder (in .onnx format).
        /// If no config is found null is returned.
        /// </summary>
        protected static Model LoadModel(string model) {
            if (File.Exists(model.StreamingAssetsPathForModel(MODEL_NAME))) {
                return ModelLoader.Load(model.StreamingAssetsPathForModel(MODEL_NAME));
            }
            return LoadFromModelAsset(model.ResourcePathForModel(MODEL_NAME));
        }

        private static C FromConfig<C>(PretrainedConfig config, Model model, BackendType backend) where C : PretrainedModel {
            try {
                return (C)Activator.CreateInstance(typeof(C), model, config, backend);
            } catch (Exception e) {
                Log.Error($"{e.GetType().Name} when trying to create class of type '{typeof(C).Name}'");
                throw e;
            }
        }

        protected static C FromPretrained<C>(string pretrainedModelNameOrPath, BackendType backend) where C : PretrainedModel {
            string configFile = Path.Combine(pretrainedModelNameOrPath, CONFIG_NAME);
            var config = LoadConfig(configFile) ?? throw new FileNotFoundException($"File '{configFile}' not found for: '{typeof(C).Name}'");
            var model = LoadModel(pretrainedModelNameOrPath) ?? throw new FileNotFoundException($"No model file for '{pretrainedModelNameOrPath}' not found for: '{typeof(C).Name}'");
            return FromConfig<C>(config, model, backend);
        }
    }
}