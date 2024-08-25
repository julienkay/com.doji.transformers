using System.IO;
using System;
using Unity.Sentis;
using UnityEngine;
using System.Collections.Generic;

namespace Doji.AI.Transformers {
    public abstract partial class PreTrainedModel : Configurable<PretrainedConfig>, IDisposable {

        public const string MODEL_NAME = "model";
        public virtual string MainInputName { get; } = "input_ids";

        public GenerationConfig GenerationConfig { get; }

        protected bool IsStateful { get; } = false;

        // Flash Attention 2 support
        protected bool SupportsFlashAttn2 { get; } = false;

        // SDPA support
        protected bool SupportsSdpa { get; } = false;

        // Has support for a `Cache` instance as `past_key_values`? Does it support a `StaticCache`?
        protected bool SupportsCacheClass { get; } = false;
        protected bool SupportsStaticCache { get; } = false;

        // Has support for a `QuantoQuantizedCache` instance as `past_key_values`
        protected bool SupportsQuantizedCache { get; } = false;


        /* In original code these are retrieved by inspecting attributes/arguments
         t.b.d. if there's a good way to implement this generically without resorting to Reflection */
        protected abstract bool AcceptsAttentionMask { get; }
        protected abstract bool HasEncoder { get; }
        protected virtual object Encoder { get; } = null;

        /// <summary>
        /// Which <see cref="BackendType"/> to run the model with.
        /// </summary>
        private BackendType Backend { get; set; } = BackendType.GPUCompute;

        /// <summary>
        /// The runtime model.
        /// </summary>
        private Model _model;
        protected Worker _worker;
        protected Ops _ops;

        public PreTrainedModel(Model model, PretrainedConfig config, GenerationConfig generationConfig = null, BackendType backend = BackendType.GPUCompute) : base(config) {
            Backend = backend;
            InitializeNetwork(model);
            GenerationConfig = generationConfig;
        }

        protected virtual void InitializeNetwork(Model model) {
            if (model == null) {
                throw new ArgumentException("Model was null", nameof(model));
            }

            _model = model;
            _worker = new Worker(_model, Backend);
            _ops = new Ops(Backend);
        }

        public virtual void Dispose() {
            _worker?.Dispose();
            _ops?.Dispose();
        }

        public abstract ModelOutput Execute(Dictionary<string, Tensor> modelInputs);

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

        private static C FromConfig<C>(PretrainedConfig config, Model model, GenerationConfig generationConfig, BackendType backend) where C : PreTrainedModel {
            try {
                return (C)Activator.CreateInstance(typeof(C), model, config, generationConfig, backend);
            } catch (Exception e) {
                Log.Error($"{e.GetType().Name} when trying to create class of type '{typeof(C).Name}'");
                throw e;
            }
        }

        protected static C FromPretrained<C>(string pretrainedModelNameOrPath, BackendType backend) where C : PreTrainedModel {
            string configFile = Path.Combine(pretrainedModelNameOrPath, CONFIG_NAME);
            string generationConfigFile = Path.Combine(pretrainedModelNameOrPath, "generation_config.json");
            var config = LoadConfig(configFile) ?? throw new FileNotFoundException($"File '{configFile}' not found for: '{typeof(C).Name}'");
            var model = LoadModel(pretrainedModelNameOrPath) ?? throw new FileNotFoundException($"Model file for '{pretrainedModelNameOrPath}' not found for: '{typeof(C).Name}'");
            var generationConfig = Load<GenerationConfig>(generationConfigFile);
            return FromConfig<C>(config, model, generationConfig, backend);
        }
    }
}