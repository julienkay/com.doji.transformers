using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class Phi3ForCausalLM : PreTrainedModel {

        protected override bool AcceptsAttentionMask => true;
        protected override bool HasEncoder => false;

        private Dictionary<string, Tensor> _inputs = new Dictionary<string, Tensor>();

        public Phi3ForCausalLM(Model model, PretrainedConfig config, BackendType backend = BackendType.GPUCompute) : base(model, config, backend) { }

        /// <summary>
        /// Instantiate a Phi3 model from a JSON configuration file.
        /// </summary>
        public static Phi3ForCausalLM FromPretrained(string model, BackendType backend = BackendType.GPUCompute) {
            return FromPretrained<Phi3ForCausalLM>(model, backend);
        }

        public override ModelOutput Execute(Dictionary<string, Tensor> modelInputs) {
            _worker.Execute(modelInputs);
            var logits = _worker.PeekOutput("logits") as TensorFloat;
            return new CausalLMOutputWithPast(logits);
        }

        public TensorFloat Execute(TensorInt inputIds, TensorInt attentionMask, TensorInt positionIds) {
            _inputs["input_ids"] = inputIds;
            _inputs["attention_mask"] = attentionMask;
            _inputs["position_ids"] = positionIds;
            _worker.Execute(_inputs);
            return _worker.PeekOutput("logits") as TensorFloat;
        }

        public TensorFloat ExecuteLayerByLayer(TensorInt inputIds, TensorInt attentionMask, TensorInt positionIds) {
            _inputs["input_ids"] = inputIds;
            _inputs["attention_mask"] = attentionMask;
            _inputs["position_ids"] = positionIds;

            var schedule = _worker.ExecuteLayerByLayer(_inputs);
            int i = 0;
            bool loop = true;
            while (loop) {
                if (i == 32) {
                    ;
                }
                try {
                    loop = schedule.MoveNext();
                } catch (Exception e) {
                    UnityEngine.Debug.LogError(e);
                    UnityEngine.Debug.LogError(i);

                    break;
                }
                i++;
            }

            return _worker.PeekOutput("logits") as TensorFloat;
        }

        protected override Dictionary<string, Tensor> PrepareInputsForGeneration(
            TensorInt inputIds,
            Kwargs kwargs)
        {
            Tensor pastKeyValues = kwargs.Get<Tensor>("past_key_values");
            TensorInt attentionMask = kwargs.Get<TensorInt>("attention_mask");
            TensorFloat inputsEmbeds = kwargs.Get<TensorFloat>("inputs_embeds");
            TensorInt cachePosition = kwargs.Get<TensorInt>("cache_position");
            TensorInt positionIds = kwargs.Get<TensorInt>("position_ids");
            return null;
        }
    }
}