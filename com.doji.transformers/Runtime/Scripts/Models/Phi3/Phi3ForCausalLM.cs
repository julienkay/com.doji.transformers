using System;
using System.Collections.Generic;
using System.Linq;
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

        public override ModelOutput Execute(Dictionary<string, object> modelInputs) {
            Dictionary<string, Tensor> tensorInputs = modelInputs.Where(kvp => kvp.Value is Tensor).ToDictionary(kvp => kvp.Key, kvp => (Tensor)kvp.Value);
            _worker.Execute(tensorInputs);
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

        protected override Dictionary<string, object> PrepareInputsForGeneration(
            TensorInt inputIds,
            Kwargs kwargs)
        {
            Cache pastKeyValues = kwargs.Get<Cache>("past_key_values");
            TensorInt attentionMask = kwargs.Get<TensorInt>("attention_mask");
            TensorFloat inputsEmbeds = kwargs.Get<TensorFloat>("inputs_embeds");
            TensorInt cachePosition = kwargs.Get<TensorInt>("cache_position");
            TensorInt positionIds = kwargs.Get<TensorInt>("position_ids");
            bool useCache = kwargs.Get("use_cache", true);

            // If we have cache: let's slice `inputIds` through `cachePosition`, to keep only the unprocessed tokens
            // Exception 1: when passing input_embeds, inputIds may be missing entries
            // Exception 2: some generation methods do special slicing of inputIds, so we don't need to do it here
            if (pastKeyValues != null) {
                if (inputsEmbeds != null) {
                    inputIds = _ops.Slice(inputIds, .., ^cachePosition.shape[0]..);
                } else if (inputIds.shape[1] != cachePosition.shape[0]) {
                    inputIds = _ops.GatherElements(inputIds, cachePosition, 1);
                }
            }

            if (attentionMask != null && positionIds == null) {
                // create positionIds on the fly for batch generation
                positionIds = _ops.Sub(_ops.CumSum(attentionMask, -1), 1);
                positionIds = _ops.MaskedFill(positionIds, _ops.Neg(attentionMask), 1);
                if (pastKeyValues != null) {
                    positionIds = _ops.Slice(positionIds, .., ^inputIds.shape[1]..);
                }
            }

            Dictionary<string, object> modelInputs;
            //if `inputsEmbeds` are passed, we only want to use them in the 1st generation step
            if (inputsEmbeds != null && cachePosition[0] == 0) {
                modelInputs = new() { { "inputs_embeds", inputsEmbeds }, { "input_ids", null } };
            } else {
                modelInputs = new() { { "input_ids", inputIds }, { "inputs_embeds", null } };
            }

            TensorShape shape;
            if (pastKeyValues is StaticCache && attentionMask.shape.rank == 2) {
                if (modelInputs["inputs_embeds"] != null) {
                    shape = (modelInputs["inputs_embeds"] as Tensor).shape;
                } else {
                    shape = (modelInputs["input_ids"] as Tensor).shape;
                }
                int batchSize = shape[0];
                int sequenceLength = shape[1];

                throw new NotImplementedException("_prepare_4d_causal_attention_mask_with_cachePosition");
                /*attention_mask = _prepare_4d_causal_attention_mask_with_cachePosition(
                    attention_mask,
                    sequence_length = sequence_length,
                    target_length = past_key_values.get_max_length(),
                    dtype = dtype,
                    device = device,
                    min_dtype = min_dtype,
                    cachePosition = cachePosition,
                    batch_size = batch_size,
                );*/
            }
            modelInputs["position_ids"] = positionIds;
            modelInputs["cache_position"] = cachePosition;
            modelInputs["past_key_values"] = pastKeyValues;
            modelInputs["use_cache"] = useCache;
            modelInputs["attention_mask"] = attentionMask;
            return modelInputs;
        }
    }
}