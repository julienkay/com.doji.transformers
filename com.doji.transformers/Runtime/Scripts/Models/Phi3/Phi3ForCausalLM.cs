using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class Phi3ForCausalLM : PreTrainedModel {

        protected override bool AcceptsAttentionMask => true;
        protected override bool HasEncoder => false;

        public Phi3ForCausalLM(Model model, PretrainedConfig config, GenerationConfig generationConfig = null, BackendType backend = BackendType.GPUCompute) : base(model, config, generationConfig, backend) { }

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

        protected override Dictionary<string, Tensor> PrepareInputsForGeneration(
            TensorInt inputIds,
            Kwargs kwargs)
        {
            Cache pastKeyValues = kwargs.Get<Cache>("past_key_values");
            TensorInt attentionMask = kwargs.Get<TensorInt>("attention_mask");
            TensorFloat inputsEmbeds = kwargs.Get<TensorFloat>("inputs_embeds");
            TensorInt cachePosition = kwargs.Get<TensorInt>("cache_position");
            TensorInt positionIds = kwargs.Get<TensorInt>("position_ids");

            // If we have cache: let's slice `inputIds` through `cachePosition`, to keep only the unprocessed tokens
            if (pastKeyValues != null) {
                if (inputsEmbeds != null) {
                    // Exception 1: when passing input_embeds, inputIds may be missing entries
                    inputIds = _ops.Slice(inputIds, .., ^cachePosition.shape[0]..);
                } else if (inputIds.shape[1] != cachePosition.shape[0]) {
                    var indices = _ops.Expand(cachePosition, new TensorShape(inputIds.shape[0], cachePosition.shape[0]));
                    inputIds = _ops.GatherElements(inputIds, indices, 0);
                } else {
                    ;// Exception 2: some generation methods do special slicing of inputIds, so we don't need to do it here
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

            Dictionary<string, Tensor> modelInputs;
            //if `inputsEmbeds` are passed, we only want to use them in the 1st generation step
            if (inputsEmbeds != null && cachePosition[0] == 0) {
                modelInputs = new() { { "inputs_embeds", inputsEmbeds }, { "input_ids", null } };
            } else {
                modelInputs = new() { { "input_ids", inputIds }, { "inputs_embeds", null } };
            }

            TensorShape shape;
            if (pastKeyValues is StaticCache && attentionMask.shape.rank == 2) {
                if (modelInputs["inputs_embeds"] != null) {
                    shape = modelInputs["inputs_embeds"].shape;
                } else {
                    shape = modelInputs["input_ids"].shape;
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

            // prepare past_key_values
            Cache cache = kwargs["past_key_values"] as Cache;
            for (int i = 0; i < 32; i++) {
                string key = $"past_key_values.{i}.key";
                string value = $"past_key_values.{i}.value";
                if (cache.GetSeqLength(i) == 0) {
                    // create empty tensors for initial loop
                    modelInputs[key] = _ops.AllocNoData<TensorFloat>(new TensorShape(inputIds.shape[0], 32, 0, 96));
                    modelInputs[value] = _ops.AllocNoData<TensorFloat>(new TensorShape(inputIds.shape[0], 32, 0, 96));
                    cache.Update(modelInputs[key], modelInputs[value], i);
                } else {
                    modelInputs[key] = cache[i].Key;
                    modelInputs[value] = cache[i].Value;
                }
            }

            modelInputs["position_ids"] = positionIds;
            modelInputs["cache_position"] = cachePosition;
            modelInputs["attention_mask"] = attentionMask;
            return modelInputs;
        }
    }
}