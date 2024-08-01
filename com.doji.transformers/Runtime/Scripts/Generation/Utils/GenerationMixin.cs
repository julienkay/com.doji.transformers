using Google.Protobuf.WellKnownTypes;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Unity.Sentis;
using Unity.Sentis.Layers;
using UnityEngine;
using UnityEngine.SocialPlatforms;

namespace Doji.AI.Transformers {
    public abstract partial class PretrainedModel {

        /// <summary>
        /// Generates sequences of token ids for models with a language modeling head.
        /// </summary>
        public void Generate(
            TensorInt inputs,
            GenerationConfig generationConfig,
            Dictionary<string, object> kwargs = null) {
            //ValidateModelClass();
            var modelKwargs = kwargs ?? new Dictionary<string, object>();
            //ValidateModelKwargs();
            //ValidateAssistant();

            var logitsProcessor = new List<LogitsProcessor>();
            var stoppingCriteria = new List<LogitsProcessor>();

            bool acceptsAttentionMask = AcceptsAttentionMask;
            bool requireAttentionMask = !modelKwargs.ContainsKey("encoder_outputs");
            bool kwargsHasAttentionMask = modelKwargs.ContainsKey("attention_mask");

            // define model inputs

        }

        /// <summary>
        /// This function extracts the model-specific `inputs` for generation.
        /// </summary>
        private void PrepareModelInputs(ref Tensor inputs, out string inputName, int bosTokenId, ref Dictionary<string, object> modelKwargs) {
            // retrieve all kwargs that are non-None or non-model input related.
            // some encoder-decoder models have different names for model and encoder
            if (Config.IsEncoderDecoder && HasEncoder /* && Encoder.MainInputName != MainInputName*/) {
                inputName = "";// Encoder.MainInputName
            } else {
                inputName = MainInputName;
            }

            string inputN = inputName;
            modelKwargs = modelKwargs.Where(kvp => kvp.Value != null || kvp.Key != inputN).ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
            var inputsKwarg = Pop(modelKwargs, inputName, null);
            if (inputsKwarg != null && inputs != null) {
                throw new ArgumentException($"`inputs`: {inputs}` were passed alongside {inputName} which is not allowed. " +
                $"Make sure to either pass {inputs} or {inputName}=...");
            } else if (inputsKwarg != null) {
                inputs = inputsKwarg as Tensor;
            }

            // In the presence of `inputs_embeds` for text models:
            // - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
            // doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
            // input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
            // - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
            // pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
            if (inputName == "input_ids" && modelKwargs.ContainsKey("inputs_embeds")) {
                if (!Config.IsEncoderDecoder) {
                    bool hasInputsEmbedsForwarding = HasInputsEmbedsForwarding();

                    if (!hasInputsEmbedsForwarding) {
                        throw new ArgumentException(
                            $"You passed `inputs_embeds` to `.generate()`, but the model class {this.GetType().Name} " +
                            "doesn't have its forwarding implemented. See the GPT2 implementation for an example " +
                            "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                        );
                    }

                    MaybeInitializeInputIdsForGeneration(ref inputs, bosTokenId, modelKwargs);
                    modelKwargs["input_ids"] = inputs;

                } else {
                    if (inputs != null) {
                        throw new ArgumentException("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.");
                    }
                }

                inputs = modelKwargs["inputs_embeds"] as Tensor;
                inputName = "inputs_embeds";
            }


            // 4. if `inputs` is still None, try to create `input_ids` from BOS token
            MaybeInitializeInputIdsForGeneration(ref inputs, bosTokenId, modelKwargs);
        }

        private TValue Pop<TKey, TValue>(Dictionary<TKey, TValue> d, TKey key, TValue defaultVal = default) {
            TValue val = defaultVal;
            if (d.TryGetValue(key, out val)) {
                d.Remove(key);
            }
            return val;
        }

        private bool HasInputsEmbedsForwarding() {
            var method = this.GetType().GetMethod("PrepareInputsForGeneration", BindingFlags.Instance | BindingFlags.NonPublic);
            var parameters = method.GetParameters();
            return parameters.Any(p => p.Name == "inputs_embeds");
        }


        /// <summary>
        /// Initializes input ids for generation, if necessary.
        /// </summary>
        private void MaybeInitializeInputIdsForGeneration(ref Tensor inputs, int? bosTokenId, Dictionary<string, object> modelKwargs) {
            if (inputs != null) {
                return;
            }

            modelKwargs.TryGetValue<string, Tensor>("encoder_outputs", out Tensor encoder_outputs);
            if (Config.IsEncoderDecoder && encoder_outputs != null) {
                // make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
                var shape = encoder_outputs.last_hidden_state.size()[:-1]
                torch.ones(shape, dtype = torch.long, device = self.device) * -100
            }

        // If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        // soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break

        if "inputs_embeds" in model_kwargs:
            return torch.ones((batch_size, 0), dtype = torch.long, device = self.device)

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        return torch.ones((batch_size, 1), dtype = torch.long, device = self.device) * bos_token_id
        }

    }
}