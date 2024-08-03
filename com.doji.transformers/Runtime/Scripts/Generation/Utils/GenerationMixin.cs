using static Doji.AI.Transformers.TensorUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public abstract partial class PretrainedModel {
        protected virtual void PrepareInputsForGeneration() {
            throw new NotImplementedException("A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`.");
        }

        /// <summary>
        /// Generates sequences of token ids for models with a language modeling head.
        /// </summary>
        public void Generate(
            TensorInt inputs,
            GenerationConfig generationConfig,
            PreTrainedTokenizerBase tokenizer = null,
            Dictionary<string, object> kwargs = null)
        {
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
            PrepareModelInputs(ref inputs, out string modelInputName, generationConfig.BosTokenId.Value, ref modelKwargs);
            TensorInt inputsTensor = inputs;
            int batchSize = inputsTensor.shape[0];

            PrepareSpecialTokens(generationConfig, kwargsHasAttentionMask);

            // decoder-only models must use left-padding for batched generation.
            /*if (Config.IsEncoderDecoder) {
                // If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
                // Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
                if (generationConfig.PadTokenTensor != null && batchSize > 1 && inputsTensor.shape.length == 2
                && torch.sum(inputsTensor[:, -1] == generationConfig.PadTokenTensor) > 0)
                {
                    Log.Warning("A decoder-only architecture is being used, but right-padding was detected! For correct " +
                        "generation results, please set `padding_side='left'` when initializing the tokenizer.");
                }
            }*/

            // 4. Define other model kwargs
            // decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
            // generating the first new token or not, and we only want to use the embeddings for the first new token)
            if (!Config.IsEncoderDecoder && modelInputName == "inputs_embeds") {
                modelKwargs["use_cache"] = true;
            } else {
                modelKwargs["use_cache"] = generationConfig.UseCache;
            }

            if (!kwargsHasAttentionMask && requireAttentionMask && acceptsAttentionMask) {
                modelKwargs["attention_mask"] = PrepareAttentionMaskForGeneration(inputsTensor, generationConfig.PadTokenTensor);
            }

            if (Config.IsEncoderDecoder && !modelKwargs.ContainsKey("encoder_outputs")) {
                // if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
                PrepareEncoderDecoderKwargsForGeneration(inputsTensor, modelKwargs, modelInputName, generationConfig);
            }

            // Prepare `input_ids` which will be used for auto-regressive generation
            TensorInt inputIds;
            if (Config.IsEncoderDecoder) {
                inputIds = PrepareDecoderInputIdsForGeneration(
                    batchSize,
                    modelInputName,
                    modelKwargs,
                    generationConfig.DecoderStartTokenTensor
                ) as TensorInt;
            } else {
                inputIds = modelInputName == "input_ids" ? inputsTensor : Pop(modelKwargs, "input_ids") as TensorInt;
            }

            if (generationConfig.TokenHealing) {
                inputIds = HealTokens(inputIds, tokenizer) as TensorInt;
            }
        }

        /// <summary>
        /// This function extracts the model-specific `inputs` for generation.
        /// </summary>
        private void PrepareModelInputs(ref TensorInt inputs, out string inputName, int bosTokenId, ref Dictionary<string, object> modelKwargs) {
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
                inputs = inputsKwarg as TensorInt;
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

                inputs = modelKwargs["inputs_embeds"] as TensorInt;
                inputName = "inputs_embeds";
            }


            // 4. if `inputs` is still None, try to create `input_ids` from BOS token
            MaybeInitializeInputIdsForGeneration(ref inputs, bosTokenId, modelKwargs);
        }

        private TValue Pop<TKey, TValue>(Dictionary<TKey, TValue> d, TKey key, TValue defaultVal = default) {
            if (d.TryGetValue(key, out TValue val)) {
                d.Remove(key);
                return val;
            } else {
                return defaultVal;
            }
        }

        private bool HasInputsEmbedsForwarding() {
            var method = GetType().GetMethod(nameof(PrepareInputsForGeneration), BindingFlags.Instance | BindingFlags.NonPublic);
            var parameters = method.GetParameters();
            return parameters.Any(p => p.Name == "inputs_embeds");
        }

        /// <summary>
        /// Initializes input ids for generation, if necessary.
        /// </summary>
        private void MaybeInitializeInputIdsForGeneration(ref TensorInt inputs, int? bosTokenId, Dictionary<string, object> modelKwargs) {
            if (inputs != null) {
                return;
            }

            modelKwargs.TryGetValue("encoder_outputs", out object encoder_outputs);
            if (Config.IsEncoderDecoder && encoder_outputs != null) {
                // make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
                //var shape = encoder_outputs.last_hidden_state.size()[:-1]
                //torch.ones(shape, dtype = torch.long = self.device) * -100
                throw new NotImplementedException();
            }

            // If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
            // soft-prompting or in multimodal implementations built on top of decoder-only language models.
            int batch_size = 1;
            foreach (var kvp in modelKwargs) {
                if (kvp.Value is Tensor) {
                    batch_size = (kvp.Value as Tensor).shape[0];
                    break;
                }
            }

            if (modelKwargs.ContainsKey("inputs_embeds")) {
                inputs = Ones<TensorInt>(new TensorShape(batch_size, 0));
            }

            if (bosTokenId == null) {
                throw new ArgumentException("`bos_token_id` has to be defined when no `input_ids` are provided.");
            }

            using TensorInt bos = new TensorInt(bosTokenId.Value);
            using TensorInt ones = Ones<TensorInt>(new TensorShape(batch_size, 1));
            TensorInt mul = TensorInt.AllocNoData(ones.shape);

            _ops.Mul(ones, bos, mul);
            inputs = mul;
        }

        /// <summary>
        /// Prepares the special tokens for generation, overwriting the generation config with their processed versions converted to tensor.
        /// Note that <paramref name="generationConfig"/> is modified in this method.
        /// </summary>
        private void PrepareSpecialTokens(GenerationConfig generationConfig, bool? kwargsHasAttentionMask = null) {
            Tensor bos_token_tensor = TensorOrNone(generationConfig.BosTokenId);
            Tensor eos_token_tensor = TensorOrNone(generationConfig.EosTokenId);
            Tensor pad_token_tensor = TensorOrNone(generationConfig.PadTokenId);
            Tensor decoder_start_token_tensor = TensorOrNone(generationConfig.DecoderStartTokenId);

            if (Config.IsEncoderDecoder) {
                decoder_start_token_tensor = decoder_start_token_tensor ?? bos_token_tensor;
            }

            if (eos_token_tensor != null && eos_token_tensor.shape.rank == 0) {
                eos_token_tensor.Reshape(eos_token_tensor.shape.Unsqueeze(0));
            }

            if (pad_token_tensor == null && eos_token_tensor != null) {
                if (kwargsHasAttentionMask == false) {
                    Log.Warning("The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.");
                }
                //pad_token_tensor = eos_token_tensor[0];
                Log.Warning($"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.");
                throw new NotImplementedException();
            }

            if (Config.IsEncoderDecoder && decoder_start_token_tensor == null) {
                throw new ArgumentException("`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation.");
            }

            if (eos_token_tensor != null && generationConfig.EosTokenId.Contains(generationConfig.PadTokenId.Value)) {
                if (kwargsHasAttentionMask == false) {
                    Log.Warning("The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.");
                }
            }

            if (eos_token_tensor != null && generationConfig.EosTokenId.Any(eosToken => eosToken < 0)) {
                Log.Warning($"`eos_token_id` should consist of positive integers, but is {eos_token_tensor}. Your generation will not stop until the maximum length is reached. Depending on other flags, it may even crash.");
            }

            generationConfig.BosTokenTensor = bos_token_tensor;
            generationConfig.EosTokenTensor = eos_token_tensor;
            generationConfig.PadTokenTensor = pad_token_tensor;
            generationConfig.DecoderStartTokenTensor = decoder_start_token_tensor;
        }

        private Tensor TensorOrNone(int? token) {
            if (token == null) {
                return null;
            }

            return new TensorInt(new TensorShape(1), new int[] { token.Value });
        }

        private Tensor TensorOrNone(int[] token) {
            if (token == null) {
                return null;
            }

            return new TensorInt(new TensorShape(token.Length), token);
        }

        private static Tensor PrepareAttentionMaskForGeneration(Tensor inputs, Tensor padTokenId) {
            // No information for attention mask inference -> return default attention mask
            var defaultAttentionMask = Ones<TensorInt>(inputs.shape);
            if (padTokenId == null) {
                return defaultAttentionMask;
            }
            bool isInputIds = inputs.shape.length == 2 && (inputs.dataType == DataType.Int);

            if (!isInputIds) {
                return defaultAttentionMask;
            }

            throw new NotImplementedException("Can't infer missing attention mask. Please provide an `attention_mask`");
            /*
            // Otherwise we have may have information -> try to infer the attention mask
            bool isPadTokenInInputs = padTokenId != null && torch.isin(inputs, torch.tensor(new[] { padTokenId.Value }, dtype: torch.@long)).any().ToBoolean();
            bool isPadTokenNotEqualToEosTokenId = eosTokenId == null || !torch.isin(torch.tensor(new[] { eosTokenId.Value }, dtype: torch.@long), torch.tensor(new[] { padTokenId.Value }, dtype: torch.@long)).any().ToBoolean();
            bool canInferAttentionMask = isPadTokenInInputs && isPadTokenNotEqualToEosTokenId;
            var attentionMaskFromPadding = inputs.ne(padTokenId.Value).to_type(torch.@long);
            var attentionMask = torch.where(torch.tensor(canInferAttentionMask), attentionMaskFromPadding, defaultAttentionMask);
            return attentionMask;*/
        }

        private static Tensor PrepareEncoderDecoderKwargsForGeneration(Tensor inputs, Dictionary<string, object> modelKwargs, string modelInputName, GenerationConfig generationConfig) {
            throw new NotImplementedException();
        }

        private static Tensor PrepareDecoderInputIdsForGeneration(int batchSize, string modelInputName, Dictionary<string, object> modelKwargs, Tensor decoderStartTokenTensor) {
            throw new NotImplementedException();
        }

        private static Tensor HealTokens(Tensor inputIds, PreTrainedTokenizerBase tokenizer) {
            throw new NotImplementedException();
        }
    }
}