using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public abstract partial class PretrainedModel {

        /// <summary>
        /// Generates sequences of token ids for models with a language modeling head.
        /// </summary>
        public void Generate(
            TensorInt inputs,
            GenerationConfig generationConfig,
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
            var method = this.GetType().GetMethod("PrepareInputsForGeneration", BindingFlags.Instance | BindingFlags.NonPublic);
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
                inputs = TensorUtils.Ones(new TensorShape(batch_size, 0), DataType.Int) as TensorInt;
            }

            if (bosTokenId == null) {
                throw new ArgumentException("`bos_token_id` has to be defined when no `input_ids` are provided.");
            }

            using TensorInt bos = new TensorInt(bosTokenId.Value);
            using TensorInt ones = TensorUtils.Ones(new TensorShape(batch_size, 1), DataType.Int) as TensorInt;
            TensorInt mul = TensorInt.AllocNoData(ones.shape);

            _ops.Mul(ones, bos, mul);
            inputs = mul;
        }

        /// <summary>
        /// Prepares the special tokens for generation, overwriting the generation config with their processed versions converted to tensor.
        /// </summary>
        public void PrepareSpecialTokens(GenerationConfig generationConfig, bool? kwargsHasAttentionMask = null) {
            Tensor bos_token_tensor = _tensor_or_none(generationConfig.BosTokenId);
            Tensor eos_token_tensor = _tensor_or_none(generationConfig.EosTokenId);
            Tensor pad_token_tensor = _tensor_or_none(generationConfig.PadTokenId);
            Tensor decoder_start_token_tensor = _tensor_or_none(generationConfig.DecoderStartTokenId);

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

        private Tensor _tensor_or_none(int? token) {
            if (token == null) {
                return null;
            }

            return new TensorInt(new TensorShape(1), new int[] { token.Value });
        }

        private Tensor _tensor_or_none(int[] token) {
            if (token == null) {
                return null;
            }

            return new TensorInt(new TensorShape(token.Length), token);
        }
    }
}