using static Doji.AI.Transformers.TensorUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public abstract partial class PreTrainedModel {

        private static readonly Dictionary<string, Type> NEED_SETUP_CACHE_CLASSES_MAPPING = new() {
            { "static", typeof(StaticCache) },
            { "sliding_window", typeof(SlidingWindowCache) },
            { "hybrid", typeof(HybridCache) },
            { "mamba", typeof( MambaCache) }
        };
        private static readonly Dictionary<string, Type> QUANT_BACKEND_CLASSES_MAPPING = new() {
            {"quanto", typeof(QuantoQuantizedCache) },
             { "HQQ", typeof(HQQQuantizedCache) }
        };

        private Cache _cache;

        protected virtual void PrepareInputsForGeneration() {
            throw new NotImplementedException($"A model class needs to define a {nameof(PrepareInputsForGeneration)} method in order to use `.Generate()`.");
        }

        /// <summary>
        /// Generates sequences of token ids for models with a language modeling head.
        /// </summary>
        public void Generate(
            TensorInt inputs,
            GenerationConfig generationConfig,
            PreTrainedModel assistantModel = null,
            object streamer = null,
            PreTrainedTokenizerBase tokenizer = null,
            Kwargs kwargs = null)
        {
            //ValidateModelClass();
            var modelKwargs = kwargs ?? new Kwargs();
            //ValidateModelKwargs();
            ValidateAssistant(assistantModel);

            var logitsProcessor = new LogitsProcessorList();
            var stoppingCriteria = new StoppingCriteriaList();

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
                inputIds = modelInputName == "input_ids" ? inputsTensor : modelKwargs.Pop("input_ids") as TensorInt;
            }

            if (generationConfig.TokenHealing == true) {
                inputIds = HealTokens(inputIds, tokenizer) as TensorInt;
            }

            if (streamer != null) {
                throw new NotImplementedException("streamer.Put()");
                //streamer.Put(inputIds);
            }

            // Prepare `max_length` depending on other stopping criteria.
            int inputIdsLength = inputIds.shape[-1];
            bool hasDefaultMaxLength = kwargs.Get("max_length") == null && generationConfig.MaxLength != null;
            bool hasDefaultMinLength = kwargs.Get("min_length") == null && generationConfig.MinLength != null;
            PrepareGeneratedLength(
                generationConfig,
                hasDefaultMaxLength,
                hasDefaultMinLength,
                modelInputName,
                inputIdsLength,
                inputsTensor
            );

            bool useDynamicCacheByDefault = false;
            string cacheName = "past_key_values";

            if (assistantModel != null && generationConfig.CacheConfig != null && SupportsDefaultDynamicCache()) {
                Log.Warning($"An assistant model is provided, using a dynamic cache instead of a cache of type='{generationConfig.CacheImplementation}'.");
                generationConfig.CacheImplementation = null;
            }

            if (generationConfig.CacheImplementation != null && modelKwargs.Get(cacheName) != null) {
                throw new ArgumentException("\"Passing both `cache_implementation` (used to initialize certain caches) and" +
                    "`{cache_name}` (a Cache object) is unsupported. Please use only one of the two.");
            } else if (generationConfig.CacheImplementation != null) {
                if (NEED_SETUP_CACHE_CLASSES_MAPPING.ContainsKey(generationConfig.CacheImplementation)) {
                    if (generationConfig.CacheImplementation == "static" && !SupportsStaticCache) {
                        throw new ArgumentException("This model does not support `cache_implementation='static'`. Please check the following " +
                            "issue: https://github.com/huggingface/transformers/issues/28981");
                    }
                    int maxBatchSize = (generationConfig.NumBeams ?? 1) * (generationConfig.NumReturnSequences ?? 1) * batchSize;
                    modelKwargs[cacheName] = GetCache(
                       generationConfig.CacheImplementation,
                       maxBatchSize,
                       generationConfig.MaxLength.Value,
                       modelKwargs);
                } else if (generationConfig.CacheImplementation == "quantized") {
                    if (!SupportsQuantizedCache) {
                        throw new ArgumentException("This model does not support the quantized cache.");
                    }
                    var cacheConfig = generationConfig.CacheConfig ?? new QuantizedCacheConfig();
                    var backend = ((QuantizedCacheConfig)cacheConfig).Backend;
                    var cacheClass = QUANT_BACKEND_CLASSES_MAPPING[backend];
                    if (backend == "quanto") {
                        throw new NotImplementedException("Selected cache backend `quanto` not supported.");
                    } else if (backend == "HQQ") {
                        throw new NotImplementedException("Selected cache backend `HQQ` not supported.");
                    }
                    modelKwargs[cacheName] = Activator.CreateInstance(cacheClass, cacheConfig);
                } else if (generationConfig.CacheImplementation == "offloaded") {
                    modelKwargs[cacheName] = new OffloadedCache();
                }
            } else if (generationConfig.CacheImplementation == null && SupportsDefaultDynamicCache()) {
                // Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
                // keeps copying the cache thus using much more memory
                var past = (string)modelKwargs.Get(cacheName, null);
                bool requires_cross_attention_cache = Config.IsEncoderDecoder || modelKwargs.ContainsKey("encoder_outputs");
                if (past == null) {
                    modelKwargs[cacheName] =
                        !requires_cross_attention_cache ?
                        new DynamicCache() :
                        new EncoderDecoderCache(new DynamicCache(), new DynamicCache());
                    useDynamicCacheByDefault = true;
                } else {
                    modelKwargs[cacheName] =
                          !requires_cross_attention_cache ?
                            DynamicCache.FromLegacyCache(past) :
                            EncoderDecoderCache.FromLegacyCache(past);
                    useDynamicCacheByDefault = true;
                }
            }

            ValidateGeneratedLength(generationConfig, inputIdsLength, hasDefaultMaxLength);

            // determine generation mode
            var generationMode = generationConfig.GetGenerationMode(assistantModel);

            if (streamer != null && (generationConfig.NumBeams.Value > 1)) {
                throw new ArgumentException("`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1.");
            }

            // prepare distribution pre_processing samplers
            var preparedLogitsProcessor = GetLogitsProcessor(
                generationConfig,
                inputIdsLength,
                encoderInputIds: inputsTensor,
                prefixAllowedTokensFn: null,
                logitsProcessor,
                modelKwargs,
                negativePromptIds: null,
                negativePromptAttentionMask: null
            );

            // prepare stopping criteria
            var preparedStoppingCriteria = GetStoppingCriteria(generationConfig, stoppingCriteria, tokenizer);

            // 10. go into different generation modes
            if (generationMode == GenerationMode.ASSISTED_GENERATION) {
                if (generationConfig.NumReturnSequences > 1) {
                    throw new ArgumentException("num_return_sequences has to be 1 when doing assisted generate, but is " + generationConfig.NumReturnSequences);
                }
                if (batchSize > 1) {
                    throw new ArgumentException("assisted generate is only supported for batch_size = 1");
                }
                if (modelKwargs.Get<bool>("use_cache") != true) {
                    throw new ArgumentException("assisted generate requires `use_cache=True`");
                }
                if (generationConfig.CacheImplementation == "static") {
                    throw new ArgumentException("assisted generate is not supported with `static_cache`");
                }
                if (IsStateful) {
                    throw new ArgumentException("assisted generation is not supported with stateful models, such as " + GetType().Name);
                }

                // 11. Get the candidate generator, given the parameterization
                var candidate_generator = GetCandidateGenerator(
                    generationConfig,
                    inputIds,
                    inputsTensor,
                    assistantModel,
                    logitsProcessor,
                    modelKwargs
                );

                // 12. prepare logits warper (if `do_sample` is `True`)
                var preparedLogitsWarper = generationConfig.DoSample
                    ? GetLogitsWarper(generationConfig)
                    : null;

                // 13. run assisted generate
                var result = AssistedDecoding(
                    inputIds,
                    candidate_generator,
                    preparedLogitsProcessor,
                    preparedLogitsWarper,
                    preparedStoppingCriteria,
                    generationConfig,
                    streamer,
                    modelKwargs
                );
            } else if (generationMode == GenerationMode.DOLA_GENERATION) {
                if (IsStateful) {
                    throw new ArgumentException("dola decoding is not supported with stateful models, such as " + GetType().Name);
                }
                var preparedLogitsWarper = generationConfig.DoSample
                    ? GetLogitsWarper(generationConfig)
                    : null;

                var result = DolaDecoding(
                    inputIds,
                    generationConfig.DolaLayers,
                    preparedLogitsProcessor,
                    preparedLogitsWarper,
                    preparedStoppingCriteria,
                    generationConfig,
                    streamer,
                    modelKwargs
                );
            } else if (generationMode == GenerationMode.CONTRASTIVE_SEARCH) {
                if (!(bool)modelKwargs.GetType().GetProperty("use_cache").GetValue(modelKwargs, null)) {
                    throw new ArgumentException("Contrastive search requires `use_cache=True`");
                }
                if (IsStateful) {
                    throw new ArgumentException("contrastive search is not supported with stateful models, such as " + GetType().Name);
                }

                var result = ContrastiveSearch(
                    inputIds,
                    preparedLogitsProcessor,
                    preparedStoppingCriteria,
                    generationConfig,
                    streamer,
                    modelKwargs
                );
            } else if (generationMode == GenerationMode.SAMPLE || generationMode == GenerationMode.GREEDY_SEARCH) {
                // 11. prepare logits warper
                var preparedLogitsWarper = generationConfig.DoSample
                    ? GetLogitsWarper(generationConfig)
                    : null;

                // 12. expand inputIds with `num_return_sequences` additional sequences per batch
                ExpandInputsForGeneration(
                    ref inputIds,
                    generationConfig.NumReturnSequences,
                    Config.IsEncoderDecoder,
                    modelKwargs
                );

                // 13. run sample (it degenerates to greedy search when `generationConfig.do_sample=False`)
                var result = Sample(
                    inputIds,
                    preparedLogitsProcessor,
                    preparedLogitsWarper,
                    preparedStoppingCriteria,
                    generationConfig,
                    streamer,
                    modelKwargs
                );
            } else if (generationMode == GenerationMode.BEAM_SAMPLE || generationMode == GenerationMode.BEAM_SEARCH) {
                // 11. prepare logits warper
                var preparedLogitsWarper = generationConfig.DoSample
                    ? GetLogitsWarper(generationConfig)
                    : null;

                // 12. prepare beam search scorer
                var beamScorer = new BeamSearchScorer(
                    batchSize,
                    generationConfig.NumBeams.Value,
                    generationConfig.LengthPenalty,
                    generationConfig.EarlyStopping,
                    generationConfig.NumReturnSequences,
                    generationConfig.MaxLength.Value
                );

                // 13. interleave inputIds with `num_beams` additional sequences per batch
                ExpandInputsForGeneration(
                    ref inputIds,
                    generationConfig.NumBeams,
                    Config.IsEncoderDecoder,
                    modelKwargs
                );

                // 14. run beam sample
                var result = BeamSearch(
                    inputIds,
                    beamScorer,
                    preparedLogitsProcessor,
                    preparedLogitsWarper,
                    preparedStoppingCriteria,
                    generationConfig,
                    modelKwargs
                );
            } else if (generationMode == GenerationMode.GROUP_BEAM_SEARCH) {
                // 11. prepare beam search scorer
                var beamScorer = new BeamSearchScorer(
                    batchSize,
                    generationConfig.NumBeams.Value,
                    generationConfig.LengthPenalty,
                    generationConfig.EarlyStopping,
                    generationConfig.NumReturnSequences,
                    generationConfig.NumBeamGroups,
                    generationConfig.MaxLength
                );

                // 12. interleave inputIds with `num_beams` additional sequences per batch
                ExpandInputsForGeneration(
                    ref inputIds,
                    generationConfig.NumBeams,
                    Config.IsEncoderDecoder,
                    modelKwargs
                );

                // 13. run beam search
                var result = GroupBeamSearch(
                    inputIds,
                    beamScorer,
                    preparedLogitsProcessor,
                    preparedStoppingCriteria,
                    generationConfig,
                    modelKwargs
                );
            } else if (generationMode == GenerationMode.CONSTRAINED_BEAM_SEARCH) {
                var finalConstraints = new List<string>();
                if (generationConfig.Constraints != null) {
                    finalConstraints.AddRange(generationConfig.Constraints);
                }

                if (generationConfig.ForceWordsIds != null) {
                    Action typeerror = () =>
                    {
                        throw new ArgumentException(
                            "`force_words_ids` has to either be a `List<List<List<int]]]` or `List<List<int]]` " +
                            "of positive integers, but is " + generationConfig.ForceWordsIds
                        );
                    };
                    throw new NotImplementedException("Constrained Beam Search");
                    /*if (!(generationConfig.ForceWordsIds is List<List<List<int>>> forceWordsIdsList) || forceWordsIdsList.Count == 0) {
                        typeerror();
                    }

                    foreach (var word_ids in generationConfig.ForceWordsIds) {
                        if (word_ids[0] is List<int>) {
                            if (!(word_ids is List<List<int>> listOfLists) || listOfLists.Count == 0) {
                                typeerror();
                            }
                            if (listOfLists.Any(token_ids => !(token_ids is List<int>))) {
                                typeerror();
                            }
                            if (listOfLists.Any(token_ids => token_ids.Any(token_id => token_id < 0))) {
                                typeerror();
                            }

                            var constraint = new DisjunctiveConstraint(listOfLists);
                            finalConstraints.Add(constraint);
                        } else {
                            if (!(word_ids is List<int> list) || list.Count == 0) {
                                typeerror();
                            }
                            if (list.Any(token_id => token_id < 0)) {
                                typeerror();
                            }

                            var constraint = new PhrasalConstraint(list);
                            finalConstraints.Add(constraint);
                        }
                    }

                    // 11. prepare beam search scorer
                    var constrained_beam_scorer = new ConstrainedBeamSearchScorer(
                        finalConstraints,
                        batchSize,
                        generationConfig.NumBeams.Value,
                        generationConfig.LengthPenalty,
                        generationConfig.EarlyStopping,
                        generationConfig.NumReturnSequences,
                        generationConfig.MaxLength
                    );

                    // 12. interleave inputIds with `num_beams` additional sequences per batch
                    ExpandInputsForGeneration(
                        ref inputIds,
                        generationConfig.NumBeams,
                        Config.IsEncoderDecoder,
                        modelKwargs
                    );

                    // 13. run beam search
                    var result = ConstrainedBeamSearch(
                        inputIds,
                        constrained_beam_scorer,
                        preparedLogitsProcessor,
                        preparedStoppingCriteria,
                        generationConfig,
                        modelKwargs
                    );*/
                }
            }

            // Convert to legacy cache if needed
            /*if (useDynamicCacheByDefault && generationConfig.ReturnLegacyCache) {
                if isinstance(result, ModelOutput) and hasattr(result, "past_key_values"):
                if isinstance(result.past_key_values, (DynamicCache, EncoderDecoderCache)):
                    result.past_key_values = result.past_key_values.to_legacy_cache()
            }
            return result*/
        }

        private object GetCandidateGenerator(GenerationConfig generationConfig, TensorInt inputIds, TensorInt inputsTensor, object assistantModel, object logitsProcessor, Kwargs modelKwargs) {
            throw new NotImplementedException();
        }

        /// <summary>
        /// This class returns a <see cref="LogitsProcessorList"/>list object that contains all relevant
        /// [`LogitsWarper`] instances used for multinomial sampling.
        /// </summary>
        private LogitsProcessorList GetLogitsWarper(GenerationConfig generationConfig) {
            // instantiate warpers list
            LogitsProcessorList warpers = new LogitsProcessorList();

            // In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
            // better score (i.e. keep len(list(generationConfig.EosTokenTensor)) + 1)
            int minTokensToKeep;
            if (generationConfig.NumBeams > 1) {
                minTokensToKeep = generationConfig.EosTokenTensor.shape[0] + 1;
            } else {
                minTokensToKeep = 1;
            }

            // the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
            // all samplers can be found in `generation_utils_samplers.py`
            if (generationConfig.Temperature != null && generationConfig.Temperature != 1.0f) {
                warpers.Add(new TemperatureLogitsWarper(generationConfig.Temperature.Value));
            }
            if (generationConfig.TopK != null && generationConfig.TopK != 0) {
                warpers.Add(new TopKLogitsWarper(generationConfig.TopK.Value, minTokensToKeep));
            }
            if (generationConfig.TopP != null && generationConfig.TopP < 1.0f) {
                warpers.Add(new TopPLogitsWarper(generationConfig.TopP.Value, minTokensToKeep));
            }
            if (generationConfig.MinP != null) {
                // Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
                warpers.Add(new MinPLogitsWarper(generationConfig.MinP.Value, minTokensToKeep));
            }
            if (generationConfig.TypicalP != null && generationConfig.TypicalP < 1.0) {
                warpers.Add(new TypicalLogitsWarper(generationConfig.TypicalP.Value, minTokensToKeep));
            }
            if (generationConfig.EpsilonCutoff != null && generationConfig.EpsilonCutoff > 0.0 && generationConfig.EpsilonCutoff < 1.0) {
                warpers.Add(new EpsilonLogitsWarper(generationConfig.EpsilonCutoff.Value, minTokensToKeep));
            }
            if (generationConfig.EtaCutoff != null && generationConfig.EtaCutoff > 0.0 && generationConfig.EtaCutoff < 1.0) {
                warpers.Add(new EtaLogitsWarper(generationConfig.EtaCutoff.Value, minTokensToKeep));
            }
            // `LogitNormalization` should always be the last logit processor, when present
            if (generationConfig.RenormalizeLogits == true) {
                warpers.Add(new LogitNormalization());
            }
            return warpers;
        }

        private object AssistedDecoding(TensorInt inputIds, object candidateGenerator, object logitsProcessor, object logitsWarper, object stoppingCriteria, GenerationConfig generationConfig, object streamer, Kwargs modelKwargs) {
            throw new NotImplementedException();
        }

        private object DolaDecoding(TensorInt inputIds, object dola_layers, object logitsProcessor, object logitsWarper, object stoppingCriteria, GenerationConfig generationConfig, object streamer, Kwargs modelKwargs) {
            throw new NotImplementedException();
        }

        private object ContrastiveSearch(TensorInt inputIds, object logitsProcessor, object stoppingCriteria, GenerationConfig generationConfig, object streamer, Kwargs modelKwargs) {
            throw new NotImplementedException();
        }

        private object Sample(TensorInt inputIds, object logitsProcessor, object logitsWarper, object stoppingCriteria, GenerationConfig generationConfig, object streamer, Kwargs modelKwargs) {
            throw new NotImplementedException();
        }

        private object BeamSearch(TensorInt inputIds, object beam_scorer, object logitsProcessor, object logitsWarper, object stoppingCriteria, GenerationConfig generationConfig, Kwargs modelKwargs) {
            throw new NotImplementedException();
        }

        private object GroupBeamSearch(TensorInt inputIds, object beam_scorer, object logitsProcessor, object stoppingCriteria, GenerationConfig generationConfig, Kwargs modelKwargs) {
            throw new NotImplementedException();
        }

        private object ConstrainedBeamSearch(TensorInt inputIds, object constrained_beam_scorer, object logitsProcessor, object stoppingCriteria, GenerationConfig generationConfig, Kwargs modelKwargs) {
            throw new NotImplementedException();
        }

        private void ExpandInputsForGeneration(ref TensorInt inputIds, int? expandSize = 1, bool isEncoderDecoder = false, Kwargs modelKwargs = null) {
            throw new NotImplementedException();
        }

        public LogitsProcessorList GetLogitsProcessor(
            GenerationConfig generationConfig,
            int inputIdsSeqLength,
            object encoderInputIds,
            Func<int, List<int>> prefixAllowedTokensFn,
            LogitsProcessorList logitsProcessor,
            Dictionary<string, object> modelKwargs = null,
            object negativePromptIds = null,
            object negativePromptAttentionMask = null)
        {
            LogitsProcessorList processors = new LogitsProcessorList();

            if (generationConfig.GuidanceScale.HasValue && generationConfig.GuidanceScale.Value != 1) {
                processors.Add(new UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    generationConfig.GuidanceScale.Value,
                    this,
                    negativePromptIds,
                    negativePromptAttentionMask,
                    (bool)modelKwargs["use_cache"]
                ));
            }

            if (generationConfig.SequenceBias != null) {
                processors.Add(new SequenceBiasLogitsProcessor(generationConfig.SequenceBias));
            }

            if (generationConfig.DiversityPenalty.HasValue && generationConfig.DiversityPenalty > 0.0f) {
                processors.Add(new HammingDiversityLogitsProcessor(
                    generationConfig.DiversityPenalty.Value,
                    generationConfig.NumBeams.Value,
                    generationConfig.NumBeamGroups.Value
                ));
            }

            if (generationConfig.EncoderRepetitionPenalty.HasValue && generationConfig.EncoderRepetitionPenalty.Value != 1.0f) {
                processors.Add(new EncoderRepetitionPenaltyLogitsProcessor(
                    generationConfig.EncoderRepetitionPenalty.Value,
                    encoderInputIds
                ));
            }

            if (generationConfig.RepetitionPenalty.HasValue && generationConfig.RepetitionPenalty.Value != 1.0f) {
                processors.Add(new RepetitionPenaltyLogitsProcessor(generationConfig.RepetitionPenalty.Value));
            }

            if (generationConfig.NoRepeatNgramSize.HasValue && generationConfig.NoRepeatNgramSize > 0) {
                processors.Add(new NoRepeatNGramLogitsProcessor(generationConfig.NoRepeatNgramSize.Value));
            }

            if (generationConfig.EncoderNoRepeatNgramSize.HasValue && generationConfig.EncoderNoRepeatNgramSize > 0) {
                processors.Add(new EncoderNoRepeatNGramLogitsProcessor(
                    generationConfig.EncoderNoRepeatNgramSize.Value,
                    encoderInputIds
                ));
            }

            if (generationConfig.BadWordsIds != null) {
                processors.Add(new NoBadWordsLogitsProcessor(
                    generationConfig.BadWordsIds,
                    generationConfig.EosTokenTensor
                ));
            }

            if (generationConfig.MinLength.HasValue && generationConfig.EosTokenTensor != null && generationConfig.MinLength > 0) {
                processors.Add(new MinLengthLogitsProcessor(
                    generationConfig.MinLength.Value,
                    generationConfig.EosTokenTensor
                ));
            }

            if (generationConfig.MinNewTokens.HasValue && generationConfig.EosTokenTensor != null && generationConfig.MinNewTokens > 0) {
                processors.Add(new MinNewTokensLengthLogitsProcessor(
                    inputIdsSeqLength,
                    generationConfig.MinNewTokens.Value,
                    generationConfig.EosTokenTensor
                ));
            }

            if (prefixAllowedTokensFn != null) {
                processors.Add(new PrefixConstrainedLogitsProcessor(
                    prefixAllowedTokensFn,
                    generationConfig.NumBeams.Value / generationConfig.NumBeamGroups.Value
                ));
            }

            if (generationConfig.ForcedBosTokenId.HasValue) {
                processors.Add(new ForcedBOSTokenLogitsProcessor(
                    generationConfig.ForcedBosTokenId.Value
                ));
            }

            if (generationConfig.ForcedEosTokenId.HasValue) {
                processors.Add(new ForcedEOSTokenLogitsProcessor(
                    generationConfig.MaxLength.Value,
                    generationConfig.ForcedEosTokenId.Value
                ));
            }

            if (generationConfig.RemoveInvalidValues == true) {
                processors.Add(new InfNanRemoveLogitsProcessor());
            }

            if (generationConfig.ExponentialDecayLengthPenalty.HasValue) {
                processors.Add(new ExponentialDecayLengthPenalty(
                    generationConfig.ExponentialDecayLengthPenalty.Value,
                    generationConfig.EosTokenTensor,
                    inputIdsSeqLength
                ));
            }

            if (generationConfig.SuppressTokens != null) {
                processors.Add(new SuppressTokensLogitsProcessor(
                    generationConfig.SuppressTokens
                ));
            }

            if (generationConfig.BeginSuppressTokens != null) {
                int beginIndex = inputIdsSeqLength;
                beginIndex = (inputIdsSeqLength > 1 || generationConfig.ForcedBosTokenId == null) ? beginIndex : beginIndex + 1;
                if (generationConfig.ForcedDecoderIds != null) {
                    beginIndex += generationConfig.ForcedDecoderIds.Last()[0];
                }
                processors.Add(new SuppressTokensAtBeginLogitsProcessor(
                    generationConfig.BeginSuppressTokens,
                    beginIndex
                ));
            }

            if (generationConfig.ForcedDecoderIds != null) {
                // TODO: Deprecate in v4.40
                Console.WriteLine("You have explicitly specified `forced_decoder_ids`. This functionality has been deprecated and will throw an error in v4.40. Please remove the `forced_decoder_ids` argument in favour of `input_ids` or `decoder_input_ids` respectively.");
                processors.Add(new ForceTokensLogitsProcessor(generationConfig.ForcedDecoderIds, true));
            }

            if (generationConfig.WatermarkingConfig != null) {
                processors.Add(new WatermarkLogitsProcessor(
                    Config.VocabSize,
                    generationConfig.WatermarkingConfig.GreenlistRatio,
                    generationConfig.WatermarkingConfig.Bias,
                    generationConfig.WatermarkingConfig.HashingKey,
                    generationConfig.WatermarkingConfig.SeedingScheme,
                    generationConfig.WatermarkingConfig.ContextWidth
                ));
            }

            processors = MergeCriteriaProcessorList(processors, logitsProcessor) as LogitsProcessorList;

            // `LogitNormalization` should always be the last logit processor, when present
            if (generationConfig.RenormalizeLogits == true) {
                processors.Add(new LogitNormalization());
            }

            return processors;
        }

        public StoppingCriteriaList GetStoppingCriteria(
           GenerationConfig generationConfig,
           StoppingCriteriaList stoppingCriteria = null,
           PreTrainedTokenizerBase tokenizer = null)
        {
            var criteria = new StoppingCriteriaList();

            if (generationConfig.MaxLength.HasValue) {
                var maxPositionEmbeddings = Config.MaxPositionEmbeddings ?? null;
                criteria.Add(new MaxLengthCriteria(generationConfig.MaxLength.Value, maxPositionEmbeddings));
            }

            if (generationConfig.MaxTime.HasValue) {
                criteria.Add(new MaxTimeCriteria(generationConfig.MaxTime.Value));
            }

            if (generationConfig.StopStrings != null) {
                if (tokenizer == null) {
                    throw new ArgumentException("There are one or more stop strings, either in the arguments to `generate` or in the model's generation config, but we could not locate a tokenizer. When generating with stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`.");
                }
                criteria.Add(new StopStringCriteria(tokenizer, generationConfig.StopStrings));
            }

            if (generationConfig.EosTokenTensor != null) {
                criteria.Add(new EosTokenCriteria(generationConfig.EosTokenId));
            }

            return MergeCriteriaProcessorList(criteria, stoppingCriteria) as StoppingCriteriaList;
        }

        public List<T> MergeCriteriaProcessorList<T>(List<T> defaultList, List<T> customList) {
            if (customList.Count == 0) {
                return defaultList;
            }

            foreach (var defaultItem in defaultList) {
                foreach (var customItem in customList) {
                    if (customItem.GetType() == defaultItem.GetType()) {
                        string objectType = customItem is StoppingCriteria ? "stopping criteria" : "logits processor";
                        throw new ArgumentException(
                            $"A custom {objectType} of type {customItem.GetType()} with values {customItem} has been passed to `.Generate()`, but it has already been created with the values {defaultItem}. {defaultItem} has been created by passing the corresponding arguments to generate or by the model's config default values. If you just want to change the default values of {objectType} consider passing them as arguments to `.Generate()` instead of using a custom {objectType}."
                        );
                    }
                }
            }

            defaultList.AddRange(customList);
            return defaultList;
        }

        private void ValidateAssistant(object assistantModel) {
            if (assistantModel == null) {
                return;
            }

            throw new NotImplementedException("Assitant model not yet supported.");
            /*if (Config.IsEncoderDecoder && !assistantModel.config.is_encoder_decoder) {
                string[] attributesToCheck = new string[] { "encoder_attention_heads", "encoder_ffn_dim", "encoder_layers"};
                attributesToCheck = [attr for attr in dir(assistant_model.config) if attr in attributes_to_check]
                bool areEqual = all(getattr(self.config, attr) == getattr(assistant_model.config, attr) for attr in attributes_to_check)
                if (!areEqual) {
                    throw new ArgumentException("The main model and the assistant don't have compatible encoder-dependent input shapes. " +
                        "Ensure you load the assistant with the correct encoder-decoder class, e.g. `AutoModelForSpeechSeq2Seq` for Whisper.");
                }
            }

            if (Config.vocab_size != assistantModel.config.vocab_size) {
                throw new ArgumentException("Make sure the main and assistant model use the same tokenizer");
            }*/
        }

        /// <summary>
        /// This function extracts the model-specific `inputs` for generation.
        /// </summary>
        private void PrepareModelInputs(ref TensorInt inputs, out string inputName, int bosTokenId, ref Kwargs modelKwargs) {
            // retrieve all kwargs that are non-None or non-model input related.
            // some encoder-decoder models have different names for model and encoder
            if (Config.IsEncoderDecoder && HasEncoder /* && Encoder.MainInputName != MainInputName*/) {
                inputName = "";// Encoder.MainInputName
            } else {
                inputName = MainInputName;
            }

            string inputN = inputName;
            modelKwargs = modelKwargs.Where(kvp => kvp.Value != null || kvp.Key != inputN);
            var inputsKwarg = modelKwargs.Pop(inputName, null);
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
                            $"You passed `inputs_embeds` to `.Generate()`, but the model class {GetType().Name} " +
                            "doesn't have its forwarding implemented. See the GPT2 implementation for an example " +
                            "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                        );
                    }

                    MaybeInitializeInputIdsForGeneration(ref inputs, bosTokenId, modelKwargs);
                    modelKwargs["input_ids"] = inputs;

                } else {
                    if (inputs != null) {
                        throw new ArgumentException("You passed `inputs_embeds` and `input_ids` to `.Generate()`. Please pick one.");
                    }
                }

                inputs = modelKwargs["inputs_embeds"] as TensorInt;
                inputName = "inputs_embeds";
            }


            // 4. if `inputs` is still None, try to create `input_ids` from BOS token
            MaybeInitializeInputIdsForGeneration(ref inputs, bosTokenId, modelKwargs);
        }

        private bool HasInputsEmbedsForwarding() {
            var method = GetType().GetMethod(nameof(PrepareInputsForGeneration), BindingFlags.Instance | BindingFlags.NonPublic);
            var parameters = method.GetParameters();
            return parameters.Any(p => p.Name == "inputs_embeds");
        }

        /// <summary>
        /// Initializes input ids for generation, if necessary.
        /// </summary>
        private void MaybeInitializeInputIdsForGeneration(ref TensorInt inputs, int? bosTokenId, Kwargs modelKwargs) {
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

        private bool SupportsDefaultDynamicCache() {
            return true;
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

        private static Tensor PrepareEncoderDecoderKwargsForGeneration(Tensor inputs, Kwargs modelKwargs, string modelInputName, GenerationConfig generationConfig) {
            throw new NotImplementedException();
        }

        private static Tensor PrepareDecoderInputIdsForGeneration(int batchSize, string modelInputName, Kwargs modelKwargs, Tensor decoderStartTokenTensor) {
            throw new NotImplementedException();
        }

        private static Tensor HealTokens(Tensor inputIds, PreTrainedTokenizerBase tokenizer) {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs validation related to the resulting generated length"
        /// </summary>
        public void ValidateGeneratedLength(GenerationConfig generationConfig, int inputIdsLength, bool hasDefaultMaxLength) {
            // Max length warnings related to poor parameterization
            if (hasDefaultMaxLength && generationConfig.MaxNewTokens == null && generationConfig.MaxLength == 20) {
                Log.Warning($"Using the model-agnostic default `max_length` (={generationConfig.MaxLength}) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.");
            }
            if (inputIdsLength >= generationConfig.MaxLength) {
                string inputIdsString = Config.IsEncoderDecoder ? "decoder_input_ids" : "input_ids";
                throw new ArgumentException($"Input length of {inputIdsString} is {inputIdsLength}, but `max_length` is set to {generationConfig.MaxLength}. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.");
            }

            // Min length warnings due to unfeasible parameter combinations
            string minLengthErrorSuffix = " Generation will stop at the defined maximum length. You should decrease the minimum length and/or increase the maximum length.";
            if (hasDefaultMaxLength) {
                minLengthErrorSuffix += $" Note that `max_length` is set to {generationConfig.MaxLength}, its default value.";
            }
            if (generationConfig.MinLength.HasValue && generationConfig.MinLength.Value > generationConfig.MaxLength) {
                Log.Warning($"Unfeasible length constraints: `min_length` ({generationConfig.MinLength}) is larger than the maximum possible length ({generationConfig.MaxLength})." + minLengthErrorSuffix);
            }
            if (generationConfig.MinNewTokens.HasValue) {
                int minLength = generationConfig.MinNewTokens.Value + inputIdsLength;
                if (minLength > generationConfig.MaxLength) {
                    Log.Warning($"Unfeasible length constraints: `min_new_tokens` ({generationConfig.MinNewTokens}), when added to the prompt length ({inputIdsLength}), is larger than the maximum possible length ({generationConfig.MaxLength})." + minLengthErrorSuffix);
                }
            }
        }

        /// <summary>
        /// Prepared max and min length in generation configs to avoid clashes between similar attributes
        /// </summary>
        public GenerationConfig PrepareGeneratedLength(
            GenerationConfig generationConfig,
            bool hasDefaultMaxLength,
            bool hasDefaultMinLength,
            string modelInputName,
            int inputIdsLength,
            Tensor inputsTensor)
        {
            if (generationConfig.MaxNewTokens.HasValue) {
                if (!hasDefaultMaxLength && generationConfig.MaxLength.HasValue) {
                    Log.Warning($"Both `max_new_tokens` (={generationConfig.MaxNewTokens}) and `max_length`(={generationConfig.MaxLength}) seem to have been set. `max_new_tokens` will take precedence. " +
                        "Please refer to the documentation for more information. " +
                        "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                    );
                }
                generationConfig.MaxLength = generationConfig.MaxNewTokens + inputIdsLength;
            }
            // if both `inputs_embeds` and `input_ids` are passed, we do not correct the length
            // otherwise we need total length [inputs-embeds-len + new-tokens-len] to not go beyond indicated `max_length``
            else if (modelInputName == "inputs_embeds" && inputIdsLength != inputsTensor.shape[1] && !Config.IsEncoderDecoder) {
                generationConfig.MaxLength -= inputsTensor.shape[1];
            }

            // same for min length
            if (generationConfig.MinNewTokens.HasValue) {
                if (!hasDefaultMinLength) {
                    Log.Warning($"Both `min_new_tokens` (={generationConfig.MinNewTokens}) and `min_length`(={generationConfig.MinLength}) seem to have been set. `min_new_tokens` will take precedence. " +
                        "Please refer to the documentation for more information. " +
                        "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                    );
                }
                generationConfig.MinLength = generationConfig.MinNewTokens + inputIdsLength;
            } else if (modelInputName == "inputs_embeds" && inputIdsLength != inputsTensor.shape[1] && !Config.IsEncoderDecoder) {
                generationConfig.MinLength = Math.Max(generationConfig.MinLength.Value - inputsTensor.shape[1], 0);
            }

            return generationConfig;
        }

        /// <summary>
        /// Sets a cache for <see cref="Generate"/>, that will persist across calls. A new cache will only be initialized a
        /// new <see cref="Generate"/> call requires a larger cache or uses a different batch size.
        /// </summary>
        public Cache GetCache(string cacheImplementation, int maxBatchSize, int maxCacheLen, Kwargs modelKwargs) {
            Type cacheCls = NEED_SETUP_CACHE_CLASSES_MAPPING[cacheImplementation];
            bool requiresCrossAttentionCache = Config.IsEncoderDecoder || modelKwargs.ContainsKey("encoder_outputs");

            Cache cacheToCheck = null;
            if (_cache != null) {
                cacheToCheck = requiresCrossAttentionCache ? ((EncoderDecoderCache)_cache).SelfAttentionCache : _cache;
            }

            if (cacheImplementation == "sliding_window") {
                maxCacheLen = Math.Min(Config.SlidingWindow.Value, maxCacheLen);
            }

            bool needNewCache = _cache == null || cacheToCheck.GetType() != cacheCls || cacheToCheck.MaxBatchSize != maxBatchSize;
            if (cacheImplementation != "mamba") {
                needNewCache = needNewCache || cacheToCheck.MaxCacheLen < maxCacheLen;
            }

            if (requiresCrossAttentionCache && _cache != null) {
                needNewCache = needNewCache || ((EncoderDecoderCache)_cache).CrossAttentionCache.MaxCacheLen != ((Tensor[])modelKwargs["encoder_outputs"])[0].shape.length;
            }

            if (needNewCache) {
                Type cacheDtype = null;//Config.PreQuantizationDtype ?? self.dtype;

                var cacheArgs = new {
                    Config,
                    maxBatchSize,
                    maxCacheLen,
                    dtype = cacheDtype
                };

                _cache = (Cache)Activator.CreateInstance(cacheCls, cacheArgs);
                if (requiresCrossAttentionCache) {
                    var encoderArgs = new {
                        Config,
                        maxBatchSize,
                        maxCacheLen = ((Tensor[])modelKwargs["encoder_outputs"])[0].shape.length,
                        dtype = cacheDtype
                    };
                    _cache = new EncoderDecoderCache(_cache, (Cache)Activator.CreateInstance(cacheCls, encoderArgs));
                }
            } else {
                _cache.Reset();
            }

            return _cache;
        }
    }
}