using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class WatermarkingConfig {
        [JsonProperty("greenlist_ratio")]
        public float GreenlistRatio { get; set; }

        [JsonProperty("bias")]
        public float Bias { get; set; }

        [JsonProperty("hashing_key")]
        public int HashingKey { get; set; }

        [JsonProperty("seeding_scheme")]
        public string SeedingScheme { get; set; }

        [JsonProperty("context_width")]
        public int ContextWidth { get; set; }
    }

    public enum GenerationMode {
        GREEDY_SEARCH,
        SAMPLE,
        BEAM_SEARCH,
        BEAM_SAMPLE,
        GROUP_BEAM_SEARCH,
        CONTRASTIVE_SEARCH,
        CONSTRAINED_BEAM_SEARCH,
        ASSISTED_GENERATION,
        DOLA_GENERATION
    }
    public enum Schedule {
        [EnumMember(Value = "heuristic")]
        Heuristic,
        [EnumMember(Value = "heuristic_transient")]
        HeuristicTransient,
        [EnumMember(Value = "constant")]
        Constant
    }
    public enum StoppingCondition { True, False, Never }

    public partial class GenerationConfig {

        /* Parameters that control the length of the output */

        [JsonProperty("max_length")]
        public int? MaxLength { get; set; }

        [JsonProperty("max_new_tokens")]
        public int? MaxNewTokens { get; set; }

        [JsonProperty("min_length")]
        public int? MinLength { get; set; }

        [JsonProperty("min_new_tokens")]
        public int? MinNewTokens { get; set; }

        [JsonProperty("early_stopping")]
        public StoppingCondition EarlyStopping { get; set; }

        [JsonProperty("max_time")]
        public float? MaxTime { get; set; }

        [JsonProperty("stop_strings")]
        public List<string> StopStrings { get; set; }


        /* Parameters that control the generation strategy used */

        [JsonProperty("do_sample")]
        public bool DoSample { get; set; }

        [JsonProperty("num_beams")]
        public int? NumBeams { get; set; }

        [JsonProperty("num_beam_groups")]
        public int? NumBeamGroups { get; set; }

        [JsonProperty("penalty_alpha")]
        public float? PenaltyAlpha { get; set; }

        [JsonProperty("use_cache")]
        public bool? UseCache { get; set; }


        /* Parameters for manipulation of the model output logits */

        [JsonProperty("temperature")]
        public float Temperature { get; set; }

        [JsonProperty("top_k")]
        public int? TopK { get; set; }

        [JsonProperty("top_p")]
        public float? TopP { get; set; }

        [JsonProperty("min_p")]
        public float? MinP { get; set; }

        [JsonProperty("typical_p")]
        public float? TypicalP { get; set; }

        [JsonProperty("epsilon_cutoff")]
        public float EpsilonCutoff { get; set; }

        [JsonProperty("eta_cutoff")]
        public float? EtaCutoff { get; set; }

        [JsonProperty("diversity_penalty")]
        public float? DiversityPenalty { get; set; }

        [JsonProperty("repetition_penalty")]
        public float? RepetitionPenalty { get; set; }

        [JsonProperty("encoder_repetition_penalty")]
        public float? EncoderRepetitionPenalty { get; set; }

        [JsonProperty("length_penalty")]
        public float? LengthPenalty { get; set; }

        [JsonProperty("no_repeat_ngram_size")]
        public int? NoRepeatNgramSize { get; set; }

        [JsonProperty("bad_words_ids")]
        public List<List<int>> BadWordsIds { get; set; }

        [JsonProperty("force_words_ids")]
        public List<List<int>> ForceWordsIds { get; set; }

        [JsonProperty("renormalize_logits")]
        public bool? RenormalizeLogits { get; set; }

        [JsonProperty("constraints")]
        public List<string> Constraints { get; set; }

        [JsonProperty("forced_bos_token_id")]
        public int? ForcedBosTokenId { get; set; }

        [JsonProperty("forced_eos_token_id")]
        public int? ForcedEosTokenId { get; set; }

        [JsonProperty("remove_invalid_values")]
        public bool? RemoveInvalidValues { get; set; }

        [JsonProperty("exponential_decay_length_penalty")]
        public (int startIndex, float decayFactor)? ExponentialDecayLengthPenalty { get; set; }

        [JsonProperty("suppress_tokens")]
        public List<int> SuppressTokens { get; set; }

        [JsonProperty("begin_suppress_tokens")]
        public List<int> BeginSuppressTokens { get; set; }

        [JsonProperty("forced_decoder_ids")]
        public List<List<int>> ForcedDecoderIds { get; set; }

        [JsonProperty("sequence_bias")]
        public object SequenceBias { get; set; }

        [JsonProperty("token_healing")]
        public bool? TokenHealing { get; set; }

        [JsonProperty("guidance_scale")]
        public float? GuidanceScale { get; set; }

        [JsonProperty("low_memory")]
        public bool? LowMemory { get; set; }
        
        [JsonProperty("watermarking_config")]
        //TODO: custom convert in case its passed as dict
        public WatermarkingConfig WatermarkingConfig { get; set; }

        /*  Parameters that define the output variables of generate */

        [JsonProperty("num_return_sequences")]
        public int? NumReturnSequences { get; set; }

        [JsonProperty("output_attentions")]
        public bool? OutputAttentions { get; set; }

        [JsonProperty("output_hidden_states")]
        public bool? OutputHiddenStates { get; set; }

        [JsonProperty("output_scores")]
        public bool? OutputScores { get; set; }

        [JsonProperty("output_logits")]
        public bool? OutputLogits { get; set; }

        [JsonProperty("return_dict_in_generate")]
        public bool? ReturnDictInGenerate { get; set; }


        /* special tokens that can be used at generation time */

        [JsonProperty("bos_token_id")]
        public int? BosTokenId { get; set; }

        [JsonProperty("eos_token_id")]
        public int[] EosTokenId { get; set; }

        [JsonProperty("pad_token_id")]
        public int? PadTokenId { get; set; }


        /* Generation parameters exclusive to encoder-decoder models */

        [JsonProperty("encoder_no_repeat_ngram_size")]
        public int? EncoderNoRepeatNgramSize { get; set; }

        [JsonProperty("decoder_start_token_id")]
        public int[] DecoderStartTokenId { get; set; }


        /* Generation parameters exclusive to assistant generation */

        [JsonProperty("num_assistant_tokens")]
        public int? NumAssistantTokens { get; set; }

        [JsonProperty("num_assistant_tokens_schedule")]
        public Schedule NumAssistantTokensSchedule { get; set; }

        [JsonProperty("prompt_lookup_num_tokens")]
        public int? PromptLookupNumTokens { get; set; }

        [JsonProperty("max_matching_ngram_size")]
        public int? MaxMatchingNgramSize { get; set; }


        /* Generation parameters exclusive to assistant generation */

        [JsonProperty("dola_layers")]
        public object DolaLayers { get; set; }


        /* Parameters specific to the caching mechanism: */

        [JsonProperty("cache_implementation")]
        public string CacheImplementation { get; set; }

        [JsonProperty("cache_config")]
        public CacheConfig CacheConfig { get; set; }

        [JsonProperty("return_legacy_cache")]
        public bool ReturnLegacyCache { get; set; }


        /* remaining attributes */

        [JsonProperty("_from_model_config")]
        public bool FromModelConfig { get; set; }

        [JsonProperty("transformers_version")]
        public string TransformersVersion { get; set; }


        [JsonIgnore]
        public Tensor BosTokenTensor { get; set; }
        [JsonIgnore]
        public Tensor EosTokenTensor { get; set; }
        [JsonIgnore]
        public Tensor PadTokenTensor { get; set; }
        [JsonIgnore]
        public Tensor DecoderStartTokenTensor { get; set; }

        public GenerationConfig() {
            // Parameters that control the length of the output
            MaxLength = 20;
            MaxNewTokens = null;
            MinLength = 0;
            MinNewTokens = null;
            EarlyStopping = StoppingCondition.False;
            MaxTime = null;
            StopStrings = null;

            // Parameters that control the generation strategy used
            DoSample = false;
            NumBeams = 1;
            NumBeamGroups = 1;
            PenaltyAlpha = null;
            UseCache = true;

            // Parameters for manipulation of the model output logits
            Temperature = 1.0f;
            TopK = 50;
            TopP = 1.0f;
            MinP = null;
            TypicalP = 1.0f;
            EpsilonCutoff = 0.0f;
            EtaCutoff = 0.0f;
            DiversityPenalty = 0.0f;
            RepetitionPenalty = 1.0f;
            EncoderRepetitionPenalty = 1.0f;
            LengthPenalty = 1.0f;
            NoRepeatNgramSize = 0;
            BadWordsIds = null;
            ForceWordsIds = null;
            RenormalizeLogits = false;
            Constraints = null;
            ForcedBosTokenId = null;
            ForcedEosTokenId = null;
            RemoveInvalidValues = false;
            ExponentialDecayLengthPenalty = null;
            SuppressTokens = null;
            BeginSuppressTokens = null;
            ForcedDecoderIds = null;
            SequenceBias = null;
            TokenHealing = false;
            GuidanceScale = null;
            LowMemory = null;
            WatermarkingConfig = null;

            // Parameters that define the output variables of `generate`
            NumReturnSequences = 1;
            OutputAttentions = false;
            OutputHiddenStates = false;
            OutputScores = false;
            OutputLogits = false;
            ReturnDictInGenerate = false;

            // Special tokens that can be used at generation time
            PadTokenId = null;
            BosTokenId = null;
            EosTokenId = null;

            // Generation parameters exclusive to encoder-decoder models
            EncoderNoRepeatNgramSize = 0;
            DecoderStartTokenId = null;

            // Assistant generation
            NumAssistantTokens = 5;
            NumAssistantTokensSchedule = Schedule.Heuristic;

            // DoLa generation
            DolaLayers = null;

            // Cache implementation
            CacheImplementation = null;
            CacheConfig = null;
            ReturnLegacyCache = true;

            //Prompt lookup decoding
            PromptLookupNumTokens = null;
            MaxMatchingNgramSize = null;

            // The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
            // interface.
            FromModelConfig = false;
        }

        /// <summary>
        /// Returns the generation mode triggered by the [`GenerationConfig`] instance.
        /// </summary>
        public GenerationMode GetGenerationMode(PreTrainedModel assistantModel = null) {
            GenerationMode generationMode;

            if (Constraints != null || ForceWordsIds != null) {
                generationMode = GenerationMode.CONSTRAINED_BEAM_SEARCH;
            } else if (NumBeams == 1) {
                if (!DoSample) {
                    if (TopK != null && TopK.Value > 1 && PenaltyAlpha != null && PenaltyAlpha.Value > 0) {
                        generationMode = GenerationMode.CONTRASTIVE_SEARCH;
                    } else {
                        generationMode = GenerationMode.GREEDY_SEARCH;
                    }
                } else {
                    generationMode = GenerationMode.SAMPLE;
                }
            } else {
                if (NumBeamGroups > 1) {
                    generationMode = GenerationMode.GROUP_BEAM_SEARCH;
                } else if (DoSample) {
                    generationMode = GenerationMode.BEAM_SAMPLE;
                } else {
                    generationMode = GenerationMode.BEAM_SEARCH;
                }
            }

            if (assistantModel != null || PromptLookupNumTokens != null) {
                if (generationMode == GenerationMode.GREEDY_SEARCH || generationMode == GenerationMode.SAMPLE) {
                    generationMode = GenerationMode.ASSISTED_GENERATION;
                } else {
                    throw new ArgumentException(
                        "You've set `assistantModel`, which triggers assisted generation. Currently, assisted generation " +
                        "is only supported with Greedy Search and Sample."
                    );
                }
            }

            if (DolaLayers != null) {
                if (generationMode == GenerationMode.GREEDY_SEARCH || generationMode == GenerationMode.SAMPLE) {
                    generationMode = GenerationMode.DOLA_GENERATION;
                } else {
                    throw new ArgumentException(
                        "You've set `dolaLayers`, which triggers DoLa generation. Currently, DoLa generation " +
                        "is only supported with Greedy Search and Sample."
                    );
                }
            }

            return generationMode;
        }
    }
}
