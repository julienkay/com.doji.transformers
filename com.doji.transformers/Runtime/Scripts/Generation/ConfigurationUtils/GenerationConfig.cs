using Newtonsoft.Json;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

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
        public string StopStrings { get; set; }


        /* Parameters that control the generation strategy used */

        [JsonProperty("do_sample")]
        public bool DoSample { get; set; }

        [JsonProperty("num_beams")]
        public int NumBeams { get; set; }

        [JsonProperty("num_beam_groups")]
        public int NumBeamGroups { get; set; }

        [JsonProperty("penalty_alpha")]
        public float? PenaltyAlpha { get; set; }

        [JsonProperty("use_cache")]
        public bool UseCache { get; set; }


        /* Parameters for manipulation of the model output logits */

        [JsonProperty("temperature")]
        public float Temperature { get; set; }

        [JsonProperty("top_k")]
        public int TopK { get; set; }

        [JsonProperty("top_p")]
        public float TopP { get; set; }

        [JsonProperty("min_p")]
        public float? MinP { get; set; }

        [JsonProperty("typical_p")]
        public float TypicalP { get; set; }

        [JsonProperty("epsilon_cutoff")]
        public float EpsilonCutoff { get; set; }

        [JsonProperty("eta_cutoff")]
        public float EtaCutoff { get; set; }

        [JsonProperty("diversity_penalty")]
        public float DiversityPenalty { get; set; }

        [JsonProperty("repetition_penalty")]
        public float RepetitionPenalty { get; set; }

        [JsonProperty("encoder_repetition_penalty")]
        public float EncoderRepetitionPenalty { get; set; }

        [JsonProperty("length_penalty")]
        public float LengthPenalty { get; set; }

        [JsonProperty("no_repeat_ngram_size")]
        public int NoRepeatNgramSize { get; set; }

        [JsonProperty("bad_words_ids")]
        public List<List<int>> BadWordsIds { get; set; }

        [JsonProperty("force_words_ids")]
        public List<List<int>> ForceWordsIds { get; set; }

        [JsonProperty("renormalize_logits")]
        public bool RenormalizeLogits { get; set; }

        [JsonProperty("constraints")]
        public List<string> Constraints { get; set; }

        [JsonProperty("forced_bos_token_id")]
        public int? ForcedBosTokenId { get; set; }

        [JsonProperty("forced_eos_token_id")]
        public int? ForcedEosTokenId { get; set; }

        [JsonProperty("remove_invalid_values")]
        public bool RemoveInvalidValues { get; set; }

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
        public bool TokenHealing { get; set; }

        [JsonProperty("guidance_scale")]
        public float? GuidanceScale { get; set; }

        [JsonProperty("low_memory")]
        public bool? LowMemory { get; set; }


        /*  Parameters that define the output variables of generate */

        [JsonProperty("num_return_sequences")]
        public int NumReturnSequences { get; set; }

        [JsonProperty("output_attentions")]
        public bool OutputAttentions { get; set; }

        [JsonProperty("output_hidden_states")]
        public bool OutputHiddenStates { get; set; }

        [JsonProperty("output_scores")]
        public bool OutputScores { get; set; }

        [JsonProperty("output_logits")]
        public bool OutputLogits { get; set; }

        [JsonProperty("return_dict_in_generate")]
        public bool ReturnDictInGenerate { get; set; }


        /* special tokens that can be used at generation time */

        [JsonProperty("bos_token_id")]
        public int? BosTokenId { get; set; }

        [JsonProperty("eos_token_id")]
        public int[] EosTokenId { get; set; }

        [JsonProperty("pad_token_id")]
        public int? PadTokenId { get; set; }


        /* Generation parameters exclusive to encoder-decoder models */
        [JsonProperty("decoder_start_token_id")]
        public int[] DecoderStartTokenId { get; set; }


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

        public enum StoppingCondition { True, False, Never }

        public GenerationConfig() {
            MaxLength = 20;
            MaxNewTokens = null;
            MinLength = 0;
            MinNewTokens = null;
            EarlyStopping = StoppingCondition.False;
            MaxTime = null;
            StopStrings = null;

            DoSample = false;
            NumBeams = 1;
            NumBeamGroups = 1;
            PenaltyAlpha = null;
            UseCache = true;

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

            NumReturnSequences = 1;
            OutputAttentions = false;
            OutputHiddenStates = false;
            OutputScores = false;
            OutputLogits = false;
            ReturnDictInGenerate = false;

            PadTokenId = null;
            BosTokenId = null;
            EosTokenId = null;
        }
    }
}
