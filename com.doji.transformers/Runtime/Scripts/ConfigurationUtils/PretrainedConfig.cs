using Newtonsoft.Json;
using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Base class for all configuration classes. Handles a few parameters common to all models' configurations
    /// as well as methods for loading configurations.
    /// </summary>
    public class PretrainedConfig {

        /* Common attributes(present in all subclasses): */

        [JsonProperty("vocab_size")]
        public int VocabSize { get; set; }

        [JsonProperty("hidden_size")]
        public int HiddenSize { get; set; }

        [JsonProperty("num_attention_heads")]
        public int NumAttentionHeads { get; set; }

        [JsonProperty("num_hidden_layers")]
        public int NumHiddenLayers { get; set; }


        [JsonProperty("_name_or_path")]
        public string NameOrPath { get; set; }

        [JsonProperty("architectures")]
        public List<string> Architectures { get; set; }

        [JsonProperty("is_encoder_decoder")]
        public bool IsEncoderDecoder { get; set; }


        /* These are actually not part of the base config class, but still convenient to have here */

        [JsonProperty("sliding_window")]
        public int? SlidingWindow { get; set; }

        [JsonProperty("max_position_embeddings")]
        public int? MaxPositionEmbeddings { get; set; }
        
        public PretrainedConfig() {
            IsEncoderDecoder = false;
        }
    }
}