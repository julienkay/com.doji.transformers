using Newtonsoft.Json;
using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Base class for all configuration classes. Handles a few parameters common to all models' configurations
    /// as well as methods for loading configurations.
    /// </summary>
    public class PretrainedConfig {

        [JsonProperty("_name_or_path")]
        public string NameOrPath { get; set; }

        [JsonProperty("architectures")]
        public List<string> Architectures { get; set; }

        [JsonProperty("is_encoder_decoder")]
        public bool IsEncoderDecoder { get; set; }

        [JsonProperty("sliding_window")]
        public int? SlidingWindow { get; set; }

        public PretrainedConfig() {
            IsEncoderDecoder = false;
        }
    }
}