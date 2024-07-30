using Newtonsoft.Json;

namespace Doji.AI.Transformers {

    public partial class GenerationConfig {
        [JsonProperty("_from_model_config")]
        public bool FromModelConfig { get; set; }

        [JsonProperty("bos_token_id")]
        public long BosTokenId { get; set; }

        [JsonProperty("eos_token_id")]
        public long[] EosTokenId { get; set; }

        [JsonProperty("pad_token_id")]
        public long PadTokenId { get; set; }

        [JsonProperty("transformers_version")]
        public string TransformersVersion { get; set; }
    }
}
