using Newtonsoft.Json;

namespace Doji.AI.Transformers {

    public class TokenizerConfig {

        [JsonProperty("bos_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token BosToken { get; set; } = new AddedToken("<|startoftext|>");

        [JsonProperty("eos_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token EosToken { get; set; } = new AddedToken("<|endoftext|>");

        [JsonProperty("unk_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token UnkToken { get; set; } = new AddedToken("<|endoftext|>");

        [JsonProperty("sep_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token SepToken { get; set; } = null;

        [JsonProperty("pad_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token PadToken { get; set; } = new AddedToken("<|endoftext|>");

        [JsonProperty("cls_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token ClsToken { get; set; } = null;

        [JsonProperty("mask_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token MaskToken { get; set; } = null;

        [JsonProperty("do_lower_case")]
        public bool DoLowerCase { get; set; }

        [JsonProperty("errors")]
        public string Errors { get; set; } = "replace";

        [JsonProperty("model_max_length")]
        public int ModelMaxLength { get; set; } = int.MaxValue;

        [JsonProperty("tokenizer_class")]
        public string TokenizerClass { get; set; }

        public static TokenizerConfig Deserialize(string json) {
            TokenizerConfig config = JsonConvert.DeserializeObject<TokenizerConfig>(json);
            return config;
        }
    }
}