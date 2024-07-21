using Newtonsoft.Json;
using System;
using System.Collections.Generic;

namespace Doji.AI.Transformers {

    public class TokenizerConfig {
        
        [JsonProperty("add_prefix_space")]
        public bool? AddPrefixSpace = null;

        [JsonProperty("add_bos_token")]
        public bool? AddBosToken = null;

        [JsonProperty("add_eos_token")]
        public bool? AddEosToken = null;

        [JsonProperty("added_tokens_decoder")]
        [JsonConverter(typeof(AddedTokensConverter))]
        public Dictionary<int, AddedToken> AddedTokensDecoder;

        [JsonProperty("bos_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token BosToken { get; set; } = null;

        [JsonProperty("eos_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token EosToken { get; set; } = null;

        [JsonProperty("unk_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token UnkToken { get; set; } = null;

        [JsonProperty("sep_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token SepToken { get; set; } = null;

        [JsonProperty("pad_token")]
        [JsonConverter(typeof(TokenConverter))]
        public Token PadToken { get; set; } = null;

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

        [JsonProperty("legacy")]
        public bool? Legacy { get; set; }

        [JsonProperty("model_max_length")]
        public int ModelMaxLength { get; set; } = int.MaxValue;

        [JsonProperty("tokenizer_class")]
        public string TokenizerClass { get; set; }

        public static TokenizerConfig Deserialize(string json) {
            TokenizerConfig config = JsonConvert.DeserializeObject<TokenizerConfig>(json);
            return config;
        }

        private class AddedTokensConverter : JsonConverter {
            public override bool CanConvert(Type objectType) {
                return true;
            }
            
            public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer) {
                var addedTokens = serializer.Deserialize<Dictionary<int, AddedToken>>(reader);
                return addedTokens;
            }

            public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer) {
                throw new NotImplementedException();
            }
        }
    }
}