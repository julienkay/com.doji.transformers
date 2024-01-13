using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System;

namespace Doji.AI.Transformers {

    public class Token {

        [JsonProperty("content")]
        public string Content { get; set; }

        public static implicit operator string(Token t) => t.Content;

        public override string ToString() {
            return Content;
        }
    }

    public class TokenConverter : JsonConverter {
        public override bool CanConvert(Type objectType) {
            return objectType == typeof(Token) || objectType == typeof(string);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer) {
            if (reader.TokenType == JsonToken.String) {
                // If the JSON represents the token as a simple string, create an TokenString instance
                string content = reader.Value.ToString();
                return new TokenString(content);
            }

            JObject obj = JObject.Load(reader);

            // Check if '__type' attribute exists
            if (obj.TryGetValue("__type", out var typeToken)) {
                string typeName = typeToken.Value<string>();

                switch (typeName) {
                    case "AddedToken":
                        return obj.ToObject<AddedToken>();
                    default:
                        throw new InvalidOperationException($"Unknown token type: {typeName}");
                }
            }

            throw new InvalidOperationException($"Unknown token type: {obj}");
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer) {
            throw new NotImplementedException();
        }
    }
}