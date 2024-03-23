using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    [JsonConverter(typeof(VocabConverter))]
    public class Vocab {

        public Dictionary<string, int> Encoder { get; set; }
        public Dictionary<int, string> Decoder { get; set; }

        public Vocab(Dictionary<string, int> encoder) {
            Encoder = encoder;
            Decoder = encoder.ToDictionary(x => x.Value, x => x.Key);
        }

        public static Vocab Deserialize(string json) {
            Dictionary<string, int> vocabEntries = JsonConvert.DeserializeObject<Dictionary<string, int>>(json);
            return new Vocab(vocabEntries);
        }
    }

    public class VocabConverter : JsonConverter<Vocab> {
        public override Vocab ReadJson(JsonReader reader, Type objectType, Vocab existingValue, bool hasExistingValue, JsonSerializer serializer) {
            var vocabEntries = serializer.Deserialize<Dictionary<string, int>>(reader);
            return new Vocab(vocabEntries);
        }
        public override void WriteJson(JsonWriter writer, Vocab value, JsonSerializer serializer) {
            var vocabEntries = value.Encoder;
            serializer.Serialize(writer, vocabEntries);
        }
    }
}