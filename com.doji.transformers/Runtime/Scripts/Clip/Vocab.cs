using Newtonsoft.Json;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

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
}