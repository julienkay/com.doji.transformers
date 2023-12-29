using Newtonsoft.Json;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    public class Vocab {
        public Dictionary<string, int> Encoder { get; set; }
        public Dictionary<int, string> Decoder { get; set; }

        public static Vocab Deserialize(string json) {
            Dictionary<string, int> vocabEntries = JsonConvert.DeserializeObject<Dictionary<string, int>>(json);
            return new Vocab() {
                Encoder = vocabEntries,
                Decoder = vocabEntries.ToDictionary(x => x.Value, x => x.Key)
            };
        }
    }
}