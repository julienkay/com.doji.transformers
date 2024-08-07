using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Represents the encoded output for a single text input.
    /// </summary>
    public class InputEncoding : Encoding {

        public InputEncoding() : base() { }
        public InputEncoding(Dictionary<string, object> dict) : base(dict) { }

        public override IEnumerable<int> InputIds {
            get {
                TryGetValue("input_ids", out var inputIds);
                return inputIds as IEnumerable<int>;
            }
        }

        public override IEnumerable<int> AttentionMask {
            get {
                TryGetValue("attention_mask", out var inputIds);
                return inputIds as IEnumerable<int>;
            }
        }

        public override IEnumerable<int> TokenTypeIds {
            get {
                TryGetValue("token_type_ids", out var inputIds);
                return inputIds as IEnumerable<int>;
            }
        }
    }
}