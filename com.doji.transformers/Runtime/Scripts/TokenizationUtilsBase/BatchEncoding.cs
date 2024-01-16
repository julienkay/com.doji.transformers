using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Holds the output of <see cref="PreTrainedTokenizerBase.Encode"/>
    /// and <see cref="PreTrainedTokenizerBase.EncodePlus"/> methods
    /// (tokens, attention_masks, etc).
    /// 
    /// This class can be used as a dictionary. In addition, this class exposes
    /// utility methods to map from word/character space to token space.
    /// </summary>
    public class BatchEncoding : Dictionary<string, ICollection<int>> {

        internal int NumTruncatedTokens { get; set; }
        internal int Length { get; set; }

        public List<int> InputIds {
            get {
                TryGetValue("input_ids", out var inputIds);
                return inputIds as List<int>;
            }
        }
        public bool PrependBatchAxis { get; set;}
        public int? NSequences { get; set; }

        public BatchEncoding() : base() { }

        public BatchEncoding(
            Dictionary<string, ICollection<int>> data,
            object encoding = null, // dummy, EmcodingFast implemented in the `tokenizers` library, which is not yet used
            bool prependBatchAxis = false,
            int? nSequences = null) : base(data)
        {
            PrependBatchAxis = prependBatchAxis;
            NSequences = nSequences;
        }

        public void Merge(Dictionary<string, ICollection<int>> dict2) {
            foreach (var kvp in dict2) {
                if (ContainsKey(kvp.Key)) {
                    foreach (var value in kvp.Value) {
                        this[kvp.Key].Add(value);
                    }
                } else {
                    this[kvp.Key] = new List<int>(kvp.Value);
                }
            }
        }
    }
}