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
    public class BatchEncoding : Dictionary<string, object> {

        public List<int> InputIds {
            get {
                TryGetValue("input_ids", out object inputIds);
                return inputIds as List<int>;
            }
        }
        public bool PrependBatchAxis { get; set;}
        public int? NSequences { get; set; }

        public BatchEncoding(
            Dictionary<string, object> data = null,
            object encoding = null, // dummy, EmcodingFast implemented in the `tokenizers` library, which is not yet used
            bool prependBatchAxis = false,
            int? nSequences = null) : base(data)
        {
            PrependBatchAxis = prependBatchAxis;
            NSequences = nSequences;
        }
    }
}