using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A dictionary that holds the output of <see cref="PreTrainedTokenizerBase.Encode"/>
    /// and <see cref="PreTrainedTokenizerBase.EncodePlus"/> methods
    /// (tokens, attention_masks, etc).
    /// </summary>
    public abstract class Encoding : Dictionary<string, object> {

        public abstract IEnumerable<int> InputIds { get; }

        internal int NumTruncatedTokens { get; set; }
        internal int Length { get; set; }

        public bool PrependBatchAxis { get; set;}
        public int? NSequences { get; set; }

        public Encoding() : base() { }
        protected Encoding(Dictionary<string, object> dict) : base(dict) { }
    }
}