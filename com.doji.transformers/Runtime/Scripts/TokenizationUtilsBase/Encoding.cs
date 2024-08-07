using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A dictionary that holds the output of <see cref="PreTrainedTokenizerBase.Encode"/>
    /// and <see cref="PreTrainedTokenizerBase.EncodePlus"/> methods
    /// (tokens, attention_masks, etc).
    /// </summary>
    public abstract class Encoding : Dictionary<string, object> {

        /// <summary>
        /// Indices of input sequence tokens in the vocabulary.
        /// These are numerical representations of tokens that will be used as the main input by most models.
        /// </summary>
        public abstract IEnumerable<int> InputIds { get; }

        /// <summary>
        /// Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        /// Only valid if 'returnAttentionMask = true' was passed to Encode() method.
        /// </summary>
        public abstract IEnumerable<int> AttentionMask { get; }

        /// <summary>
        /// Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:
        /// Only valid if 'returnTokenTypeIds = true' was passed to Encode() method.
        /// </summary>
        public abstract IEnumerable<int> TokenTypeIds { get; }

        internal int NumTruncatedTokens { get; set; }
        internal int Length { get; set; }

        public bool PrependBatchAxis { get; set;}
        public int? NSequences { get; set; }

        public Encoding() : base() { }
        protected Encoding(Dictionary<string, object> dict) : base(dict) { }
    }
}