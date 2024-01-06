using System;
using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Holds the output of <see cref="PreTrainedTokenizerBase.EncodePrompt"/>
    /// and <see cref="PreTrainedTokenizerBase.EncodePlus"/> methods
    /// (tokens, attention_masks, etc).
    /// 
    /// This class can be used as a dictionary. In addition, this class exposes
    /// utility methods to map from word/character space to token space.
    /// </summary>
    public class BatchEncoding : Dictionary<string, object> {

        public BatchEncoding(
            Dictionary<string, object> data = null,
            object encoding = null, // dummy, EmcodingFast implemented in the `tokenizers` library.
            bool prependBatchAxis = false,
            int? nSequences = null) : base(data)
        {
        
        }

        /*public static implicit operator BatchEncoding(Dictionary<string, object> data) {
            return new BatchEncoding(data);
        }*/
    }
}