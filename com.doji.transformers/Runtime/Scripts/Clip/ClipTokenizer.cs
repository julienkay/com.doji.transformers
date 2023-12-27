using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A clip tokenizer to tokenize text.
    /// </summary>
    public class ClipTokenizer : PreTrainedTokenizer {

        /// <summary>
        /// Initializes a new clip tokenizer.
        /// </summary>
        public ClipTokenizer(
            string errors = "replace",
            string unk_token= "<|endoftext|>",
            string bos_token= "<|startoftext|>",
            string eos_token= "<|endoftext|>",
            string pad_token= "<|endoftext|>")
        {
            Initialize();
        }

        
        private void Initialize() {
            
        }
    }
}