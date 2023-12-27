using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A clip tokenizer to tokenize text.
    /// </summary>
    public class ClipTokenizer : PreTrainedTokenizer {

        private BasicTokenizer _nlp;

        /// <summary>
        /// Initializes a new clip tokenizer.
        /// </summary>
        public ClipTokenizer(
            string errors = "replace",
            string unk_token = "<|endoftext|>",
            string bos_token = "<|startoftext|>",
            string eos_token = "<|endoftext|>",
            string pad_token = "<|endoftext|>")
        {
            BosToken = new AddedToken(bos_token);
            EosToken = new AddedToken(eos_token);
            UnkToken = new AddedToken(unk_token);
            Initialize();
        }

        public ClipTokenizer(
            string errors,
            AddedToken unk_token,
            AddedToken bos_token,
            AddedToken eos_token,
            AddedToken pad_token)
        {
            BosToken = bos_token;
            EosToken = eos_token;
            UnkToken = unk_token;
            Initialize();
        }

        private void Initialize() {
            // ftfy.fix_text not implemented, using BasicTokenizer instead
            _nlp = new BasicTokenizer();

        }
    }
}