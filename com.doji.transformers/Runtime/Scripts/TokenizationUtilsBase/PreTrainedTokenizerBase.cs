using System;
using System.Collections.Generic;
using Unity.Sentis.Layers;

namespace Doji.AI.Transformers {

    public enum Padding {

        /// <summary>
        /// no padding is applied
        /// </summary>
        None,

        /// <summary>
        /// pad to the longest sequence in the batch
        /// (no padding is applied if you only provide a single sequence).
        /// </summary>
        Longest,

        /// <summary>
        /// pad to a length specified by the max_length argument or the maximum length
        /// accepted by the model if no max_length is provided (max_length=None).
        /// Padding will still be applied if you only provide a single sequence.
        /// </summary>
        MaxLength
    }

    public enum Truncation {

        /// <summary>
        /// no truncation is applied
        /// </summary>
        None,

        /// <summary>
        /// truncate to a maximum length specified by the max_length argument or the maximum
        /// length accepted by the model if no max_length is provided (max_length=None).
        /// This will truncate token by token, removing a token from the longest sequence
        /// in the pair until the proper length is reached.
        /// </summary>
        LongestFirst,

        /// <summary>
        /// truncate to a maximum length specified by the max_length argument or the maximum
        /// length accepted by the model if no max_length is provided (max_length=None).
        /// This will only truncate the second sentence of a pair if a pair of sequences
        /// (or a batch of pairs of sequences) is provided.
        /// </summary>
        OnlySecond,

        /// <summary>
        /// truncate to a maximum length specified by the max_length argument or the maximum
        /// length accepted by the model if no max_length is provided (max_length=None).
        /// This will only truncate the first sentence of a pair if a pair of sequences
        /// (or a batch of pairs of sequences) is provided.
        /// </summary>
        OnlyFirst
    }

    /// <summary>
    /// Base classe common to both the slow and the fast tokenization classes.
    /// (host all the user fronting encoding methods)
    /// Special token mixing(host the special tokens logic) and BatchEncoding
    /// (wrap the dictionary of output with special method for the Fast tokenizers)
    /// </summary>
    public abstract partial class PreTrainedTokenizerBase {

        public int? ModelMaxLength { get; set; }

        protected virtual void Initialize(
            int? modelMaxLength = null,
            AddedToken bosToken = null,
            AddedToken eosToken = null,
            AddedToken unkToken = null,
            AddedToken sepToken = null,
            AddedToken padToken = null,
            AddedToken clsToken = null,
            AddedToken maskToken = null,
            Dictionary<int, AddedToken> addedTokensDecoder = null)
        {
            ModelMaxLength = modelMaxLength ?? int.MaxValue;

            //...

            InitializeSpecialTokensMixin(
                bosToken,
                eosToken,
                unkToken,
                sepToken,
                padToken,
                clsToken,
                maskToken
            );
        }

        /// <summary>
        /// Main method to tokenize and prepare a prompt for the model.
        /// </summary>
        public int EncodePrompt(
            string text,
            Padding padding = Padding.None,
            Truncation truncation= Truncation.None,
            int? maxLengh = null,
            int stride = 0,
            bool isSplitIntoWords = false,
            int? padToMultipleOf = null,
            bool? return_token_type_ids = null,
            bool? return_attention_mask = null,
            bool return_overflowing_tokens = false,
            bool return_special_tokens_mask = false,
            bool return_offsets_mapping = false,
            bool return_length = false,
            bool verbose = true)
        {
            return 0;

        }

        /// <summary>
        /// Returns the vocabulary as a dictionary of token to index.
        /// `tokenizer.get_vocab()[token]` is equivalent to
        /// `tokenizer.convert_tokens_to_ids(token)` when `token`
        /// is in the vocab.
        /// </summary>
        protected virtual Dictionary<string, int> GetVocab() {
            throw new NotImplementedException();
        }
    }
}