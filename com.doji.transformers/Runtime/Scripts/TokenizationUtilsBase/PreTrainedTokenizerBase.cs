using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Base classe common to both the slow and the fast tokenization classes.
    /// (host all the user fronting encoding methods)
    /// Special token mixing(host the special tokens logic) and BatchEncoding
    /// (wrap the dictionary of output with special method for the Fast tokenizers)
    /// </summary>
    public abstract partial class PreTrainedTokenizerBase {

        protected virtual void Initialize(
            Vocab vocab,
            string[] merges,
            string errors,
            AddedToken bosToken = null,
            AddedToken eosToken = null,
            AddedToken unkToken = null,
            AddedToken sepToken = null,
            AddedToken padToken = null,
            AddedToken clsToken = null,
            AddedToken maskToken = null,
            Dictionary<int, AddedToken> addedTokensDecoder = null)
        {

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
    }
}