using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Base class for all slow tokenizers.
    /// Handle all the shared methods for tokenization and special
    /// tokens as well as methods downloading/caching/loading
    /// pretrained tokenizers as well as adding tokens to the vocabulary.

    /// This class also contain the added tokens in a unified way on top
    /// of all tokenizers so we don't have to handle thespecific vocabulary
    /// augmentation methods of the various underlying dictionary structures
    /// (BPE, sentencepiece...).
    /// </summary>
    public class PreTrainedTokenizer : PreTrainedTokenizerBase {

        private Trie _tokensTrie;
        private Dictionary<string, int> _added_tokens_encoder;
        private Dictionary<int, AddedToken> _added_tokens_decoder;

        protected override void Initialize(
            Vocab vocab,
            string[] merges,
            string errors,
            AddedToken unkToken,
            AddedToken bosToken,
            AddedToken eosToken,
            AddedToken padToken,
            Dictionary<int, AddedToken> addedTokensDecoder = null)
        {
            _tokensTrie = new Trie();

            // init `_added_tokens_decoder` if child class did not
            if (_added_tokens_decoder == null) {
                _added_tokens_decoder = new Dictionary<int, AddedToken>();
            }

            // if a `added_tokens_decoder` is passed, we are loading from a saved tokenizer, we overwrite 
            if (addedTokensDecoder != null) {
                foreach (var token in addedTokensDecoder) {
                    _added_tokens_decoder[token.Key] = token.Value;
                }
            }
            _added_tokens_encoder = addedTokensDecoder.ToDictionary(x => (string)x.Value, x => x.Key);

            // 4 init the parent class
            base.Initialize(vocab, merges, errors, unkToken, bosToken, eosToken, padToken);


        }
    }
}
