using System;
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

        public bool? DoLowerCase { get; set; }

        private Trie _tokensTrie;
        private Dictionary<string, int> AddedTokensEncoder;
        private Dictionary<int, AddedToken> AddedTokensDecoder;

        protected override void Initialize(
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
            _tokensTrie = new Trie();

            // init `AddedTokensDecoder` if child class did not
            if (AddedTokensDecoder == null) {
                AddedTokensDecoder = new Dictionary<int, AddedToken>();
            }

            // if a `added_tokens_decoder` is passed, we are loading from a saved tokenizer, we overwrite 
            if (addedTokensDecoder != null) {
                foreach (var token in addedTokensDecoder) {
                    AddedTokensDecoder[token.Key] = token.Value;
                }
            }
            AddedTokensEncoder = addedTokensDecoder.ToDictionary(x => (string)x.Value, x => x.Key);

            // 4 init the parent class
            base.Initialize(
                vocab,
                merges,
                errors,
                bosToken,
                eosToken,
                unkToken,
                sepToken,
                padToken,
                clsToken,
                maskToken);

            // 4. If some of the special tokens are not part of the vocab, we add them, at the end.
            // the order of addition is the same as self.SPECIAL_TOKENS_ATTRIBUTES following `tokenizers`
            var tokensToAdd = AllSpecialTokensExtended
                .Where(token => !AddedTokensEncoder.ContainsKey(token))
                .ToList() as IList<Token>;

            AddTokens(tokensToAdd, specialTokens: true);
        }


        /// <summary>
        /// Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        /// it with indices starting from length of the current vocabulary.Special tokens are sometimes already in the
        /// vocab which is why they have to be handled specifically.
        /// </summary>
        private int AddTokens(IList<Token> newTokens, bool specialTokens = false) {
            int addedTokens = 0;

            if (newTokens == null) {
                return addedTokens;
            }

            Dictionary<string, int> currentVocab = GetVocab().ToDictionary(entry => entry.Key, entry => entry.Value);
            int newIdx = currentVocab.Count;


            for (int i = 0; i < newTokens.Count; i++) {
                Token t = newTokens[i];
                if (!(t is TokenString || t is AddedToken)) {
                    throw new ArgumentException($"Token {t} is not a string but a {t.GetType()}.");
                }

                if (t == "") {
                    continue;
                }

                if (t is TokenString) {
                    if (AddedTokensEncoder.ContainsKey(t)) {
                        continue;
                    } else {
                        bool isSpecial = AllSpecialTokens.Contains(t) || specialTokens;
                        t = new AddedToken(t.ToString(), rstrip: false, lstrip: false, normalized: !isSpecial, special: isSpecial);
                    }
                } else if (specialTokens) {
                    ((AddedToken)t).Special = true;
                }

                AddedToken token = t as AddedToken;

                // how does that make sense in the original code??
                //if (AddedTokensDecoder.ContainsKey(token)) {
                //    continue;
                //}

                if (!token.Special && token.Normalized && (DoLowerCase ?? false)) {
                    token.Content = token.Content.ToLower();
                }

                int tokenIndex;
                if (!currentVocab.ContainsKey(token.Content)) {
                    tokenIndex = newIdx + addedTokens;
                    currentVocab[token.Content] = tokenIndex;
                    addedTokens++;
                } else {
                    tokenIndex = currentVocab[token.Content];
                }

                if (token.Special && !AddedTokensEncoder.ContainsKey(token)) {
                    AdditionalSpecialTokens.Add(token);
                }

                AddedTokensDecoder[tokenIndex] = token;
                AddedTokensEncoder[token.Content] = tokenIndex;

                if (Verbose) {
                    Console.WriteLine($"Adding {t} to the vocabulary");
                }
            }

            UpdateTrie();
            return addedTokens;
        }

        private void UpdateTrie(List<string> uniqueNoSplitTokens = null) {
            foreach (var token in AddedTokensDecoder.Values) {
                if (!_tokensTrie.Tokens.Contains(token.Content)) {
                    _tokensTrie.Add(token.Content);
                }
            }

            if (uniqueNoSplitTokens != null) {
                foreach (var token in uniqueNoSplitTokens) {
                    if (!_tokensTrie.Tokens.Contains(token)) {
                        _tokensTrie.Add(token);
                    }
                }
            }
        }

    }
}
