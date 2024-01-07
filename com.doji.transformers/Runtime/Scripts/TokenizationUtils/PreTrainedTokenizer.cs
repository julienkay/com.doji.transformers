using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

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
        public override bool Fast { get => false; }

        private Trie _tokensTrie;
        private Dictionary<string, int> AddedTokensEncoder;
        private Dictionary<int, AddedToken> AddedTokensDecoder;

        protected override void Initialize(
            int modelMaxLength = int.MaxValue,
            Side paddingSide = Side.Right,
            Side truncationSide = Side.Right,
            List<string> modelInputNames = null,
            bool cleanUpTokenizationSpaces = true,
            bool splitSpecialTokens = false,
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
            AddedTokensDecoder ??= new Dictionary<int, AddedToken>();

            // if a `added_tokens_decoder` is passed, we are loading from a saved tokenizer, we overwrite 
            if (addedTokensDecoder != null) {
                foreach (var token in addedTokensDecoder) {
                    AddedTokensDecoder[token.Key] = token.Value;
                }
            }
            AddedTokensEncoder = AddedTokensDecoder.ToDictionary(x => (string)x.Value, x => x.Key);

            // 4 init the parent class
            base.Initialize(
                modelMaxLength,
                paddingSide,
                truncationSide,
                modelInputNames,
                cleanUpTokenizationSpaces,
                splitSpecialTokens,
                bosToken: bosToken,
                eosToken: eosToken,
                unkToken: unkToken,
                sepToken: sepToken,
                padToken: padToken,
                clsToken: clsToken,
                maskToken: maskToken
            );

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

       protected override int NumSpecialTokensToAdd(bool pair = false) {
            List<int> TokenIds0 = new List<int>();
            List<int> TokenIds1 = new List<int>();

            return BuildInputsWithSpecialTokens(TokenIds0, pair ? TokenIds1 : null).Count;
        }

        protected override BatchEncoding EncodePlus(
            string text,
            string textPair = null,
            bool addSpecialTokens = true,
            Padding padding = Padding.None,
            Truncation truncation = Truncation.None,
            int? maxLength = null,
            int stride = 0,
            bool isSplitIntoWords = false,
            int? padToMultipleOf = null,
            bool? returnTokenTypeIds = null,
            bool? returnAttentionMask = null,
            bool returnOverflowingTokens = false,
            bool returnSpecialTokensMask = false,
            bool returnOffsetsMapping = false,
            bool returnLength = false,
            bool verbose = true)
        {
            if (returnOffsetsMapping) {
                throw new NotImplementedException(
                    "returnOffsetsMapping is not available with this tokenizer. " +
                    "This feature requires a tokenizer deriving from " +
                    "transformers.PreTrainedTokenizerFast, which has not been " +
                    "ported to C# yet."
                );
            }

            List<int> firstIds = GetInputIds(text);
            List<int> secondIds = textPair != null ? GetInputIds(textPair) : null;

            return PrepareForModel(firstIds, secondIds, addSpecialTokens, padding, truncation, maxLength, stride,
                        isSplitIntoWords, padToMultipleOf, returnTokenTypeIds,
                        returnAttentionMask, returnOverflowingTokens, returnSpecialTokensMask,
                        returnOffsetsMapping, returnLength, verbose);
        }

        /// <summary>
        /// Converts a string into a sequence of tokens, using the tokenizer.
        /// Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        /// (BPE/SentencePieces/WordPieces). Takes care of added tokens.
        /// </summary>
        protected override List<string> Tokenize(string text, string textPair = null) {
            bool splitSpecialTokens = SplitSpecialTokens;

            if (DoLowerCase == true) {
                // convert non-special tokens to lowercase. Might be super slow as well?
                List<string> escapedSpecialToks = AllSpecialTokens.Select(sTok => Regex.Escape(sTok)).ToList();
                escapedSpecialToks.AddRange(
                    AddedTokensDecoder.Values
                        .Where(sTok => !sTok.Special && sTok.Normalized)
                        .Select(sTok => Regex.Escape(sTok.Content))
                );
                string pattern = "(" + string.Join("|", escapedSpecialToks) + @")|(.+?)";
                text = Regex.Replace(text, pattern, m => m.Groups[1].Success ? m.Groups[1].Value : m.Groups[2].Value.ToLower());
            }

            List<string> noSplitToken;
            List<string> tokens;

            if (splitSpecialTokens) {
                noSplitToken = new List<string>();
                tokens = new List<string> { text };
            } else {
                noSplitToken = AddedTokensEncoder.Keys.ToList();
                tokens = _tokensTrie.Split(text).ToList();
            }

            for (int i = 0; i < tokens.Count; i++) {
                string token = tokens[i];
                if (noSplitToken.Contains(token)) {
                    AddedToken tokExtended = AddedTokensDecoder.TryGetValue(AddedTokensEncoder[token], out var value)
                        ? value
                        : null;
                    string left = i > 0 ? tokens[i - 1] : null;
                    string right = i < tokens.Count - 1 ? tokens[i + 1] : null;

                    if (tokExtended is AddedToken addedToken) {
                        if (addedToken.Rstrip && right != null) {
                            tokens[i + 1] = right.TrimStart();
                        }

                        if (addedToken.Lstrip && left != null) {
                            tokens[i - 1] = left.TrimEnd();
                        }

                        if (addedToken.SingleWord && left != null && left.EndsWith(" ")) {
                            tokens[i - 1] += token;
                            tokens[i] = "";
                        } else if (addedToken.SingleWord && right != null && right.StartsWith(" ")) {
                            tokens[i + 1] = token + tokens[i + 1];
                            tokens[i] = "";
                        }
                    } else {
                        throw new InvalidOperationException(
                            $"{tokExtended} cannot be tokenized because it was not properly added " +
                            $"to the tokenizer. This means that it is not an `AddedToken` but a {tokExtended?.GetType()}"
                        );
                    }
                }
            }

            List<string> tokenizedText = new List<string>();
            foreach (string token in tokens.Where(token => !string.IsNullOrEmpty(token))) {
                if (noSplitToken.Contains(token)) {
                    tokenizedText.Add(token);
                } else {
                    tokenizedText.AddRange(_Tokenize(token));
                }
            }

            return tokenizedText;
        }

        /// <summary>
        /// Converts a string into a sequence of tokens (string), using the tokenizer.
        /// Split in words for word-based vocabulary or sub-words for sub-word-based
        /// vocabularies (BPE/SentencePieces/WordPieces).
        /// Do NOT take care of added tokens.
        /// </summary>
        protected virtual List<string> _Tokenize(string text) {
            throw new NotImplementedException($"This tokenizer does not implement {nameof(_Tokenize)}");
        }

        private List<int> GetInputIds(string text) {
            //if (text is string) {
            var tokens = Tokenize(text);
            return ConvertTokensToIds(tokens);
            /*} else if (text is IEnumerable<string> && ((IEnumerable<string>)text).Any()) {
                if (isSplitIntoWords) {
                    var tokens = ((IEnumerable<string>)text)
                        .SelectMany(t => Tokenize(t, isSplitIntoWords))
                        .ToList();
                    return ConvertTokensToIds(tokens);
                } else {
                    return ConvertTokensToIds((IEnumerable<string>)text);
                }
            } else if (text is IEnumerable<int> && ((IEnumerable<int>)text).Any()) {
                return ((IEnumerable<int>)text).ToList();
            } else {
                if (isSplitIntoWords) {
                    throw new ArgumentException($"Input {text} is not valid. Should be a string or a list/tuple of strings when `isSplitIntoWords=true`.");
                } else {
                    throw new ArgumentException($"Input {text} is not valid. Should be a string, a list/tuple of strings, or a list/tuple of integers.");
                }
            }*/
        }

        public override List<int> GetSpecialTokensMask(List<int> tokenIds0, List<int> tokenIds1, bool alreadyHasSpecialTokens = false) {
            if (alreadyHasSpecialTokens) {
                if (tokenIds1 != null) {
                    throw new ArgumentException("You should not supply a second sequence if the provided sequence of " +
                                                "ids is already formatted with special tokens for the model.");
                }
                return base.GetSpecialTokensMask(tokenIds0, tokenIds1, true);
            }
            return Enumerable.Repeat(0, (tokenIds1?.Count ?? 0) + tokenIds0.Count).ToList();
        }

        /// <summary>
        /// Converts a sequence of tokens into sequence of ids using the vocabulary.
        /// </summary>
        protected override List<int> ConvertTokensToIds(List<string> tokens) {
            if (tokens == null) {
                return null;
            }

            List<int> ids = new List<int>();
            foreach (var token in tokens) {
                ids.Add(ConvertTokenToIdWithAddedVoc(token));
            }

            return ids;
        }

        /// <summary>
        /// Converts a token string into a single integer id using the vocabulary.
        /// </summary>
        protected override int ConvertTokensToIds(string tokens) {
            if (tokens == null) {
                return -1;
            }

            return ConvertTokenToIdWithAddedVoc(tokens);
        }

        private int ConvertTokenToIdWithAddedVoc(string token) {
            if (token == null) {
                return -1;
            }

            if (AddedTokensEncoder.ContainsKey(token)) {
                return AddedTokensEncoder[token];
            }

            return ConvertTokenToId(token);
        }

        protected virtual int ConvertTokenToId(string token) {
            throw new NotImplementedException($"This tokenizer does not implement {nameof(ConvertTokenToId)}");
        }
    }
}