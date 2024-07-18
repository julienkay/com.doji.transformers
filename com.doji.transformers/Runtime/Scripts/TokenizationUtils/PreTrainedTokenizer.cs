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
    /// 
    /// This class also contain the added tokens in a unified way on top
    /// of all tokenizers so we don't have to handle the specific vocabulary
    /// augmentation methods of the various underlying dictionary structures
    /// (BPE, sentencepiece...).
    /// </summary>
    public class PreTrainedTokenizer : PreTrainedTokenizerBase {

        public bool? DoLowerCase { get; set; }
        public override bool Fast { get => false; }

        private Trie _tokensTrie;
        private Dictionary<string, int> AddedTokensEncoder;
        private Dictionary<int, AddedToken> AddedTokensDecoder;

        public PreTrainedTokenizer(
            Side paddingSide = Side.Right,
            Side truncationSide = Side.Right,
            List<string> modelInputNames = null,
            bool cleanUpTokenizationSpaces = true,
            bool splitSpecialTokens = false) : base() { }

        protected override void Initialize(TokenizerConfig config) {
            _tokensTrie = new Trie();

            // init `AddedTokensDecoder` if child class did not
            AddedTokensDecoder ??= new Dictionary<int, AddedToken>();

            // if a `added_tokens_decoder` is passed, we are loading from a saved tokenizer, we overwrite 
            /*if (addedTokensDecoder != null) {
                foreach (var token in addedTokensDecoder) {
                    AddedTokensDecoder[token.Key] = token.Value;
                }
            }*/
            AddedTokensEncoder = AddedTokensDecoder.ToDictionary(x => (string)x.Value, x => x.Key);

            // 4 init the parent class
            base.Initialize(config);

            // 4. If some of the special tokens are not part of the vocab, we add them, at the end.
            // the order of addition is the same as self.SPECIAL_TOKENS_ATTRIBUTES following `tokenizers`
            var tokensToAdd = AllSpecialTokensExtended
                .Where(token => !AddedTokensEncoder.ContainsKey(token))
                .ToList() as IList<Token>;

            AddTokens(tokensToAdd, specialTokens: true);
        }


        /// <summary>
        /// Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        /// it with indices starting from length of the current vocabulary. Special tokens are sometimes already in the
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

                if (!token.Special && token.Normalized && DoLowerCase == true) {
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

                Log.Info($"Adding {t} to the vocabulary");
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

        /// <inheritdoc cref="PreTrainedTokenizerBase.EncodePlus(PreTrainedTokenizerBase.EncodingParams)"/>
        protected override Encoding EncodePlus(EncodingParams args) {
            if (args.ReturnOffsetsMapping) {
                throw new NotImplementedException(
                    "returnOffsetsMapping is not available with this tokenizer. " +
                    "This feature requires a tokenizer deriving from " +
                    "transformers.PreTrainedTokenizerFast, which has not been " +
                    "ported to C# yet."
                );
            }

            List<int> firstIds = GetInputIds(args.Text);
            List<int> secondIds = args.TextPair != null ? GetInputIds(args.TextPair) : null;

            return PrepareForModel(args, firstIds, secondIds);
        }

        /// <inheritdoc cref="PreTrainedTokenizerBase.BatchEncodePlus(EncodingParams)"/>
        protected override Encoding BatchEncodePlus(EncodingParams args) {
            if (args.ReturnOffsetsMapping) {
                throw new NotImplementedException(
                    "returnOffsetsMapping is not available with this tokenizer. " +
                    "This feature requires a tokenizer deriving from " +
                    "transformers.PreTrainedTokenizerFast, which has not been " +
                    "ported to C# yet."
                );
            }

            System.Diagnostics.Debug.Assert(args.Text is BatchInput || args.Text is PretokenizedBatchInput);
            var batch = args.Text as IBatchInput;
            bool isPretokenized = args.Text is PretokenizedBatchInput;

            // get input ids from all sequences and flatten them into a single list
            List<(List<int> first, List<int> second)> inputIds = new List<(List<int>, List<int>)>();
            foreach(var input in batch.Sequence) {
                List<int> firstIds;
                if (isPretokenized) {
                    firstIds = ConvertTokensToIds(input as List<string>);
                } else {
                    firstIds = ConvertTokensToIds(Tokenize(input as string));
                }
                List<int> secondIds = args.TextPair != null ? GetInputIdsBatch(args.TextPair) : null;
                inputIds.Add((firstIds, secondIds));
            }

            return BatchPrepareForModel(args, inputIds);
        }

        /// <summary>
        /// Converts a string into a sequence of tokens, using the tokenizer.
        /// Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        /// (BPE/SentencePieces/WordPieces). Takes care of added tokens.
        /// </summary>
        public override List<string> Tokenize(string text) {
            bool splitSpecialTokens = SplitSpecialTokens;

            text = PrepareForTokenization(text);
            if (this is not ClipTokenizer) {
                // original code passes args dynamically and specifically checks if all have been used
                // TODO: need to figure out how to best implement that
                throw new NotImplementedException();
            }

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

        /// <summary>
        /// Get input ids for a single input input.
        /// At this point <paramref name="text"/> is either of type <see cref="SingleInput"/>
        /// or <see cref="PretokenizedSingleInput"/>
        /// </summary>
        private List<int> GetInputIds(Input text) {
            List<string> tokens;
            if (text is SingleInput input) {
                tokens = Tokenize(input.Text);
            } else if (text is PretokenizedSingleInput pretokenizedInput) {
                tokens = pretokenizedInput.PretokenizedText;
            } else {
                throw new ArgumentException($"Input {text} is not valid. Unexpected type '{text.GetType()}'.");
            }
            return ConvertTokensToIds(tokens);
        }

        /// <summary>
        /// Get input ids for a sequence/batch input input.
        /// At this point <paramref name="text"/> is either of type <see cref="BatchInput"/>
        /// or <see cref="PretokenizedBatchInput"/>
        /// </summary>
        private List<int> GetInputIdsBatch(Input text) {
            List<string> tokens;
            if (text is BatchInput input) {
                tokens = new List<string>();
                foreach (string token in input.Sequence) {
                    tokens.AddRange(Tokenize(token));
                }
            } else if (text is PretokenizedBatchInput pretokenizedInput) {
                tokens = new List<string>();
                foreach (List<string> t in pretokenizedInput.Sequence) {
                    tokens.AddRange(t);
                }
            } else {
                throw new ArgumentException($"Input {text} is not valid. Unexpected type '{text.GetType()}'.");
            }
            return ConvertTokensToIds(tokens);
        }

        /// <summary>
        /// Prepares a sequence of input id, or a pair of sequences of inputs ids so that
        /// it can be used by the model. It adds special tokens, truncates sequences if
        /// overflowing while taking into account the special tokens and manages a moving
        /// window (with user defined stride) for overflowing tokens.
        /// </summary>
        private Encoding BatchPrepareForModel(
            EncodingParams args,
            List<(List<int> firstIds, List<int> secondIds)> batchIdPairs)
        {
            EncodingParams batchArgs = new EncodingParams() {
                AddSpecialTokens        =  args.AddSpecialTokens,
                Padding                 =  Padding.None, // we pad in batch afterward
                Truncation              =  args.Truncation,
                MaxLength               =  args.MaxLength,
                Stride                  =  args.Stride,
                PadToMultipleOf         =  null, // we pad in batch afterward
                ReturnTokenTypeIds      =  args.ReturnTokenTypeIds,
                ReturnAttentionMask     =  false, // we pad in batch afterward
                ReturnOverflowingTokens =  args.ReturnOverflowingTokens,
                ReturnSpecialTokensMask =  args.ReturnSpecialTokensMask,
                ReturnOffsetsMapping    =  args.ReturnOffsetsMapping,
                ReturnLength            =  args.ReturnLength
            };
            
            BatchEncoding batchOutputs = new BatchEncoding();
            foreach ((List<int> firstIds, List<int> secondIds) in batchIdPairs) {
                InputEncoding outputs = PrepareForModel(batchArgs, firstIds, secondIds, prependBatchAxis: false);
                batchOutputs.Append(outputs);
            }

            Pad(batchOutputs,
                args.Padding,
                args.MaxLength,
                args.PadToMultipleOf,
                args.ReturnAttentionMask
            );

            return batchOutputs;
        }

        protected virtual string PrepareForTokenization(string text) {
            return text;
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