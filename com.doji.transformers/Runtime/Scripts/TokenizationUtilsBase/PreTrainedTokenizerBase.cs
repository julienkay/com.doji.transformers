using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

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

    public enum Side { Right, Left }

    /// <summary>
    /// Base class common to both the slow and the fast tokenization classes.
    /// (host all the user fronting encoding methods)
    /// Special token mixing (host the special tokens logic) implemented via
    /// <see cref="ISpecialTokensMixin"/> 
    /// BatchEncoding (wrap the dictionary of output with special method for
    /// the Fast tokenizers) via <see cref="BatchEncoding"/>
    /// </summary>
    public abstract partial class PreTrainedTokenizerBase {

        public int? ModelMaxLength { get; set; }
        public Side PaddingSide { get; set; }
        public Side TruncationSide { get; set; }

        public List<string> ModelInputNames { get; private set; } = new List<string>() { "input_ids", "token_type_ids", "attention_mask" };
        public bool CleanUpTokenizationSpaces { get; set; }
        public bool SplitSpecialTokens { get; set; }
        public HashSet<string> DeprecationWarnings { get; set; }
        public bool InTargetContextManager { get; set; }

        public abstract bool Fast { get; }

        protected virtual void Initialize(
            TokenizerConfig config,
            Side paddingSide = Side.Right,
            Side truncationSide = Side.Right,
            List<string> modelInputNames = null,
            bool cleanUpTokenizationSpaces = true,
            bool splitSpecialTokens = false,
            Dictionary<int, AddedToken> addedTokensDecoder = null)
        {
            ModelMaxLength = config.ModelMaxLength;
            PaddingSide = paddingSide;
            TruncationSide = truncationSide;
            ModelInputNames = modelInputNames ?? ModelInputNames;
            CleanUpTokenizationSpaces = cleanUpTokenizationSpaces;
            SplitSpecialTokens = splitSpecialTokens;
            DeprecationWarnings = new HashSet<string> { };
            InTargetContextManager = false;

            InitializeSpecialTokensMixin(config);
        }

        /// <summary>
        /// Returns the number of added tokens when encoding a sequence with special tokens.
        /// </summary>
        /// <param name="pair">Whether the number of added tokens should be computed in the
        /// case of a sequence pair or a single sequence.</param>
        protected virtual int NumSpecialTokensToAdd(bool pair = false) {
            throw new NotImplementedException($"This tokenizer does not implement {nameof(NumSpecialTokensToAdd)}");
        }

        /// <summary>
        /// Find the correct padding/truncation strategy
        /// </summary>
        /// <remarks>
        /// Tried to remove old/deprecated behavior from original code.
        /// </remarks>
        private void GetPaddingTruncationStrategies(ref Padding padding, ref Truncation truncation, ref int? maxLength) {
            throw new NotImplementedException();
        }

        /// <inheritdoc cref="Encode{T}(T, T, string, string, bool, Padding, Truncation, int?, int, int?, bool?, bool?, bool, bool, bool, bool)"/>
        public BatchEncoding Encode(
            Input text,
            Input textPair = null,
            string textTarget = null,
            string textPairTarget = null,
            bool addSpecialTokens = true,
            Padding padding = Padding.None,
            Truncation truncation = Truncation.None,
            int? maxLength = null,
            int stride = 0,
            int? padToMultipleOf = null,
            bool? returnTokenTypeIds = null,
            bool? returnAttentionMask = null,
            bool returnOverflowingTokens = false,
            bool returnSpecialTokensMask = false,
            bool returnOffsetsMapping = false,
            bool returnLength = false)
        {
            return Encode<Input>(text, textPair, textTarget, textPairTarget, addSpecialTokens,
                padding, truncation, maxLength, stride, padToMultipleOf, returnTokenTypeIds,
                returnAttentionMask, returnOverflowingTokens, returnSpecialTokensMask,
                returnOffsetsMapping, returnLength);
        }

        /// <summary>
        /// Main method to tokenize and prepare a prompt for the model.
        /// </summary>
        /// <remarks>
        /// PreTrainedTokenizerBase.__call__
        /// </remarks>
        public BatchEncoding Encode<T>(
            T text,
            T textPair = null,
            string textTarget = null,
            string textPairTarget = null,
            bool addSpecialTokens = true,
            Padding padding = Padding.None,
            Truncation truncation= Truncation.None,
            int? maxLength = null,
            int stride = 0,
            int? padToMultipleOf = null,
            bool? returnTokenTypeIds = null,
            bool? returnAttentionMask = null,
            bool returnOverflowingTokens = false,
            bool returnSpecialTokensMask = false,
            bool returnOffsetsMapping = false,
            bool returnLength = false) where T : Input
        {
            EncodingParams args = new EncodingParams(
                text, textPair, textTarget, textPairTarget, addSpecialTokens, padding,
                truncation, maxLength, stride, padToMultipleOf, returnTokenTypeIds,
                returnAttentionMask, returnOverflowingTokens, returnSpecialTokensMask,
                returnOffsetsMapping, returnLength
            );

            if (text == null && textTarget == null) {
                throw new ArgumentException("You need to specify either `text` or `textTarget`.");
            }

            BatchEncoding encodings = null;
            BatchEncoding targetEncodings = null;

            if (text != null) {
                // The context manager will send the inputs as normal texts and not textTarget,
                // but we shouldn't change the input mode in this case.
                if (!InTargetContextManager) {
                    SwitchToInputMode();
                }
                encodings = EncodePromptOne(args);
            }

            if (textTarget != null) {
                SwitchToTargetMode();
                targetEncodings = EncodePromptOne(args);
            }

            // Leave back tokenizer in input mode
            SwitchToInputMode();

            if (textTarget == null) {
                return encodings;
            } else if (text == null) {
                return targetEncodings;
            } else {
                encodings["labels"] = targetEncodings["input_ids"];
                return encodings;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <remarks>
        /// PretrainedTokenizerBase._call_one
        /// </remarks>
        private BatchEncoding EncodePromptOne(EncodingParams args) {
            // in contrast to the original implementation, type validity of 'text'
            // and 'textPair' can be assumed here, given our typed representations

            Input text = args.Text;
            Input textPair = args.TextPair;
            bool isBatched = args.Text is BatchInput || args.Text is PretokenizedBatchInput;

            if (isBatched) {
                if (textPair is not null && !textPair.IsBatch()) {
                    throw new ArgumentException("when tokenizing batches of text, `text_pair` must be " +
                        "a list or tuple with the same Length as `text`.");
                }
                if (textPair is not null && ((text as IBatchInput).Sequence.Count) != (textPair as IBatchInput).Sequence.Count) {
                    throw new ArgumentException("when tokenizing batches of text, `text_pair` must be " +
                        "a list or tuple with the same Length as `text`.");
                }
                if (textPair is not null) {
                    throw new NotImplementedException("Usage of textPair is not yet implemented.");
                }
                return BatchEncodePlus(args);
            } else {
                return EncodePlus(args);
            }
        }

        /// <summary>
        /// Tokenize and prepare for the model a sequence or a pair of sequences.
        /// </summary>
        /// <remarks>
        /// PretrainedTokenizerBase._encode_plus
        /// </remarks>
        protected virtual BatchEncoding EncodePlus(EncodingParams args) {
            throw new NotImplementedException($"This tokenizer does not implement {nameof(EncodePlus)}");
        }

        /// <summary>
        ///  Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
        /// </summary>
        /// <remarks>
        /// PretrainedTokenizerBase._batch_encode_plus
        /// </remarks>
        protected virtual BatchEncoding BatchEncodePlus(EncodingParams args) {
            throw new NotImplementedException($"This tokenizer does not implement {nameof(BatchEncodePlus)}");
        }

        /// <summary>
        /// Converts a string into a sequence of tokens, replacing unknown tokens with the `unk_token`.
        /// </summary>
        public virtual List<string> Tokenize(string text) {
            throw new NotImplementedException($"This tokenizer does not implement {nameof(Tokenize)}");
        }

        /// <summary>
        /// Pad a single encoded input or a batch of encoded inputs up to a
        /// predefined length or to the max sequence length in the batch.
        /// Padding side (left/right) padding token ids are defined at the
        /// tokenizer level(with `self.padding_side`, `self.pad_token_id`
        /// and `self.pad_token_type_id`).
        /// Please note that with a fast tokenizer, using the `__call__`
        /// method is faster than using a method to encode the text
        /// followed by a call to the `pad` method to get a padded encoding.
        /// </summary>
        public BatchEncoding Pad(
            BatchEncoding encodedInputs,
            Padding padding = Padding.None,
            int? maxLength = null,
            int? padToMultipleOf = null,
            bool? returnAttentionMask = null,
            string returnTensors = null)
        {
            if (Fast) {
                string warningKey = "Asking-to-pad-a-fast-tokenizer";
                if (!DeprecationWarnings.Contains(warningKey)) {
                    Log.Info($"You're using a {GetType()}. Please note that with a fast tokenizer, " +
                        $" using the `__call__` method is faster than using a method to encode the text followed by a call " +
                        $"to the `pad` method to get a padded encoding.");
                }

                DeprecationWarnings.Add(warningKey);
            }

            // The model's main input name, usually `input_ids`, has be passed for padding
            if (!encodedInputs.ContainsKey(ModelInputNames[0])) {
                throw new ArgumentException($"You should supply an encoding or a list of encodings to this method that includes" +
                    $"{ModelInputNames[0]}, but you provided {string.Join(", ", encodedInputs.Keys)}");
            }

            IList requiredInput = encodedInputs[ModelInputNames[0]] as IList;

            if (requiredInput == null || (requiredInput is ICollection collection && collection.Count == 0)) {
                if (returnAttentionMask == true) {
                    encodedInputs["attention_mask"] = new List<int>();
                }

                return new BatchEncoding(encodedInputs);
            }

            if (requiredInput != null && requiredInput[0] is not ICollection) {
                encodedInputs = _Pad(
                    encodedInputs,
                    maxLength,
                    padding,
                    padToMultipleOf,
                    returnAttentionMask
                );

                return new BatchEncoding(encodedInputs);
            }

            throw new InvalidOperationException("Unexpected requiredInput");
            // not sure if we ever need the below if we only care about inference

            /*int batchSize = requiredInput.Count;
            if (!encodedInputs.Values.All(v => (v as ICollection<object>)?.Count == batchSize)) {
                throw new InvalidOperationException("Some items in the output dictionary have a different batch size than others.");
            }

            if (padding == Padding.Longest) {
                maxLength = (requiredInput as List<object>).Max(inputs => (inputs as ICollection<object>)?.Count ?? 0);
                padding = Padding.MaxLength;
            }

            BatchEncoding batchOutputs = new BatchEncoding();
            for (int i = 0; i < batchSize; i++) {
                var inputs = encodedInputs.ToDictionary(kvp => kvp.Key, kvp => ((IList<object>)kvp.Value)[i]);
                var outputs = _Pad(
                    inputs,
                    maxLength,
                    padding,
                    padToMultipleOf,
                    returnAttentionMask
                );

                batchOutputs.Merge(outputs);
            }

            return new BatchEncoding(batchOutputs);
            */
        }

        private BatchEncoding _Pad(
            BatchEncoding encodedInputs,
            int? maxLength,
            Padding padding,
            int? padToMultipleOf,
            bool? returnAttentionMask)
        {
            if (returnAttentionMask == null) {
                returnAttentionMask = encodedInputs.ContainsKey("attention_mask");
            }

            List<int> requiredInput = encodedInputs[ModelInputNames[0]] as List<int>;

            if (padding == Padding.Longest) {
                maxLength = requiredInput.Count;
            }

            if (maxLength != null && padToMultipleOf != null && (maxLength % padToMultipleOf != 0)) {
                maxLength = ((maxLength.Value / padToMultipleOf.Value) + 1) * padToMultipleOf.Value;
            }

            bool needsToBePadded = padding != Padding.None && requiredInput.Count != maxLength;

            // Initialize attention mask if not present.
            if (returnAttentionMask == true && !encodedInputs.ContainsKey("attention_mask")) {
                encodedInputs["attention_mask"] = Enumerable.Repeat(1, requiredInput.Count).ToList();
            }

            if (needsToBePadded) {
                int difference = maxLength.Value - requiredInput.Count;

                if (PaddingSide == Side.Right) {
                    if (returnAttentionMask == true) {
                        var atnMasks = (List<int>)encodedInputs["attention_mask"];
                        encodedInputs["attention_mask"] = atnMasks.Concat(Enumerable.Repeat(0, difference)).ToList();
                    }
                    if (encodedInputs.ContainsKey("token_type_ids")) {
                        var ids = (List<int>)encodedInputs["token_type_ids"];
                        encodedInputs["token_type_ids"] = ids.Concat(Enumerable.Repeat(PadTokenTypeID, difference)).ToList();
                    }
                    if (encodedInputs.ContainsKey("special_tokens_mask")) {
                        var special = (List<int>)encodedInputs["special_tokens_mask"];
                        encodedInputs["special_tokens_mask"] = special.Concat(Enumerable.Repeat(1, difference)).ToList();
                    }
                    encodedInputs[ModelInputNames[0]] = requiredInput.Concat(Enumerable.Repeat(PadTokenId.Value, difference)).ToList();
                } else if (PaddingSide == Side.Left) {
                    if (returnAttentionMask == true) {
                        var atnMasks = (List<int>)encodedInputs["attention_mask"];
                        encodedInputs["attention_mask"] = Enumerable.Repeat(0, difference).Concat(atnMasks).ToList();
                    }
                    if (encodedInputs.ContainsKey("token_type_ids")) {
                        var ids = (List<int>)encodedInputs["token_type_ids"];
                        encodedInputs["token_type_ids"] = Enumerable.Repeat(PadTokenTypeID, difference).Concat(ids).ToList();
                    }
                    if (encodedInputs.ContainsKey("special_tokens_mask")) {
                        var special = (List<int>)encodedInputs["special_tokens_mask"];
                        encodedInputs["special_tokens_mask"] = Enumerable.Repeat(1, difference).Concat(special).ToList();
                    }
                    encodedInputs[ModelInputNames[0]] = Enumerable.Repeat(PadTokenId.Value, difference).Concat(requiredInput).ToList();
                } else {
                    throw new ArgumentException("Invalid padding strategy: " + PaddingSide);
                }
            }

            return encodedInputs;
        }

        /// <summary>
        /// Create the token type IDs corresponding to the sequences passed.
        /// Should be overridden in a subclass if the model has a special way of building those.
        /// </summary>
        /// <param name="TokenIds0"></param>
        /// <param name="TokenIds1"></param>
        /// <returns></returns>
        public virtual List<int> CreateTokenTypeIdsFromSequences(List<int> TokenIds0, List<int> TokenIds1 = null) {
            if (TokenIds1 == null) {
                return new List<int>(new int[TokenIds0.Count]);
            }

            List<int> tokenTypeIds = new List<int>(TokenIds0.Count + TokenIds1.Count);

            for (int i = 0; i < TokenIds0.Count; i++) {
                tokenTypeIds.Add(0);
            }

            for (int i = 0; i < TokenIds1.Count; i++) {
                tokenTypeIds.Add(1);
            }

            return tokenTypeIds;
        }

        /// <summary>
        /// Build model inputs from a sequence or a pair of sequence
        /// for sequence classification tasks by concatenating and
        /// adding special tokens.
        /// This implementation does not add special tokens and this method should be overridden in a subclass.
        /// </summary>
        protected virtual List<int> BuildInputsWithSpecialTokens(List<int> TokenIds0, List<int> TokenIds1 = null) {
            if (TokenIds1 == null) {
                return TokenIds0;
            }

            List<int> result = new List<int>(TokenIds0);
            result.AddRange(TokenIds1);
            return result;
        }

        /// <summary>
        /// Prepares a sequence of input id, or a pair of sequences of inputs ids
        /// so that it can be used by the model. It adds special tokens, truncates
        /// sequences if overflowing while taking into account the special tokens and
        /// manages a moving window(with user defined stride) for overflowing tokens.
        /// </summary>
        /// <remarks>
        /// Please Note, for *pair_ids* different than `None` and
        /// * truncation_strategy = longest_first * or `True`, it is not possible to return
        /// overflowing tokens. Such a combination of arguments will raise an error.
        /// </remarks>
        protected BatchEncoding PrepareForModel(
            EncodingParams args,
            List<int> ids,
            List<int> pairIds = null,
            bool prependBatchAxis = false)
        {
            bool pair = pairIds != null;
            int lenIds = ids.Count;
            int lenPairIds = pair ? pairIds.Count : 0;
            bool? returnTokenTypeIds = args.ReturnTokenTypeIds;
            bool? returnAttentionMask = args.ReturnAttentionMask;

            if (args.ReturnTokenTypeIds == true && !args.AddSpecialTokens) {
                throw new ArgumentException(
                    "Asking to return token_type_ids while setting add_special_tokens to False " +
                    "results in an undefined behavior. Please set add_special_tokens to True or " +
                    "set return_token_type_ids to None."
                );
            }

            if (args.ReturnOverflowingTokens
                && args.Truncation == Truncation.LongestFirst
                && pairIds != null) {
                throw new ArgumentException(
                    "Not possible to return overflowing tokens for a pair of sequences with the " +
                    "`longest_first`. Please select another truncation strategy than `longest_first`, " +
                    "for instance `only_second` or `only_first`."
                );
            }

            // Load from model defaults
            if (!returnTokenTypeIds.HasValue) {
                returnTokenTypeIds = ModelInputNames.Contains("token_type_ids");
            }

            if (!returnAttentionMask.HasValue) {
                returnAttentionMask = ModelInputNames.Contains("attention_mask");
            }

            BatchEncoding encodedInputs = new BatchEncoding();

            // Compute the total size of the returned encodings
            int totalLen = lenIds + lenPairIds + (args.AddSpecialTokens ? NumSpecialTokensToAdd(pair) : 0);

            // Truncation: Handle max sequence length
            List<int> overflowingTokens = null;
            if (args.Truncation != Truncation.None && args.MaxLength > 0 && totalLen > args.MaxLength) {
                Tuple<List<int>, List<int>, List<int>> truncationResult = TruncateSequences(
                    ids,
                    pairIds,
                    totalLen - args.MaxLength.Value,
                    args.Truncation,
                    args.Stride
                );

                ids = truncationResult.Item1;
                pairIds = truncationResult.Item2;
                overflowingTokens = truncationResult.Item3;
            }

            if (args.ReturnOverflowingTokens) {
                System.Diagnostics.Debug.Assert(args.MaxLength.HasValue);
                encodedInputs["overflowing_tokens"] = overflowingTokens;
                encodedInputs.NumTruncatedTokens = totalLen - args.MaxLength.Value;
            }

            // Add special tokens
            List<int> sequence;
            List<int> tokenTypeIds;
            if (args.AddSpecialTokens) {
                sequence = BuildInputsWithSpecialTokens(ids, pairIds);
                tokenTypeIds = CreateTokenTypeIdsFromSequences(ids, pairIds);
            } else {
                sequence = pair ? (List<int>)ids.Concat(pairIds) : new List<int>(ids);
                tokenTypeIds = new List<int>(Enumerable.Repeat(0, ids.Count));
                if (pair) {
                    tokenTypeIds.AddRange(Enumerable.Repeat(0, pairIds.Count));
                }
            }

            // build output dictionary
            encodedInputs["input_ids"] = sequence;
            if (returnTokenTypeIds == true) {
                encodedInputs["token_type_ids"] = tokenTypeIds;
            } 
            if (args.ReturnSpecialTokensMask) {
                if (args.AddSpecialTokens) {
                    encodedInputs["special_tokens_mask"] = GetSpecialTokensMask(ids, pairIds); ;
                } else {
                    encodedInputs["special_tokens_mask"] = Enumerable.Repeat(0, sequence.Count).ToList();
                }
            }

            // Check lengths
            EventualWarnAboutTooLongSequence(encodedInputs["input_ids"] as List<int>, args.MaxLength);

            // Padding
            if (args.Padding != Padding.None || returnAttentionMask == true) {
                encodedInputs = Pad(
                    encodedInputs,
                    maxLength: args.MaxLength,
                    padding: args.Padding,
                    padToMultipleOf: args.PadToMultipleOf,
                    returnAttentionMask: returnAttentionMask
                );
            }

            if (args.ReturnLength) {
                encodedInputs.Length = (encodedInputs["input_ids"] as ICollection).Count;
            }

            BatchEncoding batchOutputs = new BatchEncoding(
                encodedInputs, prependBatchAxis: prependBatchAxis
            );
            return batchOutputs;
        }

        public Tuple<List<int>, List<int>, List<int>> TruncateSequences(
            List<int> ids,
            List<int> pairIds = null,
            int numTokensToRemove = 0,
            Truncation Truncation = Truncation.None,
            int stride = 0)
        {
            if (numTokensToRemove <= 0) {
                return Tuple.Create(ids, pairIds, new List<int>());
            }

            Truncation strategy = Truncation;
            List<int> overflowingTokens = new List<int>();

            if (strategy == Truncation.OnlyFirst || (strategy == Truncation.LongestFirst && pairIds == null)) {
                if (ids.Count > numTokensToRemove) {
                    int windowLen = Math.Min(ids.Count, stride + numTokensToRemove);

                    if (TruncationSide == Side.Left) {
                        overflowingTokens = ids.GetRange(0, windowLen);
                        ids = ids.GetRange(numTokensToRemove, ids.Count - numTokensToRemove);
                    } else if (TruncationSide == Side.Right) {
                        overflowingTokens = ids.GetRange(ids.Count - windowLen, windowLen);
                        ids = ids.GetRange(0, ids.Count - numTokensToRemove);
                    } else {
                        throw new ArgumentException($"Invalid truncation strategy: {TruncationSide}, use 'left' or 'right'.");
                    }
                } else {
                    string errorMsg = $"We need to remove {numTokensToRemove} to truncate the input " +
                                        $"but the first sequence has a length {ids.Count}. ";

                    if (strategy == Truncation.OnlyFirst) {
                        errorMsg += $"Please select another truncation strategy than {strategy}, " +
                                    "for instance 'longest_first' or 'only_second'.";
                    }

                    Log.Error(errorMsg);
                }
            } else if (strategy == Truncation.LongestFirst) {
                Log.Info("Be aware, overflowing tokens are not returned for the setting you have chosen, " +
                    "i.e. sequence pairs with the '{Truncation.LongestFirst}' truncation strategy. " +
                    "So the returned list will always be empty even if some tokens have been removed.");

                for (int i = 0; i < numTokensToRemove; i++) {
                    if (pairIds == null || ids.Count > pairIds.Count) {
                        if (TruncationSide == Side.Right) {
                            ids.RemoveAt(ids.Count - 1);
                        } else if (TruncationSide == Side.Left) {
                            ids.RemoveAt(0);
                        } else {
                            throw new ArgumentException("Invalid truncation strategy:" + TruncationSide);
                        }
                    } else {
                        if (TruncationSide == Side.Right) {
                            pairIds.RemoveAt(pairIds.Count - 1);
                        } else if (TruncationSide == Side.Left) {
                            pairIds.RemoveAt(0);
                        } else {
                            throw new ArgumentException("Invalid truncation strategy:" + TruncationSide);
                        }
                    }
                }
            } else if (strategy == Truncation.OnlySecond && pairIds != null) {
                if (pairIds.Count > numTokensToRemove) {
                    int windowLen = Math.Min(pairIds.Count, stride + numTokensToRemove);

                    if (TruncationSide == Side.Right) {
                        overflowingTokens = pairIds.GetRange(pairIds.Count - windowLen, windowLen);
                        pairIds = pairIds.GetRange(0, pairIds.Count - numTokensToRemove);
                    } else if (TruncationSide == Side.Left) {
                        overflowingTokens = pairIds.GetRange(0, windowLen);
                        pairIds = pairIds.GetRange(numTokensToRemove, pairIds.Count - numTokensToRemove);
                    } else {
                        throw new ArgumentException("Invalid truncation strategy:" + TruncationSide);
                    }
                } else {
                    Log.Info($"We need to remove {numTokensToRemove} to truncate the input " +
                        $"but the second sequence has a length {pairIds.Count}. " +
                        $"Please select another truncation strategy than {strategy}, " +
                        $"for instance 'longest_first' or 'only_first'.");
                }
            }

            return Tuple.Create(ids, pairIds, overflowingTokens);
        }

        /// <summary>
        /// Retrieves sequence ids from a token list that has no special tokens added.
        /// This method is called when adding special tokens using the tokenizer
        /// `prepare_for_model` or `encode_plus` methods.
        /// </summary>
        /// <returns>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</returns>
        public virtual List<int> GetSpecialTokensMask(List<int> tokenIds0, List<int> tokenIds1, bool alreadyHasSpecialTokens = false) {
            if (!(alreadyHasSpecialTokens && tokenIds1 == null)) {
                throw new InvalidOperationException("You cannot use alreadyHasSpecialTokens=false with this tokenizer. " +
                                                    "Please use a slow tokenizer to activate this argument. " +
                                                    "Or set returnSpecialTokensMask=true when calling the encoding method " +
                                                    "to get the special tokens mask in any tokenizer.");
            }

            List<int> specialTokensMask = tokenIds0.Select(token => AllSpecialIds.Contains(token) ? 1 : 0).ToList();

            return specialTokensMask;
        }

        /// <summary>
        /// Depending on the input and internal state we might trigger a warning
        /// about a sequence that is too long for its corresponding model
        /// </summary>
        public void EventualWarnAboutTooLongSequence(List<int> ids, int? maxLength) {
            if (!maxLength.HasValue && ids.Count > ModelMaxLength) {
                string warningKey = "sequence-length-is-longer-than-the-specified-maximum";

                if (!DeprecationWarnings.Contains(warningKey)) {
                    Log.Info($"Token indices sequence length is longer than the specified maximum sequence length " +
                        $"for this model ({ids.Count} > {ModelMaxLength}). Running this sequence through the model will result" +
                        $"in indexing errors");
                }

                DeprecationWarnings.Add(warningKey);
            }
        }

        public virtual void SwitchToInputMode() { }
        public virtual void SwitchToTargetMode() { }

        /// <summary>
        /// Returns the vocabulary as a dictionary of token to index.
        /// `tokenizer.get_vocab()[token]` is equivalent to
        /// `tokenizer.convert_tokens_to_ids(token)` when `token`
        /// is in the vocab.
        /// </summary>
        protected virtual Dictionary<string, int> GetVocab() {
            throw new NotImplementedException($"This tokenizer does not implement {nameof(GetVocab)}");
        }
    }
}