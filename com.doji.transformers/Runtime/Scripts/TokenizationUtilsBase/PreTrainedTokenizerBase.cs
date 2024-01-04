using System;
using System.Collections.Generic;
using TextInput = System.String;
using PreTokenizedInput = System.Collections.Generic.List<string>;
using EncodedInput = System.Collections.Generic.List<string>;
using TextInputPair = System.Tuple<string, string>;
using PreTokenizedInputPair = System.Tuple<System.Collections.Generic.List<string>, System.Collections.Generic.List<string>>;
using EncodedInputPair = System.Tuple<System.Collections.Generic.List<int>, System.Collections.Generic.List<int>>;

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
    /// Special token mixing(host the special tokens logic) and BatchEncoding
    /// (wrap the dictionary of output with special method for the Fast tokenizers)
    /// </summary>
    public abstract partial class PreTrainedTokenizerBase {

        public int? ModelMaxLength { get; set; }
        public Side PaddingSide { get; set; }
        public Side TruncationSide { get; set; }

        public List<string> ModelInputNames = new List<string>() { "input_ids", "token_type_ids", "attention_mask" };
        public bool CleanUpTokenizationSpaces { get; set; }
        public bool SplitSpecialTokens { get; set; }
        public bool InTargetContextManager { get; set; }

        protected virtual void Initialize(
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
            ModelMaxLength = modelMaxLength;
            PaddingSide = paddingSide;
            TruncationSide = truncationSide;
            ModelInputNames = modelInputNames ?? new List<string>();
            CleanUpTokenizationSpaces = cleanUpTokenizationSpaces;
            SplitSpecialTokens = splitSpecialTokens;
            InTargetContextManager = false;

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
        /// <remarks>
        /// PreTrainedTokenizerBase.__call__
        /// </remarks>
        public BatchEncoding EncodePrompt(
            string text = null,
            string textPair = null,
            string textTarget = null,
            string textPairTarget = null,
            bool addSpecialTokens = true,
            Padding padding = Padding.None,
            Truncation truncation= Truncation.None,
            int? maxLength = null,
            int stride = 0,
            bool isSplitIntoWords = false,
            int? padToMultipleOf = null,
            bool? returnTokenTypesIds = null,
            bool? returnAttentionMask = null,
            bool returnOverflowingTokens = false,
            bool returnSpecialTokensMask = false,
            bool returnOffsetsMapping = false,
            bool returnLength = false,
            bool verbose = true)
        {
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
                encodings = EncodePromptOne(text, textPair);
            }

            if (textTarget != null) {
                SwitchToTargetMode();
                targetEncodings = EncodePromptOne(textTarget, textPairTarget);
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

        private BatchEncoding EncodePromptOne(
            string text,
            string textPair = null,
            bool addSpecialTokens = true,
            Padding padding = Padding.None,
            Truncation truncation = Truncation.None,
            int? maxLength = null,
            int stride = 0,
            bool isSplitIntoWords = false,
            int? padToMultipleOf = null,
            bool? returnTokenTypesIds = null,
            bool? returnAttentionMask = null,
            bool returnOverflowingTokens = false,
            bool returnSpecialTokensMask = false,
            bool returnOffsetsMapping = false,
            bool returnLength = false,
            bool verbose = true)
        {
            if (!IsValidTextInput(text)) {
                throw new ArgumentException("text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) " +
                                            "or `List<List[str]]` (batch of pretokenized examples).");
            }

            if (textPair != null && !IsValidTextInput(textPair)) {
                throw new ArgumentException("text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) " +
                                            "or `List<List[str]]` (batch of pretokenized examples).");
            }

            // only non-batched version implemented for now
            return EncodePlus(text, textPair, addSpecialTokens, padding, truncation, maxLength, stride,
                        isSplitIntoWords, padToMultipleOf, returnTokenTypesIds,
                        returnAttentionMask, returnOverflowingTokens, returnSpecialTokensMask,
                        returnOffsetsMapping, returnLength, verbose);
        }

        protected virtual BatchEncoding EncodePlus(
            string text,
            string textPair = null,
            bool addSpecialTokens = true,
            Padding padding = Padding.None,
            Truncation truncation = Truncation.None,
            int? maxLength = null,
            int stride = 0,
            bool isSplitIntoWords = false,
            int? padToMultipleOf = null,
            bool? returnTokenTypesIds = null,
            bool? returnAttentionMask = null,
            bool returnOverflowingTokens = false,
            bool returnSpecialTokensMask = false,
            bool returnOffsetsMapping = false,
            bool returnLength = false,
            bool verbose = true)
        {
            throw new NotImplementedException($"This tokenizer does not implement {nameof(EncodePlus)}");
        }

        /// <summary>
        /// Converts a string into a sequence of tokens, replacing unknown tokens with the `unk_token`.
        /// </summary>
        protected virtual List<string> Tokenize(string text, string textPair = null) {
            throw new NotImplementedException($"This tokenizer does not implement {nameof(Tokenize)}");
        }

        /// <summary>
        /// Input type checking.
        /// </summary>
        /// <remarks>
        /// Sketchy port. TODO: validate me!
        /// </remarks>
        private static bool IsValidTextInput(object t) {
            if (t is string) {
                // Strings are fine
                return true;
            } else if (t is IList<object> list) {
                // List is fine as long as it is...
                if (list.Count == 0) {
                    // ... empty
                    return true;
                } else if (list[0] is string) {
                    // ... list of strings
                    return true;
                } else if (list[0] is IList<string> innerList) {
                    // ... list with an empty list or with a list of strings
                    return innerList.Count == 0 || innerList[0] is string;
                } else if (list[0] is Tuple<object, object> tuple) {
                    return tuple.Item1 is string;
                } else {
                    return false;
                }
            } else {
                return false;
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