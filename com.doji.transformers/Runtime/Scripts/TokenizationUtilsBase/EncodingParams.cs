namespace Doji.AI.Transformers {

    public abstract partial class PreTrainedTokenizerBase {

        protected readonly struct EncodingParams {
            public string Text                  { get; }
            public string TextPair              { get; }
            public string TextTarget            { get; }
            public string TextPairTarget        { get; }
            public bool AddSpecialTokens        { get; }
            public Padding Padding              { get; }
            public Truncation Truncation        { get; }
            public int? MaxLength               { get; }
            public int Stride                   { get; }
            public bool IsSplitIntoWords        { get; }
            public int? PadToMultipleOf         { get; }
            public bool? ReturnTokenTypeIds     { get; }
            public bool? ReturnAttentionMask    { get; }
            public bool ReturnOverflowingTokens { get; }
            public bool ReturnSpecialTokensMask { get; }
            public bool ReturnOffsetsMapping    { get; }
            public bool ReturnLength            { get; }
            public bool Verbose                 { get; }

            public EncodingParams(
                string text = null,
                string textPair = null,
                string textTarget = null,
                string textPairTarget = null,
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
                Text                    = text;
                TextPair                = textPair;
                TextTarget              = textTarget;
                TextPairTarget          = textPairTarget;
                AddSpecialTokens        = addSpecialTokens;
                Padding                 = padding;
                Truncation              = truncation;
                MaxLength               = maxLength;
                Stride                  = stride;
                IsSplitIntoWords        = isSplitIntoWords;
                PadToMultipleOf         = padToMultipleOf;
                ReturnTokenTypeIds      = returnTokenTypeIds;
                ReturnAttentionMask     = returnAttentionMask;
                ReturnOverflowingTokens = returnOverflowingTokens;
                ReturnSpecialTokensMask = returnSpecialTokensMask;
                ReturnOffsetsMapping    = returnOffsetsMapping;
                ReturnLength            = returnLength;
                Verbose                 = verbose;
            }
        }
    }
}