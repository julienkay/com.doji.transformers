namespace Doji.AI.Transformers {

    public abstract partial class PreTrainedTokenizerBase {

        protected struct EncodingParams {
            public Input Text                   { get; set; }
            public Input TextPair               { get; set; }
            public string TextTarget            { get; set; }
            public string TextPairTarget        { get; set; }
            public bool AddSpecialTokens        { get; set; }
            public Padding Padding              { get; set; }
            public Truncation Truncation        { get; set; }
            public int? MaxLength               { get; set; }
            public int Stride                   { get; set; }
            public bool IsSplitIntoWords        { get; set; }
            public int? PadToMultipleOf         { get; set; }
            public bool? ReturnTokenTypeIds     { get; set; }
            public bool? ReturnAttentionMask    { get; set; }
            public bool ReturnOverflowingTokens { get; set; }
            public bool ReturnSpecialTokensMask { get; set; }
            public bool ReturnOffsetsMapping    { get; set; }
            public bool ReturnLength            { get; set; }

            public EncodingParams(
                Input text = null,
                Input textPair = null,
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
                bool returnLength = false)
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
            }
        }
    }
}