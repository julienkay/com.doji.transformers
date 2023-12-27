using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Base classe common to both the slow and the fast tokenization classes.
    /// (host all the user fronting encoding methods)
    /// Special token mixing(host the special tokens logic) and BatchEncoding
    /// (wrap the dictionary of output with special method for the Fast tokenizers)
    /// </summary>
    public class PreTrainedTokenizerBase : ISpecialTokensMixin {
        public string BosToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string EosToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string UnkToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string SepToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string PadToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string ClsToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string MaskToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public List<string> AdditionalSpecialTokens { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public bool Verbose { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
    }
}