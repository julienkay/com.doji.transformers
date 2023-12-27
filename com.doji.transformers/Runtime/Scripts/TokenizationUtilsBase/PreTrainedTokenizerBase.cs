using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Base classe common to both the slow and the fast tokenization classes.
    /// (host all the user fronting encoding methods)
    /// Special token mixing(host the special tokens logic) and BatchEncoding
    /// (wrap the dictionary of output with special method for the Fast tokenizers)
    /// </summary>
    public class PreTrainedTokenizerBase : ISpecialTokensMixin {
        public string bos_token { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string eos_token { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string unk_token { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string sep_token { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string pad_token { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string cls_token { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string mask_token { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public List<string> additional_special_tokens { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public bool verbose { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
    }
}