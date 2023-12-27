using System.Collections.Generic;

namespace Doji.AI.Transformers {
    public interface ISpecialTokensMixin {
        public string bos_token { get; set; }
        public string eos_token { get; set; }
        public string unk_token { get; set; }
        public string sep_token { get; set; }
        public string pad_token { get; set; }
        public string cls_token { get; set; }
        public string mask_token { get; set; }
        public List<string> additional_special_tokens { get; set; }
        public bool verbose { get; set; }

        public static List<string> SPECIAL_TOKENS_ATTRIBUTES = new List<string>() {
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens"
        };
    }
}