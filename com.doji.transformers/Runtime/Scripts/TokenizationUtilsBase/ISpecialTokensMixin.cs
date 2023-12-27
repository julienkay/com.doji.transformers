using System.Collections.Generic;

namespace Doji.AI.Transformers {
    public interface ISpecialTokensMixin {
        public string BosToken { get; set; }
        public string EosToken { get; set; }
        public string UnkToken { get; set; }
        public string SepToken { get; set; }
        public string PadToken { get; set; }
        public string ClsToken { get; set; }
        public string MaskToken { get; set; }
        public List<string> AdditionalSpecialTokens { get; set; }
        public bool Verbose { get; set; }

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