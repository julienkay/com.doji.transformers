using System;
using System.Collections.Generic;

namespace Doji.AI.Transformers {
    public interface ISpecialTokensMixin {

        public bool Verbose { get; set; }

        public int AddSpecialTokens(Dictionary<string, AddedToken> specialTokensDict, bool replaceAdditionalSpecialTokens = true);

        public int AddTokens(string newTokens);
        public int AddTokens(AddedToken newTokens);
        public int AddTokens(List<AddedToken> newTokens);

        public string BosToken { get; set; }
        public string EosToken { get; set; }
        public string UnkToken { get; set; }
        public string SepToken { get; set; }
        public string PadToken { get; set; }
        public string ClsToken { get; set; }
        public string MaskToken { get; set; }

        public List<string> AdditionalSpecialTokens { get; set; }

        public int? BosTokenID { get;}
        public int? EosTokenID { get; }
        public int? UnkTokenID { get; }
        public int? SepTokenID { get; }
        public int? PadTokenID { get; }
        public int PadTokenTypeID { get; }
        public int? ClsTokenID { get; }
        public int? MaskTokenID { get; }

        public List<int> AdditionalSpecialTokensIDs { get; set; }

        public Dictionary<string, string> SpecialTokensMap();
        public Dictionary<string, List<string>> SpecialTokensMap2();
    
        public List<string> AllSpecialTokens { get; }
        public List<int> AllSpecialIDs { get; }

        /*public static List<string> SPECIAL_TOKENS_ATTRIBUTES = new List<string>() {
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens"
        };*/
    }
}