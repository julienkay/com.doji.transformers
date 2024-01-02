using System;
using System.Collections.Generic;

namespace Doji.AI.Transformers {
    public interface ISpecialTokensMixin {

        public bool Verbose { get; set; }

        public int AddSpecialTokens(Dictionary<string, AddedToken> specialTokensDict, bool replaceAdditionalSpecialTokens = true);

        public int AddTokens(string newTokens);
        public int AddTokens(AddedToken newTokens);
        public int AddTokens(List<AddedToken> newTokens);

        public Token BosToken { get; set; }
        public Token EosToken { get; set; }
        public Token UnkToken { get; set; }
        public Token SepToken { get; set; }
        public Token PadToken { get; set; }
        public Token ClsToken { get; set; }
        public Token MaskToken { get; set; }

        public List<Token> AdditionalSpecialTokens { get; set; }

        public int? BosTokenID { get;}
        public int? EosTokenID { get; }
        public int? UnkTokenID { get; }
        public int? SepTokenID { get; }
        public int? PadTokenID { get; }
        public int PadTokenTypeID { get; }
        public int? ClsTokenID { get; }
        public int? MaskTokenID { get; }

        public List<int> AdditionalSpecialTokensIDs { get; set; }

        public Dictionary<string, AddedToken> SpecialTokensMap { get; }

        public Dictionary<string, Token> SpecialTokensMapExtended { get; }
        public List<Token> AllSpecialTokensExtended { get; }

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