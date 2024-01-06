using System;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    public abstract partial class PreTrainedTokenizerBase : ISpecialTokensMixin {
        public bool Verbose { get; set; }

        public Token BosToken { get; set; }
        public Token EosToken { get; set; }
        public Token UnkToken { get; set; }
        public Token SepToken { get; set; }
        public Token PadToken { get; set; }
        public Token ClsToken { get; set; }
        public Token MaskToken { get; set; }
        public List<Token> AdditionalSpecialTokens { get; set; }

        public int? BosTokenId {
            get {
                if (BosToken == null) return null;
                return ConvertTokensToIds(BosToken);
            }
            set {
                throw new NotImplementedException();
            }
        }

        public int? EosTokenId {
            get {
                if (EosToken == null)
                    return null;
                return ConvertTokensToIds(EosToken);
            }
            set {
                throw new NotImplementedException();
            }
        }

        public int? UnkTokenId {
            get {
                if (UnkToken == null)
                    return null;
                return ConvertTokensToIds(UnkToken);
            }
            set {
                throw new NotImplementedException();
            }
        }

        public int? SepTokenId {
            get {
                if (SepToken == null)
                    return null;
                return ConvertTokensToIds(SepToken);
            }
            set {
                throw new NotImplementedException();
            }
        }

        public int? PadTokenId {
            get {
                if (PadToken == null)
                    return null;
                return ConvertTokensToIds(PadToken);
            }
            set {
                throw new NotImplementedException();
            }
        }

        public int PadTokenTypeID { get; private set; }

        public int? ClsTokenId {
            get {
                if (ClsToken == null)
                    return null;
                return ConvertTokensToIds(ClsToken);
            }
            set {
                throw new NotImplementedException();
            }
        }

        public int? MaskTokenId {
            get {
                if (MaskToken == null)
                    return null;
                return ConvertTokensToIds(MaskToken);
            }
            set {
                throw new NotImplementedException();
            }
        }

        public List<int> AdditionalSpecialTokensIds { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        /// <summary>
        /// A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).
        /// Convert tokens of `tokenizers.AddedToken` type to string.
        /// </summary>
        public List<string> AllSpecialTokens {
            get {
                List<string> allToks = new List<string>();
                foreach (var s in AllSpecialTokensExtended) {
                    allToks.Add(s.ToString());
                }
                return allToks;
            }
        }

        /// <summary>
        /// List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        /// </summary>
        public List<int> AllSpecialIds {
            get {
                return ConvertTokensToIds(AllSpecialTokens);
            }
        }

        protected abstract List<int> ConvertTokensToIds(List<string> tokens);
        protected abstract int ConvertTokensToIds(string tokens);

        public int AddSpecialTokens(Dictionary<string, AddedToken> specialTokensDict, bool replaceAdditionalSpecialTokens = true) {
            throw new System.NotImplementedException();
        }

        public int AddTokens(string newTokens) {
            throw new System.NotImplementedException();
        }

        public int AddTokens(AddedToken newTokens) {
            throw new System.NotImplementedException();
        }

        public int AddTokens(List<AddedToken> newTokens) {
            throw new System.NotImplementedException();
        }

        Dictionary<string, AddedToken> ISpecialTokensMixin.SpecialTokensMap => throw new System.NotImplementedException();

        public Dictionary<string, Token> SpecialTokensMapExtended {
            get {
                return new Dictionary<string, Token> {
                    { "bos_token" , BosToken  },
                    { "eos_token" , EosToken  },
                    { "unk_token" , UnkToken  },
                    { "sep_token" , SepToken  },
                    { "pad_token" , PadToken  },
                    { "cls_token" , ClsToken  },
                    { "mask_token", MaskToken },
                };
            }
        }

        public List<Token> AllSpecialTokensExtended {
            get {
                List<Token> allTokens = new List<Token>();
                HashSet<string> seen = new HashSet<string>();

                foreach (var value in SpecialTokensMapExtended.Values) {
                    var tokenToAdd = value.ToString();
                    if (!seen.Contains(tokenToAdd)) {
                        allTokens.Add(value);
                        seen.Add(tokenToAdd);
                    }
                }

                var tokensToAdd = AdditionalSpecialTokens.Where(token => !seen.Contains(token));
                allTokens.AddRange(tokensToAdd);
                seen.UnionWith(tokensToAdd.Select(token => token.ToString()));

                return allTokens;
            }
        }

        public void InitializeSpecialTokensMixin(
            AddedToken bosToken = null,
            AddedToken eosToken = null,
            AddedToken unkToken = null,
            AddedToken sepToken = null,
            AddedToken padToken = null,
            AddedToken clsToken = null,
            AddedToken maskToken = null,
            List<Token> additionalSpecialTokens = null,
            bool verbose = false)
        {
            BosToken = bosToken;
            EosToken = eosToken;
            UnkToken = unkToken;
            SepToken = sepToken;
            PadToken = padToken;
            ClsToken = clsToken;
            MaskToken = maskToken;
            PadTokenTypeID = 0;
            Verbose= verbose;

            if (additionalSpecialTokens != null) {
                AdditionalSpecialTokens = additionalSpecialTokens;
            } else {
                AdditionalSpecialTokens = new List<Token>();
            }
            Verbose = Verbose;
        }
    }
}
