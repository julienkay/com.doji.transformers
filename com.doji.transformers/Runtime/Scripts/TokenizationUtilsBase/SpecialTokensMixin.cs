using System.Collections.Generic;
using Unity.Sentis.Layers;

namespace Doji.AI.Transformers {

    public abstract partial class PreTrainedTokenizerBase : ISpecialTokensMixin {
        public bool Verbose { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string BosToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string EosToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string UnkToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string SepToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string PadToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string ClsToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public string MaskToken { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        public List<string> AdditionalSpecialTokens { get; set; }

        public int? BosTokenID => throw new System.NotImplementedException();

        public int? EosTokenID => throw new System.NotImplementedException();

        public int? UnkTokenID => throw new System.NotImplementedException();

        public int? SepTokenID => throw new System.NotImplementedException();

        public int? PadTokenID => throw new System.NotImplementedException();

        public int PadTokenTypeID { get; private set; }

        public int? ClsTokenID => throw new System.NotImplementedException();

        public int? MaskTokenID => throw new System.NotImplementedException();

        public List<int> AdditionalSpecialTokensIDs { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }

        public List<string> AllSpecialTokens => throw new System.NotImplementedException();

        public List<int> AllSpecialIDs => throw new System.NotImplementedException();

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

        public Dictionary<string, string> SpecialTokensMap() {
            throw new System.NotImplementedException();
        }

        public Dictionary<string, List<string>> SpecialTokensMap2() {
            throw new System.NotImplementedException();
        }

        public void InitializeSpecialTokensMixin(
            AddedToken bosToken = null,
            AddedToken eosToken = null,
            AddedToken unkToken = null,
            AddedToken sepToken = null,
            AddedToken padToken = null,
            AddedToken clsToken = null,
            AddedToken maskToken = null,
            List<string> additionalSpecialTokens = null,
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
                AdditionalSpecialTokens = new List<string>();
            }
            Verbose = Verbose;
        }
    }
}
