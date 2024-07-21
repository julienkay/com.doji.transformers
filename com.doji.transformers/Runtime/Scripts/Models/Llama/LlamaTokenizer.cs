using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace Doji.AI.Transformers {

    public class LlamaTokenizer : PreTrainedTokenizer {

        private static string SPIECE_UNDERLINE = '\u2581'.ToString();

        private string VocabFilePath { get; set; }
        private bool AddBosToken { get; set; }
        private bool AddEosToken { get; set; }
        private bool AddPrefixSpace { get; set; }
        private bool Legacy { get; set; }

        private SentencePieceBpe _spModel;

        public LlamaTokenizer(
            string vocabFilePath,
            TokenizerConfig config = null,
            Side paddingSide = Side.Right,
            Side truncationSide = Side.Right,
            List<string> modelInputNames = null,
            bool cleanUpTokenizationSpaces = false,
            bool splitSpecialTokens = false,
            bool addPrefixSpace = true) : base(paddingSide, truncationSide, modelInputNames, cleanUpTokenizationSpaces, splitSpecialTokens)
        {
            config ??= new TokenizerConfig();
            config.UnkToken ??= new AddedToken("<unk>");
            config.BosToken ??= new AddedToken("<s>");
            config.EosToken ??= new AddedToken("</s>");
            config.AddPrefixSpace = AddPrefixSpace = addPrefixSpace;
            config.AddBosToken = AddBosToken = true;
            config.AddEosToken = AddEosToken = false;

            if (config.Legacy == null) {
                Log.Warning(
                    $"You are using the default legacy behaviour of the {nameof(LlamaTokenizer)}. This is" +
                    " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you." +
                    " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it" +
                    " means, and thoroughly read the reason why this was added as explained in" +
                    " https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file" +
                    " you can ignore this message"
                );
            }
            Legacy = config.Legacy ?? true;
            VocabFilePath = vocabFilePath;
            _spModel = GetSpmProcessor();

            base.Initialize(config);
        }

        // Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
        private SentencePieceBpe GetSpmProcessor() {
            if (Legacy) {
                using FileStream f = new FileStream(VocabFilePath, FileMode.Open, FileAccess.Read);
                return Tokenizer.CreateLlama(f, AddBosToken, AddEosToken) as SentencePieceBpe;
            }
            
            var tokenizer = Sentencepiece.SentencepieceUtils.LoadModifiedProto(VocabFilePath, AddBosToken, AddEosToken, addDummyPrefix: false);
            UnityEngine.Debug.Log(tokenizer.AddDummyPrefix);
            return tokenizer;
        }

        public int VocabSize {
            get {
                return _spModel.Vocab.Count;
            }
        }

        protected override Dictionary<string, int> GetVocab() {
            var vocab = new Dictionary<string, int>();
            for (int i = 0; i < VocabSize; i++) {
                string token = ConvertIdsToTokens(i);
                vocab[token] = i;
            }
            foreach(var kvp in AddedTokensEncoder) {
                vocab[kvp.Key] = kvp.Value;
            }
            return vocab;
        }


        // Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
        public override List<string> Tokenize(string text) {
            if (Legacy || string.IsNullOrEmpty(text)) {
                return base.Tokenize(text);
            }

            text = Regex.Replace(text, SPIECE_UNDERLINE, " ");
            if (AddPrefixSpace) {
                text = SPIECE_UNDERLINE + text;
            }

            var tokens = base.Tokenize(text);

            if (tokens.Count > 1 && tokens[0] == SPIECE_UNDERLINE && AllSpecialTokens.Contains(tokens[1])) {
                tokens = tokens.Skip(1).ToList();
            }
            return tokens;
        }

        // Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
        /// <summary>
        /// Returns a tokenized string.
        /// </summary>
        protected override List<string> _Tokenize(string text) {
            var tokens = _spModel.Encode(text, out string normalized);
            if (Legacy || (!(text.StartsWith(SPIECE_UNDERLINE) || text.StartsWith(" ")))) {
                return ToStringList(tokens);
            }
            tokens = _spModel.Encode(UnkToken + text, out string normalized2);
            if (tokens.Count >= UnkToken.Content.Length) {
                return ToStringList(tokens.Skip(UnkToken.Content.Length));
            } else {
                return ToStringList(tokens);
            }
        }

        private static List<string> ToStringList(IEnumerable<Microsoft.ML.Tokenizers.Token> tokens) {
            List<string> result = new List<string>();
            foreach (var token in tokens) {
                result.Add(token.Value);
            }
            return result;
        }

        protected override int ConvertTokenToId(string token) {
            return _spModel.MapTokenToId(token).Value;
        }

        protected override string ConvertIdToToken(int index) {
            return _spModel.MapIdToToken(index);
        }

        protected override List<int> BuildInputsWithSpecialTokens(List<int> TokenIds0, List<int> TokenIds1 = null) {
            List<int> bosTokenIds = AddBosToken ? new List<int> { BosTokenId.Value } : new();
            List<int> eosTokenIds = AddEosToken ? new List<int> { EosTokenId.Value } : new();

            var output = bosTokenIds.Concat(TokenIds0).Concat(eosTokenIds);

            if (TokenIds1 != null) {
                return output.Concat(bosTokenIds).Concat(TokenIds1).Concat(eosTokenIds).ToList();
            }

            return output.ToList();
        }

        public override List<int> GetSpecialTokensMask(List<int> tokenIds0, List<int> tokenIds1, bool alreadyHasSpecialTokens = false) {
            if (alreadyHasSpecialTokens) {
                return base.GetSpecialTokensMask(tokenIds0, tokenIds1, true);
            }
            List<int> bosTokenId = AddBosToken ? new List<int> { 1 } : new();
            List<int> eosTokenId = AddEosToken ? new List<int> { 1 } : new();

            if (tokenIds1 == null) {
                return bosTokenId.Concat(new int[tokenIds0.Count]).Concat(eosTokenId).ToList();

            }
            return bosTokenId
                .Concat(new int[tokenIds0.Count])
                .Concat(eosTokenId)
                .Concat(bosTokenId)
                .Concat(new int[tokenIds1.Count])
                .Concat(eosTokenId).ToList();
        }

        /// <summary>
        /// Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        /// An ALBERT sequence pair mask has the following format:
        /// ```
        /// 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        /// | first sequence    | second sequence |
        /// ```
        /// if token_ids_1 is null, only returns the first portion of the mask (0s).
        /// </summary>
        public override List<int> CreateTokenTypeIdsFromSequences(List<int> TokenIds0, List<int> TokenIds1 = null) {
            List<int> bosTokenId = AddBosToken ? new List<int> { BosTokenId.Value } : new();
            List<int> eosTokenId = AddEosToken ? new List<int> { EosTokenId.Value } : new();

            IEnumerable<int> output = new int[bosTokenId.Count + TokenIds0.Count + eosTokenId.Count];

            if (TokenIds1 != null) {
                output = output.Concat(Enumerable.Repeat(1, bosTokenId.Count + TokenIds1.Count + eosTokenId.Count));
            }

            return output.ToList();
        }
    }
}