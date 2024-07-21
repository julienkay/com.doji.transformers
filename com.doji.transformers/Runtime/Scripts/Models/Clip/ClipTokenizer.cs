using static Doji.AI.Transformers.TokenizationUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A clip tokenizer to tokenize text.
    /// Based on byte-level Byte-Pair-Encoding.
    /// </summary>
    public class ClipTokenizer : PreTrainedTokenizer {

        public int VocabSize {
            get {
                return Vocab.Encoder.Count;
            }
        }
        private Vocab Vocab { get; set; }

        private BasicTokenizer _nlp;
        private object _fixText;
        private Dictionary<int, char> _byteEncoder;
        private Dictionary<char, int> _byteDecoder;
        private Dictionary<Tuple<string, string>, int> _bpeRanks;
        private Dictionary<string, string> _cache;
        private Regex _pat;

        /// <summary>
        /// Initializes a new clip tokenizer.
        /// </summary>
        public ClipTokenizer(
            Vocab vocab,
            string merges,
            TokenizerConfig config = null,
            Side paddingSide = Side.Right,
            Side truncationSide = Side.Right,
            List<string> modelInputNames = null,
            bool cleanUpTokenizationSpaces = true,
            bool splitSpecialTokens = false) : base(paddingSide, truncationSide, modelInputNames, cleanUpTokenizationSpaces, splitSpecialTokens)
        {
            config ??= new TokenizerConfig();
            config.UnkToken ??= new AddedToken("<|endoftext|>");
            config.BosToken ??= new AddedToken("<|startoftext|>");
            config.EosToken ??= new AddedToken("<|endoftext|>");
            config.PadToken ??= new AddedToken("<|endoftext|>");

            // TODO: BasicTokenizer only a fallback, implement ftfy.fix_text instead?
            _nlp = new BasicTokenizer();
            _fixText = null;
            Vocab = vocab;
            _byteEncoder = BytesToUnicode();
            _byteDecoder = _byteEncoder.ToDictionary(x => x.Value, x => x.Key);

            // Extract relevant lines and split them into tuples
            List<Tuple<string, string>> bpeMerges = merges
                .Trim()
                .Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None)
                .Skip(1)
                .Take(49152 - 256 - 2 + 1)
                .Select(line => line.Split())
                .Select(parts => Tuple.Create(parts[0], parts[1]))
                .ToList();

            // Create the dictionary with bpe ranks
            _bpeRanks = bpeMerges
                .Select((merge, index) => new { Merge = merge, Index = index })
                .ToDictionary(item => item.Merge, item => item.Index);

            _cache = new Dictionary<string, string>() {
                { "<|startoftext|>", "<|startoftext|>" },
                { "<|endoftext|>", "<|endoftext|>" }
            };

            _pat = new Regex(
                @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
                RegexOptions.IgnoreCase
            );

            base.Initialize(config);
        }

        protected override Dictionary<string, int> GetVocab() {
            return Vocab.Encoder;
        }

        protected override List<int> BuildInputsWithSpecialTokens(List<int> TokenIds0, List<int> TokenIds1 = null) {
            List<int> bosToken = new List<int> { BosTokenId.Value };
            List<int> eosToken = new List<int> { EosTokenId.Value };

            if (TokenIds1 == null) {
                return CombineLists(bosToken, TokenIds0, eosToken);
            }

            return CombineLists(bosToken, TokenIds0, eosToken, eosToken, TokenIds1, eosToken);

        }

        private List<int> CombineLists(params List<int>[] lists) {
            List<int> result = new List<int>();
            foreach (var list in lists) {
                result.AddRange(list);
            }
            return result;
        }

        /// <summary>
        /// Return set of symbol pairs in a word.
        /// Word is represented as tuple of symbols (symbols being variable-Length strings).
        /// </summary>
        private static HashSet<Tuple<string, string>> GetPairs(List<string> word) {
            var pairs = new HashSet<Tuple<string, string>>();
            string prevChar = word[0];

            for (int i = 1; i < word.Count; i++) {
                pairs.Add(new Tuple<string, string>(prevChar, word[i]));
                prevChar = word[i];
            }

            return pairs;
        }

        /// <summary>
        /// represents a special token id
        /// </summary>
        private static readonly List<int> ONE = new List<int>() { 1 };

        public override List<int> GetSpecialTokensMask(List<int> tokenIds0, List<int> tokenIds1, bool alreadyHasSpecialTokens = false) {
            if (alreadyHasSpecialTokens) {
                return base.GetSpecialTokensMask(tokenIds0, tokenIds1, true);
            }

            if (tokenIds1 == null) {
                return ONE.Concat(new int[tokenIds0.Count]).Concat(ONE).ToList();
            }

            return ONE.Concat(new int[tokenIds0.Count]).Concat(ONE).Concat(ONE).Concat(new int[tokenIds1.Count]).Concat(ONE).ToList();
        }

        /// <summary>
        /// Create a mask from the two sequences passed.
        /// CLIP does not make use of token type ids,
        /// therefore a list of zeros is returned.
        /// </summary>
        public override List<int> CreateTokenTypeIdsFromSequences(List<int> TokenIds0, List<int> TokenIds1 = null) {
            int n;
            if (TokenIds1 == null) {
                // len(bos_token + token_ids_0 + eos_token)
                n = TokenIds0.Count + 2;
            } else {
                // len(bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token)
                n = TokenIds0.Count + TokenIds1.Count + 4;
            }
            return Enumerable.Repeat(0, n).ToList();
        }

        private string bpe(string token) {
            if (_cache.ContainsKey(token)) {
                return _cache[token];
            }
            List<string> word = token.Take(token.Length - 1).Select(c => c.ToString()).ToList();
            word.Add(token.Last() + "</w>");
            var pairs = GetPairs(word);
            if (pairs.Count == 0) {
                return token + "</w>";
            }

            while (true) {
                var bigram = pairs.OrderBy(pair => _bpeRanks.GetValueOrDefault(pair, int.MaxValue)).First();
                if (!_bpeRanks.ContainsKey(bigram)) {
                    break;
                }

                var first = bigram.Item1;
                var second = bigram.Item2;
                var newWord = new List<string>();
                var i = 0;

                while (i < word.Count) {
                    int j = word.IndexOf(first, i);
                    if (j == -1) {
                        newWord.AddRange(Slice(word, i));
                        break;
                    }

                    newWord.AddRange(Slice(word, i, j));
                    i = j;

                    if (word[i] == first && i < word.Count - 1 && word[i + 1] == second) {
                        newWord.Add(first + second);
                        i += 2;
                    } else {
                        newWord.Add(word[i]);
                        i += 1;
                    }

                }

                word = new List<string>(newWord);
                if (word.Count == 1) {
                    break;
                } else {
                    pairs = GetPairs(word);
                }
            }

            string result = string.Join(" ", word);
            _cache[token] = result;
            return result;
        }

        private List<string> Slice(List<string> word, int start) {
            if (start < 0 || start >= word.Count) {
                throw new ArgumentException("start index out of range.");
            }
            return word.GetRange(start, word.Count - start);
        }

        private List<string> Slice(List<string> word, int start, int end) {
            if (start < 0 || end >= word.Count || start > end) {
                throw new ArgumentException("start or end index out of range.");
            }
            return word.GetRange(start, end - start);
        }

        /// <summary>
        /// Tokenize a string.
        /// </summary>
        protected override List<string> _Tokenize(string text) {
            List<string> bpeTokens = new List<string>();
            if (_fixText == null) {
                text = string.Join(" ", _nlp.Tokenize(text));
            } else {
                //text = Regex.Replace(_fixText(text).ToLower(), @"\s+", " ").Trim();
                throw new NotImplementedException();
            }

            foreach (var match in _pat.Matches(text)) {
                // Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
                var token = string.Join("", System.Text.Encoding.UTF8.GetBytes(match.ToString()).Select(b => _byteEncoder[b]));
                bpeTokens.AddRange(bpe(token).Split(' '));
            }

            return bpeTokens;
        }

        /// <summary>
        /// Converts a token into an id using the vocab.
        /// </summary>
        protected override int ConvertTokenToId(string token) {
            var encoder = Vocab.Encoder;
            if (encoder.TryGetValue(token, out int id)) {
                return id;
            } else {
                return encoder.GetValueOrDefault(UnkToken);
            }
        }
    }
}