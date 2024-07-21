using static Doji.AI.Transformers.TokenizationUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A  GPT-2 tokenizer.
    /// Based on byte-level Byte-Pair-Encoding.
    /// </summary>
    public class GPT2Tokenizer : PreTrainedTokenizer {

        private Vocab Vocab { get; set; }
        private bool AddBosToken { get; set; }

        private Dictionary<int, char> _byteEncoder;
        private Dictionary<char, int> _byteDecoder;
        private Dictionary<Tuple<string, string>, int> _bpeRanks;
        private Dictionary<string, string> _cache;
        private Regex _pat;

        public GPT2Tokenizer(
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
            config.BosToken ??= new AddedToken("<|endoftext|>");
            config.EosToken ??= new AddedToken("<|endoftext|>");
            config.AddPrefixSpace = false;
            config.AddBosToken = AddBosToken = false;

            Vocab = vocab;
            _byteEncoder = BytesToUnicode();
            _byteDecoder = _byteEncoder.ToDictionary(x => x.Value, x => x.Key);

            // Extract relevant lines and split them into tuples
            var m = merges
                .Trim()
                .Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

            List<Tuple<string, string>> bpeMerges = m
                .Skip(1).Take(m.Length - 2) // m[1:-1] 
                .Select(line => line.Split())
                .Select(parts => Tuple.Create(parts[0], parts[1]))
                .ToList();

            // Create the dictionary with bpe ranks
            _bpeRanks = bpeMerges
                .Select((merge, index) => new { Merge = merge, Index = index })
                .ToDictionary(item => item.Merge, item => item.Index);

            _cache = new();

            _pat = new Regex(
                @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            );

            base.Initialize(config);
        }

        protected override Dictionary<string, int> GetVocab() {
            return Vocab.Encoder;
        }

        protected override List<int> BuildInputsWithSpecialTokens(List<int> TokenIds0, List<int> TokenIds1 = null) {
            List<int> bosTokenIds = AddBosToken ? new List<int> { BosTokenId.Value } : new();

            var output = bosTokenIds.Concat(TokenIds0);

            if (TokenIds1 == null) {
                return output.ToList();
            }

            return output.Concat(bosTokenIds).Concat(TokenIds1).ToList();
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

            if (!AddBosToken) {
                base.GetSpecialTokensMask(tokenIds0, tokenIds1, false);
            }


            if (tokenIds1 == null) {
                return ONE.Concat(new int[tokenIds0.Count]).ToList();
            }

            return ONE.Concat(new int[tokenIds0.Count]).Concat(ONE).Concat(new int[tokenIds1.Count]).ToList();
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