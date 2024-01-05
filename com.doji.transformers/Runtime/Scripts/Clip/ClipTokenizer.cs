using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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
            string[] merges,
            string errors = "replace",
            string unkToken = "<|endoftext|>",
            string bosToken = "<|startoftext|>",
            string eosToken = "<|endoftext|>",
            string padToken = "<|endoftext|>") : this
        (
            vocab,
            merges,
            errors,
            unkToken: new AddedToken(unkToken),
            bosToken: new AddedToken(bosToken),
            eosToken: new AddedToken(eosToken),
            padToken: new AddedToken(padToken)
        ) { }

        public ClipTokenizer(
            Vocab vocab,
            string[] merges,
            string errors,
            AddedToken unkToken,
            AddedToken bosToken,
            AddedToken eosToken,
            AddedToken padToken)
        {
            Initialize(
                vocab,
                merges,
                errors,
                unkToken,
                bosToken,
                eosToken,
                padToken
            );
        }

        protected void Initialize(
            Vocab vocab,
            string[] merges,
            string errors,
            AddedToken unkToken,
            AddedToken bosToken,
            AddedToken eosToken,
            AddedToken padToken,
            Dictionary<int, AddedToken> addedTokensDecoder = null,
            int modelMaxLength = int.MaxValue,
            Side paddingSide = Side.Right,
            Side truncationSide = Side.Right,
            List<string> modelInputNames = null,
            bool cleanUpTokenizationSpaces = true,
            bool splitSpecialTokens = false)
        {
            BosToken = bosToken;
            EosToken = eosToken;
            UnkToken = unkToken;
            PadToken = padToken;

            // TODO: BasicTokenizer only a fallback, implement ftfy.fix_text instead?
            _nlp = new BasicTokenizer();
            _fixText = null;
            Vocab = vocab;
            _byteEncoder = BytesToUnicode();
            _byteDecoder = _byteEncoder.ToDictionary(x => x.Value, x => x.Key);

            // Extract relevant lines and split them into tuples
            List<Tuple<string, string>> bpeMerges = merges
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

            base.Initialize(
                modelMaxLength,
                paddingSide,
                truncationSide,
                modelInputNames,
                cleanUpTokenizationSpaces,
                splitSpecialTokens,
                unkToken,
                bosToken,
                eosToken,
                padToken
            );
        }

        protected override Dictionary<string, int> GetVocab() {
            return Vocab.Encoder;
        }

        /// <summary>
        /// Returns list of utf-8 byte and a mapping to unicode strings.
        /// We specifically avoids mapping to whitespace/control
        /// characters the bpe code barfs on.
        /// 
        /// The reversible bpe codes work on unicode strings.  This means you need a large # of unicode characters in your vocab
        /// if you want to avoid UNKs.When you're at something like a 10B token dataset you end up needing around 5K for
        /// decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab.To avoid that, we want lookup
        /// tables between utf-8 bytes and unicode strings.
        /// 
        /// TODO: Look into C# equivalent for @lru_cache()
        /// </summary>
        private static Dictionary<int, char> BytesToUnicode() {
            List<int> bs = Enumerable.Range('!', '~' + 1)
                .Concat(Enumerable.Range('¡', '¬' + 1))
                .Concat(Enumerable.Range('®', 'ÿ' + 1))
                .ToList();

            List<int> cs = new List<int>(bs);
            int n = 0;

            for (int b = 0; b < 256; b++) {
                if (!bs.Contains(b)) {
                    bs.Add(b);
                    cs.Add(256 + n);
                    n++;
                }
            }

            Dictionary<int, char> result = new Dictionary<int, char>();
            for (int i = 0; i < bs.Count; i++) {
                result.Add(bs[i], (char)cs[i]);
            }

            return result;
        }

        /// <summary>
        /// Return set of symbol pairs in a word.
        /// Word is represented as tuple of symbols (symbols being variable-length strings).
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
                var token = string.Join("", Encoding.UTF8.GetBytes(match.ToString()).Select(b => _byteEncoder[b]));
                bpeTokens.AddRange(bpe(token).Split(' '));
            }

            return bpeTokens;
        }
    }
}