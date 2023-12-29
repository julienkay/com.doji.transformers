using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Unity.Sentis.Layers;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A clip tokenizer to tokenize text.
    /// Based on byte-level Byte-Pair-Encoding.
    /// </summary>
    public class ClipTokenizer : PreTrainedTokenizer {

        private BasicTokenizer _nlp;
        private Vocab _vocab;
        private Dictionary<int, char> _byteEncoder;
        private Dictionary<char, int> _byteDecoder;
        private Dictionary<Tuple<string, string>, int> bpeRanks;
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
            AddedToken padToken) : base()
        {
            BosToken = bosToken;
            EosToken = eosToken;
            UnkToken = unkToken;
            PadToken = padToken;

            // ftfy.fix_text not implemented, using BasicTokenizer instead
            _nlp = new BasicTokenizer();
            _vocab = vocab;
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
            bpeRanks = bpeMerges
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
    }
}