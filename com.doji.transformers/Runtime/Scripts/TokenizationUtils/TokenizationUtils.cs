using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace Doji.AI.Transformers {

    internal static class TokenizationUtils {

        /// <summary>
        /// Returns list of utf-8 byte and a mapping to unicode strings.
        /// We specifically avoid mapping to whitespace/control
        /// characters the bpe code barfs on.
        /// 
        /// The reversible bpe codes work on unicode strings.
        /// This means you need a large # of unicode characters in your
        /// vocab if you want to avoid UNKs. When you're at something
        /// like a 10B token dataset you end up needing around 5K for
        /// decent coverage. This is a significant percentage of your
        /// normal, say, 32K bpe vocab.To avoid that, we want lookup
        /// tables between utf-8 bytes and unicode strings.
        /// 
        /// TODO: cache this
        /// </summary>
        public static Dictionary<int, char> BytesToUnicode() {
            List<int> bs = GetRange(33, 127) // ! to ~
                .Concat(GetRange(161, 173))  // ¡ to ¬
                .Concat(GetRange(174, 256))  // ® to ÿ
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

        private static IEnumerable<int> GetRange(int start, int end) {
            return Enumerable.Range(start, end - start);
        }

        /// <summary>
        /// Checks whether char <paramref name="c"/> is a whitespace character.
        /// </summary>
        public static bool IsWhitespace(char c) {
            // \t, \n, and \r are treated as whitespace since they are generally considered as such.
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                return true;
            }

            // Additional check for Unicode category "Zs" (space separators).
            if (char.GetUnicodeCategory(c) == UnicodeCategory.SpaceSeparator) {
                return true;
            }

            return false;
        }

        /// <summary>
        /// Checks whether char <paramref name="c"/> is a control character.
        /// </summary>
        public static bool IsControl(char c) {
            // These are technically control characters, but we count them as whitespace characters.
            if (c == '\t' || c == '\n' || c == '\r') {
                return false;
            }

            return char.IsControl(c);
        }

        /// <summary>
        /// Checks whether `char` is a punctuation character.
        /// </summary>
        public static bool IsPunctuation(char c) {
            int cp = Convert.ToInt32(c);

            // We treat all non-letter/number ASCII as punctuation.
            // Characters such as "^", "$", and "`" are not in the Unicode
            // Punctuation class but we treat them as punctuation anyways, for
            // consistency.
            if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
                return true;
            }

            return char.GetUnicodeCategory(c).IsPunctuation();
        }

        private static bool IsPunctuation(this UnicodeCategory cat) {
            return cat switch {
                UnicodeCategory.ClosePunctuation or
                UnicodeCategory.ConnectorPunctuation or
                UnicodeCategory.DashPunctuation or
                UnicodeCategory.FinalQuotePunctuation or
                UnicodeCategory.InitialQuotePunctuation or
                UnicodeCategory.OpenPunctuation or
                UnicodeCategory.OtherPunctuation => true,
                _ => false,
            };
        }

        /// <summary>
        /// Checks whether the last character in text is one of a
        /// punctuation, control or whitespace character.
        /// </summary>
        public static bool IsEndOfWord(string text) {
            char lastChar = text[text.Length - 1];
            return IsControl(lastChar) || IsPunctuation(lastChar) || IsWhitespace(lastChar);
        }

        /// <summary>
        /// Checks whether the first character in text is one of a
        /// punctuation, control or whitespace character.
        /// </summary>
        public static bool IsStartOfWord(string text) {
            char firstChar = text[0];
            return IsControl(firstChar) || IsPunctuation(firstChar) || IsWhitespace(firstChar);
        }

        /// <summary>
        /// Inserts one token to an ordered list if it does not already exist.
        /// Note: <paramref name="tokenList"/> must be sorted.
        /// </summary>
        public static void InsertOneTokenToOrderedList(List<string> tokenList, string newToken) {
            int insertionIndex = tokenList.BinarySearch(newToken, StringComparer.Ordinal);

            // Checks if newToken is already in the ordered tokenList
            if (insertionIndex >= 0 && insertionIndex < tokenList.Count && tokenList[insertionIndex] == newToken) {
                // newToken is in tokenList, don't add
                return;
            } else {
                // bitwise complement to get the correct insertion point if the item is not found in the list.
                tokenList.Insert(~insertionIndex, newToken);
            }
        }
    }
}