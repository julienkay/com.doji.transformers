using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using static Doji.AI.Transformers.TokenizationUtils;

namespace Doji.AI.Transformers {

    public class BasicTokenizer {

        /// <summary>
        /// Whether or not to lowercase the input when tokenizing.
        /// </summary>
        private bool _doLowerCase { get; }

        /// <summary>
        /// Collection of tokens which will never be split during tokenization.
        /// </summary>
        private List<string> _neverSplit { get; }

        /// <summary>
        /// Whether or not to tokenize Chinese characters.
        /// </summary>
        private bool _tokenizeChineseChars { get; }

        /// <summary>
        /// Whether or not to strip all accents.
        /// If this option is not specified, then it will be determined
        /// by the value for `lowercase` (as in the original BERT).
        /// </summary>
        private bool? _stripAccents { get; }

        /// <summary>
        /// In some instances we want to skip the basic punctuation splitting
        /// so that later tokenization can capture the full context of the words,
        /// such as contractions.
        /// </summary>
        private bool _doSplitOnPunc { get; }
 
        public BasicTokenizer(
            bool doLowerCase = true,
            List<string> neverSplit = null,
            bool tokenizeChineseChars = false,
            bool? stripAccents = null,
            bool doSPlitOnPunc = true
        ) {
            _doLowerCase = doLowerCase;
            _neverSplit = neverSplit ?? new List<string>();
            _tokenizeChineseChars = tokenizeChineseChars;
            _stripAccents = stripAccents;
            _doSplitOnPunc = doSPlitOnPunc;
        }

        public List<string> Tokenize(string text) {
            text = CleanText(text);

            //TODO: implement tokenize chinese chars
            if (_tokenizeChineseChars) {
                throw new NotImplementedException("BasicTokenizer currently does not support chinese characters.");
            }

            // prevents treating the same character with different unicode codepoints as different characters
            string unicodeNormalized = text.Normalize(NormalizationForm.FormC);
            List<string> origTokens = WhitespaceTokenize(unicodeNormalized);
            List<string> splitTokens = new List<string>();

            foreach (string token in origTokens) {
                if (!_neverSplit.Contains(token)) {
                    string processedToken = token;
                    if (_doLowerCase) {
                        processedToken = token.ToLower();
                        if (_stripAccents != false) {
                            processedToken = RunStripAccents(processedToken);
                        }
                        //.AddRange(RunSplitOnPunc(processedToken, _neverSplit));
                    } else if (_stripAccents == true) {
                        processedToken = RunStripAccents(processedToken);
                    }
                    splitTokens.AddRange(RunSplitOnPunc(processedToken, _neverSplit));
                }
            }

            var outputTokens = WhitespaceTokenize(string.Join(" ", splitTokens));
            return outputTokens;
        }

        /// <summary>
        /// Strips accents from a piece of text.
        /// </summary>
        private static string RunStripAccents(string text) {
            text = text.Normalize(NormalizationForm.FormD);
            StringBuilder output = new StringBuilder();

            foreach (char c in text) {
                UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(c);
                if (cat == UnicodeCategory.NonSpacingMark) {
                    continue;
                }
                output.Append(c);
            }

            return output.ToString();
        }

        /// <summary>
        /// Splits punctuation on a piece of text.
        /// </summary>
        private List<string> RunSplitOnPunc(string text, List<string> neverSplit = null) {
            if (!_doSplitOnPunc || (neverSplit != null && neverSplit.Contains(text))) {
                return new List<string> { text };
            }

            List<char> chars = new List<char>(text.ToCharArray());
            int i = 0;
            bool startNewWord = true;
            List<List<char>> output = new List<List<char>>();

            while (i < chars.Count) {
                char currentChar = chars[i];

                if (IsPunctuation(currentChar)) {
                    output.Add(new List<char> { currentChar });
                    startNewWord = true;
                } else {
                    if (startNewWord) {
                        output.Add(new List<char>());
                    }

                    startNewWord = false;
                    output[^1].Add(currentChar);
                }

                i++;
            }

            return output.Select(x => new string(x.ToArray())).ToList();
        }

        /// <summary>
        /// Performs invalid character removal and whitespace cleanup on text.
        /// </summary>
        private static string CleanText(string text) {
            StringBuilder output = new StringBuilder();

            foreach (char c in text) {
                int cp = Convert.ToInt32(c);

                if (cp == 0 || cp == 0xFFFD || IsControl(c)) {
                    continue;
                }

                if (IsWhitespace(c)) {
                    output.Append(" ");
                } else {
                    output.Append(c);
                }
            }

            return output.ToString();
        }

        /// <summary>
        /// Runs basic whitespace cleaning and splitting on a piece of text.
        /// </summary>
        private static List<string> WhitespaceTokenize(string text) {
            text = text.Trim();

            if (string.IsNullOrEmpty(text)) {
                return new List<string>();
            }

            string[] tokens = text.Split();
            return new List<string>(tokens);
        }
    }
}