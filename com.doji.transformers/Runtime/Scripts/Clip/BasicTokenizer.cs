using System.Collections.Generic;

namespace Doji.AI.Transformers {

    public class BasicTokenizer {

        /// <summary>
        /// Whether or not to lowercase the input when tokenizing.
        /// </summary>
        private bool DoLowerCase { get; }

        /// <summary>
        /// Collection of tokens which will never be split during tokenization.
        /// </summary>
        private List<string> NeverSplit { get; }

        /// <summary>
        /// Whether or not to tokenize Chinese characters.
        /// </summary>
        public bool TokenizeCHineseChars { get; }

        /// <summary>
        /// Whether or not to strip all accents.
        /// If this option is not specified, then it will be determined
        /// by the value for `lowercase` (as in the original BERT).
        /// </summary>
        private bool? StripAccents { get; }

        /// <summary>
        /// In some instances we want to skip the basic punctuation splitting
        /// so that later tokenization can capture the full context of the words,
        /// such as contractions.
        /// </summary>
        private bool DoSplitOnPunc { get; }
 
        public BasicTokenizer(
            bool doLowerCase = true,
            List<string> neverSplit = null,
            bool tokenizeChineseChars = true,
            bool? stripAccents = null,
            bool doSPlitOnPunc = true
        ) {
            DoLowerCase = doLowerCase;
            NeverSplit = neverSplit;
            TokenizeCHineseChars = tokenizeChineseChars;
            StripAccents = stripAccents;
            DoSplitOnPunc = doSPlitOnPunc;
        }

        public void Tokenize(string text, bool neverSplit) {
            throw new System.NotImplementedException();
        }
    }
}