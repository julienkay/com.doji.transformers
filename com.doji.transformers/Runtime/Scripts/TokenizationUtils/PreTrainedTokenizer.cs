namespace Doji.AI.Transformers {

    /// <summary>
    /// Base class for all slow tokenizers.
    /// Handle all the shared methods for tokenization and special
    /// tokens as well as methods downloading/caching/loading
    /// pretrained tokenizers as well as adding tokens to the vocabulary.

    /// This class also contain the added tokens in a unified way on top
    /// of all tokenizers so we don't have to handle thespecific vocabulary
    /// augmentation methods of the various underlying dictionary structures
    /// (BPE, sentencepiece...).
    /// </summary>
    public class PreTrainedTokenizer : PreTrainedTokenizerBase {

        private Trie _tokensTrie;

        public PreTrainedTokenizer() {

        }
    }
}
