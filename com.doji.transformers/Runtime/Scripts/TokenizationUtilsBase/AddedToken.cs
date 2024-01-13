using Newtonsoft.Json;

namespace Doji.AI.Transformers {

    /// <summary>
    /// AddedToken represents a token to be added to a Tokenizer
    /// An AddedToken can have special options defining the
    /// way it should behave.
    /// The `normalized` will default to `not special` if it is not specified,
    /// similarly to the definition in `tokenizers`.
    /// </summary>
    public class AddedToken : Token {

        [JsonProperty("single_word")]
        public bool SingleWord { get; set; }

        [JsonProperty("lstrip")]
        public bool Lstrip { get; set; }

        [JsonProperty("rstrip")]
        public bool Rstrip { get; set; }

        public bool Special { get; set; }

        [JsonProperty("normalized")]
        public bool Normalized { get; set; }

        [JsonProperty("__type")]
        public string Type { get; set; }

        public AddedToken(
            string content,
            bool singleWord = false,
            bool lstrip = false,
            bool rstrip = false,
            bool special = false,
            bool? normalized = null)
        {
            Content = content;
            SingleWord = singleWord;
            Lstrip = lstrip;
            Rstrip = rstrip;
            Special = special;
            Normalized = normalized ?? !special;
        }
    }
}