namespace Doji.AI.Transformers {

    /// <summary>
    /// AddedToken represents a token to be added to a Tokenizer
    /// An AddedToken can have special options defining the
    /// way it should behave.
    /// The `normalized` will default to `not special` if it is not specified,
    /// similarly to the definition in `tokenizers`.
    /// </summary>
    public class AddedToken {

        public string Content { get; }
        public bool Single_Word { get; }
        public bool Lstrip { get; }
        public bool Rstrip { get; }
        public bool Special { get; }
        public bool Normalized { get; }


        public AddedToken(
            string content,
            bool single_word = false,
            bool lstrip = false,
            bool rstrip = false,
            bool special = false,
            bool? normalized = null)
        {
            Content = content;
            Single_Word = single_word;
            Lstrip = lstrip;
            Rstrip = rstrip;
            Special = special;
            Normalized = normalized ?? !special;
        }

        public static implicit operator string(AddedToken t) => t.Content;
    }
}