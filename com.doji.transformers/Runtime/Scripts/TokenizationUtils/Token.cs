namespace Doji.AI.Transformers {

    public abstract class Token {

        public string Content { get; set; }

        public static implicit operator string(Token t) => t.Content;

        public override string ToString() {
            return Content;
        }
    }
}