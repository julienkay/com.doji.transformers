namespace Doji.AI.Transformers {

    public class TokenString : Token {

        public TokenString(string content) {
            Content = content;
        }

        public override string ToString() {
            return Content;
        }
    }
}