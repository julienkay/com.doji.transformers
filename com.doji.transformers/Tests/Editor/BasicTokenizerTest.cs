using NUnit.Framework;
using System.Collections.Generic;

namespace Doji.AI.Transformers.Editor.Tests {

    public class BasicTokenizerTest {

        [Test]
        public void TestEncodeSimple() {
            BasicTokenizer tokenizer = new BasicTokenizer();
            var tokens = tokenizer.Tokenize("a cat");
            List<string> expected = new List<string>() { "a", "cat" };
            CollectionAssert.AreEqual(expected, tokens);
        }

    }
}