using NUnit.Framework;

namespace Doji.AI.Transformers.Editor.Tests {

    public class TrieTest {

        [Test]
        public void TestTrieAddSingle() {
            Trie trie = new Trie();
            trie.Add("Hello 友達");

            string expected = "{\"H\":{\"e\":{\"l\":{\"l\":{\"o\":{\" \":{\"友\":{\"達\":{\"\":1}}}}}}}}}";
            Assert.AreEqual(expected, trie.ToString(), "Trie does not match expected value after adding 'Hello 友達'.");
        }

        [Test]
        public void TestTrieAddMultiple() {
            Trie trie = new Trie();
            trie.Add("Hello 友達");
            trie.Add("Hello");

            string expected = "{\"H\":{\"e\":{\"l\":{\"l\":{\"o\":{\"\":1,\" \":{\"友\":{\"達\":{\"\":1}}}}}}}}}";
            Assert.AreEqual(expected, trie.ToString(), "Trie does not match expected value after adding 'Hello'.");
        }

        [Test]
        public void TestTrieIdempotent() {
            Trie trie = new Trie();
            trie.Add("Hello World 123 !§$%");
            string first = trie.ToString();
            trie.Add("Hello World 123 !§$%");
            string second = trie.ToString();
            
            Assert.AreEqual(first, second, "Trie is not idempotent. Adding twice the same word changed the trie.");
        }
    }
}