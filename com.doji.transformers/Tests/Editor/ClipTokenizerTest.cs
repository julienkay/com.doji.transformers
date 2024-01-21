using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// ClipTokenizer test with a reduced vocabulary.
    /// </summary>
    public class ClipTokenizerTest {

        public static IEnumerable TokenizeTestData {
            get {
                yield return new TestCaseData("lower newer").Returns(new List<string>() { "lo", "w", "er</w>", "n", "e", "w", "er</w>" });
                yield return new TestCaseData("lone loner").Returns(new List<string>() { "lo", "n", "e</w>", "lo", "n", "er</w>" });
                yield return new TestCaseData("new low never hover").Returns(new List<string>() { "n", "e", "w</w>", "low</w>", "n", "e", "v", "er</w>", "h", "o", "v", "er</w>" });
            }
        }
        public static IEnumerable EncodeTestData {
            get {
                yield return new TestCaseData("lower newer").Returns(new List<int>() { 21, 10, 2, 16, 9, 3, 2, 16, 22 });
                yield return new TestCaseData("lone loner").Returns(new List<int>() { 21, 10, 9, 20, 10, 9, 16, 22 });
                yield return new TestCaseData("new low never hover").Returns(new List<int>() { 21, 9, 3, 12, 15, 9, 3, 20, 16, 20, 1, 20, 16, 22 });
            }
        }

        [Test]
        [TestCaseSource(nameof(TokenizeTestData))]
        public List<string> TestTokenize(string text) {
            ClipTokenizer t = CreateTokenizer();
            List<string> tokens = t.Tokenize(text);
            return tokens;
        }

        [Test]
        public void TestEncodeType() {
            ClipTokenizer t = CreateTokenizer();
            var encoding = t.Encode("lower newer");

            Assert.IsTrue(encoding.ContainsKey("input_ids"), "Encoded ids not found in 'input_ids'.");
            object encodedIds = encoding["input_ids"];
            Assert.IsTrue(encodedIds is ICollection, "Unexpected type for encoded text.");
        }

        [Test]
        [TestCaseSource(nameof(EncodeTestData))]
        public List<int> TestEncode(string text) {
            ClipTokenizer t = CreateTokenizer();
            var encodedIds = t.Encode(text).InputIds;
            return encodedIds;
        }

        [Test]
        public void TestEncodeBatch() {
            ClipTokenizer t = CreateTokenizer();
            List<string> prompts = new List<string>() { "lower newer", "lone loner", "new low never hover" };
            var encodedIds = t.Encode<BatchInput>(prompts).InputIds;

            //TODO: verify whether we should actually return a flattened list here and not individual lists per sequence
            List<int> exptected = new List<int>() { 21, 10, 2, 16, 9, 3, 2, 16, 22, 21, 10, 9, 20, 10, 9, 16, 22, 21, 9, 3, 12, 15, 9, 3, 20, 16, 20, 1, 20, 16, 22 };
            CollectionAssert.AreEqual(encodedIds, exptected);
        }

        [Test]
        public void TestEncodeBatchPadding() {
            ClipTokenizer t = CreateTokenizer();
            List<string> prompts = new List<string>() { "lower newer", "lone loner", "new low never hover" };
            var encodedIds = t.Encode<BatchInput>(prompts, padding: Padding.MaxLength, maxLength: 77).InputIds;

            List<int> exptected = new List<int>() { 21, 10, 2, 16, 9, 3, 2, 16, 22, 22, 22, 22, 22, 22, 21, 10, 9, 20, 10, 9, 16, 22, 22, 22, 22, 22, 22, 22, 21, 9, 3, 12, 15, 9, 3, 20, 16, 20, 1, 20, 16, 22 };
            CollectionAssert.AreEqual(encodedIds, exptected);
        }

        /// <summary>
        /// Creates a basic ClipTokenizer with a reduced vocabulary.
        /// </summary>
        /// <returns></returns>
        private ClipTokenizer CreateTokenizer() {
            string[] vocabList = { "l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "lo", "l</w>", "w</w>", "r</w>", "t</w>", "low</w>", "er</w>", "lowest</w>", "newer</w>", "wider", "<unk>", "<|startoftext|>", "<|endoftext|>" };
            string merges = "#version: 0.2\n" + "l o\n" + "lo w</w>\n" + "e r</w>\n";
            Dictionary<string, int> vocabTokens = new Dictionary<string, int>();
            for (int i = 0; i < vocabList.Length; i++) {
                vocabTokens[vocabList[i]] = i;
            }

            Vocab vocab = new Vocab(vocabTokens);
            ClipTokenizer tokenizer = new ClipTokenizer(vocab, merges);
            return tokenizer;
        }
    }
}