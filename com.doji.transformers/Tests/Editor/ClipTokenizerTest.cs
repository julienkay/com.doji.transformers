using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

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

        private static List<string> BatchInput = new List<string>() { "lower newer", "lone loner", "new low never hover" };
        private static List<string> RoundtripInput = new List<string>() { "lower newer" };

        public static IEnumerable EncodeBatchTestData {
            get {
                yield return new TestCaseData(BatchInput, Padding.None).Returns(
                    new List<List<int>>() {
                        new List<int> { 21, 10, 2, 16, 9, 3, 2, 16, 22 },
                        new List<int> { 21, 10, 9, 20, 10, 9, 16, 22 },
                        new List<int> { 21, 9, 3, 12, 15, 9, 3, 20, 16, 20, 1, 20, 16, 22 }
                    }
                );
                yield return new TestCaseData(BatchInput, Padding.Longest).Returns(
                    new List<List<int>>() {
                        new List<int> { 21, 10, 2, 16, 9,  3,  2, 16, 22, 22, 22, 22, 22, 22 },
                        new List<int> { 21, 10, 9, 20, 10, 9, 16, 22, 22, 22, 22, 22, 22, 22 },
                        new List<int> { 21,  9, 3, 12, 15, 9,  3, 20, 16, 20,  1, 20, 16, 22 }
                    }
                );
                yield return new TestCaseData(BatchInput, Padding.MaxLength).Returns(
                    new List<List<int>>() {
                        new List<int> {
                            21, 10, 2, 16, 9, 3, 2, 16, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22
                        },
                        new List<int> {
                            21, 10, 9, 20, 10, 9, 16, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22
                        },
                        new List<int> {
                            21, 9, 3, 12, 15, 9, 3, 20, 16, 20, 1, 20, 16, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                            22, 22, 22, 22
                        }
                    }
                );
            }
        }

        private static Dictionary<string, int> VocabTokens = new Dictionary<string, int>() {
            { "l", 0 }, { "o", 1 }, { "w", 2 }, { "e", 3 }, { "r", 4 }, { "s", 5 }, { "t", 6 }, { "i", 7 }, { "d", 8 }, { "n", 9 },
            { "lo", 10 }, { "l</w>", 11 }, { "w</w>", 12 }, { "r</w>", 13 }, { "t</w>", 14 }, { "low</w>", 15 }, { "er</w>", 16 },
            { "lowest</w>", 17 }, { "newer</w>", 18 }, { "wider",  19 }, { "<unk>", 20 }, { "<|startoftext|>", 21 }, { "<|endoftext|>", 22 }
        };

        private static string Merges = "#version: 0.2\n" + "l o\n" + "lo w</w>\n" + "e r</w>\n";

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
            Assert.IsTrue(encoding is InputEncoding, "Unexpected type for encoding.");
        }

        [Test]
        [TestCaseSource(nameof(EncodeTestData))]
        public IEnumerable<int> TestEncode(string text) {
            ClipTokenizer t = CreateTokenizer();
            InputEncoding encoding = t.Encode(text) as InputEncoding;
            var encodedIds = encoding.InputIds;
            return encodedIds;
        }

        [Test]
        [TestCaseSource(nameof(EncodeBatchTestData))]
        public List<List<int>> TestEncodeBatch(List<string> prompts, Padding padding) {
            ClipTokenizer t = CreateTokenizer();
            BatchEncoding encoding = t.Encode<BatchInput>(prompts, padding: padding, maxLength: 77) as BatchEncoding;
            var encodedIds = encoding["input_ids"] as List<List<int>>;
            return encodedIds;
        }

        [Test]
        public void TestEncodeRoundtrip([ValueSource(nameof(RoundtripInput))] string prompt) {
            ClipTokenizer t = CreateTokenizer();
            var result = t.Decode(t.Encode(prompt).InputIds.ToList(), skipSpecialTokens: true);
            Assert.That(result, Is.EqualTo(prompt));
        }

        /// <summary>
        /// Creates a basic ClipTokenizer with a reduced vocabulary.
        /// </summary>
        /// <returns></returns>
        private ClipTokenizer CreateTokenizer() {
            Vocab vocab = new Vocab(VocabTokens);
            TokenizerConfig config = new TokenizerConfig();
            config.UnkToken = "<unk>";
            ClipTokenizer tokenizer = new ClipTokenizer(vocab, Merges, config);
            return tokenizer;
        }
    }
}