using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;

namespace Doji.AI.Transformers.Editor.Tests {

    public class ClipTokenizerTest {

        [Test]
        public void TestTokenize() {
            ClipTokenizer t = CreateTokenizer();
            List<string> tokens = t.Tokenize("lower newer");

            List<string> exptected = new List<string>() { "lo", "w", "er</w>", "n", "e", "w", "er</w>" };
            CollectionAssert.AreEqual(tokens, exptected);
        }

        [Test]
        public void TestEncode() {
            ClipTokenizer t = CreateTokenizer();
            var encoding = t.Encode("lower newer");

            Assert.IsTrue(encoding.ContainsKey("input_ids"), "Encoded ids not found in 'input_ids'.");

            object encodedIds = encoding["input_ids"];
            Assert.IsTrue(encodedIds is ICollection, "Unexpected type for encoded text.");

            List<int> exptected = new List<int>() { 21, 10, 2, 16, 9, 3, 2, 16, 22 };
            CollectionAssert.AreEqual(encodedIds as ICollection, exptected);
        }

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